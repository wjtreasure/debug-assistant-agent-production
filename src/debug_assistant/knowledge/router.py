from __future__ import annotations

"""Deterministic Incident capability detection, routing, and query building."""

from typing import Any, Literal
import re

from pydantic import BaseModel, ConfigDict, Field

from .contracts import KnowledgeQuery, KnowledgeSource, PriorContext
from .rag import DomainKnowledgeRetriever
from .static_graph import InMemoryStaticGraph, StaticKnowledgeGraphTools


CapabilityState = Literal["AVAILABLE", "EMPTY", "UNAVAILABLE", "FAILED"]


_TOOL_CAPABILITY_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "check_service_connectivity": ("service_connectivity",),
    "get_service_topology": ("structured_service_topology",),
    "get_resources": ("kubernetes_resources",),
    "get_app_yaml": ("application_yaml",),
    "describe_resource": ("resource_descriptions",),
    "get_alerts": ("telemetry",),
    "get_error_logs": ("telemetry",),
    "get_service_dependencies": ("structured_service_topology",),
    "check_node_service_status": ("resource_descriptions",),
    "get_cluster_configuration": ("resource_descriptions",),
    "list_code_files": ("application_code",),
    "repo_tree": ("application_code",),
    "grep": ("application_code",),
    "read_file": ("application_code",),
    "symbol_search": ("application_code",),
    "code_search": ("application_code",),
    "inspect_symbol_context": ("application_code",),
}


class IncidentEntities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    service: str | None = None
    module: str | None = None
    version: str | None = None
    repo: str | None = None
    error_code: str | None = None
    request_id: str | None = None
    namespace: str | None = None


class CapabilitySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    logs_available: bool = False
    metrics_available: bool = False
    trace_available: bool = False
    k8s_available: bool = False
    code_available: bool = False
    domain_rag_available: bool = False
    incident_memory_available: bool = False
    static_kg_available: bool = False
    available_code_tools: tuple[str, ...] = ()
    available_knowledge_sources: tuple[KnowledgeSource, ...] = ()
    # Runtime-visible lifecycle state.  The legacy booleans above remain as a
    # compatibility projection for callers that only need capability presence.
    # ``states`` is the authoritative planner/evaluation surface.
    states: dict[str, CapabilityState] = Field(default_factory=dict)

    def capability_state(self, capability: str) -> CapabilityState:
        """Return a fail-closed state for a named data capability."""
        explicit = self.states.get(str(capability))
        if explicit is not None:
            return explicit
        # Compatibility instances created by older callers do not have the
        # state map.  Presence is still represented truthfully; no implicit
        # EMPTY/AVAILABLE claim is made for an unknown capability.
        legacy = {
            "domain_rag": self.domain_rag_available,
            "incident_memory": self.incident_memory_available,
            "static_kg": self.static_kg_available,
            "application_code": bool(self.available_code_tools),
        }
        if capability in legacy:
            return "AVAILABLE" if legacy[capability] else "UNAVAILABLE"
        return "UNAVAILABLE"

    def planner_tool_names(self, tools: Any) -> tuple[str, ...]:
        """Return only tools backed by currently AVAILABLE capabilities.

        ``finalize_diagnosis`` is a control operation, not a data capability,
        and is always visible.  A tool whose source is EMPTY, UNAVAILABLE, or
        FAILED is omitted so the Planner cannot turn a known boundary into a
        business conclusion or a repeated low-value call.
        """
        names: list[str] = []
        for spec in tools.specs():
            name = str(spec.name)
            if name == "finalize_diagnosis":
                names.append(name)
                continue
            if name == "knowledge_retrieval":
                if any(self.capability_state(source) == "AVAILABLE"
                       for source in self.available_knowledge_sources):
                    names.append(name)
                continue
            required = _TOOL_CAPABILITY_REQUIREMENTS.get(name)
            if required is None:
                # Unknown optional tools are not advertised by default.  This
                # keeps a new registry addition fail-closed until its source
                # semantics are explicitly mapped.
                continue
            if all(self.capability_state(source) == "AVAILABLE" for source in required):
                if name not in {"list_code_files", "inspect_symbol_context"}:
                    names.append(name)
                elif name in self.available_code_tools:
                    names.append(name)
        return tuple(dict.fromkeys(names))

    def observe_tool(self, tool_name: str, observation: Any) -> "CapabilitySnapshot":
        """Update one capability from a real tool Observation.

        A successful empty result is ``EMPTY``.  A missing snapshot record is
        ``UNAVAILABLE``; other execution/provider failures are ``FAILED``.
        Neither state is converted into a causal negative claim.
        """
        metadata = dict(getattr(observation, "metadata", {}) or {})
        if str(tool_name) == "knowledge_retrieval":
            diagnostics = dict(metadata.get("knowledge_diagnostics") or {})
            requested = tuple(str(item) for item in diagnostics.get("requested_sources", ()))
            if not requested:
                return self
            counts = dict(diagnostics.get("per_source_count") or {})
            reasons = str(diagnostics.get("reason") or "")
            states = dict(self.states)
            for source in requested:
                if f"{source}_unavailable" in reasons:
                    state: CapabilityState = "UNAVAILABLE"
                elif f"{source}_failed" in reasons:
                    state = "FAILED"
                elif not counts.get(source, 0):
                    state = "EMPTY"
                else:
                    state = "AVAILABLE"
                states[source] = state
            return self.model_copy(update={"states": states})
        required = _TOOL_CAPABILITY_REQUIREMENTS.get(str(tool_name))
        if not required:
            return self
        if metadata.get("status") == "UNAVAILABLE" or getattr(observation, "error_type", None) == "snapshot_unavailable":
            state: CapabilityState = "UNAVAILABLE"
        elif metadata.get("semantic_negative") is True:
            state = "EMPTY"
        elif bool(getattr(observation, "ok", False)):
            state = "AVAILABLE"
        else:
            state = "FAILED"
        states = dict(self.states)
        for capability in required:
            states[capability] = state
        return self.model_copy(update={"states": states})

    @classmethod
    def detect(cls, case: Any, tools: Any, *, source_workspace_available: bool = False,
               domain_rag: Any | None = None, knowledge_store: Any | None = None,
               static_graph: Any | None = None) -> "CapabilitySnapshot":
        evidence_sources = {str(item).casefold() for item in getattr(case, "evidence_sources", ())}
        get = lambda name: tools.get(name) is not None
        code_tools = tuple(
            name for name in ("repo_tree", "grep", "read_file", "symbol_search", "code_search")
            if get(name)
        ) if source_workspace_available else ()
        def source_state(source_names: set[str], tool_names: tuple[str, ...]) -> CapabilityState:
            if not source_names.intersection(evidence_sources):
                return "UNAVAILABLE"
            return "AVAILABLE" if any(get(name) for name in tool_names) else "FAILED"

        states: dict[str, CapabilityState] = {
            "structured_service_topology": source_state({"structured_service_topology"}, ("get_service_topology",)),
            "service_connectivity": source_state({"service_connectivity"}, ("check_service_connectivity",)),
            "kubernetes_resources": source_state({"kubernetes_resources"}, ("get_resources",)),
            "resource_descriptions": source_state({"resource_descriptions"}, ("describe_resource", "check_node_service_status", "get_cluster_configuration")),
            "application_yaml": source_state({"application_yaml"}, ("get_app_yaml",)),
            "telemetry": source_state({"telemetry", "logs", "error_logs", "alerts"}, ("get_alerts", "get_error_logs")),
            "logs": source_state({"telemetry", "logs", "error_logs"}, ("get_error_logs", "get_alerts")),
            "metrics": source_state({"telemetry", "metrics", "alerts"}, ("get_alerts",)),
            "trace": "UNAVAILABLE" if not {"telemetry", "trace", "traces"}.intersection(evidence_sources) else "FAILED",
            "application_code": (
                "UNAVAILABLE" if "application_code" not in evidence_sources or not source_workspace_available
                else ("AVAILABLE" if code_tools else "FAILED")
            ),
        }
        knowledge = []
        if domain_rag is not None:
            knowledge.append("domain_rag")
            states["domain_rag"] = "AVAILABLE"
        else:
            states["domain_rag"] = "UNAVAILABLE"
        if knowledge_store is not None:
            knowledge.append("incident_memory")
            states["incident_memory"] = "AVAILABLE"
        else:
            states["incident_memory"] = "UNAVAILABLE"
        if static_graph is not None:
            knowledge.append("static_kg")
            states["static_kg"] = "AVAILABLE"
        else:
            states["static_kg"] = "UNAVAILABLE"
        return cls(
            logs_available=states["logs"] == "AVAILABLE", metrics_available=states["metrics"] == "AVAILABLE",
            trace_available=states["trace"] == "AVAILABLE", k8s_available=states["kubernetes_resources"] == "AVAILABLE",
            code_available=states["application_code"] == "AVAILABLE",
            domain_rag_available=domain_rag is not None,
            incident_memory_available=knowledge_store is not None,
            static_kg_available=static_graph is not None,
            available_code_tools=code_tools,
            available_knowledge_sources=tuple(dict.fromkeys(knowledge)),
            states=states,
        )


class RouterDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    telemetry_priority: int = Field(ge=0, le=100)
    service_priority: int = Field(ge=0, le=100)
    runtime_priority: int = Field(ge=0, le=100)
    dependency_priority: int = Field(ge=0, le=100)
    code_priority: int = Field(ge=0, le=100)
    knowledge_needed: bool = False
    requested_knowledge_sources: tuple[KnowledgeSource, ...] = ()
    reason: str = ""


class IncidentEntityExtractor:
    """Extract syntax-shaped entities without a case or fault answer map."""

    _SERVICE = re.compile(r"\bservice(?:_name| name)?\s*[:=]?\s*([A-Za-z][A-Za-z0-9_.-]*)", re.I)
    _MODULE = re.compile(r"\bmodule\s*[:=]\s*([A-Za-z][A-Za-z0-9_.-]*)", re.I)
    _REPO = re.compile(r"\brepo(?:sitory)?\s*[:=]\s*([A-Za-z0-9_.\-/]+)", re.I)
    _VERSION = re.compile(r"\b(v\d+(?:\.\d+){0,3}|\d+\.\d+(?:\.\d+)?)\b", re.I)
    _ERROR = re.compile(r"\b(?:error|err|code|status(?:_code)?)\s*[:=]?\s*([A-Z][A-Z0-9_.-]{2,}|\d{3})\b", re.I)
    _REQUEST = re.compile(r"\b(?:request[_ -]?id|trace[_ -]?id)\s*[:=]\s*([A-Za-z0-9_.:-]+)", re.I)

    def extract(self, case: Any, *, structured: dict[str, Any] | None = None) -> IncidentEntities:
        structured = dict(structured or {})
        text = " ".join(str(getattr(case, item, "") or "") for item in ("summary", "system", "namespace"))
        pick = lambda pattern: (pattern.search(text).group(1) if pattern.search(text) else None)
        return IncidentEntities(
            service=structured.get("service") or pick(self._SERVICE),
            module=structured.get("module") or pick(self._MODULE),
            version=structured.get("version") or pick(self._VERSION),
            repo=structured.get("repo") or pick(self._REPO),
            error_code=structured.get("error_code") or pick(self._ERROR),
            request_id=structured.get("request_id") or pick(self._REQUEST),
            namespace=structured.get("namespace") or getattr(case, "namespace", None),
        )


class HybridRouter:
    """Capability-constrained priority hints; never emits a root cause."""

    def route(self, case: Any, entities: IncidentEntities, capabilities: CapabilitySnapshot, *, direct_evidence_available: bool = False) -> RouterDecision:
        summary = str(getattr(case, "summary", "") or "").casefold()
        explicit_direct = direct_evidence_available or any(
            marker in summary for marker in ("direct evidence", "verified evidence", "fully observed")
        )
        sources = capabilities.available_knowledge_sources
        needed = bool(sources) and not explicit_direct
        priorities = {
            "telemetry": 80 if capabilities.logs_available or capabilities.metrics_available or capabilities.trace_available else 10,
            "service": 75 if entities.service else 40,
            "runtime": 70 if capabilities.k8s_available else 20,
            "dependency": 80 if entities.service and (capabilities.static_kg_available or capabilities.k8s_available) else 30,
            "code": 85 if capabilities.code_available else 15,
        }
        if not needed:
            reason = "knowledge pre-retrieval skipped: no Knowledge source or direct evidence is available"
        else:
            reason = "knowledge pre-retrieval requested by available capability sources"
        return RouterDecision(
            telemetry_priority=priorities["telemetry"], service_priority=priorities["service"],
            runtime_priority=priorities["runtime"], dependency_priority=priorities["dependency"],
            code_priority=priorities["code"], knowledge_needed=needed,
            requested_knowledge_sources=sources, reason=reason,
        )


class KnowledgeQueryBuilder:
    def __init__(self, *, max_queries: int = 3, top_k: int = 5, token_budget: int | None = 1200) -> None:
        self.max_queries = max(1, int(max_queries))
        self.top_k = max(1, min(100, int(top_k)))
        self.token_budget = None if token_budget is None else max(1, int(token_budget))

    def build(self, case: Any, entities: IncidentEntities, decision: RouterDecision, *, evidence_gap: str = "", token_budget: int | None = None) -> tuple[KnowledgeQuery, ...]:
        if not decision.knowledge_needed or not decision.requested_knowledge_sources:
            return ()
        parts = [str(getattr(case, "summary", "") or "").strip()]
        for label, value in (
            ("service", entities.service), ("module", entities.module), ("version", entities.version),
            ("repo", entities.repo), ("error_code", entities.error_code), ("namespace", entities.namespace),
        ):
            if value:
                parts.append(f"{label}:{value}")
        if evidence_gap.strip():
            parts.append(f"evidence gap:{evidence_gap.strip()}")
        query = KnowledgeQuery(
            incident_id=str(getattr(case, "case_id")), query_text=" ".join(parts)[:4000],
            service=entities.service, module=entities.module, fault_type=None,
            software_version=entities.version, repo_commit=None,
            requested_sources=decision.requested_knowledge_sources,
            top_k=self.top_k,
            token_budget=max(1, int(token_budget if token_budget is not None else self.token_budget or 1)),
        )
        return (query,)


class KnowledgeCoordinator:
    """Fan-out/fan-in capability layer for Memory, RAG, and Static KG."""

    def __init__(self, *, knowledge_store: Any | None = None, domain_rag: DomainKnowledgeRetriever | None = None, static_graph: Any | None = None, max_per_source: int = 2) -> None:
        self.knowledge_store = knowledge_store
        self.domain_rag = domain_rag
        self.static_graph = static_graph
        self.max_per_source = max(1, int(max_per_source))

    @property
    def available_sources(self) -> tuple[KnowledgeSource, ...]:
        values = []
        if self.domain_rag is not None:
            values.append("domain_rag")
        if self.knowledge_store is not None:
            values.append("incident_memory")
        if self.static_graph is not None:
            values.append("static_kg")
        return tuple(values)

    def retrieve(self, query: KnowledgeQuery, *, mode: str = "lexical"):
        from .contracts import KnowledgeRetrievalDiagnostics, KnowledgeRetrievalResult
        started = __import__("time").monotonic()
        requested = query.requested_sources or self.available_sources
        candidates = []
        effective = []
        reasons = []
        counts: dict[str, int] = {}
        for source in requested:
            try:
                if source == "incident_memory":
                    if self.knowledge_store is None:
                        reasons.append("incident_memory_unavailable")
                        counts[source] = 0
                    else:
                        result = self.knowledge_store.retrieve(query)
                        rows = list(result.candidates)
                        candidates.extend(rows)
                        counts[source] = len(rows)
                        effective.append(source)
                elif source == "domain_rag":
                    if self.domain_rag is None:
                        reasons.append("domain_rag_unavailable")
                        counts[source] = 0
                    else:
                        result = self.domain_rag.retrieve(query, mode=mode)
                        rows = list(result.candidates)
                        candidates.extend(rows)
                        counts[source] = len(rows)
                        effective.append(source)
                        if result.diagnostics.degraded:
                            reasons.append(result.diagnostics.reason or "domain_rag_degraded")
                elif source == "static_kg":
                    if self.static_graph is None:
                        reasons.append("static_kg_unavailable")
                        counts[source] = 0
                    else:
                        service = query.service
                        if not service:
                            rows = []
                        else:
                            graph = self.static_graph
                            result = graph.get_neighbors(f"service:{service}", version=query.software_version)
                            rows = list(result.candidates)
                        candidates.extend(rows)
                        counts[source] = len(rows)
                        effective.append(source)
            except Exception as exc:
                reasons.append(f"{source}_failed:{type(exc).__name__}")
                counts[source] = 0
        deduped = []
        seen = set()
        for candidate in candidates:
            key = candidate.parent_id or candidate.candidate_id
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        deduped.sort(key=lambda item: (-(item.rerank_score if item.rerank_score is not None else item.retrieval_score or 0.0), item.candidate_id))
        groups: dict[str, list[Any]] = {}
        for candidate in deduped:
            groups.setdefault(candidate.source_type, []).append(candidate)
        ordered = []
        for index in range(max((len(value) for value in groups.values()), default=0)):
            for source in requested:
                rows = groups.get(source, [])
                if index < len(rows) and sum(item.source_type == source for item in ordered) < self.max_per_source:
                    ordered.append(rows[index])
        packed = []
        budget = query.token_budget * 4
        used = 0
        stop_reason = "candidates_exhausted"
        for candidate in ordered:
            remaining = budget - used
            if remaining <= 0:
                stop_reason = "token_ceiling"
                break
            if len(candidate.content) > remaining:
                continue
            packed.append(candidate)
            used += len(candidate.content)
            if len(packed) >= query.top_k:
                stop_reason = "top_k_reached"
                break
        diagnostics = KnowledgeRetrievalDiagnostics(
            requested_sources=tuple(requested), effective_sources=tuple(dict.fromkeys(effective)),
            degraded=bool(reasons), reason=";".join(dict.fromkeys(reason for reason in reasons if reason)),
            per_source_counts=counts, latency_ms=(__import__("time").monotonic() - started) * 1000,
            token_estimate=(used + 3) // 4, version_filters={key: value for key, value in {"software_version": query.software_version, "repo_commit": query.repo_commit}.items() if value},
            before_dedup=len(candidates), after_dedup=len(deduped), retrieval_mode=mode,
            per_source_count={source: sum(item.source_type == source for item in packed) for source in requested},
            diversity_applied=len({item.source_type for item in packed}) > 1,
            token_budget=query.token_budget, packed_candidates=len(packed),
            dropped_candidates=max(0, len(ordered) - len(packed)),
            packing_stop_reason=stop_reason,
        )
        return KnowledgeRetrievalResult(candidates=tuple(packed), diagnostics=diagnostics)

    def prior_context(self, query: KnowledgeQuery, *, mode: str = "lexical", context_id: str | None = None, result=None) -> PriorContext:
        result = result or self.retrieve(query, mode=mode)
        return PriorContext(
            context_id=context_id or f"prior-{query.incident_id}", candidates=result.candidates,
            source_types=tuple(dict.fromkeys(item.source_type for item in result.candidates)),
            retrieval_query=query,
            provenance=tuple({(item.provenance.source, item.provenance.source_id, item.provenance.path): item.provenance for item in result.candidates}.values()),
            packed_chars=sum(len(item.content) for item in result.candidates),
            packed_tokens=result.diagnostics.token_estimate,
        )
