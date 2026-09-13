from __future__ import annotations

"""Deterministic Incident capability detection, routing, and query building."""

from typing import Any
import re

from pydantic import BaseModel, ConfigDict, Field

from .contracts import KnowledgeQuery, KnowledgeSource, PriorContext
from .rag import DomainKnowledgeRetriever
from .static_graph import InMemoryStaticGraph, StaticKnowledgeGraphTools


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
        logs = bool({"telemetry", "logs", "error_logs"} & evidence_sources) and (get("get_error_logs") or get("get_alerts"))
        metrics = bool({"telemetry", "metrics", "alerts"} & evidence_sources) and get("get_alerts")
        trace = bool({"telemetry", "trace", "traces"} & evidence_sources)
        k8s = bool({"kubernetes_resources", "resource_descriptions", "application_yaml"} & evidence_sources) and get("get_resources")
        code = "application_code" in evidence_sources and source_workspace_available and bool(code_tools)
        knowledge = []
        if domain_rag is not None:
            knowledge.append("domain_rag")
        if knowledge_store is not None:
            knowledge.append("incident_memory")
        if static_graph is not None:
            knowledge.append("static_kg")
        return cls(
            logs_available=bool(logs), metrics_available=bool(metrics), trace_available=bool(trace),
            k8s_available=bool(k8s), code_available=bool(code),
            domain_rag_available=domain_rag is not None,
            incident_memory_available=knowledge_store is not None,
            static_kg_available=static_graph is not None,
            available_code_tools=code_tools,
            available_knowledge_sources=tuple(dict.fromkeys(knowledge)),
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
