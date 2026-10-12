from __future__ import annotations

from dataclasses import asdict, replace
from copy import deepcopy
import json
import math
from pathlib import Path
import time
from typing import Any, Literal

from debug_assistant.agent.final_review import (
    FinalReviewAgent, ReviewSchemaError, enforce_review_consistency,
)
from debug_assistant.agent.reflection import IncidentReflectionAgent, ReflectionContractExhausted
from debug_assistant.skills.projection import load_skill_views
from debug_assistant.agent.planner import (
    NativeToolPlanner, PlannerContractError,
    PlannerContractExhausted,
)
from debug_assistant.config import RuntimeConfig
from debug_assistant.context.models import ContextView
from debug_assistant.context.manager import ContextManager
from debug_assistant.context.projection import IncidentProjectionPolicy
from debug_assistant.harness.trace import TraceRecorder
from debug_assistant.harness.budget import BudgetController
from debug_assistant.harness.dynamic_budget import (
    BudgetState, DynamicBudgetController, ProgressMarker, PromptBudgetExceeded,
    RunBudgetAdmissionExceeded,
    progress_made, resolve_model_capability,
)
from debug_assistant.harness.deadline import RunDeadline
from debug_assistant.harness.retry import RetryPolicy
from debug_assistant.harness.provider_health import ToolCircuitBreaker
from debug_assistant.incidents.contracts import (
    CompletionContract, FactRequirement, EvidenceConstraint,
    Contradiction, IncidentCase, IncidentEvidence, IncidentHypothesis, IncidentMetrics,
    IncidentRunResult, ReflectionFeedback,
    ReviewDecision, RootCauseCandidate, SemanticDecisionRecord, SemanticEvidenceRecord,
    DiagnosticReportReview,
    SourceClaim, TerminalOutcome, VerificationObligation, build_verification_key,
)
from debug_assistant.incidents.verification_dag import (
    VerificationDAGHarnessAdapter,
    VerificationDAGState,
    VerificationTask,
    VerificationTaskExecution,
)
from debug_assistant.incidents.rca_state import RCAState, RCAStateError
from debug_assistant.incidents.semantic_transition import (
    SemanticTransitionError, SemanticTraceBuffer, resolve_hypothesis,
    validate_disposition_conflicts,
)
from debug_assistant.incidents.evidence_pipeline import (
    EvidencePipeline, EvidencePipelineError, EvidenceSet,
)
from debug_assistant.incidents.finalization import (
    FinalizationDecision, FinalizationFacts, FinalizationPolicy,
)
from debug_assistant.incidents.report import build_diagnostic_report
from debug_assistant.context.requests import ContextResolutionError
from debug_assistant.agent.requirement_targeting import targeting_feedback, requirement_table
from debug_assistant.incidents.planning_session import PlanningSession, LogicalTurnRetryBudget
from debug_assistant.incidents.continuation import ContinuationPolicy
from debug_assistant.incidents.provider_gateway import (
    ProviderOperationalGateway, ProviderCallLimitExceeded, aggregate_provider_usage,
)
from debug_assistant.incidents.action_runtime import (
    ActionRuntime, ActionExecutionContext, InvestigationUse, DuplicateAcquisition,
)
from debug_assistant.incidents.verification_plan import (
    PlanAdmissionDecision, VerificationPlanError, compile_planned_action,
    registry_schema_sha256, EnrichedVerificationPlan,
)
from debug_assistant.incidents.obligation_admission import (
    CanonicalObligationProposal,
    ObligationAdmissionDecision,
    ObligationAdmissionPolicy,
    ObligationProposalError,
    obligation_id_for_key,
)
from debug_assistant.incidents.progress_policy import ProgressDecision, ProgressFacts, ProgressPolicy
from debug_assistant.models import ActionKind, ActionProposal, AgentState, TaskSpec, ToolObservation
from debug_assistant.memory.evidence_boundary import is_canonical_source_evidence
from debug_assistant.memory.evidence_memory import EvidenceMemory
from debug_assistant.memory.evidence_projector import (
    EvidenceProjector, EvidenceProposal, EvidenceProposalError,
    evidence_relationship_key,
)
from debug_assistant.memory.observation_store import ObservationStore
from debug_assistant.knowledge import (
    CapabilitySnapshot, HybridRouter, IncidentEntityExtractor,
    KnowledgeCoordinator, KnowledgeQueryBuilder, KnowledgeRetrievalTool,
)
from debug_assistant.llm.base import (
    LLMDeadlineExceeded, LLMError, LLMOutputError,
)
from debug_assistant.skills.investigation import (
    DomainAssetCatalog,
    DomainSkillRegistry,
)
from debug_assistant.tools.cloudops_snapshot import CloudOpsSnapshotToolRegistry


# Runtime-owned convergence controls. The provider adapter carries the
# reasoning limit as a bounded structured completion envelope.
CONVERGE_MAX_REASONING_TOKENS = 2_000
CONVERGE_MAX_OUTPUT_TOKENS = 3_000
CONVERGE_MAX_READ_FILE_CALLS = 3
# L5 v1: DeepSeek-V4's hidden reasoning can consume the entire completion
# envelope before emitting a native tool call.  Converge turns therefore use
# the provider's explicit non-thinking mode; exploration behavior is unchanged.
CONVERGE_ENABLE_THINKING = False
# L5 v3: once the source-investigation window has reached its bounded
# convergence turn, do not spend additional LLM calls on reflection branches.
# This keeps one terminal planner call plus Review inside the hard call budget.
CONVERGE_START_STEP = 7
CONVERGE_DISABLE_REFLECTIONS = True


def _prompt_accounting_residual(
    *,
    actual_prompt_tokens: int | None,
    components: dict[str, int | None],
    provider_overhead_tokens: int | None,
) -> tuple[int | None, bool]:
    """Return actual prompt residual and whether the attributed total is too high."""
    if actual_prompt_tokens is None:
        return None, False
    attributed = sum(
        int(value) for value in components.values() if value is not None
    )
    overhead = max(0, int(provider_overhead_tokens or 0))
    residual = int(actual_prompt_tokens) - attributed - overhead
    return residual, residual < 0


def _provider_failure_category(exc: BaseException) -> str:
    """Classify transport, provider-contract, and semantic failures separately."""
    if isinstance(exc, LLMOutputError):
        return "provider_contract_failure"
    if isinstance(exc, LLMError):
        return "provider_transport_failure"
    return "semantic_adjudication_failure"


def _usage_token_value(usage: dict[str, Any], *keys: str) -> int | None:
    """Read a token field from top-level or common nested Provider usage."""
    for key in keys:
        value = usage.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        for key in keys:
            value = details.get(key)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
    return None


def _grounded_contradiction_ids(
    proposed_ids: Any,
    projected_facts: Any,
    hypothesis_scopes: set[str] | frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    """Keep only contradiction claims grounded in canonical Evidence relations.

    Planner contradiction fields are semantic proposals.  They cannot turn a
    neutral or partial observation into a Runtime-owned blocking contradiction.
    When one Evidence ID has multiple relationship projections, prefer the
    projection for the current hypothesis scope; stale branch projections must
    not block the revised RCA.
    """
    from .semantic_transition import grounded_contradiction_ids
    return grounded_contradiction_ids(proposed_ids or (), projected_facts or (), hypothesis_scopes)


def _source_investigation_declared(
    *,
    source_obligation: Any | None,
    source_claims: Any,
    cited_source_ids: Any,
    proposed_status: str,
) -> bool:
    """Return whether the Planner explicitly entered a source branch."""
    return bool(
        source_obligation is not None
        or tuple(source_claims or ())
        or tuple(cited_source_ids or ())
        or str(proposed_status or "unknown") != "unknown"
    )

# The CloudOps project has one active asset source.  Keeping the manifest
# default here makes every harness construction use the same canonical
# registry; callers may only replace it with another explicitly versioned
# manifest for a separate experiment.
def _safe_planner_metadata_for_trace(exc: Exception) -> dict[str, Any] | None:
    """Keep planner diagnostics useful without copying model output to Trace."""
    metadata = getattr(exc, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    safe: dict[str, Any] = {}
    for key in (
        "error_type", "error_layer", "index", "tool", "arguments_type", "actions_type",
        "actions_count", "child_types", "output_shape", "repair_rejection_reason",
        "raw_output", "parsed_output", "normalized_output", "validation_error",
        "repair_prompt", "repair_output", "repair_result", "requirement_feedback", "capability_feedback",
    ):
        if key in metadata:
            safe[key] = metadata[key]
    raw_errors = metadata.get("validation_errors")
    if isinstance(raw_errors, list):
        compact_errors = []
        for error in raw_errors:
            if isinstance(error, dict):
                location = ".".join(str(part) for part in error.get("loc", ())) or "item"
                compact_errors.append({
                    "loc": location,
                    "type": str(error.get("type", "validation_error")),
                })
            elif isinstance(error, str):
                compact_errors.append(error)
            else:
                compact_errors.append(type(error).__name__)
        safe["validation_errors"] = compact_errors
    return safe or None


DiagnosisHarnessConfig = RuntimeConfig


# Kept for callers and tests written against the first Incident vertical slice.


_OBLIGATION_TRANSITIONS = {
    "OPEN": {
        "OPEN", "SATISFIED", "BLOCKED_BY_CAPABILITY",
        "NOT_APPLICABLE_BY_CAPABILITY", "SUPERSEDED_BY_REPLAN",
    },
    "SATISFIED": {"SATISFIED"},
    "BLOCKED_BY_CAPABILITY": {
        "BLOCKED_BY_CAPABILITY",
        "NOT_APPLICABLE_BY_CAPABILITY",
    },
    "NOT_APPLICABLE_BY_CAPABILITY": {"NOT_APPLICABLE_BY_CAPABILITY"},
    "SUPERSEDED_BY_REPLAN": {"SUPERSEDED_BY_REPLAN"},
}
_CONTRADICTION_TRANSITIONS = {
    "OPEN": {"OPEN", "RESOLVED", "EXPLAINED"},
    "RESOLVED": {"RESOLVED"},
    "EXPLAINED": {"EXPLAINED"},
}

# This is a generic verification claim for any incident whose causal mechanism
# may be implemented in the bound application source.  It is deliberately not
# tied to a benchmark case, service name, or fault taxonomy.
_SOURCE_MECHANISM_OBLIGATION_CLAIM = (
    "Verify the application-level causal mechanism with source evidence."
)


def _is_source_mechanism_obligation(obligation: VerificationObligation) -> bool:
    """Identify the one Runtime-owned source obligation by canonical claim."""
    return obligation.claim.strip().casefold() == _SOURCE_MECHANISM_OBLIGATION_CLAIM.casefold()


def _is_source_action_obligation(obligation: VerificationObligation) -> bool:
    """Identify any open Obligation whose answer must come from application code.

    This is intentionally broader than the Runtime-owned source-coverage
    Obligation above.  Planner-proposed intermediate questions such as source
    file discovery must receive the same source Tool route, while their
    lifecycle remains ordinary EvidenceSet/RCAState lifecycle.
    """
    text = " ".join(
        str(getattr(obligation, field, "") or "")
        for field in (
            "claim", "evidence_requirement", "verification_question", "evidence_type",
        )
    ).casefold()
    return any(marker in text for marker in (
        "source evidence", "source code", "application source", "source file",
        "source contains", "source inspection", "source pattern", "read_file",
        "code behavior", "code-level", "code defect",
    ))


def _obligation_capability_gaps(proposal: dict[str, Any], capabilities: Any) -> tuple[str, ...]:
    """Return capability boundaries for an otherwise valid obligation proposal.

    This is a Runtime capability check, not a diagnosis rule.  The Planner may
    describe an evidence question in natural language, but it cannot keep an
    OPEN lifecycle for a structured evidence type whose only provider source is
    already EMPTY/UNAVAILABLE/FAILED.
    """
    evidence_type = str(proposal.get("evidence_type", "") or "").casefold()
    text = " ".join(
        str(proposal.get(name, "") or "")
        for name in ("evidence_type", "evidence_requirement", "verification_question")
    ).casefold()
    required: list[str] = []
    if (
        "log" in evidence_type
        or "log" in text
        or "runtime behavior" in text
        or "runtime evidence" in text
    ):
        required.append("logs")
    source_markers = (
        "source evidence", "source code", "source_code", "application source",
        "application_code", "source file", "source inspection", "source pattern",
        "read_file", "code behavior", "code_behavior", "code defect",
    )
    # A log question may mention a "code-level" delay as a pattern to rule
    # out. That still needs only the captured log capability; the phrase is
    # not evidence that an application source workspace is required. Keep
    # application_code fail-closed for explicit source/code evidence types or
    # source markers, while allowing a non-log code-level question to retain
    # the historical capability boundary.
    source_required = any(
        marker in evidence_type or marker in text for marker in source_markers
    ) or (
        "code-level" in text
        and not any(
            marker in evidence_type or marker in text
            for marker in ("log", "runtime behavior", "runtime evidence")
        )
    )
    if source_required:
        # ``read_file`` is an executable Tool, not a CapabilitySnapshot key.
        # Tool visibility is checked separately against the bound source
        # registry; treating it as a capability here incorrectly converted a
        # valid source obligation into NOT_APPLICABLE_BY_CAPABILITY.
        required.append("application_code")
    gaps = []
    for capability in dict.fromkeys(required):
        state = str(capabilities.capability_state(capability))
        if state != "AVAILABLE":
            gaps.append(capability)
    return tuple(gaps)


def _normalize_obligation_capability_status(
    obligation: VerificationObligation,
    capabilities: Any,
) -> tuple[VerificationObligation, tuple[str, ...]]:
    """Reconcile an existing open Obligation with newly observed capabilities.

    Capability state can become EMPTY/FAILED after an Obligation was created
    (for example, a captured log query may return no records).  Rehydrating
    that stale OPEN item would make the Planner pursue an impossible fact until
    no-progress or budget termination.  This changes only lifecycle
    applicability; it does not assert a diagnosis or satisfy Evidence.
    """
    if obligation.status not in {"OPEN", "BLOCKED_BY_CAPABILITY"}:
        return obligation, ()
    gaps = _obligation_capability_gaps(
        {
            "evidence_type": obligation.evidence_type,
            "evidence_requirement": obligation.evidence_requirement,
            "verification_question": obligation.verification_question,
        },
        capabilities,
    )
    if not gaps:
        return obligation, ()
    return obligation.model_copy(update={
        "status": "NOT_APPLICABLE_BY_CAPABILITY",
        "supporting_evidence_ids": (),
        "blocked_capabilities": tuple(dict.fromkeys(
            (*obligation.blocked_capabilities, *gaps)
        )),
    }), gaps




class ReviewProgressRequired(RuntimeError):
    """The bounded post-review loop tried to submit an unchanged candidate."""


class ConvergenceBudgetExhausted(RuntimeError):
    """A convergence response exhausted its bounded output envelope."""


class FinalizationBlocked(RuntimeError):
    """The canonical FinalizationPolicy rejected Candidate publication."""

    def __init__(self, decision: FinalizationDecision):
        self.decision = decision
        super().__init__(", ".join(reason.code for reason in decision.reasons))


class DiagnosisHarness:
    """Bounded CloudOps diagnosis runtime with a separate final-review boundary.

    This is the sole CloudOps diagnosis runtime. SWE execution is isolated in
    the sibling project and is not part of this import graph.
    """

    def __init__(self, llm, review_llm=None, *, model: str = "", reflection_model: str = "",
                 review_model: str = "", transport_profile: str = "semantic_plan",
                 config: DiagnosisHarnessConfig | None = None,
                 topology_path: str | Path | None = None, search_engine=None,
                 knowledge_store=None, domain_rag=None, static_graph=None,
                 knowledge_coordinator: KnowledgeCoordinator | None = None):
        self.llm = llm
        self.review_llm = review_llm or llm
        self.model = model
        self.reflection_model = reflection_model or model
        self.review_model = review_model
        self.config = config or DiagnosisHarnessConfig()
        if transport_profile != "semantic_plan":
            raise ValueError("incident Runtime requires structured interpretation/planning transport")
        self.transport_profile = transport_profile
        # Optional semantic retrieval is an environment capability.  The
        # default CloudOps path still creates a task-scoped lexical index and
        # degrades explicitly when no semantic engine is injected.
        self.search_engine = search_engine
        self.knowledge_coordinator = knowledge_coordinator or KnowledgeCoordinator(
            knowledge_store=knowledge_store, domain_rag=domain_rag,
            static_graph=static_graph,
        )
        self.topology_path = Path(topology_path) if topology_path else (
            Path(__file__).resolve().parents[3] / "data" / "online_boutique" / "service_topology.json"
        )

    def run(
        self,
        case: IncidentCase,
        *,
        mode: Literal["baseline", "dag"] | None = None,
    ) -> IncidentRunResult:
        execution_mode = mode or self.config.execution_mode
        if execution_mode not in {"baseline", "dag"}:
            raise ValueError("mode must be baseline or dag")
        tools = CloudOpsSnapshotToolRegistry(
            case.runtime_data_dir, self.topology_path, search_engine=self.search_engine,
        )
        # Keep this state aligned with the provider-facing CloudOps registry;
        # source-discovery tools are advertised only when the case binds code.
        source_workspace_available = bool(
            getattr(getattr(tools, "source_binding", None), "available", False)
        )
        entities = IncidentEntityExtractor().extract(case)
        capabilities = CapabilitySnapshot.detect(
            case, tools, source_workspace_available=source_workspace_available,
            domain_rag=self.knowledge_coordinator.domain_rag,
            knowledge_store=self.knowledge_coordinator.knowledge_store,
            static_graph=self.knowledge_coordinator.static_graph,
        )
        # The registry remains the execution authority; the snapshot is the
        # provider-facing capability projection and starts fail-closed.
        available_code_tools = tuple(
            name for name in tools.planner_visible_code_tools()
            if capabilities.capability_state("application_code") == "AVAILABLE"
        )
        router_decision = HybridRouter().route(case, entities, capabilities)
        query_builder = KnowledgeQueryBuilder()
        # The Knowledge tool receives the current prior ceiling.  It remains a
        # normal Tool and never enters the Evidence ledger.
        knowledge_tool = KnowledgeRetrievalTool(self.knowledge_coordinator, case.case_id)
        knowledge_enabled = bool(self.knowledge_coordinator.available_sources)
        if knowledge_enabled:
            tools.register(knowledge_tool)
        planner = NativeToolPlanner(
            self.llm, tools, self.model,
            public_fault_taxonomy=self.config.features.public_fault_taxonomy,
            planner_contract_version=self.config.features.planner_contract_version,
            transport_profile=self.transport_profile,
        )
        provider_gateway = ProviderOperationalGateway(self.llm, transport_profile=self.transport_profile,
            max_context_chars=self.config.max_context_chars,
            max_request_input_tokens=self.config.max_request_input_tokens,
            max_llm_calls=self.config.max_llm_calls)
        planning_session = PlanningSession(planner, provider_gateway)

        reviewer = FinalReviewAgent(
            self.review_llm, self.review_model,
            provider_gateway=provider_gateway.bind(self.review_llm, "review"),
        )
        reflection_agent = IncidentReflectionAgent(
            self.llm, self.reflection_model,
            provider_gateway=provider_gateway.bind(self.llm, "reflection"),
        )
        trace = TraceRecorder(self.config.trace_dir, case.case_id)
        asset_manifest_path = self.config.domain_asset_manifest_path
        try:
            domain_assets = DomainAssetCatalog.from_manifest(asset_manifest_path)
            domain_registry = DomainSkillRegistry.from_manifest(asset_manifest_path)
            trace.record("DOMAIN_ASSET_PARITY", {
                "manifest_path": domain_assets.manifest_path,
                "manifest_sha256": domain_assets.manifest_sha256,
                "tool_count": len(domain_assets.tool_ids),
                "skill_count": len(domain_assets.skill_ids),
                "excluded_node_count": domain_assets.excluded_node_count,
                "registry_skill_count": len(domain_registry.skills),
            })
        except (OSError, ValueError) as exc:
            trace.record("DOMAIN_ASSET_PARITY_UNAVAILABLE", {
                "manifest_path": asset_manifest_path,
                "error_type": type(exc).__name__,
                "reason": str(exc),
            })
            raise RuntimeError(
                "the active CloudOps runtime requires the packaged DomainAssetManifest"
            ) from exc
        deadline = RunDeadline(self.config.max_wall_time_seconds)
        state = AgentState(TaskSpec(
            task_id=case.case_id, issue=case.summary, repo_path=case.runtime_data_dir,
            repo_name=case.system, metadata={"task_kind": "incident", "namespace": case.namespace},
        ))
        observation_store = ObservationStore()
        pending_capability_blocks: set[str] = set()
        evidence_memory = EvidenceMemory()
        evidence_projector = EvidenceProjector()
        state.evidence = evidence_memory.pinned
        incident_projection = IncidentProjectionPolicy()
        context_manager = ContextManager(
            self.config.context,
            enable_catalog=self.config.features.context_manager,
            enable_model_selection=True,
            enable_budget_packing=self.config.features.context_manager,
            enable_lifecycle=self.config.features.context_manager,
            enable_projection=self.config.features.context_projection_v2,
            projection_policy=incident_projection,
        )
        interpretation_view = ContextView("interpretation")
        bootstrap_pending_observation_ids: tuple[str, ...] = ()
        actions: list[dict[str, Any]] = []
        rejected_actions: list[dict[str, Any]] = []

        tool_circuit = ToolCircuitBreaker(
            failure_threshold=self.config.tool_circuit_failure_threshold,
        )
        action_runtime = ActionRuntime(
            tools, action_ledger=actions, circuit=tool_circuit,
            snapshot_identity=f"frozen:{Path(case.runtime_data_dir).resolve()}",
            retry_policy=RetryPolicy(
                max_attempts=self.config.tool_retry_attempts,
                base_delay_seconds=self.config.retry_base_delay_seconds,
                max_delay_seconds=self.config.retry_max_delay_seconds,
            ),
        )
        active_skill = ""
        prompt_tokens = completion_tokens = 0
        run_cost = 0.0
        cost_measured = False
        review_rounds = 0
        review_recovery_cycles = 0
        reflection_calls = 0
        reflection_delta_accepts = 0
        reflection_no_delta_count = 0
        consecutive_reflection_no_delta = 0
        reflection_feedback = ""
        reflection_constraint = ""
        runtime_action_constraint = ""
        runtime_tool_visibility_override: tuple[str, ...] | None = None
        reflection_signatures: set[str] = set()
        reflection_trigger = ""
        duplicate_calls = 0
        blocked_tool_call_count = 0
        schema_repair_count = 0
        provider_calls = 0
        budget_rejected_attempts = 0
        budget_rejection_keys: set[tuple[str, int, str]] = set()
        schema_repair_calls = 0
        provider_failures = 0
        provider_contract_failures = 0
        semantic_adjudication_failures = 0
        token_accounting_error_count = 0
        unattributed_prompt_total = 0
        token_telemetry: list[dict[str, Any]] = []
        provider_event_cursors: dict[str, int] = {}
        provider_calls_by_stage: dict[str, int] = {}
        obligation_created_count = 0
        obligation_blocked_count = 0
        blocking_contradiction_count = 0
        previous_hypothesis: IncidentHypothesis | None = None
        first_pass_accepted = False
        evidence_sufficient_step = None
        evidence_sufficient_tool_calls = None
        final_candidate_step = None
        llm_calls = 0
        planner_calls = 0
        planner_contract_retry_used = False

        contract_repair_calls = 0
        review_calls = 0
        # The cursor is initialized before the terminal-report review helper;
        # candidate Review assigns the same cursor for each normal round.
        review_record_cursor = 0
        termination_reason = ""
        run_started = time.monotonic()
        runtime_phase = "INIT"
        state_transition_count = 0
        converge_read_file_calls = 0
        dag_harness = VerificationDAGHarnessAdapter()
        dag_state: VerificationDAGState | None = None
        canonical_rca_state = RCAState(incident_id=case.case_id)

        def active_hypothesis() -> IncidentHypothesis | None:
            return canonical_rca_state.active_hypothesis_projection()

        def hypothesis_history() -> tuple[IncidentHypothesis, ...]:
            return canonical_rca_state.hypothesis_projections()

        def obligation_projection() -> dict[str, VerificationObligation]:
            return {
                item.id: item.model_copy(deep=True)
                for item in canonical_rca_state.obligations
            }
        # FinalizationPolicy is the canonical semantic publication boundary.
        # Convergence remains operational telemetry and cannot authorize a
        # Candidate independently.
        last_convergence_decision = None
        terminal_convergence_decision = None
        evidence_pipeline = EvidencePipeline(
            observation_store=observation_store,
            evidence_memory=evidence_memory,
            projector=evidence_projector,
            deterministic_fact_adapters=("source_range_capture",),
        )

        # One admitted Evidence may answer more than one canonical
        # Obligation.  Keep one relation record per (Evidence, Obligation)
        # instead of overwriting the first relation by evidence_id alone.
        projected_facts: dict[tuple[Any, ...], Any] = {}
        evidence_admission_rejections = 0
        evidence_sets: dict[tuple[str, str], EvidenceSet] = {}

        dag_task_metrics = {
            "planned": 0,
            "completed": 0,
            "blocked": 0,
            "failed": 0,
            "local_replans": 0,
        }

        def project_obligation_ids_to_dag(reason: str) -> None:
            """Project canonical Obligation IDs into execution state."""
            nonlocal dag_state
            if dag_state is None:
                return
            before = set(dag_state.obligation_ids)
            dag_state = dag_harness.register_obligation_ids(
                dag_state,
                (item.id for item in canonical_rca_state.obligations),
            )
            trace.record("DAG_OBLIGATION_IDS_PROJECTED", {
                "source": "rca_state_authority",
                "reason": reason,
                "registered_obligation_ids": sorted(
                    set(dag_state.obligation_ids) - before
                ),
                "obligation_count": len(canonical_rca_state.obligations),
                "open_critical_obligation_ids": list(
                    canonical_rca_state.open_critical_obligation_ids
                ),
            })

        def transition_phase(target: str, reason: str, **metadata: Any) -> None:
            """Record the only Runtime-owned PLAN/REFLECT/FINALIZE/REVIEW edges."""
            nonlocal runtime_phase, state_transition_count
            target = str(target)
            if target == runtime_phase:
                return
            state_transition_count += 1
            trace.record("STATE_TRANSITION", {
                "from": runtime_phase,
                "to": target,
                "reason": reason,
                "transition_index": state_transition_count,
                **metadata,
            })
            runtime_phase = target
        budget = BudgetController(
            max_steps=self.config.max_steps,
            max_tool_calls=self.config.max_tool_calls,
            max_llm_calls=self.config.max_llm_calls,
            max_total_tokens=self.config.max_total_tokens,
            max_wall_time_seconds=max(1, math.ceil(self.config.max_wall_time_seconds)),
            finalization_reserve_seconds=math.floor(self.config.finalization_reserve_seconds),
            started_at=state.started_at,
        )
        budget_allocation = self.config.budget_allocation
        model_capability = resolve_model_capability(self.llm, model=self.model)
        dynamic_budget = DynamicBudgetController(
            model_capability,
            max_context_chars=self.config.max_context_chars,
            max_request_input_tokens=self.config.max_request_input_tokens,
            max_total_tokens=self.config.max_total_tokens,
            max_llm_calls=self.config.max_llm_calls,
            max_wall_time_seconds=self.config.max_wall_time_seconds,
            max_cost_per_incident=self.config.max_cost_per_incident,
            terminal_reserve_tokens=self.config.terminal_reserve_tokens,
            terminal_reserve_llm_calls=self.config.terminal_reserve_llm_calls,
            terminal_reserve_seconds=self.config.finalization_reserve_seconds,
            budget_allocation=budget_allocation,
            pressure_ratio=self.config.context_pressure_ratio,
            hard_pressure_ratio=self.config.hard_pressure_ratio,
            started_at=state.started_at,
        )
        provider_gateway.attempt_admission = dynamic_budget.api_attempt_rejection
        trace.record("INCIDENT_STARTED", {
            "case_id": case.case_id, "summary": case.summary, "system": case.system,
            "namespace": case.namespace, "evidence_sources": case.evidence_sources,
            "execution_mode": execution_mode,
            "context_projection_mode": self.config.context_projection_mode,
            "tool_exposure_policy_version": self.config.tool_exposure_policy_version,
            "model_capability": model_capability.as_dict(),
            "terminal_reserve": dynamic_budget.terminal_reserve_payload(),
            "budget_lifecycle": budget_allocation.as_dict(),
        })
        transition_phase("PLAN", "incident_started")
        if knowledge_enabled:
            trace.record("CAPABILITY_DETECTED", capabilities.model_dump(mode="json"))
            trace.record("ROUTER_DECISION", {
                **router_decision.model_dump(mode="json"),
                "entities": entities.model_dump(mode="json"),
            })
        prior_context = None
        initial_prior_capacity = dynamic_budget.available_prior_capacity(
            critical_state_tokens=dynamic_budget.estimate_json_tokens({
                "incident": case.summary, "entities": entities.model_dump(mode="json"),
                "router": router_decision.model_dump(mode="json"),
            }),
            verified_evidence_tokens=0,
            required_control_tokens=dynamic_budget.terminal_reserve_tokens,
        )
        if initial_prior_capacity is not None:
            initial_prior_budget = max(1, min(100_000, initial_prior_capacity))
            knowledge_tool.max_token_budget = initial_prior_budget
        else:
            initial_prior_budget = None
        pre_retrieval_queries = (
            query_builder.build(case, entities, router_decision, token_budget=initial_prior_budget)
            if knowledge_enabled else ()
        )
        for query in pre_retrieval_queries:
            trace.record("KNOWLEDGE_QUERY_BUILT", {
                "query": query.model_dump(mode="json"), "stage": "pre_retrieval",
            })
        if pre_retrieval_queries:
            query = pre_retrieval_queries[0]
            retrieval_result = self.knowledge_coordinator.retrieve(query)
            prior_context = self.knowledge_coordinator.prior_context(
                query, result=retrieval_result,
            )
            trace.record("KNOWLEDGE_RETRIEVED", {
                "stage": "pre_retrieval",
                **retrieval_result.diagnostics.model_dump(mode="json"),
                "candidate_count": len(retrieval_result.candidates),
            })
            if retrieval_result.diagnostics.degraded:
                trace.record("KNOWLEDGE_RETRIEVAL_DEGRADED", {
                    "stage": "pre_retrieval",
                    "reason": retrieval_result.diagnostics.reason,
                })
            trace.record("PRIOR_CONTEXT_PACKED", {
                "stage": "pre_retrieval", "context_id": prior_context.context_id,
                "source_types": list(prior_context.source_types),
                "candidate_count": len(prior_context.candidates),
                "packed_chars": prior_context.packed_chars,
                "packed_tokens": prior_context.packed_tokens,
            })

        def _optional_int(mapping: dict[str, Any], *keys: str) -> int | None:
            for key in keys:
                if key in mapping and mapping[key] is not None:
                    try:
                        return int(mapping[key])
                    except (TypeError, ValueError):
                        return None
            return None

        def _prompt_components(breakdown: dict[str, Any]) -> dict[str, int | None]:
            explicit = breakdown.get("prompt_components") or {}
            token_breakdown = dict(
                breakdown.get("token_breakdown")
                or breakdown.get("last_token_breakdown")
                or {}
            )
            def first(*keys: str) -> int | None:
                for key in keys:
                    value = explicit.get(key, breakdown.get(key, token_breakdown.get(key)))
                    if value is not None:
                        try:
                            return int(value)
                        except (TypeError, ValueError):
                            return None
                return None
            return {
                "system_tokens": first("system_tokens", "system"),
                "tool_schema_tokens": first("tool_schema_tokens", "tools", "tool_schema"),
                "control_state_tokens": first("control_state_tokens", "control_state"),
                "diagnostic_context_tokens": first("diagnostic_context_tokens", "diagnostic_context", "user"),
            }

        def record_provider_events(stage: str) -> None:
            """Consume canonical gateway records once for every production role."""
            nonlocal provider_calls, provider_failures, provider_contract_failures
            records = [item for item in provider_gateway.records if item.role == stage]
            cursor = provider_event_cursors.get(stage, 0)
            fresh = records[cursor:]
            provider_event_cursors[stage] = len(records)
            for item in fresh:
                provider_calls += item.attempt_count
                provider_calls_by_stage[stage] = provider_calls_by_stage.get(stage, 0) + item.attempt_count
                if item.status == "failed":
                    if item.failure_layer == "transport_provider":
                        provider_failures += 1
                    elif item.failure_layer in {"parse", "contract_schema"}:
                        provider_contract_failures += 1
                trace.record("PROVIDER_CALL", {
                    "role": item.role, "request_kind": item.request_kind,
                    "turn_id": item.turn_id, "state_snapshot_hash": item.state_snapshot_hash,
                    "request_sha256": item.request_sha256,
                    "status": item.status, "failure_layer": item.failure_layer,
                    "error_type": item.error_type, "attempt_count": item.attempt_count,
                    "elapsed_seconds": item.elapsed_seconds, "usage": item.usage_dict,
                    "attempt_details": item.attempt_details,
                    "redacted": True,
                })

        def record_provider_failure(
            stage: str,
            *,
            exc: BaseException | None = None,
        ) -> str:
            nonlocal provider_calls, provider_failures
            nonlocal provider_contract_failures, semantic_adjudication_failures
            failures_before = provider_failures
            contract_before = provider_contract_failures
            semantic_before = semantic_adjudication_failures
            add_usage(self.llm, stage=stage, call_status="failed")
            category = _provider_failure_category(exc) if exc is not None else "provider_transport_failure"
            if (
                category == "provider_transport_failure"
                and provider_failures == failures_before
            ):
                provider_failures += 1
            elif (
                category == "provider_contract_failure"
                and provider_contract_failures == contract_before
            ):
                provider_contract_failures += 1
            elif (
                category == "semantic_adjudication_failure"
                and semantic_adjudication_failures == semantic_before
            ):
                semantic_adjudication_failures += 1
            return category

        def add_usage(usage_source, *, stage: str = "unknown", prompt_breakdown: dict[str, Any] | None = None,
                      call_status: str = "completed") -> None:
            nonlocal prompt_tokens, completion_tokens, run_cost, cost_measured
            nonlocal token_accounting_error_count, unattributed_prompt_total
            run_tokens_before = prompt_tokens + completion_tokens
            role_records = [item for item in provider_gateway.records if item.role == stage]
            fresh_records = role_records[provider_event_cursors.get(stage, 0):]
            if not fresh_records:
                return
            usage = aggregate_provider_usage(fresh_records)
            actual_prompt_tokens = _optional_int(usage, "prompt_tokens", "input_tokens")
            actual_completion_tokens = _optional_int(usage, "completion_tokens", "output_tokens")
            cached_tokens = _usage_token_value(
                usage, "cached_tokens", "cache_read_input_tokens"
            )
            provider_overhead_tokens = _usage_token_value(
                usage, "provider_overhead_tokens"
            )
            raw_cost = usage.get("cost", usage.get("total_cost", usage.get("estimated_cost")))
            try:
                actual_cost = None if raw_cost is None else float(raw_cost)
            except (TypeError, ValueError):
                actual_cost = None
            prompt_tokens += actual_prompt_tokens or 0
            completion_tokens += actual_completion_tokens or 0
            if actual_cost is not None:
                run_cost += max(0.0, actual_cost)
                cost_measured = True
            dynamic_budget.set_run_usage(
                tokens_used=prompt_tokens + completion_tokens,
                llm_calls_used=llm_calls,
                cost_used=run_cost,
            )
            breakdown = dict(prompt_breakdown or getattr(usage_source, "last_prompt_breakdown", {}) or {})
            components = _prompt_components(breakdown)
            estimated_prompt_tokens = breakdown.get("estimated_prompt_tokens")
            if estimated_prompt_tokens is None:
                estimated_prompt_tokens = sum(value or 0 for value in components.values())
            estimated_prompt_tokens = int(estimated_prompt_tokens or 0)
            accounting_delta, accounting_error = _prompt_accounting_residual(
                actual_prompt_tokens=actual_prompt_tokens,
                components=components,
                provider_overhead_tokens=provider_overhead_tokens,
            )
            if accounting_delta is not None:
                unattributed_prompt_total += accounting_delta
            if accounting_error:
                token_accounting_error_count += 1
            telemetry = {
                "stage": stage,
                **components,
                "provider_overhead_tokens": provider_overhead_tokens,
                "unattributed_prompt_tokens": accounting_delta,
                "token_accounting_error": accounting_error,
                "accounting_delta": accounting_delta,
                "estimated_prompt_tokens": estimated_prompt_tokens,
                "actual_prompt_tokens": actual_prompt_tokens,
                "actual_completion_tokens": actual_completion_tokens,
                "cached_tokens": cached_tokens,
                "call_status": call_status,
            }
            token_telemetry.append(telemetry)
            record_provider_events(stage)
            dynamic_budget.record_actual_usage(
                stage=stage,
                estimated_prompt_tokens=estimated_prompt_tokens,
                actual_prompt_tokens=actual_prompt_tokens,
                actual_completion_tokens=actual_completion_tokens,
                actual_cost=actual_cost,
            )
            after_state = dynamic_budget.decision(
                stage,
                estimated_prompt_tokens=estimated_prompt_tokens,
                tokens_used=prompt_tokens,
                llm_calls_used=llm_calls,
                cost_used=run_cost,
            )
            trace.record("LLM_CALL_USAGE", {
                "phase": stage,
                "provider": model_capability.provider or type(self.llm).__name__,
                "model": model_capability.model or self.model,
                "estimated_prompt_tokens": estimated_prompt_tokens,
                "actual_prompt_tokens": actual_prompt_tokens,
                "actual_completion_tokens": actual_completion_tokens,
                "cached_tokens": cached_tokens,
                "system_tokens": components["system_tokens"],
                "tool_schema_tokens": components["tool_schema_tokens"],
                "control_state_tokens": components["control_state_tokens"],
                "diagnostic_context_tokens": components["diagnostic_context_tokens"],
                "provider_overhead_tokens": provider_overhead_tokens,
                "unattributed_prompt_tokens": accounting_delta,
                "token_accounting_error": accounting_error,
                "accounting_delta": accounting_delta,
                "call_status": call_status,
                "actual_cost": actual_cost,
                "run_tokens_before": run_tokens_before,
                "run_tokens_after": prompt_tokens + completion_tokens,
                "run_cost_after": run_cost,
                "context_state": after_state.state.value,
                "context_breakdown": dict(breakdown.get("token_breakdown") or breakdown),
                "compaction_applied": after_state.state is not BudgetState.NORMAL,
                "prior_pruned": False,
                "rehydrated_evidence_count": 0,
                "cache_hit": bool(cached_tokens and cached_tokens > 0),
                "duplicate_blocked": False,
            })
            capacity = ProgressPolicy.evaluate(operational_facts("post_usage"))
            if capacity.action == "stop":
                raise RuntimeError(capacity.reasons[0] + " exceeded after LLM response")

        def operational_facts(boundary: str, **overrides) -> ProgressFacts:
            """Project operational counters; policy owns continuation decisions."""
            values = dict(
                boundary=boundary, tokens=prompt_tokens + completion_tokens,
                token_limit=self.config.max_total_tokens, llm_calls=llm_calls,
                llm_limit=self.config.max_llm_calls, tool_calls=state.tool_calls,
                tool_limit=self.config.max_tool_calls, steps=state.step,
                step_limit=self.config.max_steps, cost=run_cost,
                cost_limit=dynamic_budget.max_cost_per_incident,
                review_calls=review_calls, review_limit=self.config.review_reserve_calls,
                deadline_available=not deadline.expired(), no_progress_count=state.no_progress_count,
                no_progress_limit=self.config.max_no_progress if self.config.features.no_progress_detection else 0,
                repeated_count=state.repeated_actions,
                repeat_limit=self.config.max_duplicate_actions,
            )
            values.update(overrides)
            return ProgressFacts(**values)

        def budget_snapshot(boundary: str) -> Any:
            snapshot = budget.snapshot(
                steps=state.step, tool_calls=state.tool_calls,
                llm_calls=llm_calls, tokens=prompt_tokens + completion_tokens,
            )
            # The generic ratio-based controller is intentionally conservative,
            # but a code investigation can spend many low-cost exploration turns
            # before the provider reaches its ratio threshold.  L5 applies a
            # Runtime-owned step boundary so the next planner call is bounded
            # convergence regardless of provider-side token accounting.
            current = active_hypothesis()
            source_ready = bool(current is not None and (
                current.source_mechanism_status in {"sufficient", "not_applicable"}
                and not any(
                    _is_source_mechanism_obligation(item)
                    and item.blocks_finalization
                    for item in current.verification_obligations
                )
            ))
            phase = ProgressPolicy.evaluate(operational_facts(
                "phase", phase=snapshot.phase, phase_start_step=CONVERGE_START_STEP,
                verification_ready=source_ready or not source_workspace_available,
            )).phase
            if phase != snapshot.phase:
                if phase == "source_verify" and boundary != "source_verify_hold":
                    trace.record("INVESTIGATION_PHASE_HELD", {
                        "boundary": boundary,
                        "requested_phase": "converge",
                        "effective_phase": "source_verify",
                        "reason": "source_relevance_gate_open",
                        "source_mechanism_status": current.source_mechanism_status if current else "unknown",
                    })
                return replace(snapshot, phase=phase)
            return snapshot

        def budget_payload(snapshot) -> dict[str, Any]:
            dynamic_state = dynamic_budget.decision(
                "runtime", tokens_used=prompt_tokens + completion_tokens, llm_calls_used=llm_calls,
                cost_used=run_cost,
            )
            return {
                "remaining_steps": max(0, self.config.max_steps - snapshot.steps_used),
                **dynamic_budget.model_status(llm_calls_used=llm_calls),
                "steps_used": snapshot.steps_used,
                "tool_calls_used": snapshot.tool_calls_used,
                "llm_calls_used": snapshot.llm_calls_used,
                "tokens_used": snapshot.tokens_used,
                "wall_time_seconds": snapshot.wall_time_seconds,
                "remaining_ratio": snapshot.remaining_ratio,
                "phase": snapshot.phase,
                "seconds_until_forced_finalization": snapshot.seconds_until_forced_finalization,
                "context_state": dynamic_state.state.value,
                "input_hard_capacity": dynamic_state.input_hard_capacity,
                "cost_used": run_cost,
                "max_cost_per_incident": dynamic_budget.max_cost_per_incident,
                "remaining_run_cost": dynamic_state.remaining_run_cost,
                "terminal_reserve_tokens": dynamic_state.terminal_reserve_tokens,
                "terminal_reserve_llm_calls": dynamic_state.terminal_reserve_llm_calls,
                "budget_lifecycle": dynamic_budget.lifecycle_payload(
                    tokens_used=prompt_tokens + completion_tokens,
                    llm_calls_used=llm_calls,
                ),
            }

        def request_budget_status() -> dict[str, Any]:
            # Local ContextRequests/corrections are billed before the enclosing
            # semantic turn publishes usage. Read the one physical Gateway ledger
            # so the next request cannot reuse stale counters from that turn.
            usage = aggregate_provider_usage(provider_gateway.records)
            dynamic_budget.set_run_usage(
                tokens_used=int(usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)),
                llm_calls_used=sum(record.attempt_count for record in provider_gateway.records),
                cost_used=float(usage.get("cost", 0.0)),
            )
            return {**dynamic_budget.model_status(),
                    "remaining_steps": max(0, self.config.max_steps - state.step)}

        def record_budget_rejection(stage: str, reason: str = "") -> None:
            nonlocal budget_rejected_attempts
            if stage in {"planner", "reflection", "review", "contract_repair"}:
                rejection_key = (str(stage), int(state.step), str(reason or "local_budget_admission"))
                if rejection_key in budget_rejection_keys:
                    return
                budget_rejection_keys.add(rejection_key)
                budget_rejected_attempts += 1
                token_telemetry.append({
                    "stage": stage,
                    "system_tokens": None,
                    "tool_schema_tokens": None,
                    "control_state_tokens": None,
                    "diagnostic_context_tokens": None,
                    "provider_overhead_tokens": None,
                    "unattributed_prompt_tokens": None,
                    "token_accounting_error": False,
                    "accounting_delta": None,
                    "estimated_prompt_tokens": None,
                    "actual_prompt_tokens": None,
                    "actual_completion_tokens": None,
                    "cached_tokens": None,
                    "call_status": "rejected",
                })
                trace.record("LLM_ATTEMPT_REJECTED", {
                    "stage": stage,
                    "reason": reason or "local_budget_admission",
                    "failure_category": "run_budget_admission_exceeded",
                    "provider_calls": 0,
                    "call_status": "rejected",
                })

        def admit(stage: str, *, count_llm: bool = False,
                  allow_targeted_recovery: bool = False) -> None:
            nonlocal llm_calls, planner_calls, review_calls
            capacity = ProgressPolicy.evaluate(operational_facts("admission", stage=stage))
            if "global_deadline" in capacity.reasons:
                raise TimeoutError("global run deadline exceeded")
            if stage == "review":
                try:
                    dynamic_budget.begin_review(
                        tokens_used=prompt_tokens + completion_tokens,
                        llm_calls_used=llm_calls,
                    )
                except ValueError as exc:
                    lifecycle = dynamic_budget.lifecycle_payload(
                        tokens_used=prompt_tokens + completion_tokens,
                        llm_calls_used=llm_calls,
                    )
                    trace.record("BUDGET_LIFECYCLE_VIOLATION", {
                        "stage": stage,
                        "message": str(exc),
                        "budget_lifecycle": lifecycle,
                    })
                    record_budget_rejection(stage, str(exc))
                    raise RunBudgetAdmissionExceeded(replace(
                        dynamic_budget.decision(
                            stage,
                            tokens_used=prompt_tokens + completion_tokens,
                            llm_calls_used=llm_calls,
                            cost_used=run_cost,
                        ),
                        state=BudgetState.HARD_PRESSURE,
                        reason="review handoff exceeded resolved investigation allocation",
                        breakdown={"budget_lifecycle": lifecycle},
                    )) from exc
            snapshot = budget_snapshot("pre_" + stage)
            dynamic_budget.set_run_usage(
                tokens_used=prompt_tokens + completion_tokens,
                llm_calls_used=llm_calls,
                cost_used=run_cost,
            )
            dynamic_state = dynamic_budget.decision(
                stage, tokens_used=prompt_tokens, llm_calls_used=llm_calls,
                cost_used=run_cost,
            )
            if dynamic_state.state is not BudgetState.NORMAL:
                trace.record("RUN_BUDGET_PRESSURE", {
                    "stage": stage,
                    "context_state": dynamic_state.state.value,
                    "reason": dynamic_state.reason,
                    "remaining_run_tokens": dynamic_state.remaining_run_tokens,
                    "remaining_run_cost": dynamic_state.remaining_run_cost,
                    "terminal_reserve_tokens": dynamic_state.terminal_reserve_tokens,
                    "terminal_reserve_llm_calls": dynamic_state.terminal_reserve_llm_calls,
                })
                if "terminal reserve" in dynamic_state.reason:
                    trace.record("TERMINAL_RESERVE_PROTECTED", {
                        "stage": stage,
                        "reason": dynamic_state.reason,
                        "remaining_run_tokens": dynamic_state.remaining_run_tokens,
                        "remaining_run_cost": dynamic_state.remaining_run_cost,
                    })
                    # Exploration and semantic recovery must never consume
                    # the terminal path.  FINALIZE is deterministic and is
                    # therefore deliberately not admitted through this gate;
                    # callers either finalize an already-valid hypothesis or
                    # terminate INCONCLUSIVE.
            capacity = ProgressPolicy.evaluate(operational_facts(
                "admission", stage=stage, count_llm=count_llm,
                terminal_reserve_available="terminal reserve" not in dynamic_state.reason,
                targeted_recovery_allowed=(allow_targeted_recovery or (stage == "planner"
                    and dynamic_budget.lifecycle_phase == "consolidation")),
            ))
            if capacity.action == "stop":
                reason = capacity.reasons[0]
                if count_llm:
                    record_budget_rejection(stage, reason)
                if reason in {"terminal_reserve_protected", "review_call_reserve_exhausted"}:
                    raise RunBudgetAdmissionExceeded(dynamic_state)
                raise RuntimeError(reason + " exceeded")
            if count_llm:
                llm_calls += 1
                if stage == "planner":
                    planner_calls += 1
                elif stage == "review":
                    review_calls += 1
                    dynamic_budget.set_review_calls_used(review_calls)


        def provider_timeout(configured: float) -> float:
            remaining = deadline.remaining()
            timeout = min(float(configured), remaining)
            if timeout <= 0:
                raise LLMDeadlineExceeded("diagnosis run deadline exceeded")
            return timeout

        def account_review_attempts(reviewer: FinalReviewAgent) -> None:
            """Keep logical admission separate from ledger-derived physical usage."""
            nonlocal llm_calls, review_calls, schema_repair_count
            nonlocal schema_repair_calls
            call_count = max(1, int(getattr(reviewer, "last_call_count", 1) or 1))
            extra_calls = call_count - 1
            llm_calls += extra_calls
            attempts = tuple(getattr(reviewer, "last_attempts", ()) or ())
            repaired_calls = sum(1 for item in attempts if bool(item.get("schema_repair")))
            # If a hard process boundary kills the child before its state can
            # be copied back, the parent still knows that the Review object
            # entered its bounded repair path from this marker/call count.
            if repaired_calls == 0 and bool(getattr(reviewer, "last_repair_attempted", False)):
                repaired_calls = 1
            schema_repair_count += repaired_calls
            schema_repair_calls += repaired_calls
            dynamic_budget.set_review_calls_used(review_calls)
            accounting = provider_gateway.accounting(
                item for item in provider_gateway.records[review_record_cursor:]
                if item.role == "review"
            )
            trace.record("REVIEW_ATTEMPTS_ACCOUNTED", {
                "logical_review_calls": review_calls,
                "round": review_rounds,
                "role_invocations": call_count,
                "physical_review_attempts": accounting["provider_attempts"],
                "provider_accounting": accounting,
                "schema_repair_calls": repaired_calls,
                "schema_repairs_consume_logical_round": False,
            })

        def materialize_review_attempts(reviewer: FinalReviewAgent, attempt_count: int,
                                        *, effective_timeout: float,
                                        started_at: float,
                                        terminal_error: Exception | None = None) -> None:
            """Recover Review stage telemetry lost when a child is hard-killed."""
            attempt_count = max(0, int(attempt_count))
            if attempt_count <= int(getattr(reviewer, "last_call_count", 0) or 0):
                return
            attempts = [dict(item) for item in getattr(reviewer, "last_attempts", ())]
            while len(attempts) < attempt_count:
                call_index = len(attempts) + 1
                attempts.append({
                    "stage": "first_pass" if call_index == 1 else "schema_repair",
                    "call_index": call_index,
                    "schema_repair": call_index > 1,
                    "effective_timeout": effective_timeout,
                    "elapsed_seconds": 0.0,
                    "status": "schema_invalid" if call_index == 1 and attempt_count > 1 else "failed",
                    "failure_type": "schema_validation" if call_index == 1 and attempt_count > 1 else "",
                })
            if terminal_error is not None:
                attempts[-1].update({
                    "status": "failed",
                    "failure_type": type(terminal_error).__name__,
                    "elapsed_seconds": max(0.0, time.monotonic() - started_at),
                })
                reviewer.last_failure_type = type(terminal_error).__name__
            reviewer.last_call_count = attempt_count
            reviewer.last_attempts = attempts
            if attempt_count > 1:
                reviewer.last_repair_attempted = True

        def record_review_trace(*, round_number: int, configured_timeout: float,
                                effective_timeout: float, remaining_run_deadline: float,
                                started_at: float, reviewer: FinalReviewAgent,
                                terminal_error: Exception | None = None) -> None:
            """Project Review's internal stage telemetry into the run trace."""
            attempts = [dict(item) for item in getattr(reviewer, "last_attempts", ())]
            if not attempts:
                # A process-level hard kill can happen before child state is
                # returned. Preserve the outer boundary even in that case.
                attempts = [{
                    "stage": "first_pass",
                    "call_index": 1,
                    "schema_repair": False,
                    "effective_timeout": effective_timeout,
                    "elapsed_seconds": max(0.0, time.monotonic() - started_at),
                    "status": "failed" if terminal_error else "completed",
                    "failure_type": type(terminal_error).__name__ if terminal_error else "",
                }]
            first_pass_elapsed = sum(
                float(item.get("elapsed_seconds", 0.0) or 0.0)
                for item in attempts if item.get("stage") == "first_pass"
            )
            repair_elapsed = sum(
                float(item.get("elapsed_seconds", 0.0) or 0.0)
                for item in attempts if item.get("stage") == "schema_repair"
            )
            if getattr(reviewer, "last_normalization_actions", None):
                trace.record("REVIEW_REASONING_METADATA_NORMALIZED", {
                    "round": round_number,
                    "actions": list(reviewer.last_normalization_actions),
                })
            if getattr(reviewer, "last_symbol_normalizations", None):
                trace.record("MODEL_OUTPUT_CANONICALIZED", {
                    "role": "review", "events": reviewer.last_symbol_normalizations,
                })
            if (
                getattr(reviewer, "last_metadata_drops", None)
                or getattr(reviewer, "last_metadata_warnings", None)
            ):
                trace.record("REVIEW_REASONING_METADATA_DROPPED", {
                    "round": round_number,
                    "dropped_fields": list(getattr(reviewer, "last_metadata_drops", ())),
                    "warnings": list(getattr(reviewer, "last_metadata_warnings", ())),
                })
            for item in attempts:
                failure_type = str(item.get("failure_type") or "")
                trace.record("REVIEW_STAGE", {
                    "round": round_number,
                    "stage": item.get("stage", "unknown"),
                    "call_index": int(item.get("call_index", 0) or 0),
                    "schema_repair": bool(item.get("schema_repair")),
                    "configured_timeout": float(configured_timeout),
                    "effective_timeout": item.get("effective_timeout"),
                    "remaining_run_deadline": float(remaining_run_deadline),
                    "elapsed_seconds": float(item.get("elapsed_seconds", 0.0) or 0.0),
                    "first_pass_elapsed": first_pass_elapsed,
                    "repair_elapsed": repair_elapsed,
                    "status": item.get("status", "unknown"),
                    "failure_type": failure_type,
                    "deterministic_normalized": bool(
                        getattr(reviewer, "last_deterministic_normalized", False)
                    ),
                    "prompt_breakdown": dict(getattr(reviewer, "last_prompt_breakdown", {}) or {}),
                    "prompt_breakdowns": list(getattr(reviewer, "last_prompt_breakdowns", ()) or ()),
                })
            trace.record(
                "REVIEW_FAILED" if terminal_error else "REVIEW_COMPLETED",
                {
                    "round": round_number,
                    "configured_timeout": float(configured_timeout),
                    "effective_timeout": float(effective_timeout),
                    "remaining_run_deadline": float(remaining_run_deadline),
                    "elapsed_seconds": max(0.0, time.monotonic() - started_at),
                    "first_pass_elapsed": first_pass_elapsed,
                    "repair_elapsed": repair_elapsed,
                    "call_index": max(int(item.get("call_index", 0) or 0) for item in attempts),
                    "schema_repair": any(bool(item.get("schema_repair")) for item in attempts),
                    "failure_type": (
                        type(terminal_error).__name__ if terminal_error
                        else str(getattr(reviewer, "last_failure_type", "") or "")
                    ),
                    "repair_attempted": bool(getattr(reviewer, "last_repair_attempted", False)),
                    "deterministic_normalized": bool(
                        getattr(reviewer, "last_deterministic_normalized", False)
                    ),
                    "usage": dict(getattr(reviewer, "last_usage", {}) or {}),
                    "prompt_breakdown": dict(getattr(reviewer, "last_prompt_breakdown", {}) or {}),
                    "prompt_breakdowns": list(getattr(reviewer, "last_prompt_breakdowns", ()) or ()),
                },
            )

        def update_capability_state(tool_name: str, observation: ToolObservation) -> None:
            nonlocal capabilities
            previous_states = dict(capabilities.states)
            capabilities = capabilities.observe_tool(tool_name, observation)
            if getattr(observation, "error_type", None) == "snapshot_unavailable":
                trace.record("CAPABILITY_QUERY_BOUNDARY", {
                    "tool": tool_name,
                    "observation_id": observation.observation_id,
                    "reason": "snapshot_query_missing; source_capability_preserved",
                    "states": dict(capabilities.states),
                })
            changed = {
                key: value for key, value in capabilities.states.items()
                if previous_states.get(key) != value
            }
            if changed:
                trace.record("CAPABILITY_STATE_UPDATED", {
                    "tool": tool_name,
                    "states": changed,
                    "observation_id": observation.observation_id,
                })

        def execute_tool(arguments: dict[str, Any], tool_name: str, *,
                         permitted_tools: tuple[str, ...] | None = None,
                         visible_tools: tuple[str, ...] | None = None,
                         requested_for: InvestigationUse = InvestigationUse(),
                         action_record: dict[str, Any] | None = None) -> ToolObservation | DuplicateAcquisition:
            nonlocal blocked_tool_call_count
            available = tuple(capabilities.planner_tool_names(tools))
            observation = action_runtime.execute(
                ActionExecutionContext(
                    action=ActionProposal(
                        kind=ActionKind.TOOL, skill=active_skill, reason="runtime_execution",
                        tool=tool_name, arguments=arguments,
                    ),
                    permitted_tool_names=available if permitted_tools is None else permitted_tools,
                    visible_tool_names=available if visible_tools is None else visible_tools,
                    capability_tool_names=available,
                    absolute_deadline=deadline.absolute_deadline,
                    requested_for=requested_for,
                ),
                before_execution=lambda: admit("tool"),
                record=trace.record, action_record=action_record,
            )
            if isinstance(observation, DuplicateAcquisition):
                return observation
            if observation.error_type == "tool_circuit_open":
                blocked_tool_call_count += 1
            update_capability_state(tool_name, observation)
            return observation

        def admit_tool_observation(
            observation: ToolObservation, *, tool_name: str,
            arguments: dict[str, Any], obligation: Any | None = None,
            hypothesis: Any | None = None,
        ) -> tuple[str, ...]:
            """Capture execution facts only; the next Planner proposes meaning."""
            if isinstance(observation, DuplicateAcquisition):
                return observation
            if observation.error_type == "tool_circuit_open":
                pending_capability_blocks.add(tool_name)
            observation = replace(observation, metadata={
                **dict(observation.metadata or {}),
                "acquisition_hypothesis_scopes": (
                    evidence_projector.hypothesis_scope_fingerprints(hypothesis)
                    if hypothesis is not None else ()
                ),
            })
            state.observations.append(observation)
            observation_store.add(observation)
            trace.record("TOOL_OBSERVATION", observation)
            return ()

        def admit_planner_evidence(proposal: EvidenceProposal, *, turn_input, proposal_index, hypothesis=None) -> tuple[str, ...]:
            """Submit new admission or explicit reuse to the same EvidencePipeline."""
            visible = set(turn_input.context_view.visible_observation_ids)
            reusable = set(turn_input.context_view.visible_evidence_refs)
            admitted_ids: list[str] = []
            hypothesis = hypothesis or active_hypothesis()
            proposal = proposal.model_copy(update={"hypothesis_id": evidence_projector.hypothesis_id(hypothesis)})
            try:
                if proposal.evidence_ref is not None:
                    evidence = evidence_memory.get(proposal.evidence_ref)
                    if evidence is None or evidence.evidence_id not in reusable:
                        raise EvidenceProposalError("proposal requires visible canonical Evidence")
                    observation_id = evidence.raw_observation_id
                else:
                    if len(proposal.observation_refs) != 1 or proposal.observation_refs[0] not in visible:
                        raise EvidenceProposalError("proposal requires one visible Observation")
                    observation_id = proposal.observation_refs[0]
                observation = observation_store.get(observation_id)
                if observation is None or not observation.ok:
                    raise EvidenceProposalError("proposal requires a successful stored Observation")
                obligation = canonical_rca_state.obligation(proposal.obligation_id)
            except (EvidenceProposalError, RCAStateError, ValueError) as exc:
                trace.record("EVIDENCE_PROPOSAL_REJECTED", {
                    "proposal_index": proposal_index, "obligation_id": proposal.obligation_id,
                    "observation_refs": list(proposal.observation_refs), "reason": str(exc),
                    "reason_code": "planner_evidence_binding_invalid",
                })
                return ()
            tool_name = observation.tool
            target = self._target(dict(observation.metadata.get("arguments") or {}))
            projected = incident_projection.compact_content(observation)
            evidence_id = f"ev-{len(evidence_memory.pinned) + 1:03d}"
            trace.record("EVIDENCE_PROPOSAL", {
                "observation_id": observation.observation_id,
                "obligation_id": str(getattr(obligation, "id", None) or getattr(obligation, "obligation_id", "") or ""),
                **proposal.model_dump(mode="json"),
            })
            try:
                relation_input = dict(
                    proposal=proposal, obligation=obligation, hypothesis=hypothesis,
                    incident_id=case.case_id, existing_facts=tuple(projected_facts.values()),
                    rca_state=canonical_rca_state, hypothesis_scopes=(
                        evidence_projector.hypothesis_scope_fingerprints(hypothesis)
                        if hypothesis is not None else None
                    ),
                )
                if proposal.evidence_ref is not None:
                    pipeline_result = evidence_pipeline.reuse(
                        **relation_input, reusable_evidence_ids=reusable,
                    )
                else:
                    pipeline_result = evidence_pipeline.admit(
                        observation, **relation_input, evidence_id=evidence_id,
                        visible_observation_ids=visible, evidence_kwargs={
                            "kind": str(observation.metadata.get("context_kind") or "OBSERVATION"),
                            "source": tool_name,
                            "summary": projected,
                            "excerpt": projected,
                            "target": target,
                            "tags": ["task_kind:incident"],
                            "enforce_admission": self.config.features.evidence_lifecycle,
                            "deduplicate": self.config.features.evidence_lifecycle,
                        },
                    )
            except (EvidencePipelineError, ValueError) as exc:
                trace.record("EVIDENCE_PROPOSAL_REJECTED", {
                    "proposal_index": proposal_index, "obligation_id": proposal.obligation_id,
                    "observation_id": observation.observation_id,
                    "tool": tool_name,
                    "reason_code": getattr(exc, "code", "proposal_validation_failed"),
                    "reason": str(exc),
                })
                return ()
            if pipeline_result.duplicate:
                trace.record("EVIDENCE_PROPOSAL_DUPLICATE", {
                    "observation_id": observation.observation_id,
                    "obligation_id": obligation.id,
                    "evidence_id": pipeline_result.evidence_id,
                })
                return (pipeline_result.evidence_id,)
            shared_evidence = evidence_memory.evidence_for_observation(
                observation.observation_id
            )
            projected_fact = pipeline_result.projected_fact
            if shared_evidence is not None and projected_fact is not None:
                admitted_ids.append(shared_evidence.evidence_id)
                relationship_key = evidence_relationship_key(projected_fact)
                projected_facts[relationship_key] = projected_fact
                trace.record("EVIDENCE_PROJECTED", {
                    **projected_fact.model_dump(mode="json"),
                    "relationship_key": list(relationship_key),
                })
                item = IncidentEvidence(
                    evidence_id=shared_evidence.evidence_id,
                    source=shared_evidence.source,
                    target=shared_evidence.target or target,
                    summary=shared_evidence.summary,
                    observation_id=(
                        shared_evidence.raw_observation_id
                        or observation.observation_id
                    ),
                )
                trace.record("EVIDENCE_REUSED" if proposal.evidence_ref else "EVIDENCE_ADDED", item)
                qualify_current_evidence_set(
                    obligation,
                    hypothesis,
                    reason="planner_evidence_admitted",
                )

            return tuple(admitted_ids)



        def incident_evidence() -> tuple[IncidentEvidence, ...]:
            """Project the shared ledger only at Review/serialization boundaries."""
            return tuple(
                IncidentEvidence(
                    evidence_id=item.evidence_id,
                    source=item.source,
                    target=item.target or "",
                    summary=item.summary,
                    observation_id=item.raw_observation_id or "",
                    excerpt=item.excerpt,
                    file=item.file,
                    start_line=item.source_start_line,
                    end_line=item.source_end_line,
                    raw_observation_id=item.raw_observation_id,
                    truncation=item.excerpt_truncated,
                    tags=tuple(item.tags),
                    provenance=dict(item.provenance),
                )
                for item in evidence_memory.pinned
            )

        def record_semantic_decision(selection, prior_hypothesis, committed_hypothesis) -> None:
            """Persist model meaning after the whole semantic transaction validates.

            Evidence bodies and provenance stay in their existing stores.  This
            journal only records the model's relationship-scoped interpretation
            plus canonical refs, so a later turn can recover meaning without a
            second summarizer call or a second semantic owner.
            """
            nonlocal canonical_rca_state
            facts = tuple(projected_facts.values())
            interpretations: list[SemanticEvidenceRecord] = []
            for proposal in tuple(getattr(selection, "evidence_proposals", ()) or ()):
                if proposal.evidence_ref:
                    evidence = evidence_memory.get(proposal.evidence_ref)
                else:
                    ref = proposal.observation_refs[0] if proposal.observation_refs else ""
                    evidence = evidence_memory.evidence_for_observation(ref)
                if evidence is None:
                    # Invalid/unadmitted proposals never become canonical
                    # semantic history; the admission path already recorded
                    # their rejection.
                    continue
                candidates = [
                    fact for fact in facts
                    if fact.evidence_id == evidence.evidence_id
                    and fact.obligation_id == proposal.obligation_id
                    and fact.hypothesis_id == evidence_projector.hypothesis_id(committed_hypothesis)
                ]
                fact = candidates[-1] if candidates else None
                if fact is None:
                    # A duplicate/no-op proposal has no newly admitted
                    # relationship to journal; the prior canonical record is
                    # already recoverable by its Evidence ref.
                    continue
                interpretations.append(SemanticEvidenceRecord(
                    evidence_id=evidence.evidence_id,
                    observation_id=evidence.raw_observation_id or evidence.evidence_id,
                    semantic_claim=proposal.semantic_claim,
                    hypothesis_relation=proposal.hypothesis_relation,
                    obligation_relation=proposal.relation,
                    covered_fact_slots=tuple(getattr(fact, "covered_fact_slots", ()) or proposal.covered_fact_slots),
                    reason=proposal.reason,
                    rationale=proposal.rationale,
                    source_spans=tuple(proposal.source_spans),
                ))
            transition = getattr(selection, "hypothesis_transition", None)
            kind = transition.kind if transition is not None else "KEEP_CURRENT"
            reason = " ".join(str(getattr(selection, "reason", "") or "").split()).strip()
            if not reason:
                reason = "semantic decision retained without additional hypothesis rationale"
            changed_ids = tuple(
                dict.fromkeys(
                    str(item.obligation_id)
                    for item in (getattr(transition, "obligation_dispositions", ()) or ())
                    if str(item.obligation_id)
                )
            )
            record = SemanticDecisionRecord(
                record_id=f"sem-{len(canonical_rca_state.semantic_records) + 1:04d}",
                state_version=canonical_rca_state.event_count,
                transition_kind=kind,
                hypothesis_id=evidence_projector.hypothesis_id(committed_hypothesis),
                hypothesis_revision=int(getattr(committed_hypothesis, "revision", 0) or 0),
                hypothesis_claim=str(getattr(committed_hypothesis, "claim", "") or ""),
                change_reason=reason,
                evidence=tuple(interpretations),
                changed_obligation_ids=changed_ids,
            )
            canonical_rca_state = canonical_rca_state.reduce({
                "kind": "ADD_SEMANTIC_RECORD",
                "payload": {"record": record.model_dump(mode="json")},
            })
            trace.record("SEMANTIC_DECISION_RECORDED", record.model_dump(mode="json"))

        def source_evidence_ids() -> tuple[str, ...]:
            """Return canonical CODE Evidence from verified source-reading tools."""
            return evidence_memory.source_evidence_ids()

        def evaluate_source_claims(claims: tuple[SourceClaim, ...] | list[SourceClaim] = ()) -> dict[str, Any]:
            """Check claim/evidence coverage without judging claim truth.

            A source claim is covered only when every line in its declared file
            range is contained by the union of its cited canonical ``read_file``
            CODE Evidence ranges.  This is intentionally an agreement check
            between the Planner declaration and observed source Evidence; Gold
            and fault labels never participate.
            """
            def canonical_path(value: str) -> str:
                raw = str(value or "").replace("\\", "/").strip().lstrip("./")
                # Source tools expose paths relative to the bound ``code`` root.
                # Accept one explicit root prefix, but never recursively strip it.
                if raw == "code":
                    return "."
                if raw.startswith("code/"):
                    raw = raw[len("code/"):]
                return raw or "."

            canonical_by_id = {
                item.evidence_id: item for item in evidence_memory.pinned
                if is_canonical_source_evidence(None, item, evidence_memory=evidence_memory)
            }
            results: list[dict[str, Any]] = []
            for claim in tuple(claims or ()):
                wanted_path = canonical_path(claim.file)
                intervals: list[tuple[int, int, str]] = []
                missing_ids: list[str] = []
                for evidence_id in claim.evidence_ids:
                    evidence = canonical_by_id.get(str(evidence_id))
                    if evidence is None:
                        missing_ids.append(str(evidence_id))
                        continue
                    if canonical_path(evidence.file or "") != wanted_path:
                        missing_ids.append(str(evidence_id))
                        continue
                    start = evidence.source_start_line
                    end = evidence.source_end_line
                    if not isinstance(start, int) or not isinstance(end, int):
                        missing_ids.append(str(evidence_id))
                        continue
                    intervals.append((start, end, evidence.evidence_id))
                intervals.sort()
                cursor = claim.start_line
                for start, end, _ in intervals:
                    if end < cursor:
                        continue
                    if start > cursor:
                        break
                    cursor = max(cursor, end + 1)
                    if cursor > claim.end_line:
                        break
                covered = bool(intervals) and cursor > claim.end_line and not missing_ids
                results.append({
                    "file": wanted_path,
                    "symbol": claim.symbol,
                    "start_line": claim.start_line,
                    "end_line": claim.end_line,
                    "claim": claim.claim,
                    "evidence_ids": list(claim.evidence_ids),
                    "covered": covered,
                    "missing_or_mismatched_evidence_ids": sorted(set(missing_ids)),
                })
            return {
                "claims": results,
                "claim_count": len(results),
                "covered_claim_count": sum(1 for item in results if item["covered"]),
                "all_claims_covered": bool(results) and all(item["covered"] for item in results),
                "source_evidence_ids": sorted(canonical_by_id),
            }

        # Provider payloads may contain broad prose gaps. They are retained for
        # audit and convergence diagnostics
        # but are never admitted as canonical obligations or closure signals.
        unstructured_obligation_gaps: tuple[str, ...] = ()

        def reduce_runtime_hypothesis(
            proposed: IncidentHypothesis, *, reason: str,
        ) -> IncidentHypothesis:
            """Commit a Runtime-normalized semantic snapshot through RCAState.

            Provider objects are proposals. Every accepted semantic change is
            reduced here before consumers receive a detached projection.
            """
            nonlocal canonical_rca_state
            try:
                for evidence in evidence_memory.pinned:
                    if evidence.evidence_id not in canonical_rca_state.evidence_ids:
                        canonical_rca_state = canonical_rca_state.reduce({
                            "kind": "LINK_EVIDENCE",
                            "payload": {"evidence_id": evidence.evidence_id},
                        })
                canonical_by_id = {
                    item.id: item for item in canonical_rca_state.obligations
                }
                normalized_obligations = []
                for item in proposed.verification_obligations:
                    current = canonical_by_id.get(item.id)
                    if current is None:
                        raise RCAStateError(
                            f"hypothesis references an unadmitted Obligation: {item.id}"
                        )
                    elif current != item:
                        # Terminal/capability decisions are Runtime-owned. A
                        # stale Planner projection cannot reopen them.
                        if current.status in {
                            "SATISFIED", "SUPERSEDED_BY_REPLAN",
                        } and item.status != current.status:
                            item = current
                        elif current.status == "BLOCKED_BY_CAPABILITY" and item.status == "OPEN":
                            item = current
                        else:
                            canonical_rca_state = canonical_rca_state.reduce({
                                "kind": "UPDATE_OBLIGATION",
                                "payload": {
                                    "obligation_id": item.id,
                                    "status": item.status,
                                    "supporting_evidence_ids": list(item.supporting_evidence_ids),
                                    "blocked_capabilities": list(item.blocked_capabilities),
                                },
                            })
                            current = canonical_rca_state.obligation(item.id)
                    normalized_obligations.append(current)
                normalized = proposed.model_copy(update={
                    "verification_obligations": tuple(normalized_obligations),
                })
                canonical_rca_state = canonical_rca_state.reduce({
                    "kind": "SET_CONTRADICTIONS",
                    "payload": {
                        "contradictions": [
                            item.model_dump() for item in normalized.contradictions
                        ],
                    },
                })
                canonical_rca_state = canonical_rca_state.reduce({
                    "kind": "SET_HYPOTHESIS",
                    "payload": {"hypothesis": normalized.model_dump()},
                })
            except RCAStateError as exc:
                trace.record("RCA_STATE_REJECTED", {
                    "reason": reason,
                    "error": str(exc),
                })
                raise RuntimeError(f"RCAState rejected semantic update: {exc}") from exc
            project_obligation_ids_to_dag(reason)
            committed = canonical_rca_state.active_hypothesis_projection()
            if committed is None:
                raise RuntimeError("RCAState did not commit an active hypothesis")
            trace.record("RCA_STATE_UPDATED", {
                "source": "rca_state_reducer",
                "reason": reason,
                "incident_id": case.case_id,
                "event_count": canonical_rca_state.event_count,
                "obligation_count": len(canonical_rca_state.obligations),
                "task_count": len(dag_state.tasks) if dag_state is not None else 0,
                "hypothesis_count": len(canonical_rca_state.hypotheses),
                "open_critical_obligation_ids": list(
                    canonical_rca_state.open_critical_obligation_ids
                ),
            })
            return committed

        def qualify_current_evidence_set(
            obligation: Any,
            hypothesis: Any | None = None,
            *,
            reason: str,
        ) -> EvidenceSet:
            """Aggregate admitted projected facts and apply the Runtime rule."""
            nonlocal canonical_rca_state
            obligation_id = str(
                getattr(obligation, "id", None)
                or getattr(obligation, "obligation_id", "")
                or ""
            )
            hypothesis_id = str(
                getattr(hypothesis, "hypothesis_id", None)
                or getattr(hypothesis, "id", None)
                or ""
            )
            allowed_scopes = set(
                evidence_projector.hypothesis_scope_fingerprints(hypothesis)
            )
            facts = tuple(
                item for item in projected_facts.values()
                if item.obligation_id == obligation_id
                and (not allowed_scopes or item.hypothesis_fingerprint in allowed_scopes)
            )
            completion_prerequisites_met = True
            completion_blocker = ""
            # Source line coverage is a deterministic prerequisite. It can
            # veto coverage completion but cannot create semantic coverage.
            if _is_source_mechanism_obligation(obligation):
                source_capability_gaps = []
                if not source_workspace_available:
                    source_capability_gaps.append("source_workspace")
                if "application_code" not in case.evidence_sources:
                    source_capability_gaps.append("application_code")
                if "read_file" not in available_code_tools:
                    source_capability_gaps.append("read_file")
                source_claims = tuple(getattr(hypothesis, "source_claims", ()) or ())
                source_coverage = evaluate_source_claims(source_claims)
                if (
                    not source_capability_gaps
                    and source_claims
                    and not source_coverage["all_claims_covered"]
                ):
                    completion_prerequisites_met = False
                    completion_blocker = (
                        "declared source claim lacks complete deterministic line coverage"
                    )
            try:
                evidence_set = evidence_pipeline.qualify(
                    obligation=obligation,
                    evidence=facts,
                    hypothesis_id=hypothesis_id,
                    hypothesis_scopes=(
                        allowed_scopes if hypothesis is not None else None
                    ),
                    completion_prerequisites_met=completion_prerequisites_met,
                    completion_blocker=completion_blocker,
                    resolved_contradiction_ids=(
                        item.evidence_id
                        for item in canonical_rca_state.contradictions
                        if item.status != "OPEN"
                    ),
                )
            except EvidencePipelineError as exc:
                trace.record("EVIDENCE_PIPELINE_REJECTED", {
                    "obligation_id": obligation_id,
                    "reason_code": exc.code,
                    "reason": str(exc),
                })
                evidence_set = evidence_pipeline.qualify(
                    obligation=obligation,
                    evidence=(),
                    hypothesis_id=hypothesis_id,
                    hypothesis_scopes=(
                        allowed_scopes if hypothesis is not None else None
                    ),
                )
            evidence_sets[(obligation_id, hypothesis_id)] = evidence_set
            trace.record("EVIDENCE_SET_QUALIFIED", {
                "obligation_id": obligation_id,
                "hypothesis_id": hypothesis_id,
                "reason": reason,
                **evidence_set.model_dump(mode="json"),
            })
            try:
                rca_events = evidence_pipeline.propose_rca_events(
                    evidence_set,
                    facts=facts,
                    state=canonical_rca_state,
                )
                for event in rca_events:
                    canonical_rca_state = canonical_rca_state.reduce(event)
                    if event.kind == "UPDATE_OBLIGATION":
                        trace.record("OBLIGATION_SATISFIED", {
                            "source": "evidence_pipeline_coverage",
                            "obligation_id": obligation_id,
                            "evidence_ids": list(evidence_set.supporting_evidence_ids),
                        })
                # Refresh only the DAG's canonical Obligation-ID references.
                # Task serialization cannot write semantic lifecycle state.
                project_obligation_ids_to_dag("evidence_set_qualification")
            except RCAStateError as exc:
                trace.record("EVIDENCE_SET_STATE_REJECTED", {
                    "obligation_id": obligation_id,
                    "reason": str(exc),
                })
            return evidence_set

        def completion_provenance_profiles():
            from debug_assistant.tools.provenance import registry_provenance_profiles, observation_provenance_profiles
            return registry_provenance_profiles(tools, capabilities.planner_tool_names(tools)) + observation_provenance_profiles(
                [{'tool': obs.tool, 'ok': obs.ok, 'metadata': obs.metadata} for obs in observation_store.all()])

        def build_hypothesis(selection) -> IncidentHypothesis:
            """Normalize one canonical Planner proposal into RCAState input."""
            nonlocal canonical_rca_state
            nonlocal previous_hypothesis, obligation_created_count
            nonlocal obligation_blocked_count, blocking_contradiction_count
            nonlocal unstructured_obligation_gaps

            known_ids = {item.evidence_id for item in evidence_memory.pinned}
            old = active_hypothesis()
            # Source-obligation lifecycle is Runtime-owned.  The provider may
            # propose a status for audit purposes, but it cannot close or waive
            # this obligation.  Keep the capability decision deterministic and
            # reuse EvidenceMemory's canonical read_file/CODE predicate.
            source_ids = source_evidence_ids()
            cited_source_ids = tuple(
                evidence_id for evidence_id in source_ids
                if evidence_id in set(selection.supporting_evidence_ids)
            )
            source_capability_gaps: list[str] = []
            if not source_workspace_available:
                source_capability_gaps.append("source_workspace")
            if "application_code" not in case.evidence_sources:
                source_capability_gaps.append("application_code")
            if "read_file" not in available_code_tools:
                source_capability_gaps.append("read_file")
            source_capability_gaps = list(dict.fromkeys(source_capability_gaps))
            source_capability_available = not source_capability_gaps

            supplied = list(selection.verification_obligations or ())
            current_obligations = obligation_projection()
            next_obligations: dict[str, VerificationObligation] = {}
            rejected_gaps: list[str] = []

            selection_hypothesis_fingerprints = set(
                evidence_projector.hypothesis_scope_fingerprints(selection)
            )
            for proposal in supplied:
                try:
                    canonical_proposal = ObligationAdmissionPolicy.canonicalize(
                        proposal,
                        related_hypothesis=selection.current_hypothesis,
                    )
                except ObligationProposalError as exc:
                    claim = " ".join(str(getattr(proposal, "claim", "") or "").split()).strip()
                    if claim:
                        rejected_gaps.append(claim)
                    trace.record("OBLIGATION_PROPOSAL_REJECTED", {
                        "claim": claim,
                        "reason": "non_atomic_obligation",
                        "missing_fields": list(exc.missing_fields),
                    })
                    continue
                claim = canonical_proposal.claim
                target_component = canonical_proposal.target_component
                verification_question = canonical_proposal.verification_question
                evidence_type = canonical_proposal.evidence_type
                trace.record("OBLIGATION_PROPOSAL_CANONICALIZED", {
                    "claim": claim,
                    "target_component": target_component,
                    "related_hypothesis": canonical_proposal.related_hypothesis,
                    "active_hypothesis_component": selection.candidate_component,
                    "lifecycle_fields_runtime_owned": True,
                    "reason": "planner_proposal_to_runtime_identity_inputs",
                })
                verification_key = build_verification_key(
                    target_component=target_component,
                    verification_question=verification_question,
                    evidence_type=evidence_type,
                    related_hypothesis=selection.current_hypothesis,
                )
                matching_priors = tuple(
                    item for item in (*next_obligations.values(), *current_obligations.values())
                    if item.canonical_verification_key
                    and item.canonical_verification_key == verification_key
                )

                # Prefer the live instance when an older superseded branch
                # and a reactivated instance share the same structured key.
                prior = next(
                    (
                        item for item in matching_priors
                        if item.status != "SUPERSEDED_BY_REPLAN"
                    ),
                        matching_priors[0] if matching_priors else None,
                )

                if prior is not None:
                    duplicate_admission = ObligationAdmissionPolicy.admit(
                        proposal,
                        provenance_profiles=completion_provenance_profiles(), related_hypothesis=selection.current_hypothesis,
                        existing_obligations=matching_priors,
                    )
                    if duplicate_admission.status == "REJECTED":
                        trace.record("OBLIGATION_PROPOSAL_REJECTED", {
                            "claim": claim, "reason": "common_admission_rejected",
                            "admission_reasons": list(duplicate_admission.reasons),
                        })
                        continue

                if prior is not None and prior.status == "SUPERSEDED_BY_REPLAN":
                    # A superseded branch is historical state, not a live
                    # answer to the same structured question. Reusing its
                    # terminal ID would silently project a non-open
                    # obligation and make the Planner repeat the same Action
                    # forever. Reactivate the structured question as a new
                    # Runtime instance while keeping natural-language
                    # hypothesis prose out of the identity key.
                    trace.record("OBLIGATION_REACTIVATION", {
                        "previous_obligation_id": prior.id,
                        "verification_key": verification_key,
                        "reason": "superseded_branch_reintroduced_same_structured_question",
                    })
                    prior = None
                existing_ids = {
                    item.id for item in (
                        *canonical_rca_state.obligations,
                        *next_obligations.values(),
                        *current_obligations.values(),
                    )
                }
                obligation_identity = verification_key
                if (
                    prior is None
                    and obligation_id_for_key(obligation_identity) in existing_ids
                ):
                    identity_disambiguator = (
                        f"runtime_instance:{obligation_created_count + 1}"
                    )
                else:
                    identity_disambiguator = ""
                if prior is None:
                    source_requirement = _is_source_mechanism_obligation(canonical_proposal)
                    admission_capability_gaps = (
                        tuple(source_capability_gaps) if source_requirement
                        else _obligation_capability_gaps(canonical_proposal.model_dump(), capabilities)
                    )
                    admission = ObligationAdmissionPolicy.admit(
                        proposal,
                        provenance_profiles=completion_provenance_profiles(),
                        related_hypothesis=selection.current_hypothesis,
                        existing_obligations=tuple(
                            item for item in (
                                *next_obligations.values(), *current_obligations.values()
                            )
                            if item.status != "SUPERSEDED_BY_REPLAN"
                        ),
                        critical=True,
                        required_tools=("read_file",) if source_requirement else (),
                        available_tools=capabilities.planner_tool_names(tools),
                        capability_gaps=admission_capability_gaps,
                        identity_disambiguator=identity_disambiguator,
                    )
                    if not admission.admitted:
                        trace.record("OBLIGATION_PROPOSAL_REJECTED", {
                            "claim": claim,
                            "reason": "common_admission_rejected",
                            "admission_reasons": list(admission.reasons),
                            "capability_gaps": list(admission_capability_gaps),
                            "capability_states": {name: capabilities.capability_state(name)
                                                  for name in admission_capability_gaps},
                        })
                        continue
                    canonical_rca_state = canonical_rca_state.reduce(admission.accepted_event())
                    obligation = canonical_rca_state.obligation(admission.obligation_id)
                else:
                    # A follow-up binds to an existing canonical question; it
                    # does not replace its identity with Provider metadata.
                    obligation = canonical_rca_state.obligation(prior.id)
                proposed_status = next(
                    (
                        value for path, value in getattr(
                            selection, "planner_control_rejections", ()
                        )
                        if path.endswith(".status")
                    ),
                    "OPEN",
                )
                # Planner status/evidence projections are never lifecycle
                # writes. SATISFIED can only be produced by the EvidenceSet
                # completion reducer below.
                if prior is not None and prior.status != "OPEN":
                    # Planner proposals can restate a semantic question but
                    # cannot reopen or supersede a Runtime-owned lifecycle.
                    if prior.status == "SATISFIED":
                        if proposed_status == "OPEN":
                            trace.record("OBLIGATION_REOPEN_REJECTED", {
                                "obligation_id": prior.id,
                                "from": prior.status,
                                "to": "OPEN",
                                "reason": "terminal_runtime_state_preserved",
                            })
                        else:
                            trace.record("OBLIGATION_TERMINAL_REPLAY_REJECTED", {
                                "obligation_id": prior.id,
                                "status": proposed_status,
                                "reason": "terminal_runtime_state_preserved",
                            })
                    obligation = prior
                if _is_source_mechanism_obligation(obligation):
                    if source_capability_available:
                        # Strict source verification may only preserve a
                        # SATISFIED status that was already produced by the
                        # shared EvidenceSet completion path.  Merely having
                        # a read_file Evidence or complete source coverage is
                        # a deterministic prerequisite; it is not semantic
                        # RCA closure.
                        prior_source_ids = tuple(
                            evidence_id for evidence_id in (prior.supporting_evidence_ids if prior else ())
                            if evidence_id in set(source_ids)
                        )
                        runtime_status = (
                            "SATISFIED"
                            if obligation.status == "SATISFIED"
                            else "OPEN"
                        )
                        runtime_support = (
                            tuple(dict.fromkeys(prior_source_ids))
                            if runtime_status == "SATISFIED" else ()
                        )
                        runtime_blocked = ()
                    else:
                        runtime_status = "NOT_APPLICABLE_BY_CAPABILITY"
                        runtime_support = ()
                        runtime_blocked = tuple(source_capability_gaps)
                    obligation = obligation.model_copy(update={
                        "claim": _SOURCE_MECHANISM_OBLIGATION_CLAIM,
                        "critical": True,
                        "verification_role": "NECESSARY",
                        "status": runtime_status,
                        "supporting_evidence_ids": runtime_support,
                        "blocked_capabilities": runtime_blocked,
                    })
                    if (
                        proposed_status != runtime_status
                        or (prior is not None and prior.status != runtime_status)
                    ):
                        trace.record("SOURCE_OBLIGATION_STATUS_OVERRIDDEN", {
                            "obligation_id": obligation.id,
                            "proposed_status": proposed_status,
                            "previous_runtime_status": prior.status if prior else None,
                            "runtime_status": runtime_status,
                            "capability_available": source_capability_available,
                            "capability_gaps": list(source_capability_gaps),
                            "cited_source_evidence_ids": list(cited_source_ids),
                            "reason": "runtime_owned_source_obligation",
                        })
                    # Do not apply generic transition/terminal checks to a
                    # source obligation. A stale provider projection is
                    # normalized to the Runtime-owned OPEN/SATISFIED/N/A state;
                    # strict SATISFIED can only be replayed after EvidenceSet
                    # completion, never synthesized from source coverage.
                    if obligation.id not in current_obligations:
                        obligation_created_count += 1
                        trace.record("OBLIGATION_CREATED", obligation.model_dump())
                    elif prior != obligation:
                        trace.record("OBLIGATION_TRANSITION", {
                            "obligation_id": obligation.id,
                            "from": prior.status if prior else None,
                            "to": obligation.status,
                            "reason": "runtime_source_obligation_normalization",
                        })
                    next_obligations[obligation.id] = obligation
                    continue
                capability_gaps = _obligation_capability_gaps(
                    canonical_proposal.model_dump(), capabilities,
                )
                if capability_gaps and obligation.status != "SATISFIED":
                    runtime_status = "NOT_APPLICABLE_BY_CAPABILITY"
                    obligation = obligation.model_copy(update={
                        "status": runtime_status,
                        "supporting_evidence_ids": (),
                        "blocked_capabilities": capability_gaps,
                    })
                    trace.record("OBLIGATION_STATUS_OVERRIDDEN", {
                        "obligation_id": obligation.id,
                        "proposed_status": proposed_status,
                        "runtime_status": runtime_status,
                        "capability_gaps": list(capability_gaps),
                        "evidence_type": evidence_type,
                        "reason": "runtime_capability_boundary",
                    })
                if obligation.id not in current_obligations:
                    obligation_created_count += 1
                    trace.record("OBLIGATION_CREATED", obligation.model_dump())
                next_obligations[obligation.id] = obligation

            # Natural-language evidence gaps are diagnostics only. They never
            # become semantic obligations; every active obligation must arrive
            # through the structured proposal contract above.
            for gap in selection.required_evidence_gaps:
                claim = " ".join(str(gap).split()).strip()
                if not claim:
                    continue
                rejected_gaps.append(claim)
                trace.record("OBLIGATION_PROPOSAL_REJECTED", {
                    "claim": claim,
                    "reason": "unstructured_evidence_gap",
                })

            # Rehydrate canonical critical obligations into the next
            # provider projection.  The Planner may omit an existing
            # obligation while proposing a follow-up action, but that cannot
            # erase the Runtime/RCAState semantic requirement.  This is a
            # state projection only: it creates no obligation and changes no
            # lifecycle status.
            for canonical in canonical_rca_state.obligations:
                normalized_canonical, capability_gaps = (
                    _normalize_obligation_capability_status(canonical, capabilities)
                )
                if capability_gaps:
                    canonical = normalized_canonical
                    trace.record("OBLIGATION_STATUS_OVERRIDDEN", {
                        "obligation_id": canonical.id,
                        "proposed_status": "OPEN",
                        "runtime_status": canonical.status,
                        "capability_gaps": list(capability_gaps),
                        "reason": "runtime_capability_boundary_rehydration",
                    })
                if canonical.blocks_finalization and canonical.id not in next_obligations:
                    next_obligations[canonical.id] = canonical
                    trace.record("OBLIGATION_REHYDRATED", {
                        "obligation_id": canonical.id,
                        "status": canonical.status,
                        "reason": "preserve_open_critical_runtime_state",
                    })

            unstructured_obligation_gaps = tuple(dict.fromkeys(rejected_gaps))

            # Source coverage is a Runtime-owned capability/evidence decision.
            # The LLM field is retained for audit only and cannot close the
            # source obligation or declare it not applicable.
            proposed_source_status = getattr(
                selection, "source_mechanism_status", "unknown"
            ) or "unknown"
            active_source_claims = tuple(getattr(selection, "source_claims", ()) or ())
            if not active_source_claims and old is not None:
                active_source_claims = tuple(old.source_claims or ())
            source_coverage = evaluate_source_claims(active_source_claims)
            if active_source_claims and not source_coverage["all_claims_covered"]:
                raise SemanticTransitionError("source_claim_provenance_invalid",
                    json.dumps(source_coverage, ensure_ascii=False))
            trace.record("SOURCE_MECHANISM_RELEVANCE_EVALUATED", {
                "capability_available": source_capability_available,
                "capability_gaps": list(source_capability_gaps),
                **source_coverage,
                "reason": "runtime_file_and_complete_line_range_coverage_only",
            })
            claim_source_ids = tuple(dict.fromkeys(
                evidence_id
                for claim in active_source_claims
                for evidence_id in claim.evidence_ids
                if evidence_id in set(source_ids)
            ))
            source_support_ids = tuple(dict.fromkeys((*cited_source_ids, *claim_source_ids)))
            source_obligation = next(
                (
                    item for item in (*next_obligations.values(), *current_obligations.values())
                    if _is_source_mechanism_obligation(item)
                ),
                None,
            )
            source_declared = _source_investigation_declared(
                source_obligation=source_obligation,
                source_claims=active_source_claims,
                cited_source_ids=cited_source_ids,
                proposed_status=proposed_source_status,
            )
            source_status = (
                "not_applicable" if not source_capability_available
                else "sufficient"
                if source_coverage["all_claims_covered"]
                or (not active_source_claims and bool(source_support_ids))
                else "gap"
                if source_declared
                else "unknown"
            )
            if proposed_source_status != source_status:
                trace.record("SOURCE_MECHANISM_STATUS_OVERRIDDEN", {
                    "proposed_status": proposed_source_status,
                    "runtime_status": source_status,
                    "capability_available": source_capability_available,
                    "capability_gaps": list(source_capability_gaps),
                    "cited_source_evidence_ids": list(source_support_ids),
                    "source_claim_count": len(active_source_claims),
                    "reason": (
                        "runtime_owned_source_coverage"
                        if active_source_claims
                        else "runtime_owned_canonical_source_evidence"
                    ),
                })
            # A normal causal mechanism is not proof that the diagnosis is a
            # source investigation.  Source routing is declared only by an
            # explicit source status/claim, cited source Evidence, or an
            # already Runtime-owned source Obligation.  Otherwise a generic
            # mechanism such as "unverified" must not collapse the live Tool
            # surface to read_file before an exact path exists.
            # A source-specific Evidence Obligation may already have been
            # admitted and closed by the normal EvidenceSet path. When the
            # Planner merely confirms that canonical source Evidence supports
            # the mechanism, but declares no SourceClaim and no Runtime-owned
            # source coverage Obligation exists yet, do not append a second
            # synthetic coverage branch with no EvidenceSet closure path.
            # Explicit SourceClaims and an already-open source Obligation keep
            # the strict range-coverage lifecycle.
            source_coverage_already_grounded = bool(
                source_status == "sufficient"
                and source_support_ids
                and not active_source_claims
                and source_obligation is None
            )
            if source_coverage_already_grounded:
                trace.record("SOURCE_OBLIGATION_SUPPRESSED", {
                    "source_evidence_ids": list(source_support_ids),
                    "reason": (
                        "canonical source Evidence already supports the mechanism; "
                        "no SourceClaim or existing Runtime source Obligation requires "
                        "a separate coverage branch"
                    ),
                    "lifecycle_change": "none",
                })
            source_should_track = source_declared and not source_coverage_already_grounded
            if source_should_track:
                if source_status == "not_applicable":
                    desired_status = "NOT_APPLICABLE_BY_CAPABILITY"
                    desired_support = ()
                else:
                    current_status = source_obligation.status if source_obligation else "OPEN"
                    desired_status = (
                        "SATISFIED" if current_status == "SATISFIED" else "OPEN"
                    )
                    desired_support = (
                        tuple(source_obligation.supporting_evidence_ids)
                        if desired_status == "SATISFIED" else ()
                    )
                desired_blocked = (
                    tuple(source_capability_gaps)
                    if desired_status == "NOT_APPLICABLE_BY_CAPABILITY" else ()
                )
                if source_obligation is None:
                    source_proposal = CanonicalObligationProposal(
                        claim=_SOURCE_MECHANISM_OBLIGATION_CLAIM,
                        evidence_requirement="Canonical source Evidence covering every declared source claim range.",
                        target_component="application source",
                        verification_question=_SOURCE_MECHANISM_OBLIGATION_CLAIM,
                        evidence_type="complete source claim coverage",
                        completion_contract=CompletionContract(required_facts=(FactRequirement(
                            slot="source_range_coverage", role="OBSERVATION",
                            description="Canonical source Evidence covers all declared source ranges.",
                            evidence_constraints=EvidenceConstraint(source_tools=("read_file",),
                                context_kinds=("CODE",), information_sources=("source_read",)),
                        ),)),
                    )
                    admission = ObligationAdmissionPolicy.admit(
                        source_proposal,
                        provenance_profiles=completion_provenance_profiles(),
                        related_hypothesis=selection.current_hypothesis,
                        existing_obligations=canonical_rca_state.obligations,
                        required_tools=("read_file",),
                        available_tools=available_code_tools,
                        capability_gaps=tuple(source_capability_gaps),
                    )
                    trace.record("SOURCE_OBLIGATION_ADMISSION", {
                        "status": admission.status, "reasons": list(admission.reasons),
                        "obligation_id": admission.obligation_id,
                        "capability_gaps": list(source_capability_gaps),
                    })
                    if admission.admitted:
                        canonical_rca_state = canonical_rca_state.reduce(admission.accepted_event())
                        source_obligation = canonical_rca_state.obligation(admission.obligation_id)
                        obligation_created_count += 1
                        trace.record("OBLIGATION_CREATED", source_obligation.model_dump())
                    elif admission.status == "DUPLICATE":
                        source_obligation = canonical_rca_state.obligation(admission.obligation_id)
                if source_obligation is not None and (
                    source_obligation.status != desired_status
                    or source_obligation.supporting_evidence_ids != desired_support
                    or source_obligation.blocked_capabilities != desired_blocked
                ):
                    updated = source_obligation.model_copy(update={
                        "claim": _SOURCE_MECHANISM_OBLIGATION_CLAIM,
                        "critical": True,
                        "status": desired_status,
                        "supporting_evidence_ids": desired_support,
                        "blocked_capabilities": desired_blocked,
                    })
                    trace.record("OBLIGATION_TRANSITION", {
                        "obligation_id": source_obligation.id,
                        "from": source_obligation.status,
                        "to": desired_status,
                        "reason": "source_capability_and_evidence_check",
                    })
                    source_obligation = updated
                if source_obligation is not None:
                    next_obligations[source_obligation.id] = source_obligation

            # Planner support/evidence fields are proposals only.  They must
            # never close an Obligation or make the Hypothesis sufficient.
            # The only active closure path is
            # EvidenceProposal -> admission -> EvidenceSet -> RCAState reducer.
            planner_support_ids = tuple(
                evidence_id for evidence_id in selection.supporting_evidence_ids
                if evidence_id in known_ids and evidence_id.startswith("ev-")
            )
            if planner_support_ids:
                trace.record("PLANNER_SUPPORT_PROPOSAL", {
                    "evidence_ids": list(planner_support_ids),
                    "accepted_for": (
                        "provider_projection"
                        if not self.config.features.evidence_qualification
                        else "audit_only"
                    ),
                    "reason": "planner evidence links cannot mutate canonical lifecycle",
                })

            # A Planner citation is not allowed to create or close an
            # EvidenceSet.  Once a canonical EvidenceSet is already
            # SATISFIES, however, its admitted Evidence members are the
            # Runtime-known context for the completed verification question.
            # Project only the intersection explicitly cited by the Planner;
            # this lets a candidate cite the complete closed fact set while
            # keeping the EvidenceSet reducer as the sole closure authority.
            closed_evidence_ids = {
                evidence_id
                for item in canonical_rca_state.obligations
                if item.status == "SATISFIED"
                for (obligation_id, _), evidence_set in evidence_sets.items()
                if obligation_id == item.id and evidence_set.status == "SATISFIES"
                for evidence_id in evidence_set.evidence_ids
                if evidence_id in known_ids
            }
            closed_planner_support_ids = tuple(
                evidence_id for evidence_id in planner_support_ids
                if evidence_id in closed_evidence_ids
            )
            if closed_planner_support_ids:
                trace.record("PLANNER_SUPPORT_PROJECTED_FROM_CLOSED_EVIDENCE", {
                    "evidence_ids": list(closed_planner_support_ids),
                    "closed_evidence_ids": sorted(closed_evidence_ids),
                    "reason": "explicit Planner citations intersect a Runtime-closed EvidenceSet",
                    "lifecycle_mutation": False,
                })

            for obligation in next_obligations.values():
                if obligation.status == "BLOCKED_BY_CAPABILITY":
                    obligation_blocked_count += 1

            proposed_structured_contradictions = tuple(selection.contradictions or ())
            proposed_contradiction_ids = tuple(dict.fromkeys((
                *(item.evidence_id for item in proposed_structured_contradictions),
                *selection.contradicting_evidence_ids,
            )))
            grounded_contradiction_ids = _grounded_contradiction_ids(
                proposed_contradiction_ids,
                projected_facts.values(),
                set(evidence_projector.hypothesis_scope_fingerprints(selection)),
            )
            rejected_contradiction_ids = tuple(
                evidence_id for evidence_id in proposed_contradiction_ids
                if evidence_id not in set(grounded_contradiction_ids)
            )
            if rejected_contradiction_ids:
                trace.record("CONTRADICTION_PROPOSAL_REJECTED", {
                    "evidence_ids": list(rejected_contradiction_ids),
                    "grounded_evidence_ids": list(grounded_contradiction_ids),
                    "reason": (
                        "Planner contradiction must be backed by a canonical "
                        "Evidence relation marked CONTRADICTS for the current "
                        "hypothesis scope"
                    ),
                    "accepted_for": "audit_only",
                    "lifecycle_mutation": False,
                })
            structured_contradictions = tuple(
                item for item in proposed_structured_contradictions
                if item.evidence_id in set(grounded_contradiction_ids)
            )
            structured_ids = {
                item.evidence_id for item in structured_contradictions
            }
            structured_contradictions += tuple(
                Contradiction(
                    evidence_id=evidence_id,
                    claim="Evidence directly contradicts the current diagnosis.",
                    severity="BLOCKING",
                    status="OPEN",
                )
                for evidence_id in grounded_contradiction_ids
                if evidence_id not in structured_ids
            )
            flat_contradictions = tuple(grounded_contradiction_ids)
            if (
                not structured_contradictions
                and old is not None
                and old.contradictions
            ):
                # A same-hypothesis Planner projection cannot silently clear a
                # still-open contradiction.  A changed hypothesis gets a new
                # contradiction projection, while the prior snapshot remains
                # auditable in RCAState.hypotheses.
                structured_contradictions = tuple(old.contradictions)
                flat_contradictions = tuple(
                    item.evidence_id for item in structured_contradictions
                    if item.blocks_finalization
                )
                trace.record("CONTRADICTION_PROJECTION_PRESERVED", {
                    "evidence_ids": list(flat_contradictions),
                    "reason": "same_hypothesis_projection_cannot_clear_open_contradiction",
                })
            for contradiction in structured_contradictions:
                if (
                    not contradiction.evidence_id.startswith("ev-")
                    or contradiction.evidence_id not in known_ids
                ):
                    raise RuntimeError(
                        "contradiction must cite an existing ev-* Evidence ID"
                    )
            prior_contradictions = {
                item.evidence_id: item for item in (old.contradictions if old else ())
            }
            canonical_contradictions: list[Contradiction] = []
            for contradiction in structured_contradictions:
                prior = prior_contradictions.get(contradiction.evidence_id)
                if prior is not None and contradiction.status not in _CONTRADICTION_TRANSITIONS[prior.status]:
                    # Provider projections cannot reopen a Runtime-owned
                    # terminal contradiction. Preserve the canonical terminal
                    # record and continue with the current hypothesis; this is
                    # lifecycle normalization, not semantic resolution.
                    trace.record("CONTRADICTION_PROJECTION_REUSED", {
                        "evidence_id": contradiction.evidence_id,
                        "from": prior.status,
                        "proposed": contradiction.status,
                        "reason": "terminal_contradiction_cannot_reopen",
                    })
                    contradiction = prior
                canonical_contradictions.append(contradiction)
                if prior is None:
                    trace.record("CONTRADICTION_CREATED", contradiction.model_dump())
                elif prior != contradiction:
                    trace.record("CONTRADICTION_TRANSITION", {
                        "evidence_id": contradiction.evidence_id,
                        "from": prior.model_dump(),
                        "to": contradiction.model_dump(),
                    })
            structured_contradictions = tuple(canonical_contradictions)
            flat_contradictions = tuple(
                item.evidence_id for item in structured_contradictions
                if item.blocks_finalization
            )
            blocking = {
                item.evidence_id for item in structured_contradictions
                if item.blocks_finalization
            }
            blocking_contradiction_count = len(blocking)

            # Only canonical Obligations are blockers. Typed rejected proposals
            # abort the complete staged transaction before publication.
            required_projection = tuple(dict.fromkeys(
                item.claim for item in next_obligations.values()
                if item.blocks_finalization
            ))
            if unstructured_obligation_gaps:
                trace.record("UNSTRUCTURED_OBLIGATION_GAPS", {
                    "gaps": list(dict.fromkeys(unstructured_obligation_gaps)),
                    "accepted_for": "planner_hint_and_telemetry_only",
                    "blocks_finalization": False,
                })
            active_obligation_ids = {
                item.id for item in canonical_rca_state.obligations
                if item.status != "SUPERSEDED_BY_REPLAN"
            }
            admitted_support_ids = tuple(dict.fromkeys(
                fact.evidence_id
                for fact in projected_facts.values()
                if fact.obligation_id in active_obligation_ids
                and fact.hypothesis_fingerprint in selection_hypothesis_fingerprints
                and str(getattr(fact, "relation", "")).upper() == "SUPPORTS"
                and str(getattr(fact, "hypothesis_relation", "")).upper() == "SUPPORTS"
                and str(getattr(fact, "fulfillment", "")).upper()
                in {"PARTIAL", "SATISFIES"}
            ))
            runtime_supporting_ids = admitted_support_ids
            if selection.evidence_sufficiency == "sufficient":
                trace.record("PLANNER_CONTROL_FIELD_REJECTED", {
                    "field": "evidence_sufficiency",
                    "value": "sufficient",
                    "reason": "Runtime derives hypothesis sufficiency from canonical state",
                })
            hypothesis = IncidentHypothesis(
                hypothesis_id=selection.hypothesis_id, revision=selection.revision,
                claim=selection.current_hypothesis,
                # Provider-facing display projection only: the semantic gap is
                # owned by canonical OPEN obligations, not by Planner prose.
                evidence_gap=(required_projection[0] if required_projection else ""),
                component=selection.candidate_component,
                fault=(selection.candidate_fault
                       or selection.candidate_fault_code
                       or selection.candidate_fault_explanation),
                fault_code=selection.candidate_fault_code,
                fault_explanation=selection.candidate_fault_explanation,
                mechanism=selection.candidate_mechanism,
                supporting_evidence_ids=runtime_supporting_ids,
                contradicting_evidence_ids=flat_contradictions,
                required_gaps=required_projection,
                evidence_sufficient=False,
                mechanism_category=selection.mechanism_category,
                source_mechanism_status=source_status,
                source_claims=active_source_claims,
                verification_obligations=tuple(next_obligations.values()),
                contradictions=structured_contradictions,
            )
            runtime_evidence_sufficient = bool(
                runtime_supporting_ids
                and not required_projection
                and not blocking
                and all(
                    bool(str(value).strip())
                    for value in (
                        selection.candidate_component,
                        selection.candidate_fault or selection.candidate_fault_code,
                        selection.candidate_mechanism,
                    )
                )
            )
            hypothesis = hypothesis.model_copy(update={
                "evidence_sufficient": runtime_evidence_sufficient,
                "status": "supported" if runtime_evidence_sufficient else hypothesis.status,
            })
            if old is not None:
                same_identity = selection.hypothesis_transition.kind in {"KEEP_CURRENT", "REFINE"}
                old_blocking = {
                    item.evidence_id for item in old.contradictions if item.blocks_finalization
                }
                new_blocking = bool(blocking - old_blocking)
                stable_rounds = old.stable_rounds + 1 if same_identity and not new_blocking else 0
            else:
                stable_rounds = 0
            hypothesis = hypothesis.model_copy(update={"stable_rounds": stable_rounds})
            previous_hypothesis = old
            updated = reduce_runtime_hypothesis(
                hypothesis,
                reason="planner_" + selection.hypothesis_transition.kind.lower(),
            )

            return updated

        def canonical_rca_state_reset(obligation_id):
            nonlocal canonical_rca_state
            canonical_rca_state = canonical_rca_state.reduce({
                "kind": "REOPEN_OBLIGATION",
                "payload": {"obligation_id": obligation_id, "reason": "hypothesis_revision_requires_requalification"},
            })

        def semantic_transition(transition, proposed, operation):
            """Evaluate against detached stores; publish only a validated complete result."""
            nonlocal canonical_rca_state, evidence_memory, evidence_pipeline, projected_facts, evidence_sets
            nonlocal trace, dag_state, previous_hypothesis, obligation_created_count, obligation_blocked_count
            nonlocal blocking_contradiction_count, unstructured_obligation_gaps, reflection_constraint, reflection_delta_accepts
            before_state = canonical_rca_state
            resolved = resolve_hypothesis(before_state, transition, proposed)
            live = (
                canonical_rca_state, evidence_memory, evidence_pipeline,
                projected_facts, evidence_sets, trace,
                dag_state, previous_hypothesis, obligation_created_count,
                obligation_blocked_count, blocking_contradiction_count, unstructured_obligation_gaps,
                reflection_constraint, reflection_delta_accepts,
            )
            evidence_memory = deepcopy(evidence_memory)
            evidence_pipeline = EvidencePipeline(
                observation_store=observation_store, evidence_memory=evidence_memory,
                projector=evidence_projector, deterministic_fact_adapters=("source_range_capture",),
            )
            projected_facts = dict(projected_facts)
            evidence_sets = dict(evidence_sets)
            trace = SemanticTraceBuffer()
            dag_state = None  # DAG and committed trace are post-publication projections.
            try:
                if transition.kind != 'KEEP_CURRENT' and before_state.contradictions:
                    # Contradiction status belongs to admitted relationships in
                    # one hypothesis scope, not to an Evidence body forever.
                    canonical_rca_state = canonical_rca_state.reduce({
                        'kind': 'SET_CONTRADICTIONS', 'payload': {'contradictions': []}})
                    trace.record('CONTRADICTION_SCOPE_INVALIDATED', {
                        'evidence_ids': [item.evidence_id for item in before_state.contradictions],
                        'previous_scopes': list(evidence_projector.hypothesis_scope_fingerprints(
                            before_state.active_hypothesis_projection())),
                        'next_scopes': list(evidence_projector.hypothesis_scope_fingerprints(resolved)),
                        'reason': 'hypothesis_revision_requires_requalification'})
                # A new revision carries no automatic satisfaction/coverage inheritance.
                for item in before_state.obligations:
                    if transition.kind != "KEEP_CURRENT" and item.status == "SATISFIED":
                        canonical_rca_state = canonical_rca_state.reduce({
                            "kind": "REOPEN_OBLIGATION",
                            "payload": {"obligation_id": item.id, "reason": "hypothesis_revision_requires_requalification"},
                        })
                result = operation(resolved)
                trace.validate()
                validate_disposition_conflicts(transition, before_state, canonical_rca_state)
                superseded = {item.obligation_id for item in transition.obligation_dispositions if item.action == "SUPERSEDE"}
                if any(kind == "OBLIGATION_SATISFIED" and payload.get("obligation_id") in superseded
                       for kind, payload in trace.records):
                    raise SemanticTransitionError("semantic_transition_conflict", "SUPERSEDE + SATISFIED")
                for disposition in transition.obligation_dispositions:
                    if disposition.action == "SUPERSEDE":
                        canonical_rca_state = canonical_rca_state.reduce({
                            "kind": "SUPERSEDE_OBLIGATION",
                            "payload": {"obligation_id": disposition.obligation_id},
                        })
                        trace.record("OBLIGATION_TRANSITION", {
                            "obligation_id": disposition.obligation_id, "to": "SUPERSEDED_BY_REPLAN",
                            "reason": disposition.reason, "source": "validated_disposition",
                        })
                staged = (
                canonical_rca_state, evidence_memory, evidence_pipeline,
                projected_facts, evidence_sets, trace,
                dag_state, previous_hypothesis, obligation_created_count,
                obligation_blocked_count, blocking_contradiction_count, unstructured_obligation_gaps,
                reflection_constraint, reflection_delta_accepts,
            )
            finally:
                (
                canonical_rca_state, evidence_memory, evidence_pipeline,
                projected_facts, evidence_sets, trace,
                dag_state, previous_hypothesis, obligation_created_count,
                obligation_blocked_count, blocking_contradiction_count, unstructured_obligation_gaps,
                reflection_constraint, reflection_delta_accepts,
            ) = live
            # No fallible semantic validation follows publication. Stores remain sole owners.
            staged_memory, staged_trace = staged[1], staged[5]
            evidence_memory.publish_staged(staged_memory)
            canonical_rca_state = staged[0]
            projected_facts, evidence_sets = staged[3], staged[4]
            (previous_hypothesis, obligation_created_count, obligation_blocked_count,
             blocking_contradiction_count, unstructured_obligation_gaps,
             reflection_constraint, reflection_delta_accepts) = staged[7:]
            project_obligation_ids_to_dag("semantic_transition_published")
            staged_trace.publish(trace)
            trace.record("SEMANTIC_TRANSITION_COMMITTED", {
                "kind": transition.kind, "hypothesis_id": resolved.hypothesis_id,
                "revision": resolved.revision, "event_count": canonical_rca_state.event_count,
            })
            return result

        def validate_reflection_feedback(feedback: ReflectionFeedback) -> None:
            known_ids = {item.evidence_id for item in evidence_memory.pinned}
            ids = list(feedback.supporting_evidence_ids) + list(feedback.contradicting_evidence_ids)
            ids.extend(feedback.hypothesis_delta.supporting_evidence_ids)
            ids.extend(feedback.hypothesis_delta.contradicting_evidence_ids)
            ids.extend(
                evidence_id
                for item in feedback.proposed_obligations
                for evidence_id in item.supporting_evidence_ids
            )
            ids.extend(item.evidence_id for item in feedback.contradiction_updates)
            ids.extend(
                evidence_id
                for item in feedback.obligation_reviews
                for evidence_id in item.supporting_evidence_ids
            )
            ids.extend(item.evidence_id for item in feedback.contradiction_reviews)
            bad = [item for item in ids if not item.startswith("ev-") or item not in known_ids]
            if bad:
                raise ValueError("Reflection referenced unknown Evidence IDs: " + ", ".join(sorted(set(bad))))

        def apply_reflection_delta(feedback: ReflectionFeedback, trigger: str) -> bool:
            nonlocal reflection_constraint
            current = active_hypothesis()
            delta = feedback.hypothesis_delta
            transition = delta.transition
            if transition is None:
                # Operational / audit-only feedback can carry no semantic mutation.
                if any(getattr(delta, key) for key in ("claim", "component", "fault", "fault_code",
                       "fault_explanation", "mechanism", "source_mechanism_status")) or feedback.proposed_obligations or feedback.contradiction_updates:
                    trace.record("REFLECTION_DELTA_REJECTED", {"reason": "hypothesis_transition_required"})
                    return False
                for gap in feedback.proposed_evidence_gaps:
                    admission = ObligationAdmissionPolicy.admit(gap, provenance_profiles=completion_provenance_profiles(), related_hypothesis=current.claim if current else "")
                    trace.record("REFLECTION_GAP_ADMISSION", {"trigger": trigger, "status": admission.status,
                                 "reasons": list(admission.reasons), "accepted_for": "audit_only"})
                reflection_constraint = feedback.next_action_constraint
                return False
            if current is None:
                proposed = IncidentHypothesis(evidence_gap="", **{
                    key: getattr(delta, key) for key in ("claim", "component", "fault", "fault_code",
                    "fault_explanation", "mechanism", "source_mechanism_status") if getattr(delta, key) is not None
                })
            else:
                proposed = current
            def stage_reflection(resolved):
                return stage_reflection_delta(feedback, trigger, resolved)
            try:
                return semantic_transition(transition, proposed, stage_reflection)
            except (SemanticTransitionError, RCAStateError, ValueError, RuntimeError) as exc:
                trace.record("REFLECTION_DELTA_REJECTED", {"reason": str(exc)})
                return False

        def stage_reflection_delta(feedback: ReflectionFeedback, trigger: str, resolved: IncidentHypothesis) -> bool:
            """Apply only a genuinely new Runtime-owned Reflection Delta."""
            nonlocal obligation_created_count, blocking_contradiction_count
            nonlocal reflection_constraint, reflection_delta_accepts
            nonlocal canonical_rca_state
            current = active_hypothesis()
            current_obligations = obligation_projection()
            before = (
                self._hypothesis_semantic_fingerprint(current) if current else None,
                tuple(sorted((item.id, item.claim.casefold(), item.status)
                             for item in current_obligations.values())),
                tuple(sorted((item.evidence_id, item.status, item.severity)
                             for item in (current.contradictions if current else ()))),
                reflection_constraint,
            )
            constraint = feedback.next_action_constraint

            next_obligations = dict(current_obligations)
            new_obligation_ids: list[str] = []
            admissions: list[ObligationAdmissionDecision] = []
            evidence_tools = tuple(
                name for name in capabilities.planner_tool_names(tools)
                if name not in {"finalize_diagnosis", "knowledge_retrieval"}
            )
            proposals = [
                *(item for item in feedback.proposed_evidence_gaps),
                *(item for item in feedback.proposed_obligations),
            ]
            for proposal in proposals:
                admission = ObligationAdmissionPolicy.admit(
                    proposal,
                    provenance_profiles=completion_provenance_profiles(),
                    related_hypothesis=current.claim if current else "",
                    existing_obligations=tuple(next_obligations.values()),
                    required_tools=tuple(getattr(proposal, "suggested_tools", ())),
                    available_tools=evidence_tools,
                )
                trace.record("REFLECTION_GAP_ADMISSION", {
                    "trigger": trigger, "status": admission.status,
                    "obligation_id": admission.obligation_id,
                    "reasons": list(admission.reasons),
                })
                if admission.status == "REJECTED" and proposal in feedback.proposed_obligations:
                    raise SemanticTransitionError("obligation_admission_rejected", str(admission.reasons))
                if not admission.admitted:
                    continue
                supporting = tuple(getattr(proposal, "supporting_evidence_ids", ()) or ())
                admissions.append(admission)
                obligation = VerificationObligation.model_validate(admission.accepted_event().payload)
                next_obligations[obligation.id] = obligation
                new_obligation_ids.append(obligation.id)
                if supporting:
                    trace.record("REFLECTION_SUPPORT_PROPOSAL", {
                        "obligation_id": obligation.id,
                        "evidence_ids": list(supporting),
                        "accepted_for": "audit_only",
                    })

            next_contradictions = {
                item.evidence_id: item
                for item in (current.contradictions if current else ())
            }
            for update in feedback.contradiction_updates:
                prior = next_contradictions.get(update.evidence_id)
                if prior is not None and update.status not in _CONTRADICTION_TRANSITIONS[prior.status]:
                    trace.record("REFLECTION_DELTA_REJECTED", {
                        "trigger": trigger,
                        "reason": "invalid_contradiction_transition",
                        "evidence_id": update.evidence_id,
                        "from": prior.status,
                        "to": update.status,
                    })
                    return False
                next_contradictions[update.evidence_id] = Contradiction(
                    evidence_id=update.evidence_id,
                    claim=update.claim,
                    severity=update.severity,
                    status=update.status,
                )

            delta = feedback.hypothesis_delta
            updated = resolved.model_copy(update={"supporting_evidence_ids": ()})
            accepted_fields: list[str] = []
            if current is not None:
                changes: dict[str, Any] = {}
                for field in (
                    "claim", "component", "fault", "fault_code",
                    "fault_explanation", "mechanism", "source_mechanism_status",
                ):
                    value = getattr(delta, field, None)
                    if isinstance(value, str) and value.strip():
                        if value != getattr(current, field):
                            changes[field] = value.strip()
                            accepted_fields.append(field)
                proposed_support_ids = tuple(dict.fromkeys(
                    (*feedback.supporting_evidence_ids,
                     *delta.supporting_evidence_ids)
                ))
                if proposed_support_ids:
                    trace.record("REFLECTION_SUPPORT_PROPOSAL", {
                        "evidence_ids": list(proposed_support_ids),
                        "accepted_for": "audit_only",
                    })
                contradiction_ids = tuple(dict.fromkeys(
                    (*current.contradicting_evidence_ids,
                     *feedback.contradicting_evidence_ids,
                     *delta.contradicting_evidence_ids)
                ))
                if contradiction_ids != current.contradicting_evidence_ids:
                    changes["contradicting_evidence_ids"] = contradiction_ids
                    accepted_fields.append("contradicting_evidence_ids")
                if next_obligations != current_obligations:
                    changes["verification_obligations"] = tuple(next_obligations.values())
                    changes["required_gaps"] = tuple(
                        item.claim for item in next_obligations.values()
                        if item.blocks_finalization
                    )
                    accepted_fields.append("verification_obligations")
                if next_contradictions != {
                    item.evidence_id: item for item in current.contradictions
                }:
                    changes["contradictions"] = tuple(next_contradictions.values())
                    accepted_fields.append("contradictions")
                if changes:
                    updated = updated.model_copy(update=changes)
            elif new_obligation_ids:
                accepted_fields.append("proposed_obligations")

            if constraint and constraint != reflection_constraint:
                reflection_constraint = constraint
                accepted_fields.append("next_action_constraint")

            after = (
                self._hypothesis_semantic_fingerprint(updated) if updated else None,
                tuple(sorted((item.id, item.claim.casefold(), item.status)
                             for item in next_obligations.values())),
                tuple(sorted((item.evidence_id, item.status, item.severity)
                             for item in (updated.contradictions if updated else next_contradictions.values()))),
                reflection_constraint,
            )
            if before == after and updated.revision == (current.revision if current else 0):
                trace.record("REFLECTION_NO_DELTA", {
                    "trigger": trigger,
                    "reason": "canonical_runtime_state_unchanged",
                })
                return False

            for admission in admissions:
                canonical_rca_state = canonical_rca_state.reduce(admission.accepted_event())
                obligation_created_count += 1
                trace.record("OBLIGATION_CREATED", {
                    **admission.accepted_event().payload, "source": "rca_state_reducer",
                    "reason": "reflection_delta_proposal", "trigger": trigger,
                })
            blocking_contradiction_count = sum(
                1 for item in (updated.contradictions if updated else next_contradictions.values())
                if item.blocks_finalization
            )
            if updated is not None:
                updated = updated.model_copy(update={
                    "stable_rounds": 0,
                    "verification_obligations": tuple(next_obligations.values()),
                    "required_gaps": tuple(
                        item.claim for item in next_obligations.values()
                        if item.blocks_finalization
                    ),
                    "contradictions": tuple(next_contradictions.values()),
                    "contradicting_evidence_ids": tuple(
                        item.evidence_id for item in next_contradictions.values()
                        if item.blocks_finalization
                    ),
                })
                updated = reduce_runtime_hypothesis(
                    updated, reason="reflection_delta",
                )
            reflection_delta_accepts += 1
            trace.record("REFLECTION_DELTA_ACCEPTED", {
                "trigger": trigger,
                "accepted_fields": sorted(set(accepted_fields)),
                "new_obligation_ids": new_obligation_ids,
                "runtime_hypothesis_version": len(canonical_rca_state.hypotheses),
                "next_action_constraint": reflection_constraint,
                "investigation_direction_changed": bool(new_obligation_ids or
                    set(accepted_fields).intersection({"component", "fault", "fault_code", "mechanism", "contradictions"})),
            })
            return True

        def mark_blocked_capability(tool_name: str) -> None:
            nonlocal obligation_blocked_count
            current = active_hypothesis()
            if current is None:
                return
            updated_obligations = []
            changed = False
            for obligation in current.verification_obligations:
                if obligation.status != "OPEN":
                    updated_obligations.append(obligation)
                    continue
                if _is_source_mechanism_obligation(obligation):
                    updated = obligation.model_copy(update={
                        "status": "NOT_APPLICABLE_BY_CAPABILITY",
                        "blocked_capabilities": tuple(dict.fromkeys(
                            (*obligation.blocked_capabilities, tool_name)
                        )),
                    })
                    updated_obligations.append(updated)
                    changed = True
                    trace.record("SOURCE_OBLIGATION_STATUS_OVERRIDDEN", {
                        "obligation_id": obligation.id,
                        "proposed_status": obligation.status,
                        "runtime_status": "NOT_APPLICABLE_BY_CAPABILITY",
                        "capability_gaps": list(updated.blocked_capabilities),
                        "reason": "source_tool_capability_lost",
                        "tool": tool_name,
                    })
                    trace.record("OBLIGATION_TRANSITION", {
                        "obligation_id": obligation.id,
                        "from": obligation.status,
                        "to": "NOT_APPLICABLE_BY_CAPABILITY",
                        "reason": "tool_circuit_open",
                        "tool": tool_name,
                    })
                    continue
                updated = obligation.model_copy(update={
                    "status": "BLOCKED_BY_CAPABILITY",
                    "blocked_capabilities": tuple(dict.fromkeys(
                        (*obligation.blocked_capabilities, tool_name)
                    )),
                })
                updated_obligations.append(updated)
                changed = True
                obligation_blocked_count += 1
                trace.record("OBLIGATION_TRANSITION", {
                    "obligation_id": obligation.id,
                    "from": obligation.status,
                    "to": "BLOCKED_BY_CAPABILITY",
                    "reason": "tool_circuit_open",
                    "tool": tool_name,
                })
            if not changed:
                return
            current = current.model_copy(update={
                "verification_obligations": tuple(updated_obligations),
                "required_gaps": tuple(item.claim for item in updated_obligations if item.blocks_finalization),
                "source_mechanism_status": (
                    "not_applicable"
                    if any(
                        _is_source_mechanism_obligation(item)
                        and item.status == "NOT_APPLICABLE_BY_CAPABILITY"
                        for item in updated_obligations
                    )
                    else current.source_mechanism_status
                ),
            })
            current = reduce_runtime_hypothesis(
                current, reason="capability_blocked",
            )
            trace.record("HYPOTHESIS_UPDATED", {
                "reason": "capability_blocked",
                "hypothesis": current.model_dump(),
            })



        def feasible_skill_ids() -> tuple[str, ...]:
            """One capability filter shared by normal planning and recovery."""
            return domain_registry.feasible_skill_ids(
                available_tools=capabilities.planner_tool_names(tools),
            )

        def trigger_reflection(*, review_feedback_text: str = "", **trigger_facts) -> str:
            nonlocal active_skill
            nonlocal reflection_calls, reflection_feedback, reflection_trigger, schema_repair_count, schema_repair_calls
            nonlocal llm_calls
            nonlocal reflection_no_delta_count, consecutive_reflection_no_delta
            current = active_hypothesis()
            trigger_decision = ProgressPolicy.evaluate(operational_facts(
                "reflection_trigger",
                contradiction_ids=tuple(item.evidence_id for item in canonical_rca_state.contradictions
                                        if item.blocks_finalization),
                **trigger_facts,
            ))
            if trigger_decision.reflection_trigger is None:
                return ""
            reason = trigger_decision.reflection_trigger.value
            provider = getattr(reflection_agent, "llm", None)
            signature = json.dumps({
                "trigger": reason,
                "hypothesis": self._material_hypothesis_fingerprint(current) if current else None,
                "evidence": sorted(item.evidence_id for item in evidence_memory.pinned),
                "review": review_feedback_text,
            }, sort_keys=True, ensure_ascii=False, default=str)
            permission = finalization_decision()
            recovery = ProgressPolicy.evaluate(operational_facts(
                "recovery", stage="reflection", recovery_count=reflection_calls,
                recovery_limit=self.config.max_reflection_calls,
                phase=budget_snapshot("reflection_schedule").phase,
                finalization=permission,
                targeted_recovery_allowed=not permission.allowed,
                recovery_enabled=self.config.enable_reflection,
                recovery_phase_restricted=CONVERGE_DISABLE_REFLECTIONS and self.config.converge_disable_reflections,
                recovery_capability_available=hasattr(provider, "complete_structured"),
                recovery_duplicate=signature in reflection_signatures,
                consecutive_recovery_count=consecutive_reflection_no_delta,
                consecutive_recovery_limit=self.config.max_consecutive_reflection_calls,
            ))
            if recovery.action == "stop":
                blocked_without_delta = "max_consecutive_reflection_without_runtime_delta" in recovery.reasons
                trace.record("REFLECTION_BLOCKED" if blocked_without_delta else "REFLECTION_SKIPPED", {
                    "reason": recovery.reasons[0], "trigger": reason,
                    "converge_start_step": CONVERGE_START_STEP,
                    "max_consecutive_reflection_calls": self.config.max_consecutive_reflection_calls,
                })
                if blocked_without_delta:
                    transition_phase("PLAN", "reflection_blocked_without_delta")
                return ""
            reflection_signatures.add(signature)
            reflection_trigger = reason
            reflection_calls += 1
            transition_phase("REFLECT", reason, reflection_call=reflection_calls)
            trace.record("REFLECTION_START", {
                "call": reflection_calls, "trigger": reason,
                "evidence_count": len(evidence_memory.pinned),
            })
            snapshot = {
                "current_hypothesis": current.model_dump() if current else None,
                "relevant_evidence": [item.model_dump(exclude={"excerpt"}) for item in incident_evidence()
                                      if current is None or not current.supporting_evidence_ids
                                      or item.evidence_id in set(current.supporting_evidence_ids)
                                      or item.evidence_id in set(current.contradicting_evidence_ids)],
                "required_gaps": list(current.required_gap_projection()) if current else [],
                "verification_obligations": [item.model_dump() for item in (current.verification_obligations if current else ())],
                "contradictions": [item.model_dump() for item in (current.contradictions if current else ())],
                "source_mechanism_coverage": self._source_mechanism_coverage(
                    case,
                    current,
                    source_workspace_available=source_workspace_available,
                    available_code_tools=available_code_tools,
                    source_evidence_ids=source_evidence_ids(),
                ),
                "tool_health": tool_circuit.summary(),
                "trigger_reason": reason,
                "recent_structured_actions": actions[-6:],
                "review_feedback": review_feedback_text or None,
                "prior_knowledge": prior_context.model_dump(mode="json") if prior_context else None,
            }
            reflection_views = load_skill_views(feasible_skill_ids(), tools.official_aliases)
            snapshot["feasible_skill_cards"] = [json.loads(view.card_json) for view in reflection_views]
            snapshot["selected_semantic_projection"] = next(
                (json.loads(view.semantic_json) for view in reflection_views if view.skill_id == active_skill), None,
            )
            snapshot["rca_state"] = canonical_rca_state.context_projection()
            snapshot["capabilities"] = capabilities.planner_projection()
            snapshot["hypothesis_scope"] = evidence_projector.hypothesis_scope_fingerprints(current)
            snapshot["existing_evidence"] = evidence_memory.projection(
                facts=projected_facts.values(), evidence_ids=canonical_rca_state.evidence_ids)
            from debug_assistant.tools.provenance import registry_provenance_profiles, provenance_profile_catalog
            snapshot["source_profiles"] = provenance_profile_catalog(registry_provenance_profiles(
                tools, capabilities.planner_tool_names(tools)))
            snapshot["available_tools"] = list(capabilities.planner_tool_names(tools))
            reflection_prior_capacity = dynamic_budget.available_prior_capacity(
                critical_state_tokens=dynamic_budget.estimate_json_tokens({
                    "hypothesis": snapshot["current_hypothesis"],
                    "gaps": snapshot["required_gaps"],
                }),
                verified_evidence_tokens=dynamic_budget.estimate_json_tokens(
                    snapshot["relevant_evidence"]
                ),
                required_control_tokens=dynamic_budget.estimate_json_tokens({
                    "actions": snapshot["recent_structured_actions"],
                    "contradictions": snapshot["contradictions"],
                }),
            )
            bounded_reflection_prior = self._bound_prior_context(
                prior_context, reflection_prior_capacity, dynamic_budget.estimate_tokens,
            )
            snapshot["prior_knowledge"] = (
                bounded_reflection_prior.model_dump(mode="json")
                if bounded_reflection_prior else None
            )
            reflection_context = context_manager.build(
                state, evidence_memory, observation_store,
                max_context_chars=self.config.planner_context_chars,
                max_context_tokens=dynamic_budget.context_token_budget(),
                token_estimator=dynamic_budget.estimate_tokens,
                include_agent_control_state=False, rca_state=canonical_rca_state,
                role="reflection", max_request_chars=self.config.max_context_chars,
                requested_ids=tuple(current.supporting_evidence_ids) if current else (),
                request_renderer=lambda view: reflection_agent.render_envelope(snapshot, view),
            )
            try:
                admit("reflection", count_llm=True)
                timeout = provider_timeout(self.config.planner_llm_timeout_seconds)
                feedback = reflection_agent.reflect(
                    snapshot, context_view=reflection_context.context_view,
                    logical_timeout_seconds=timeout,
                    max_output_tokens=self.config.reflection_max_completion_tokens,
                    prompt_budget=dynamic_budget,
                )
                extra_calls = max(0, int(getattr(reflection_agent, "last_call_count", 1)) - 1)
                add_usage(
                    reflection_agent, stage="reflection",
                    prompt_breakdown=getattr(reflection_agent, "last_prompt_breakdown", None),
                )
                if extra_calls:
                    llm_calls += extra_calls
                    schema_repair_count += extra_calls
                    schema_repair_calls += extra_calls
                    if ProgressPolicy.evaluate(operational_facts("post_usage")).action == "stop":
                        raise RuntimeError("max_llm_calls exceeded during reflection repair")
                    dynamic_budget.set_run_usage(
                        tokens_used=prompt_tokens + completion_tokens,
                        llm_calls_used=llm_calls,
                        cost_used=run_cost,
                    )
                if getattr(reflection_agent, "last_normalization_actions", None):
                    trace.record("REFLECTION_REASONING_METADATA_NORMALIZED", {
                        "call": reflection_calls, "trigger": reason,
                        "actions": list(reflection_agent.last_normalization_actions),
                    })
                if getattr(reflection_agent, "last_symbol_normalizations", None):
                    trace.record("MODEL_OUTPUT_CANONICALIZED", {
                        "role": "reflection", "events": reflection_agent.last_symbol_normalizations,
                    })
                if (
                    getattr(reflection_agent, "last_metadata_drops", None)
                    or getattr(reflection_agent, "last_metadata_warnings", None)
                ):
                    trace.record("REFLECTION_REASONING_METADATA_DROPPED", {
                        "call": reflection_calls, "trigger": reason,
                        "dropped_items": list(getattr(reflection_agent, "last_metadata_drops", ())),
                        "warnings": list(getattr(reflection_agent, "last_metadata_warnings", ())),
                    })
                validate_reflection_feedback(feedback)
                if feedback.skill_reselection and feedback.skill_reselection not in feasible_skill_ids():
                    raise ValueError("Reflection Skill reselection requires a feasible canonical ID")
                reflection_feedback = json.dumps(feedback.model_dump(), ensure_ascii=False)
                delta_accepted = apply_reflection_delta(feedback, reason)
                if feedback.skill_reselection and feedback.skill_reselection != active_skill:
                    trace.record("SKILL_RESELECTION_ACCEPTED", {
                        "from": active_skill, "to": feedback.skill_reselection,
                        "source": "reflection_proposal", "semantic_guidance_visible": "next_planner_turn",
                    })
                    active_skill = feedback.skill_reselection
                if delta_accepted:
                    consecutive_reflection_no_delta = 0
                else:
                    reflection_no_delta_count += 1
                    consecutive_reflection_no_delta += 1
                trace.record("REFLECTION_RESULT", {
                    "call": reflection_calls, "trigger": reason,
                    "delta_accepted": delta_accepted,
                    "consecutive_no_delta": consecutive_reflection_no_delta,
                    "usage": dict(getattr(reflection_agent, "last_usage", {}) or {}),
                    "prompt_breakdown": dict(getattr(reflection_agent, "last_prompt_breakdown", {}) or {}),
                    "prompt_breakdowns": list(getattr(reflection_agent, "last_prompt_breakdowns", ()) or ()),
                    "prompt_snapshots": [item.as_dict() for item in reflection_agent.last_prompt_snapshots],
                    **feedback.model_dump(),
                })
                transition_phase("PLAN", "reflection_completed", delta_accepted=delta_accepted)
                return reflection_feedback
            except RunBudgetAdmissionExceeded as exc:
                add_usage(reflection_agent, stage="reflection", call_status="failed")
                llm_calls = max(0, llm_calls - 1)
                reflection_calls = max(0, reflection_calls - 1)
                dynamic_budget.set_run_usage(
                    tokens_used=prompt_tokens + completion_tokens,
                    llm_calls_used=llm_calls,
                    cost_used=run_cost,
                )
                trace.record("BUDGET_ADMISSION_REJECTED", {
                    "stage": "reflection",
                    "reason": exc.decision.reason,
                    "budget_before_call": exc.decision.remaining_run_tokens,
                    "estimated_call_cost": exc.decision.breakdown,
                    "budget_after_call": max(
                        0,
                        exc.decision.remaining_run_tokens
                        - exc.decision.breakdown.get("required_run_tokens", 0),
                    ),
                })
                return ""
            except PromptBudgetExceeded as exc:
                add_usage(reflection_agent, stage="reflection", call_status="failed")
                llm_calls = max(0, llm_calls - 1)
                reflection_calls = max(0, reflection_calls - 1)
                dynamic_budget.set_run_usage(
                    tokens_used=prompt_tokens + completion_tokens,
                    llm_calls_used=llm_calls,
                    cost_used=run_cost,
                )
                trace.record("CONTEXT_PRESSURE", {
                    "stage": "reflection",
                    "context_state": exc.decision.state.value,
                    "reason": exc.decision.reason,
                    "estimated_prompt_tokens": exc.decision.estimated_prompt_tokens,
                    "input_hard_capacity": exc.decision.input_hard_capacity,
                    "breakdown": exc.decision.breakdown,
                    "compaction_applied": True,
                })
                return ""
            except ReflectionContractExhausted as exc:
                add_usage(reflection_agent, stage="reflection", call_status="failed")
                transition_phase("INCONCLUSIVE", "reflection_contract_exhausted")
                trace.record("REPAIR_FAILED", {
                    "layer": "reflection_contract",
                    "logical_turn": reflection_calls,
                    "repair_attempts": 1,
                    "reason": "reflection_schema_repair_exhausted",
                })
                trace.record("REFLECTION_CONTRACT_EXHAUSTED", {
                    "call": reflection_calls, "trigger": reason,
                    "error_type": type(exc).__name__,
                    "failure_category": "reflection_contract_exhausted",
                })
                raise
            except LLMDeadlineExceeded as exc:
                add_usage(reflection_agent, stage="reflection", call_status="failed")
                transition_phase("PLAN", "reflection_provider_timeout")
                trace.record("REFLECTION_PROVIDER_FAILURE", {
                    "call": reflection_calls, "trigger": reason,
                    "error_type": type(exc).__name__,
                    "failure_category": "provider_timeout",
                    "message": str(exc),
                })
                # A provider timeout is recoverable at the runtime boundary:
                # preserve the prior hypothesis and continue through the
                # bounded planner/finalization guards. It is not a schema
                # rejection and must not consume a contract-repair slot.
                return ""
            except Exception as exc:
                add_usage(reflection_agent, stage="reflection", call_status="failed")
                transition_phase("PLAN", "reflection_rejected_schema")
                trace.record("REFLECTION_REJECTED_SCHEMA", {
                    "call": reflection_calls, "trigger": reason,
                    "error_type": type(exc).__name__, "message": str(exc),
                })
                # A previous successful reflection remains visible as control
                # context, but it is not fresh feedback for this trigger. Do
                # not let a failed/timeout call reset the progress guard.
                return ""

        def finalization_decision() -> FinalizationDecision:
            """Evaluate the one canonical Candidate publication contract."""
            nonlocal last_convergence_decision
            hypothesis = active_hypothesis()
            source_obligation = next(
                (
                    item for item in canonical_rca_state.obligations
                    if _is_source_mechanism_obligation(item)
                ),
                None,
            )
            source_required = bool(
                source_obligation
                and source_obligation.status != "NOT_APPLICABLE_BY_CAPABILITY"
            )
            source_complete = True
            if source_required and source_obligation is not None and hypothesis is not None:
                cited_source_ids = set(source_evidence_ids()).intersection(
                    source_obligation.supporting_evidence_ids
                )
                source_coverage = evaluate_source_claims(hypothesis.source_claims)
                source_complete = bool(
                    source_obligation.status == "SATISFIED"
                    and cited_source_ids
                    and (
                        source_coverage["all_claims_covered"]
                        if hypothesis.source_claims else True
                    )
                )
            scopes = tuple(
                evidence_projector.hypothesis_scope_fingerprints(hypothesis)
                if hypothesis is not None else ()
            )
            decision = FinalizationPolicy.evaluate(
                canonical_rca_state,
                evidence_memory=evidence_memory,
                evidence_sets=evidence_sets.values(),
                projected_facts=projected_facts.values(),
                facts=FinalizationFacts(
                    hypothesis_scope_fingerprints=scopes,
                    source_coverage_required=source_required,
                    source_coverage_complete=source_complete,
                    taxonomy_required=self.config.features.public_fault_taxonomy,
                ),
            )
            convergence = ProgressPolicy.evaluate(operational_facts(
                "convergence", finalization=decision,
                hypothesis_stable=bool(hypothesis and hypothesis.stable_rounds >= 1),
                open_obligation_ids=tuple(
                    canonical_rca_state.open_critical_obligation_ids
                ),
                contradiction_ids=tuple(
                    item.evidence_id
                    for item in canonical_rca_state.contradictions
                    if item.blocks_finalization
                ),
            )).convergence
            last_convergence_decision = convergence
            if self.config.features.semantic_convergence_v2:
                trace.record("CONVERGENCE_DECISION", {
                    **convergence.model_dump(mode="json"),
                    "source": "canonical_finalization_policy",
                })
            else:
                trace.record("PRELIMINARY_CONVERGENCE_DECISION", {
                    **convergence.model_dump(mode="json"),
                    "source": "canonical_finalization_policy",
                })
            trace.record("FINALIZATION_POLICY_EVALUATED", {
                "step": state.step,
                **decision.model_dump(mode="json"),
            })
            return decision

        def record_terminal_convergence(*, terminal_status: str, reason: str) -> None:
            """Emit one final A2 decision with explicit blocking semantics.

            The A1 path keeps its preliminary event above.  A2 additionally
            records the terminal Runtime outcome, including budget and
            capability bounds, so a safety stop cannot look like a semantic
            PASS merely because the loop reached its max-step boundary.
            """
            nonlocal terminal_convergence_decision
            if (
                not self.config.features.semantic_convergence_v2
                or execution_mode != "dag"
                or self.config.context_projection_mode != "task_aware"
            ):
                terminal_convergence_decision = None
                return
            previous = last_convergence_decision
            permission = finalization_decision()
            current = active_hypothesis()
            capability_blocked = bool(
                canonical_rca_state.blocked_capability_obligation_ids
                if hasattr(canonical_rca_state, "blocked_capability_obligation_ids")
                else any(
                    item.status == "BLOCKED_BY_CAPABILITY"
                    for item in canonical_rca_state.obligations
                )
            )
            decision = ProgressPolicy.evaluate(operational_facts(
                "terminal", finalization=permission,
                terminal_status=terminal_status, terminal_reason=reason,
                previous_convergence=previous,
                hypothesis_stable=bool(current and current.stable_rounds >= 1),
                open_obligation_ids=tuple(
                    canonical_rca_state.open_critical_obligation_ids
                ),
                contradiction_ids=tuple(
                    item.evidence_id
                    for item in canonical_rca_state.contradictions
                    if item.blocks_finalization
                ),
                capability_blocked=capability_blocked,
            )).convergence
            trace.record("CONVERGENCE_DECISION", {
                **decision.model_dump(mode="json"),
                "source": "runtime_terminal_policy",
                "canonical_reused": decision is previous,
                "terminal_runtime_status": terminal_status,
                "termination_reason": reason,
            })
            terminal_convergence_decision = decision

        def finalize_candidate(source: str) -> RootCauseCandidate:
            """Publish only the Candidate projected by FinalizationPolicy."""
            transition_phase("FINALIZE", source)
            decision = finalization_decision()
            publication = ProgressPolicy.evaluate(operational_facts(
                "publication", finalization=decision, stage=source,
            ))
            if publication.action == "stop":
                trace.record("FINALIZATION_REJECTED", {
                    "source": source,
                    "reasons": [item.model_dump(mode="json") for item in decision.reasons],
                })
                raise FinalizationBlocked(decision)
            candidate = decision.candidate
            trace.record("FINALIZATION_VALIDATED", {
                "source": source,
                "candidate": candidate.model_dump(),
                "decision": decision.model_dump(mode="json"),
            })
            trace.record("ROOT_CAUSE_CANDIDATE", {
                **candidate.model_dump(), "source": source,
            })
            return candidate

        def observe_action_progress(*, canonical_progress: bool,
                                    acquisition_progress: bool = False) -> RootCauseCandidate | None:
            """Forward one operational decision; recovery still submits reducer events."""
            nonlocal termination_reason
            version_before_recovery = canonical_rca_state.event_count
            if state.repeated_actions:
                trigger_reflection(repeated_count=state.repeated_actions)
            canonical_progress = canonical_progress or canonical_rca_state.event_count != version_before_recovery
            facts = operational_facts("progress", canonical_progress=canonical_progress,
                                      acquisition_progress=acquisition_progress)
            decision = ProgressPolicy.evaluate(facts)
            state.no_progress_count = decision.no_progress_count
            trace.record("PROGRESS_POLICY_DECISION", {
                "action": decision.action, "reasons": list(decision.reasons),
                "no_progress_count": decision.no_progress_count,
                "canonical_progress": canonical_progress, "acquisition_progress": acquisition_progress,
            })
            if decision.action == "stop":
                raise RuntimeError("investigation no-progress limit exceeded: " + ", ".join(decision.reasons))
            if decision.action == "finalize":
                termination_reason = "semantic_no_progress_sufficient_hypothesis"
                return finalize_candidate(termination_reason)
            if decision.action == "retry":
                version_before = canonical_rca_state.event_count
                trigger_reflection(no_progress_count=state.no_progress_count)
                permission = finalization_decision()
                decision = ProgressPolicy.evaluate(replace(
                    facts, finalization=permission, recovery_available=False,
                    canonical_progress=canonical_rca_state.event_count != version_before,
                ))
                state.no_progress_count = decision.no_progress_count
                if decision.action == "stop":
                    raise RuntimeError("investigation no-progress limit exceeded")
                if decision.action == "finalize":
                    ensure_candidate = permission.candidate
                    assert ensure_candidate is not None
                    termination_reason = "semantic_no_progress_sufficient_hypothesis"
                    return finalize_candidate(termination_reason)
            return None







        def investigate(step_limit: int, tool_limit: int, review_feedback: str = "",
                        review_baseline: RootCauseCandidate | None = None,
                        review_evidence_ids: frozenset[str] = frozenset(),
                        review_hypothesis: IncidentHypothesis | None = None) -> RootCauseCandidate:
            nonlocal active_skill, evidence_sufficient_step, evidence_sufficient_tool_calls
            nonlocal evidence_admission_rejections
            nonlocal converge_read_file_calls
            nonlocal final_candidate_step, termination_reason
            nonlocal duplicate_calls
            nonlocal planner_contract_retry_used
            nonlocal prior_context
            nonlocal llm_calls, planner_calls
            nonlocal consecutive_reflection_no_delta

            nonlocal runtime_action_constraint
            nonlocal runtime_tool_visibility_override

            nonlocal dag_state, interpretation_view, bootstrap_pending_observation_ids
            start_steps, start_tools = state.step, state.tool_calls
            prompt_compaction_attempts = 0
            local_recovery_offered = False

            def progress_marker() -> ProgressMarker:
                current = active_hypothesis()
                hypothesis_identity = (
                    json.dumps(self._material_hypothesis_fingerprint(current), sort_keys=True)
                    if current is not None else ""
                )
                scopes = set(evidence_projector.hypothesis_scope_fingerprints(current))
                return ProgressMarker(
                    qualified_requirements=frozenset((item.obligation_id, slot)
                        for item in evidence_sets.values()
                        if set(item.hypothesis_scope_fingerprints) == scopes
                        for slot, evidence_ids in item.coverage.items() if evidence_ids),
                    evidence_ids=frozenset(item.evidence_id for item in evidence_memory.pinned),
                    hypothesis=(hypothesis_identity,),
                    open_obligations=frozenset(
                        item.id for item in (current.verification_obligations if current else ())
                        if item.status == "OPEN"
                    ),
                    blocking_contradictions=frozenset(
                        item.evidence_id for item in (current.contradictions if current else ())
                        if item.blocks_finalization and item.status == "OPEN"
                    ),
                    components=frozenset(
                        value for value in ((current.component,) if current else ()) if value
                    ),
                )

            def skill_tool_surface(skill_id: str) -> tuple[str, ...]:
                """Return one Skill's permission- and capability-valid Tool surface."""
                allowed = capabilities.planner_tool_names(tools)
                names = list(domain_registry.tool_names_for_context(
                    skill_id,
                    available_tools=allowed,
                ))
                names.extend(
                    tools.official_aliases[action["tool"]]
                    for node in domain_registry.load_skill(skill_id).graph_nodes
                    for action in node["actions"]
                    if tools.official_aliases.get(action["tool"]) in set(allowed)
                )
                if "read_file" in names:
                    names.extend(
                        name for name in available_code_tools if name in set(allowed)
                    )
                if knowledge_enabled and "knowledge_retrieval" in set(allowed):
                    names.append("knowledge_retrieval")
                if "finalize_diagnosis" in set(allowed):
                    names.append("finalize_diagnosis")
                return tuple(dict.fromkeys(names))

            def visible_tool_names() -> tuple[str, ...] | None:
                """Expose the union of capability-feasible canonical Skill surfaces."""
                allowed = set(capabilities.planner_tool_names(tools))
                names = {
                    tool_name
                    for skill_id in feasible_skill_ids()
                    for tool_name in skill_tool_surface(skill_id)
                }
                if runtime_tool_visibility_override is not None:
                    requested = set(runtime_tool_visibility_override)
                    available_override = requested.intersection(allowed)
                    # A Router family can contain tools whose backing
                    # capability became EMPTY/UNAVAILABLE after the prior
                    # observation. Never send an empty native Tool schema:
                    # fall back to the currently available registry slice and
                    # let the next Runtime admission decide the action.
                    names = available_override or allowed
                # Keep the surface non-empty when a manifest has no applicable
                # Skill projection for a newly registered executable Tool.
                if not names and allowed:
                    names = allowed
                # Finalization is a Runtime permission, not merely a Planner
                # convention.  When the canonical predicate is false, do not
                # expose the terminal Tool for another doomed proposal.  The
                # taxonomy/evidence blockers remain visible in control state;
                # the Planner must select an evidence-changing action.
                try:
                    finalize_allowed = planner_finalization_projection().allowed
                except NameError:
                    finalize_allowed = False
                if not finalize_allowed:
                    names.discard("finalize_diagnosis")
                return tuple(sorted(names.intersection(allowed)))

            def load_domain_skill(skill_id: str):
                if not skill_id:
                    return None
                try:
                    skill = domain_registry.load_skill(skill_id)
                except KeyError:
                    return None
                trace.record("SKILL_LOADED", {
                    "step": state.step + 1,
                    "skill_id": skill.skill_id,
                    "manifest_sha256": domain_registry.manifest_sha256,
                    "node_count": len(skill.nodes),
                    "source_sha256": skill.source_sha256,
                    "derived_sha256": skill.derived_sha256,
                })
                trace.record("SKILL_CONTEXT_SELECTED", {
                    "step": state.step + 1,
                    "skill_id": skill.skill_id,
                    "source": "canonical_skill_projection",
                })
                return skill

            def ensure_review_progress(candidate: RootCauseCandidate,
                                        current_hypothesis: IncidentHypothesis | None) -> None:
                current_evidence_ids = {
                    item.evidence_id for item in evidence_memory.pinned
                }
                decision = ProgressPolicy.evaluate(operational_facts(
                    "review",
                    baseline_candidate_fingerprint=(self._candidate_semantic_fingerprint(review_baseline)
                                                    if review_baseline else None),
                    candidate_fingerprint=self._candidate_semantic_fingerprint(candidate),
                    baseline_hypothesis_fingerprint=(self._hypothesis_semantic_fingerprint(review_hypothesis)
                                                     if review_hypothesis else None),
                    hypothesis_fingerprint=(self._hypothesis_semantic_fingerprint(current_hypothesis)
                                            if current_hypothesis else None),
                    baseline_evidence_ids=tuple(sorted(review_evidence_ids)),
                    evidence_ids=tuple(sorted(current_evidence_ids)),
                ))
                if decision.action == "continue":
                    return
                trace.record("REVIEW_REFINALIZE_BLOCKED", {
                    "step": state.step,
                    "reason": decision.reasons[0],
                    "baseline_candidate": review_baseline.model_dump(),
                    "candidate": candidate.model_dump(),
                    "baseline_evidence_ids": sorted(review_evidence_ids),
                    "current_evidence_ids": sorted(current_evidence_ids),
                })
                raise ReviewProgressRequired(
                    "Review feedback was not addressed before re-finalization"
                )

            def planner_finalization_projection() -> FinalizationDecision:
                """Expose the exact typed Runtime decision to the Planner."""
                return finalization_decision()

            def planner_action_requirement() -> dict[str, Any]:
                """Describe the one open semantic question the next Tool must target."""
                current = active_hypothesis()
                open_obligations = tuple(
                    item for item in (
                        canonical_rca_state.obligations
                        if canonical_rca_state.obligations
                        else (current.verification_obligations if current else ())
                    )
                    if item.status == "OPEN" and item.blocks_finalization
                )
                active = open_obligations[0] if open_obligations else None
                projection = planner_finalization_projection()
                finalization_allowed = projection.allowed
                blocking_contradictions = tuple(
                    item for item in canonical_rca_state.contradictions
                    if item.blocks_finalization and item.status == "OPEN"
                )
                missing_public_fault_code = bool(
                    self.config.features.public_fault_taxonomy
                    and current
                    and not current.fault_code.strip()
                    and not open_obligations
                    and not blocking_contradictions
                )
                semantic_support_missing = bool(
                    current
                    and any(
                        item.code == "direct_causal_support_missing"
                        for item in projection.reasons
                    )
                )
                # Do not tell the Planner to acquire evidence after the same
                # Runtime predicate has explicitly allowed finalization. When
                # the predicate is false without an OPEN obligation, retain
                # the current gap as a recovery hint so the Planner can create
                # or target the missing structured verification question.
                needs_evidence = bool(
                    active
                    or (
                        current
                        and not finalization_allowed
                        and not missing_public_fault_code
                    )
                )
                return {
                    "hypothesis": current.claim if current else "",
                    "obligation_id": active.id if active else "",
                    "open_obligation": active.claim if active else "",
                    "verification_question": (
                        active.verification_question if active else ""
                    ),
                    "evidence_need": (
                        active.evidence_requirement
                        if active else (
                            "Revise the current evidence-supported RCA and populate the exact "
                            "canonical public fault_code; the canonical EvidenceSet is closed, "
                            "so do not create another Obligation or request a duplicate Tool."
                            if missing_public_fault_code else
                            "Acquire a new Evidence fact that directly supports the "
                            "current RCA, or revise the RCA to match the strongest "
                            "observed Evidence; do not repeat an executed action."
                            if semantic_support_missing else
                            (current.evidence_gap if needs_evidence and current else "")
                        )
                    ),
                    "must_acquire_new_canonical_evidence": needs_evidence,
                    "must_not_repeat_acquired_data": True,
                }





            def execute_plan(result, turn_input, hypothesis) -> None:
                """Execute one admitted plan through the sole DAG and ActionRuntime."""
                nonlocal dag_state, active_skill, prior_context, duplicate_calls
                nonlocal converge_read_file_calls, runtime_action_constraint
                nonlocal runtime_tool_visibility_override
                continuation = result.continuation
                plan = continuation.admitted_plan
                if plan is None or len(result.skill_selections) != 1:
                    raise RuntimeError("admitted plan requires one committed semantic selection")
                preflight = ContinuationPolicy.execution_barrier(
                    canonical_rca_state, continuation,
                    target_obligation_id=plan.tasks[0].target_obligation_id,
                    state_snapshot_hash=turn_input.state_snapshot_hash,
                )
                if preflight.action != "execute":
                    trace.record("SEMANTIC_BARRIER", {
                        "step": state.step, "reason_code": preflight.reason,
                        "committed_state_version": canonical_rca_state.event_count,
                        "observation_ids": [],
                    })
                    transition_phase("PLAN", preflight.reason)
                    return
                selection = result.skill_selections[0]
                previous_skill = active_skill
                active_skill = selection.skill
                load_domain_skill(active_skill)
                if previous_skill != active_skill:
                    trace.record("SKILL_TRANSITION", {
                        "step": state.step, "from": previous_skill or None,
                        "to": active_skill, "reason": selection.reason,
                    })
                trace.record("DOMAIN_SKILL_SELECTED", {
                    "step": state.step, "skill_id": active_skill,
                    "selection_authority": "planner_proposal_runtime_validated",
                    "ordering": "before_tool_execution",
                })
                if hypothesis is None:
                    raise RuntimeError("execution plan has no committed hypothesis")
                if dag_state is None:
                    dag_state = dag_harness.initialize(
                        (), obligation_ids=(item.id for item in canonical_rca_state.obligations),
                    )
                else:
                    project_obligation_ids_to_dag("plan_admission")
                # Provider task IDs are local to the plan; DAG history is run scoped.
                task_ids = {task.task_id: f"vt-{state.step:04d}-{task.task_id}" for task in plan.tasks}
                dag_state = dag_harness.add_tasks(dag_state, (
                    VerificationTask(
                        task_id=task_ids[task.task_id], obligation_id=task.target_obligation_id,
                        dependencies=tuple(task_ids[key] for key in task.dependencies),
                    ) for task in plan.tasks
                ))
                dag_task_metrics["planned"] += len(plan.tasks)
                for task in plan.tasks:
                    trace.record("DAG_TASK_PLANNED", {
                        "task_id": task_ids[task.task_id], "plan_task_id": task.task_id,
                        "obligation_id": task.target_obligation_id,
                        "dependencies": list(task.dependencies), "step": state.step,
                    })
                trace.record("EXECUTION_BURST_STARTED", {
                    "step": state.step, "task_count": len(plan.tasks),
                    "task_ids": list(task_ids.values()),
                    "committed_state_version": plan.committed_state_version,
                    "transport_profile": self.transport_profile,
                })
                burst_observation_ids: list[str] = []
                executed = 0
                barrier = None
                burst_error = None
                while barrier is None:
                    pending = [task for task in plan.tasks
                               if not dag_state.task(task_ids[task.task_id]).terminal]
                    task = next((item for item in pending
                                 if dag_state.task(task_ids[item.task_id]).status == "READY"), None)
                    barrier_check = ContinuationPolicy.execution_barrier(
                        canonical_rca_state, continuation,
                        target_obligation_id=task.target_obligation_id if task else None,
                        state_snapshot_hash=turn_input.state_snapshot_hash,
                        ready=task is not None, complete=not pending,
                    )
                    if barrier_check.action != "execute":
                        barrier = barrier_check
                        break
                    try:
                        upstream = {
                            key: asdict(observation_store.get(dag_state.task(task_ids[key]).result_ref))
                            for key in task.dependencies
                            if dag_state.task(task_ids[key]).status == "COMPLETED"
                            and observation_store.get(dag_state.task(task_ids[key]).result_ref) is not None
                        }
                        call = compile_planned_action(
                            task, tools,
                            state_fields={**canonical_rca_state.context_projection(), "incident_id": case.case_id},
                            upstream_observations=upstream,
                        )
                    except VerificationPlanError as exc:
                        barrier = ContinuationPolicy.execution_barrier(
                            canonical_rca_state, continuation,
                            target_obligation_id=task.target_obligation_id, binding_reason=exc.reason_code,
                        )
                        break
                    trace.record("ACTION_COMPILED", {
                        "step": state.step, "task_id": task_ids[task.task_id],
                        "plan_task_id": task.task_id, "operation": call.name,
                        "arguments": call.arguments, "compiler": "deterministic",
                        "provider_calls": 0,
                    })
                    permitted = skill_tool_surface(task.skill_id)
                    context = ActionExecutionContext(
                        action=ActionProposal(
                            kind=ActionKind.TOOL, skill=task.skill_id, reason="admitted_plan_task",
                            tool=call.name, arguments=call.arguments,
                        ),
                        permitted_tool_names=permitted,
                        visible_tool_names=turn_input.visible_tool_names,
                        capability_tool_names=tuple(capabilities.planner_tool_names(tools)),
                        absolute_deadline=deadline.absolute_deadline,
                        requested_for=InvestigationUse(task.target_obligation_id, task.target_fact_slot,
                            tuple(evidence_projector.hypothesis_scope_fingerprints(hypothesis)),
                            task.task_id, task.node_id),
                    )
                    admission = action_runtime.admit(context)
                    if not admission.allowed:
                        if admission.reason_code == "ACTION_ACQUISITION_ALREADY_ACQUIRED":
                            duplicate_calls += 1
                            state.repeated_actions += 1
                        if admission.duplicate is not None:
                            trace.record("ACQUISITION_DUPLICATE", {**asdict(admission.duplicate),
                                "requested_for": asdict(context.requested_for)})
                            if not interpretation_view.is_observation_visible(admission.duplicate.existing_observation_id):
                                trace.record("EXISTING_ACQUISITION_NOT_VISIBLE", {
                                    **asdict(admission.duplicate), "reason_code": "existing_acquisition_not_visible"})
                        rejected_actions.append({
                            "step": state.step, "tool": call.name,
                            "arguments": call.arguments, "reason": admission.reason_code,
                            "existing_observation_id": admission.duplicate.existing_observation_id if admission.duplicate else None,
                        })
                        trace.record("ACTION_REJECTED", rejected_actions[-1])
                        barrier = ContinuationPolicy.execution_barrier(
                            canonical_rca_state, continuation,
                            target_obligation_id=task.target_obligation_id,
                            binding_reason=admission.reason_code,
                        )
                        break
                    capacity = ProgressPolicy.evaluate(operational_facts(
                        "admission", stage="tool",
                        tool_limit=min(self.config.max_tool_calls, start_tools + tool_limit),
                    ))
                    if capacity.action == "stop":
                        barrier = ContinuationPolicy.execution_barrier(
                            canonical_rca_state, continuation,
                            target_obligation_id=task.target_obligation_id,
                            binding_reason=capacity.reasons[0],
                        )
                        break
                    read_phase = budget_snapshot("plan_source_read_gate").phase
                    if call.name == "read_file" and read_phase in {"converge", "source_verify"}:
                        read_limit = (self.config.converge_max_read_file_calls
                                      if read_phase == "converge" else self.config.source_verify_max_read_file_calls)
                        if converge_read_file_calls >= read_limit:
                            barrier = ContinuationPolicy.execution_barrier(
                                canonical_rca_state, continuation,
                                target_obligation_id=task.target_obligation_id,
                                binding_reason="source_read_file_budget_exhausted",
                            )
                            break
                        converge_read_file_calls += 1
                    dag_state, selected = dag_harness.select_ready(dag_state, task_ids[task.task_id])
                    assert selected is not None
                    action = {
                        "step": state.step, "skill": task.skill_id, "skill_reason": selection.reason,
                        "tool": call.name, "arguments": admission.arguments,
                        "obligation_id": task.target_obligation_id,
                        "target_fact_slot": task.target_fact_slot,
                        "requested_for": asdict(context.requested_for),
                        "semantic_intents": [item.model_dump() for item in task.semantic_intents],
                        "dag_task_id": selected.task_id, "plan_task_id": task.task_id,
                        "action_rationale": selection.evidence_gap,
                        "expected_information_gain": selection.expected_information_gain,
                        "action_class": "acquisition", "information_gain": "pending_semantic",
                    }
                    actions.append(action)
                    try:
                        observation = execute_tool(
                            admission.arguments, call.name, permitted_tools=permitted,
                            visible_tools=turn_input.visible_tool_names,
                            requested_for=context.requested_for, action_record=action,
                        )
                    except Exception as exc:
                        burst_error = exc
                        dag_state = dag_harness.apply_execution(dag_state, VerificationTaskExecution(
                            task_id=selected.task_id, status="FAILED", reason=type(exc).__name__,
                        ))
                        dag_task_metrics["failed"] += 1
                        barrier = ContinuationPolicy.execution_barrier(
                            canonical_rca_state, continuation,
                            target_obligation_id=task.target_obligation_id, failed=True,
                        )
                        break
                    if isinstance(observation, DuplicateAcquisition):
                        barrier = ContinuationPolicy.execution_barrier(canonical_rca_state, continuation,
                            target_obligation_id=task.target_obligation_id,
                            binding_reason="acquisition_already_acquired")
                        break
                    state.tool_calls += 1
                    state.repeated_actions = 0
                    admit_tool_observation(
                        observation, tool_name=call.name, arguments=admission.arguments,
                        obligation=canonical_rca_state.obligation(task.target_obligation_id),
                        hypothesis=hypothesis,
                    )
                    dag_state = dag_harness.apply_execution(dag_state, VerificationTaskExecution(
                        task_id=selected.task_id, status="COMPLETED" if observation.ok else "FAILED",
                        result_ref=observation.observation_id, reason=observation.error_type or "",
                    ))
                    dag_task_metrics["completed" if observation.ok else "failed"] += 1
                    executed += 1
                    burst_observation_ids.append(observation.observation_id)
                    action.update({
                        "observation_id": observation.observation_id,
                        "outcome": "ok" if observation.ok else "error",
                        "error_type": observation.error_type,
                        "observation_status": observation.metadata.get("status"),
                        "new_evidence_count": 0, "obligation_progress_count": 0,
                        "hypothesis_changed": False, "new_evidence_ids": [],
                        "semantic_information_gain": False,
                    })
                    trace.record("DAG_TASK_EXECUTED", {
                        "task_id": selected.task_id, "status": dag_state.task(selected.task_id).status,
                        "result_ref": observation.observation_id, "dag_state_metrics": dag_state.metrics(),
                    })
                    if call.name == "knowledge_retrieval":
                        retrieved = getattr(knowledge_tool, "last_result", None)
                        retrieved_context = getattr(knowledge_tool, "last_prior_context", None)
                        if retrieved_context is not None:
                            prior_context = retrieved_context
                            trace.record("KNOWLEDGE_RETRIEVED", {
                                "stage": "on_demand",
                                **retrieved.diagnostics.model_dump(mode="json"),
                                "candidate_count": len(retrieved.candidates),
                            })
                            if retrieved.diagnostics.degraded:
                                trace.record("KNOWLEDGE_RETRIEVAL_DEGRADED", {
                                    "stage": "on_demand",
                                    "reason": retrieved.diagnostics.reason,
                                })
                            trace.record("PRIOR_CONTEXT_PACKED", {
                                "stage": "on_demand",
                                "context_id": retrieved_context.context_id,
                                "source_types": list(retrieved_context.source_types),
                                "candidate_count": len(retrieved_context.candidates),
                                "packed_chars": retrieved_context.packed_chars,
                                "packed_tokens": retrieved_context.packed_tokens,
                            })
                    if call.name == "code_search":
                        retrieval_metadata = dict(observation.metadata or {})
                        trace.record("RETRIEVAL_MODE", {
                            "requested_mode": retrieval_metadata.get("requested_mode", "lexical"),
                            "effective_mode": retrieval_metadata.get("effective_mode", "lexical"),
                            "degraded": bool(retrieval_metadata.get("degraded", False)),
                            "reason": retrieval_metadata.get("reason", ""),
                        })
                        trace.record("RETRIEVAL_RESULT_SUMMARY", {
                            "tool": call.name,
                            "matches": retrieval_metadata.get("matches", 0),
                            "candidate_only": retrieval_metadata.get("information_source") == "candidate_retrieval",
                        })
                    if not observation.ok:
                        barrier = ContinuationPolicy.execution_barrier(
                            canonical_rca_state, continuation,
                            target_obligation_id=task.target_obligation_id, failed=True,
                        )
                trace.record("EXECUTION_BURST_COMPLETED", {
                    "step": state.step, "executed_task_count": executed,
                    "observation_ids": burst_observation_ids, "reason_code": barrier.reason,
                })
                trace.record("SEMANTIC_BARRIER", {
                    "step": state.step, "reason_code": barrier.reason,
                    "target_obligation_id": barrier.target_obligation_id,
                    "committed_state_version": canonical_rca_state.event_count,
                    "observation_ids": burst_observation_ids,
                })
                runtime_action_constraint = json.dumps({
                    "reason_code": barrier.reason, "observation_ids": burst_observation_ids,
                    "open_obligation_ids": list(canonical_rca_state.open_critical_obligation_ids),
                }, sort_keys=True)
                runtime_tool_visibility_override = None
                transition_phase("PLAN", barrier.reason)
                if burst_error is not None:
                    raise burst_error


            while True:
                progress_before_action = progress_marker()
                for blocked_tool in sorted(pending_capability_blocks):
                    mark_blocked_capability(blocked_tool)
                pending_capability_blocks.clear()
                permission = finalization_decision()
                runtime_budget = budget_snapshot("investigation_loop")
                dynamic_budget.set_run_usage(
                    tokens_used=prompt_tokens + completion_tokens,
                    llm_calls_used=llm_calls, cost_used=run_cost,
                )
                pressure_before_planner = dynamic_budget.decision(
                    "planner_admission", tokens_used=prompt_tokens,
                    llm_calls_used=llm_calls, cost_used=run_cost,
                )
                phase_remaining = dynamic_budget.lifecycle_payload(
                    tokens_used=prompt_tokens + completion_tokens, llm_calls_used=llm_calls
                )["phase_remaining"]["llm_calls"]
                review_seconds = self.config.obligation_review_min_seconds + self.config.llm_cleanup_margin_seconds
                closing_pressure = (runtime_budget.phase == "finalize" or phase_remaining <= 3
                    or state.tool_calls >= min(self.config.max_tool_calls, start_tools + tool_limit)
                    or deadline.remaining() <= self.config.finalization_reserve_seconds)
                if (not permission.allowed and closing_pressure and active_hypothesis() is not None
                        and (incident_evidence() or any(row.get("observation_id") for row in actions))
                        and state.step < min(self.config.max_steps, start_steps + step_limit)
                        and deadline.remaining() > review_seconds
                        and dynamic_budget.lifecycle_phase in {"investigation", "investigation_recovery"}):
                    dynamic_budget.begin_consolidation(
                        llm_calls_used=sum(row.attempt_count for row in provider_gateway.records),
                        review_seconds=review_seconds)
                    trace.record("BUDGET_CONSOLIDATION_STARTED", {
                        "step": state.step, "budget": budget_payload(runtime_budget),
                        "tool_calls_allowed": 0, "review_permission": permission.allowed})
                consolidation = dynamic_budget.lifecycle_phase == "consolidation"
                continuation = ProgressPolicy.evaluate(operational_facts(
                    "loop", finalization=permission, phase=runtime_budget.phase,
                    consolidation_available=consolidation,
                    step_limit=min(self.config.max_steps, start_steps + step_limit),
                    tool_limit=min(self.config.max_tool_calls, start_tools + tool_limit),
                    hard_pressure=pressure_before_planner.state is BudgetState.HARD_PRESSURE,
                    automatic_finalization=execution_mode == "dag",
                    post_sufficiency_calls=(state.tool_calls - evidence_sufficient_tool_calls
                                            if evidence_sufficient_tool_calls is not None else None),
                    post_sufficiency_limit=self.config.max_post_sufficiency_tool_calls,
                ))
                trace.record("PROGRESS_POLICY_DECISION", {
                    "boundary": "loop", "action": continuation.action,
                    "reasons": list(continuation.reasons),
                })
                if continuation.action == "stop":
                    reason = continuation.reasons[0]
                    if reason == "global_deadline":
                        raise TimeoutError("global run deadline exceeded")
                    raise RuntimeError(reason + " budget exhausted before finalization")
                if continuation.action == "finalize":
                    candidate = permission.candidate
                    assert candidate is not None
                    final_candidate_step = state.step
                    reason = continuation.reasons[0]
                    termination_reason = (
                        "runtime_convergence_auto_finalize"
                        if reason == "canonical_finalization_allowed"
                        else "step_budget_sufficient_hypothesis" if reason == "max_steps"
                        else reason + "_sufficient_hypothesis"
                    )
                    ensure_review_progress(candidate, active_hypothesis())
                    if reason == "canonical_finalization_allowed":
                        trace.record("RUNTIME_CONVERGENCE_AUTO_FINALIZE", {
                            "step": state.step, "candidate": candidate.model_dump(),
                            "reason": "canonical FinalizationPolicy allows publication",
                        })
                    return finalize_candidate(termination_reason)
                tool_budget_remaining = max(0, tool_limit - (state.tool_calls - start_tools))
                agent_steps_remaining = max(0, step_limit - (state.step - start_steps))
                dynamic_state = dynamic_budget.decision(
                    "planner_context", tokens_used=prompt_tokens,
                    llm_calls_used=llm_calls,
                    cost_used=run_cost,
                )
                current_evidence_payload = [
                    item.model_dump(mode="json") for item in incident_evidence()
                ]
                prior_capacity = dynamic_budget.available_prior_capacity(
                    critical_state_tokens=dynamic_budget.estimate_json_tokens({
                        "incident": {"summary": case.summary, "system": case.system},
                        "hypothesis": (
                            active_hypothesis().model_dump()
                            if active_hypothesis() is not None else None
                        ),
                    }),
                    verified_evidence_tokens=dynamic_budget.estimate_json_tokens(current_evidence_payload),
                    required_control_tokens=dynamic_budget.estimate_json_tokens({
                        "actions": actions[-8:], "rejected_actions": rejected_actions[-4:],
                        "obligations": canonical_rca_state.obligations,
                    }),
                    output_protocol_reserve=dynamic_budget.capability.protocol_safety_reserve_tokens,
                )
                bounded_prior = self._bound_prior_context(
                    prior_context, prior_capacity, dynamic_budget.estimate_tokens,
                )
                if prior_context is not None and bounded_prior is not None and (
                    len(bounded_prior.candidates) != len(prior_context.candidates)
                    or bounded_prior.packed_tokens != prior_context.packed_tokens
                ):
                    trace.record("PRIOR_PRUNED", {
                        "stage": "planner",
                        "before_candidates": len(prior_context.candidates),
                        "after_candidates": len(bounded_prior.candidates),
                        "before_tokens": prior_context.packed_tokens,
                        "after_tokens": bounded_prior.packed_tokens,
                        "available_prior_capacity": prior_capacity,
                        "context_state": dynamic_state.state.value,
                    })
                planner_finalization = planner_finalization_projection()
                finalization_allowed = planner_finalization.allowed
                finalization_blockers = [
                    item.model_dump(mode="json")
                    for item in planner_finalization.reasons
                ]
                terminal_projection_allowed = finalization_allowed
                if not terminal_projection_allowed and active_hypothesis() is not None:
                    # Public taxonomy completion has one bounded semantic
                    # repair case: all canonical evidence/obligation gates
                    # are already clear and only the exact taxonomy code is
                    # missing. Permit the declared semantic request to provide
                    # that code without forcing a second no-op Tool call.
                    # Any evidence, contradiction, or lifecycle blocker keeps
                    # semantic-only projection disabled.
                    current = active_hypothesis()
                    assert current is not None
                    if (
                        self.config.features.public_fault_taxonomy
                        and not current.fault_code.strip()
                        and {
                            item.code for item in planner_finalization.reasons
                        } == {"taxonomy_code_missing"}
                    ):
                        terminal_projection_allowed = True
                        trace.record("SEMANTIC_CODE_COMPLETION_ALLOWED", {
                            "step": state.step + 1,
                            "reason": "only_public_fault_code_missing",
                            "supporting_evidence_ids": list(
                                current.supporting_evidence_ids
                            ),
                        })
                control_state = self._planner_control_state(
                    case, canonical_rca_state, actions, review_feedback,
                    tool_budget_remaining=tool_budget_remaining,
                    agent_steps_remaining=agent_steps_remaining,
                    llm_calls_remaining=max(0, self.config.max_llm_calls - llm_calls),
                    rejected_actions=rejected_actions,
                    available_evidence_sources=case.evidence_sources,
                    source_workspace_available=source_workspace_available,
                    available_code_tools=available_code_tools,
                    capability_states=capabilities.states,
                    visible_tool_names=visible_tool_names(),
                    feasible_skill_ids=feasible_skill_ids(),
                    source_evidence_ids=source_evidence_ids(),
                    reflection_feedback=reflection_feedback,
                    tool_health=tool_circuit.summary(),
                    reflection_trigger=reflection_trigger,
                    next_action_constraint=(
                        runtime_action_constraint or reflection_constraint
                    ),
                    finalization_allowed=finalization_allowed,
                    finalization_blockers=finalization_blockers,
                    active_action_requirement=planner_action_requirement(),
                    canonical_task_context=(
                        execution_mode == "dag"
                        and self.config.context_projection_mode == "task_aware"
                    ),
                    dag_state=dag_state,
                    citable_evidence_ids=tuple(item.evidence_id for item in incident_evidence()),
                    evidence_verdicts=tuple(
                        {
                            "evidence_id": fact.evidence_id,
                            "obligation_id": fact.obligation_id,
                            "relation": str(getattr(fact, "relation", "")),
                            "fulfillment": str(getattr(fact, "fulfillment", "")),
                            "fact": str(getattr(fact, "fact", ""))[:360],
                        }
                        for fact in projected_facts.values()
                    ),
                    domain_skill_id=active_skill,
                )
                exposed_tools = visible_tool_names()
                if exposed_tools is not None:
                    trace.record("TOOL_EXPOSURE_SLICE", {
                        "step": state.step + 1,
                        "active_skill": active_skill,
                        "tools": list(exposed_tools),
                        "policy_version": self.config.tool_exposure_policy_version,
                        "registry_authority": "CloudOpsSnapshotToolRegistry",
                        "reason": (
                            "runtime_replan_tool_surface"
                            if runtime_tool_visibility_override is not None
                            else "stage_and_capability_aware_provider_catalog"
                        ),
                        "runtime_override": (
                            list(runtime_tool_visibility_override)
                            if runtime_tool_visibility_override is not None else []
                        ),
                    })
                # The current role envelope is accounted for by ContextManager's
                # request renderer. No legacy control-prefix reservation.
                planner_context_char_limit = self.config.planner_context_chars
                review_recovery_focus = bool(review_feedback)
                focused_runtime_action = bool(
                    (runtime_action_constraint and runtime_tool_visibility_override)
                    or review_recovery_focus
                )
                context_char_budget = (
                    min(planner_context_char_limit, 12000)
                    if focused_runtime_action
                    else planner_context_char_limit
                )
                dynamic_context_tokens = dynamic_budget.context_token_budget(
                    state=dynamic_state.state,
                )
                closure_ids = tuple(dict.fromkeys(
                    evidence_id
                    for evidence_id in (
                        *(
                            active_hypothesis().supporting_evidence_ids
                            if active_hypothesis() is not None else ()
                        ),
                        *(
                            evidence_id
                            for claim in (
                                active_hypothesis().source_claims
                                if active_hypothesis() is not None else ()
                            )
                            for evidence_id in claim.evidence_ids
                        ),
                        *(
                            source_evidence_ids()
                            if runtime_budget.phase in {"source_verify", "converge", "verify_only", "finalize"}
                            else ()
                        ),
                        # RCAState is the canonical admitted-Evidence index.
                        # A Planner hypothesis can temporarily omit a partial
                        # or revised-branch fact even though that fact is still
                        # needed to understand why the closed obligation was
                        # satisfied. Retain a bounded recent tail during
                        # closure so ContextManager cannot evict the exact
                        # configuration that produced the diagnosis. This is
                        # prompt visibility only; it does not promote support
                        # or alter Evidence/Obligation lifecycle.
                        *(
                            canonical_rca_state.evidence_ids[-8:]
                            if runtime_budget.phase in {"source_verify", "converge", "verify_only", "finalize"}
                            else ()
                        ),
                    )
                    if evidence_id in {item.evidence_id for item in evidence_memory.pinned}
                ))
                if runtime_budget.phase in {"source_verify", "converge", "verify_only", "finalize"}:
                    trace.record("CLOSURE_CONTEXT_PACK", {
                        "phase": runtime_budget.phase,
                        "retained_evidence_ids": list(closure_ids),
                        "canonical_state_evidence_tail": list(
                            canonical_rca_state.evidence_ids[-8:]
                        ),
                        "reason": "retain hypothesis/source-claim Evidence during bounded closure",
                    })
                def semantic_input_projection():
                    current_permission = finalization_decision()
                    projection = {
                                **canonical_rca_state.context_projection(),
                                "incident": {"summary": case.summary, "namespace": case.namespace, "system": case.system},
                                "capabilities": capabilities.planner_projection(),
                                "state_version": canonical_rca_state.event_count,
                                "hypothesis_scope": evidence_projector.hypothesis_scope_fingerprints(active_hypothesis()),
                                "incident_id": case.case_id,
                                "review_feedback": review_feedback,
                                "reflection_feedback": reflection_feedback,
                                "action_admission_feedback": (json.loads(runtime_action_constraint)
                                    if runtime_action_constraint else None),
                                "evidence_hypothesis_id": evidence_projector.hypothesis_id(active_hypothesis()),
                                "pending_observation_ids": list(dict.fromkeys((
                                    *bootstrap_pending_observation_ids,
                                    *(row['observation_id'] for row in actions
                                      if row.get('step') == state.step and row.get('observation_id')
                                      and observation_store.get(row['observation_id']).ok),
                                ))),
                                "visible_observations": interpretation_view.reference_metadata.get("visible_observations", []),
                                "existing_evidence": evidence_memory.projection(
                                    facts=projected_facts.values(), evidence_ids=canonical_rca_state.evidence_ids,
                                ),
                                "acquisitions": action_runtime.acquisition_projection(actions),
                                "acquisition_snapshot_identity": action_runtime.snapshot_identity,
                                "evidence_gaps": [
                                    {"obligation_id": item.id, "required_fact_slots": list(item.required_fact_slots),
                                     "question": item.verification_question,
                                     "coverage": {
                                         slot: sorted({key for evidence_set in evidence_sets.values()
                                                       if evidence_set.obligation_id == item.id
                                                       and evidence_set.hypothesis_id == evidence_projector.hypothesis_id(active_hypothesis())
                                                       and set(evidence_set.hypothesis_scope_fingerprints) == set(evidence_projector.hypothesis_scope_fingerprints(active_hypothesis()))
                                                       for key in evidence_set.coverage.get(slot, ())})
                                         for slot in item.required_fact_slots
                                     }, "blocked_capabilities": list(item.blocked_capabilities)}
                                    for item in canonical_rca_state.obligations if item.status == "OPEN"
                                ],
                                "finalization_readiness": {
                                    "allowed": current_permission.allowed,
                                    "conditions": current_permission.conditions.model_dump(mode="json"),
                                    "reasons": [reason.model_dump(mode="json") for reason in current_permission.reasons],
                                },
                            }

                    rows = requirement_table(projection)
                    allowed = {}
                    observations = {obs.observation_id: obs for obs in observation_store.all()}
                    projection['admissible_observation_refs'] = [ref for ref, obs in observations.items()
                        if evidence_projector.incident_observation_available(obs)]
                    for evidence in evidence_memory.pinned:
                        obs = observation_store.get(evidence.raw_observation_id)
                        if obs is not None:
                            observations[evidence.evidence_id] = obs
                    for ref, obs in observations.items():
                        allowed[ref] = []
                        for row in rows:
                            obligation = canonical_rca_state.obligation(row['obligation_id'])
                            try:
                                evidence_pipeline._validate_constraints(obligation, (row['slot'],), obs)
                            except EvidencePipelineError:
                                continue
                            allowed[ref].append([row['obligation_id'], row['slot']])
                    projection['source_requirement_slots'] = allowed
                    return projection

                semantic_metadata = semantic_input_projection()
                def render_interpretation_context(view):
                    envelope_input = planner.create_interpretation_input(
                        state, view, visible_tool_names=exposed_tools,
                        active_skill_id=active_skill, feasible_skill_ids=feasible_skill_ids(),
                        skill_tool_names_by_id={skill_id: skill_tool_surface(skill_id)
                                               for skill_id in feasible_skill_ids()},
                        semantic_projection=semantic_metadata,
                        budget_projection=request_budget_status(),
                    )
                    return planner.render_interpretation_envelope(
                        envelope_input, terminal_projection_allowed=terminal_projection_allowed,
                        context_resolution_remaining=PlanningSession.CONTEXT_ROUND_LIMIT,
                        consolidation_allowed=consolidation,
                    )
                context_build_kwargs = dict(
                    local_requests=context_manager.retained_local_requests(
                        interpretation_view, evidence_memory, observation_store),
                    max_context_chars=(
                        None if dynamic_context_tokens is not None else context_char_budget
                    ),
                    max_context_tokens=dynamic_context_tokens,
                    token_estimator=dynamic_budget.estimate_tokens,
                    max_steps=step_limit, max_tool_calls=tool_limit,
                    requested_ids=(*closure_ids, *bootstrap_pending_observation_ids, *(row["observation_id"] for row in actions
                        if row.get("step") == state.step and row.get("observation_id"))),
                    include_agent_control_state=False,
                    role="interpretation", max_request_chars=self.config.max_context_chars,
                    evidence_references=semantic_metadata["existing_evidence"],
                    reference_ids=tuple(row['existing_observation_id'] for row in rejected_actions[-6:]
                                        if row.get('existing_observation_id')),
                    request_renderer=render_interpretation_context,
                    pressure_state=dynamic_state.state.value,
                    verification_dag_state=(
                        dag_state
                        if self.config.context_projection_mode == "task_aware"
                        else None
                    ),
                    rca_state=canonical_rca_state,
                )
                context_result = context_manager.build(state, evidence_memory, observation_store, **context_build_kwargs)
                interpretation_view = context_result.context_view
                context = interpretation_view.text
                if dynamic_state.state is not BudgetState.NORMAL:
                    trace.record("CONTEXT_PRESSURE", {
                        "stage": "planner",
                        "context_state": dynamic_state.state.value,
                        "reason": dynamic_state.reason,
                        "budget_tokens": context_result.budget_tokens,
                        "used_tokens": context_result.used_tokens,
                        "prior_capacity": prior_capacity,
                    })
                if dynamic_state.state is not BudgetState.NORMAL or context_result.dropped:
                    trace.record("CONTEXT_COMPACTED", {
                        "stage": "planner",
                        "context_state": dynamic_state.state.value,
                        "budget_tokens": context_result.budget_tokens,
                        "used_tokens": context_result.used_tokens,
                        "dropped_count": len(context_result.dropped),
                        "selected_count": len(context_result.selected),
                        "eviction_count": context_result.eviction_count,
                        "rehydrated_evidence_count": context_result.breakdown.get(
                            "rehydrated_item_count", 0
                        ),
                    })
                trace.record("CONTEXT_BUILT", {
                    "stage": "planner", "budget_chars": context_result.budget_chars,
                    "used_chars": context_result.used_chars,
                    "catalog_size": context_result.catalog_size,
                    "working_set_size": context_result.working_set_size,
                    "selected": context_result.selected, "dropped": context_result.dropped,
                    "invalid_requested_ids": context_result.invalid_requested_ids,
                    "breakdown": context_result.breakdown,
                    "known_context_chars": context_result.known_context_chars,
                    "active_item_count": context_result.active_item_count,
                    "cold_item_count": context_result.cold_item_count,
                    "eviction_count": context_result.eviction_count,
                    "projection_count": context_result.projection_count,
                    "display_coverage": context_result.display_coverage,
                    "remaining_tool_calls": tool_budget_remaining,
                    "remaining_agent_steps": agent_steps_remaining,
                    "planner_call_index": state.step + 1,
                    "available_evidence_sources": list(case.evidence_sources),
                    "source_workspace_available": source_workspace_available,
                    "available_code_tools": list(available_code_tools),
                    "diagnostic_context_chars": len(context_result.text),
                    "control_state_chars": 0,
                    "planner_context_chars": len(context),
                    "planner_context_char_limit": context_char_budget,
                    "focused_runtime_action": focused_runtime_action,
                    "review_recovery_focus": review_recovery_focus,
                    "budget_tokens": context_result.budget_tokens,
                    "used_tokens": context_result.used_tokens,
                    "context_state": dynamic_state.state.value,
                    "input_hard_capacity": dynamic_state.input_hard_capacity,
                    "prior_capacity": prior_capacity,
                    "prompt_snapshot": (
                        context_result.prompt_snapshot.as_dict()
                        if context_result.prompt_snapshot is not None else None
                    ),
                })
                rehydrated = [
                    item for item in context_result.selected
                    if str(item.get("projection_reason", "")).startswith("rehydrated_")
                ]
                if rehydrated:
                    trace.record("OBSERVATION_REHYDRATED", {
                        "stage": "planner",
                        "count": len(rehydrated),
                        "evidence_ids": [item.get("id") for item in rehydrated],
                        "reason": "active_context_projection",
                    })
                planner_phase = budget_snapshot("planner_start").phase
                converge_budget_kwargs: dict[str, Any] = {}
                if planner_phase in {"converge", "source_verify"}:
                    phase_is_converge = planner_phase == "converge"
                    phase_reasoning_limit = (
                        self.config.converge_max_reasoning_tokens
                        if phase_is_converge else self.config.source_verify_max_output_tokens
                    )
                    phase_output_limit = (
                        self.config.converge_max_output_tokens
                        if phase_is_converge else self.config.source_verify_max_output_tokens
                    )
                    phase_read_limit = (
                        self.config.converge_max_read_file_calls
                        if phase_is_converge else self.config.source_verify_max_read_file_calls
                    )
                    remaining_reads = max(
                        0, phase_read_limit - converge_read_file_calls
                    )
                    converge_budget_kwargs["max_output_tokens"] = phase_output_limit
                    converge_budget_kwargs["enable_thinking"] = CONVERGE_ENABLE_THINKING
                    trace.record("CONVERGE_BUDGET_APPLIED", {
                        "step": state.step + 1,
                        "phase": planner_phase,
                        "max_reasoning_tokens": phase_reasoning_limit,
                        "provider_completion_cap_tokens": phase_output_limit,
                        "enable_thinking": CONVERGE_ENABLE_THINKING,
                        "reasoning_mode": "disabled",
                        "max_read_file_calls": phase_read_limit,
                        "read_file_calls_used": converge_read_file_calls,
                        "read_file_calls_remaining": remaining_reads,
                        "converge_start_step": CONVERGE_START_STEP,
                        "reflections_disabled": bool(
                            CONVERGE_DISABLE_REFLECTIONS
                            and self.config.converge_disable_reflections
                        ),
                        "enforcement": "runtime_adapter_completion_cap_and_action_gate",
                    })

                prompt_budget_retry = False
                contract_retry_for_step = False

                def record_planner_request_snapshot(status: str) -> None:
                    snapshots = list(
                        getattr(planning_session, "last_request_snapshots", ()) or ()
                    )
                    if not snapshots:
                        snapshots = list(getattr(planner, "last_request_snapshots", ()) or ())
                    if not snapshots:
                        snapshot = getattr(planner, "last_request_snapshot", None)
                        snapshots = [snapshot] if snapshot else []
                    for snapshot in snapshots:
                        if not snapshot:
                            continue
                        payload = dict(snapshot)
                        payload["runtime_status"] = status
                        payload["step"] = state.step + 1

                        trace.record("PLANNER_REQUEST_SNAPSHOT", payload)
                    if not snapshots and status == "timeout_before_snapshot":
                        trace.record("PLANNER_REQUEST_SNAPSHOT_UNAVAILABLE", {
                            "step": state.step + 1,
                            "reason": "provider process terminated before child state returned",
                            "contract_retry": contract_retry_for_step,
                        })

                while True:
                    trace.record("PLANNER_CALL_STARTED", {
                        "step": state.step + 1,
                        "transport_profile": planner.transport_profile.value,
                        "logical_timeout_seconds": self.config.planner_llm_timeout_seconds,
                        "contract_retry": contract_retry_for_step,
                        "budget": budget_payload(budget_snapshot("planner_start")),
                    })
                    try:
                        admit(
                            "planner", count_llm=True,
                            allow_targeted_recovery=contract_retry_for_step,
                        )
                        planner_timeout = provider_timeout(self.config.planner_llm_timeout_seconds)
                        planner_completion_cap = self.config.planner_max_completion_tokens
                        if focused_runtime_action:
                            planner_completion_cap = min(planner_completion_cap, 1800)
                        # A Review rejection supplies a bounded correction
                        # target. Keep the recovery call inside the remaining
                        # investigation allocation instead of reserving a
                        # full exploratory completion after the context and
                        # Review call have already consumed most of it.
                        if review_recovery_focus:
                            planner_completion_cap = min(planner_completion_cap, 1600)
                        planner_kwargs = {
                            "max_output_tokens": planner_completion_cap,
                            "active_skill_id": active_skill,
                            "feasible_skill_ids": feasible_skill_ids(),
                            "skill_tool_names_by_id": {
                                skill_id: skill_tool_surface(skill_id)
                                for skill_id in feasible_skill_ids()
                            },
                            "context_snapshot": (
                                context_result.prompt_snapshot.as_dict()
                                if context_result.prompt_snapshot is not None else {}
                            ),
                            "semantic_projection": semantic_input_projection(),
                            "budget_projection": control_state.get(
                                "REMAINING_BUDGET", {}
                            ),
                            "action_ledger": tuple(
                                control_state.get("EXECUTED_ACTION_SUMMARY", {}).get(
                                    "shown", ()
                                )
                            ),
                            "terminal_projection_allowed": terminal_projection_allowed,
                            **dict(converge_budget_kwargs),
                        }
                        if review_recovery_focus:
                            planner_kwargs["max_output_tokens"] = min(
                                int(planner_kwargs.get("max_output_tokens", planner_completion_cap)),
                                1600,
                            )
                        turn_input = planner.create_interpretation_input(
                            state,
                            interpretation_view,
                            visible_tool_names=exposed_tools,
                            active_skill_id=planner_kwargs["active_skill_id"],
                            feasible_skill_ids=planner_kwargs["feasible_skill_ids"],
                            skill_tool_names_by_id=planner_kwargs["skill_tool_names_by_id"],
                            context_snapshot=planner_kwargs["context_snapshot"],
                            semantic_projection=planner_kwargs["semantic_projection"],
                            budget_projection=planner_kwargs["budget_projection"],
                            action_ledger=planner_kwargs["action_ledger"],
                        )
                        # Avoid attributing a prior turn's request to a
                        # timeout or exception in the current Provider call.
                        planner.last_request_snapshot = None
                        planning_session.last_request_snapshots = ()
                        committed_hypothesis = None

                        def commit_interpretation(selection, response):
                            nonlocal committed_hypothesis, bootstrap_pending_observation_ids
                            nonlocal evidence_admission_rejections
                            nonlocal evidence_sufficient_step, evidence_sufficient_tool_calls
                            prior_hypothesis = active_hypothesis()
                            proposed = (prior_hypothesis if selection.hypothesis_transition.kind == "KEEP_CURRENT"
                                        else IncidentHypothesis(claim=selection.current_hypothesis, evidence_gap=""))
                            def stage_planner(resolved):
                                for acknowledgement in selection.operational_acknowledgements:
                                    observation = observation_store.get(acknowledgement.observation_ref)
                                    if (observation is None or not observation.ok
                                            or not interpretation_view.is_observation_visible(observation.observation_id)):
                                        raise SemanticTransitionError("operational_acknowledgement_invalid")
                                    trace.record("OBSERVATION_OPERATIONALLY_ACKNOWLEDGED", acknowledgement.model_dump())
                                staged_selection = replace(selection, hypothesis_id=resolved.hypothesis_id,
                                                           revision=resolved.revision)
                                # CURRENT evidence stays scoped to the previous revision.
                                for proposal_index, proposal in enumerate(selection.evidence_proposals):
                                    if proposal.hypothesis_target == "CURRENT":
                                        if prior_hypothesis is None:
                                            raise SemanticTransitionError("current_hypothesis_missing")
                                        admit_planner_evidence(proposal, turn_input=turn_input, proposal_index=proposal_index, hypothesis=prior_hypothesis)
                                for item in canonical_rca_state.obligations:
                                    if selection.hypothesis_transition.kind != "KEEP_CURRENT" and item.status == "SATISFIED":
                                        canonical_rca_state_reset(item.id)
                                build_hypothesis(staged_selection)
                                proposed_hypothesis = active_hypothesis()
                                for proposal_index, proposal in enumerate(selection.evidence_proposals):
                                    if proposal.hypothesis_target == "PROPOSED":
                                        admit_planner_evidence(proposal, turn_input=turn_input, proposal_index=proposal_index, hypothesis=proposed_hypothesis)
                                # New revision qualification never borrows previous coverage.
                                for obligation in canonical_rca_state.obligations:
                                    qualify_current_evidence_set(obligation, proposed_hypothesis,
                                                                 reason="semantic_transition_qualification")
                                record_semantic_decision(
                                    selection, prior_hypothesis, proposed_hypothesis,
                                )
                            try:
                                semantic_transition(selection.hypothesis_transition, proposed, stage_planner)
                            except (SemanticTransitionError, RCAStateError, RuntimeError, ValueError) as exc:
                                error = PlannerContractError(str(exc), validation_errors=[str(exc)])
                                error.metadata["error_type"] = getattr(exc, "code", "semantic_transition_invalid")
                                details = getattr(exc, 'details', {})
                                if isinstance(details, dict) and details.get('capability_gaps'):
                                    error.metadata['capability_feedback'] = {
                                        'required_capabilities': details['capability_gaps'],
                                        'capability_states': details['capability_states'],
                                    }
                                index = details.get('proposal_index') if isinstance(details, dict) else None
                                if isinstance(index, int) and 0 <= index < len(selection.evidence_proposals):
                                    proposal = selection.evidence_proposals[index]
                                    projection = turn_input.semantic_projection
                                    refs = [row['requirement_ref'] for row in requirement_table(projection)
                                            if row['obligation_id'] == proposal.obligation_id
                                            and row['slot'] in proposal.covered_fact_slots]
                                    error.metadata['requirement_feedback'] = targeting_feedback(projection, index,
                                        proposal.obligation_id, refs, proposal.hypothesis_target)
                                    error.metadata['requirement_feedback']['rejected_refs'] = refs
                                    error.metadata['requirement_feedback']['reason_code'] = error.metadata['error_type']
                                trace.record("SEMANTIC_TRANSITION_REJECTED", {"reason_code": error.metadata["error_type"],
                                                                            "reason": str(exc), "details": getattr(exc, "details", {})})
                                raise error from exc
                            committed_hypothesis = active_hypothesis()
                            bootstrap_pending_observation_ids = ()
                            evidence_admission_rejections = 0
                            permission = finalization_decision()
                            if permission.allowed and evidence_sufficient_step is None:
                                evidence_sufficient_step = state.step + 1
                                evidence_sufficient_tool_calls = state.tool_calls
                                trace.record("EVIDENCE_SUFFICIENT", {
                                    "step": state.step + 1, "evidence_sufficient": True,
                                    "hypothesis": committed_hypothesis.model_dump(),
                                })
                        def authorize_contract_retry(exc: BaseException) -> None:
                            nonlocal contract_retry_for_step, planner_contract_retry_used
                            contract_retry_for_step = True
                            planner_contract_retry_used = True
                            planner_metadata = _safe_planner_metadata_for_trace(exc)
                            trace.record("PLANNER_CONTRACT_RETRY", {
                                "step": state.step + 1,
                                "retry_index": 1,
                                "reason": "core structured Planner contract failure",
                                "planner_metadata": planner_metadata or {},
                            })
                            admit(
                                "planner", count_llm=True,
                                allow_targeted_recovery=True,
                            )

                        def resolve_local_context(batch, previous_input, round_index):
                            nonlocal interpretation_view, context_result, turn_input
                            # Publish physical usage before admitting another model request.
                            add_usage(planner, stage='planner', prompt_breakdown=planner.last_prompt_breakdown)
                            before_version = canonical_rca_state.event_count
                            context_result = context_manager.resolve_context_requests(
                                batch, view=previous_input.context_view, state=state,
                                memory=evidence_memory, observation_store=observation_store,
                                build_kwargs={**context_build_kwargs, 'request_renderer': lambda view:
                                    planner.render_interpretation_envelope(planner.with_context_view(previous_input, view),
                                        terminal_projection_allowed=planner_kwargs.get('terminal_projection_allowed'),
                                        context_resolution_remaining=PlanningSession.CONTEXT_ROUND_LIMIT - round_index)})
                            interpretation_view = context_result.context_view
                            turn_input = planner.with_context_view(previous_input, interpretation_view)
                            if canonical_rca_state.event_count != before_version:
                                raise RuntimeError('local context resolution cannot commit semantics')
                            trace.record('LOCAL_CONTEXT_RESOLVED', {
                                'turn_id': previous_input.turn_id, 'context_resolution_round': round_index,
                                'state_version': before_version, 'requests': batch.model_dump(mode='json'),
                                'visible_source_ranges': interpretation_view.visible_source_ranges,
                                'provider_calls': 0, 'tool_calls': 0, 'semantic_commits': 0})
                            admit('planner', count_llm=True)
                            return turn_input

                        def local_context_feedback(exc, round_index):
                            trace.record('LOCAL_CONTEXT_RESOLUTION_FEEDBACK', {
                                'context_resolution_round': round_index, **exc.feedback,
                                'tool_calls': 0, 'semantic_commits': 0})
                            admit('planner', count_llm=True)

                        retry_budget = LogicalTurnRetryBudget()
                        logical_record_cursor = len(provider_gateway.records)
                        interpretation_result = planning_session.run(
                            turn_input, logical_timeout_seconds=planner_timeout,
                            prompt_budget=dynamic_budget, max_output_tokens=planner_kwargs.get("max_output_tokens"),
                            terminal_projection_allowed=planner_kwargs.get("terminal_projection_allowed"),
                            consolidation_allowed=consolidation, budget_status_provider=request_budget_status,
                            on_contract_retry=authorize_contract_retry, retry_budget=retry_budget,
                            on_interpretation_commit=commit_interpretation,
                            on_context_request=resolve_local_context,
                            on_context_feedback=local_context_feedback,
                        )
                        interpretation_snapshots = interpretation_result.request_snapshots
                        # Publish the sole reasoning request usage before deterministic Phase B.
                        add_usage(planner, stage="planner", prompt_breakdown=planner.last_prompt_breakdown)
                        selected = interpretation_result.skill_selections[0]
                        previous_skill = active_skill
                        active_skill = selected.skill
                        load_domain_skill(active_skill)
                        if previous_skill != active_skill:
                            trace.record("SKILL_TRANSITION", {"step": state.step + 1, "from": previous_skill or None,
                                "to": active_skill, "reason": selected.reason})
                        permission = finalization_decision()
                        if permission.allowed:
                            result = replace(interpretation_result, planner_result=replace(
                                interpretation_result.planner_result, plan=EnrichedVerificationPlan(tasks=()),
                                continuation=PlanAdmissionDecision("finalize", "canonical_finalization_allowed",
                                    committed_state_version=canonical_rca_state.event_count)))
                        elif consolidation:
                            if interpretation_result.next_action_intent is not None:
                                raise RuntimeError("consolidation cannot admit a Tool intent")
                            trace.record("BUDGET_CONSOLIDATION_BLOCKED", {
                                "step": state.step + 1,
                                "reasons": [item.model_dump(mode="json") for item in permission.reasons]})
                            result = replace(interpretation_result, planner_result=replace(
                                interpretation_result.planner_result, plan=EnrichedVerificationPlan(tasks=()),
                                continuation=PlanAdmissionDecision("replan", "budget_consolidation",
                                    committed_state_version=canonical_rca_state.event_count)))
                        elif (selected.hypothesis_transition.kind == "INITIAL" and canonical_rca_state.obligations
                              and interpretation_view.visible_observation_ids):
                            bootstrap_pending_observation_ids = interpretation_view.visible_observation_ids
                            trace.record("BOOTSTRAP_INTERPRETATION_PENDING", {
                                "observation_ids": list(interpretation_view.visible_observation_ids),
                                "state_version": canonical_rca_state.event_count, "tool_calls": 0})
                            result = replace(interpretation_result, planner_result=replace(
                                interpretation_result.planner_result, plan=EnrichedVerificationPlan(tasks=()),
                                continuation=PlanAdmissionDecision("replan", "bootstrap_retained_observation_interpretation",
                                    committed_state_version=canonical_rca_state.event_count)))
                        else:
                            local_available = not local_recovery_offered and context_manager.has_local_recovery(
                                interpretation_view, observation_store)
                            external_probe = planner.create_turn_input(
                                state, '', visible_tool_names=visible_tool_names(), active_skill_id=active_skill,
                                feasible_skill_ids=feasible_skill_ids(),
                                skill_tool_names_by_id={skill_id: skill_tool_surface(skill_id) for skill_id in feasible_skill_ids()},
                                semantic_projection=semantic_input_projection()) if local_available else None
                            if local_available and not any(row['available_choice_refs'] for row in
                                    external_probe.semantic_projection.get('gap_choice_surface', ())):
                                local_recovery_offered = True
                                trace.record('LOCAL_CONTEXT_RECOVERY_AVAILABLE', {'state_version': canonical_rca_state.event_count,
                                    'reason_code': 'local_context_recovery_available'})
                                result = replace(interpretation_result, planner_result=replace(
                                    interpretation_result.planner_result, plan=EnrichedVerificationPlan(tasks=()),
                                    continuation=PlanAdmissionDecision('replan', 'local_context_recovery_available',
                                        committed_state_version=canonical_rca_state.event_count)))
                            else:
                                # A closed Tool gap can still admit semantic progress.
                                # ContinuationPolicy admits the empty plan; ProgressPolicy
                                # bounds actual noops instead of forcing Reflection here.
                                def committed_planning_input():
                                    return planner.create_turn_input(
                                        state, "", visible_tool_names=visible_tool_names(),
                                        active_skill_id=active_skill, feasible_skill_ids=feasible_skill_ids(),
                                        skill_tool_names_by_id={skill_id: skill_tool_surface(skill_id) for skill_id in feasible_skill_ids()},
                                        context_snapshot=planner_kwargs["context_snapshot"], semantic_projection=semantic_input_projection(),
                                        budget_projection=budget_payload(budget_snapshot("action_planning_start")),
                                        action_ledger=tuple(actions))
                                turn_input = committed_planning_input()
                                committed_hypothesis = active_hypothesis()
                                selected = replace(selected, skill=active_skill)
                                turn_input = replace(turn_input, turn_id=interpretation_result.turn_input.turn_id)
                                generated_version = canonical_rca_state.event_count
                                trace.record("PLANNER_GAP_SURFACE", {"step": state.step + 1, "turn_id": turn_input.turn_id,
                                    "visible_choice_count": len(turn_input.action_choices),
                                    "gap_choices": turn_input.semantic_projection.get("gap_choice_surface", []),
                                    "acquisitions": turn_input.semantic_projection.get("acquisitions", [])})
                                compilation_started = time.monotonic()
                                compilation_failure = None
                                try:
                                    compiled = planner.compile_next_action(turn_input,
                                        interpretation_result.next_action_intent, interpretation_result.response)
                                except PlannerContractError as exc:
                                    intent = interpretation_result.next_action_intent
                                    compilation_failure = VerificationPlanError(
                                        exc.metadata.get('error_type') or 'plan_compilation_rejected', detail=json.dumps({
                                            'error': str(exc), 'operation': intent.operation if intent else None,
                                            'target_obligation_id': intent.obligation_id if intent else None,
                                            'target_fact_slot': intent.slot if intent else None,
                                            'validation_details': exc.metadata.get('validation_details', [])}, sort_keys=True))
                                    compiled = replace(interpretation_result.planner_result,
                                        plan=EnrichedVerificationPlan(tasks=()))
                                trace.record("NEXT_ACTION_INTENT_COMPILED", {
                                    "turn_id": turn_input.turn_id, "state_version": generated_version,
                                    "operation": interpretation_result.next_action_intent.operation
                                        if interpretation_result.next_action_intent else None,
                                    "task_count": len(compiled.plan.tasks), "llm_calls": 0,
                                    "latency_seconds": time.monotonic() - compilation_started})
                                planned = replace(interpretation_result, planner_result=compiled, turn_input=turn_input)
                                planning_session.last_request_snapshots = interpretation_snapshots
                                # Re-read Runtime owners; do not accept model-copied snapshot metadata.
                                current_input = planner.create_turn_input(state, "", visible_tool_names=visible_tool_names(),
                                    active_skill_id=active_skill, feasible_skill_ids=feasible_skill_ids(),
                                    skill_tool_names_by_id={skill_id: skill_tool_surface(skill_id) for skill_id in feasible_skill_ids()},
                                    context_snapshot=planner_kwargs["context_snapshot"], semantic_projection=semantic_input_projection(),
                                    budget_projection=json.loads(turn_input.budget_projection_json), action_ledger=tuple(actions))
                                continuation = ContinuationPolicy.admit_plan(canonical_rca_state, planned.plan,
                                    state_snapshot_hash=current_input.state_snapshot_hash,
                                    generated_state_version=generated_version, generated_snapshot_hash=turn_input.state_snapshot_hash,
                                    schema_sha256_by_operation={spec.name: registry_schema_sha256(spec) for spec in tools.specs()},
                                    registry=tools, acquisition_projection=current_input.semantic_projection,
                                    compilation_failure=compilation_failure)
                                records = tuple(provider_gateway.records[logical_record_cursor:])
                                result = replace(planned, planner_result=replace(planned.planner_result,
                                    skill_selections=(selected,), continuation=continuation),
                                    request_snapshots=planning_session.last_request_snapshots, provider_calls=records,
                                    usage=tuple(sorted(aggregate_provider_usage(records).items())), contract_retry_count=retry_budget.used,
                                    context_resolution_rounds=interpretation_result.context_resolution_rounds)
                        trace.record("PLAN_ADMISSION_DECISION", {"step": state.step + 1,
                            "action": result.continuation.action, "reason_code": result.continuation.reason,
                            "detail": result.continuation.detail,
                            "committed_state_version": canonical_rca_state.event_count,
                            "planning_snapshot_hash": turn_input.state_snapshot_hash,
                            "task_count": len(result.plan.tasks), "target_obligation_ids":
                            [task.target_obligation_id for task in result.continuation.admitted_plan.tasks]
                            if result.continuation.admitted_plan else []})
                        record_planner_request_snapshot("completed")
                        break
                    except RunBudgetAdmissionExceeded as exc:
                        llm_calls = max(0, llm_calls - 1)
                        planner_calls = max(0, planner_calls - 1)
                        dynamic_budget.set_run_usage(
                            tokens_used=prompt_tokens + completion_tokens,
                            llm_calls_used=llm_calls,
                            cost_used=run_cost,
                        )
                        trace.record("BUDGET_ADMISSION_REJECTED", {
                            "stage": "planner",
                            "step": state.step + 1,
                            "failure_category": "run_budget_admission_exceeded",
                            "reason": exc.decision.reason,
                            "budget_before_call": exc.decision.remaining_run_tokens,
                            "estimated_call_cost": exc.decision.breakdown,
                            "budget_after_call": max(
                                0,
                                exc.decision.remaining_run_tokens
                                - exc.decision.breakdown.get("required_run_tokens", 0),
                            ),
                        })
                        record_budget_rejection(
                            "planner",
                            exc.decision.reason or "run_budget_admission_exceeded",
                        )
                        permission = finalization_decision()
                        salvage = ProgressPolicy.evaluate(operational_facts(
                            "salvage", stage="planner_admission", finalization=permission,
                        ))
                        if salvage.action == "stop":
                            raise
                        termination_reason = "pre_call_budget_admission_sufficient_hypothesis"
                        assert permission.candidate is not None
                        ensure_review_progress(permission.candidate, active_hypothesis())
                        return finalize_candidate(termination_reason)
                    except PromptBudgetExceeded as exc:
                        if committed_hypothesis is not None:
                            record_planner_request_snapshot("planning_budget_rejected")
                            raise
                        # Admission happens before the agent can render its
                        # complete provider payload. This was a local
                        # compaction attempt, not an LLM call.
                        llm_calls = max(0, llm_calls - 1)
                        planner_calls = max(0, planner_calls - 1)
                        prompt_compaction_attempts += 1
                        decision = exc.decision
                        trace.record("CONTEXT_PRESSURE", {
                            "stage": "planner",
                            "failure_category": "prompt_capacity_exceeded",
                            "context_state": decision.state.value,
                            "reason": decision.reason,
                            "estimated_prompt_tokens": decision.estimated_prompt_tokens,
                            "input_hard_capacity": decision.input_hard_capacity,
                            "breakdown": decision.breakdown,
                            "compaction_applied": True,
                        })
                        record_budget_rejection(
                            "planner",
                            decision.reason or "prompt_capacity_exceeded",
                        )
                        if ProgressPolicy.evaluate(operational_facts(
                            "recovery", stage="prompt_compaction", recovery_count=prompt_compaction_attempts - 1,
                            recovery_limit=2,
                        )).action == "stop":
                            raise RuntimeError(
                                "hard prompt capacity remains exceeded after bounded context compaction"
                            ) from exc
                        prompt_budget_retry = True
                        break
                    except LLMDeadlineExceeded as exc:
                        record_planner_request_snapshot("timeout")
                        trace.record("PLANNER_CALL_TIMEOUT", {
                            "step": state.step + 1,
                            "error_type": type(exc).__name__, "message": str(exc),
                        })
                        permission = finalization_decision()
                        salvage = ProgressPolicy.evaluate(operational_facts(
                            "salvage", stage="planner_timeout", finalization=permission,
                        ))
                        if salvage.action == "stop":
                            raise
                        final_candidate_step = state.step
                        termination_reason = "planner_timeout_sufficient_hypothesis"
                        trace.record("TIMEOUT_FINALIZATION_FALLBACK", {
                            "error_type": type(exc).__name__, "message": str(exc),
                            "hypothesis": (
                                active_hypothesis().model_dump()
                                if active_hypothesis() is not None else None
                            ),
                        })
                        assert permission.candidate is not None
                        ensure_review_progress(permission.candidate, active_hypothesis())
                        return finalize_candidate(
                            "sufficient_hypothesis_timeout_fallback"
                        )
                    except PlannerContractError as exc:
                        record_planner_request_snapshot("contract_error")
                        planner_failure = {
                            "step": state.step + 1,
                            "error_type": type(exc).__name__, "message": str(exc),
                        }
                        planner_metadata = _safe_planner_metadata_for_trace(exc)
                        if planner_metadata is not None:
                            planner_failure["planner_metadata"] = planner_metadata
                        trace.record("PLANNER_CALL_FAILED", planner_failure)
                        if isinstance(exc, PlannerContractExhausted):
                            trace.record("PLANNER_CONTRACT_EXHAUSTED", {
                                "step": state.step + 1,
                                "retry_count": 1,
                                "planner_metadata": planner_metadata or {},
                            })
                            trace.record("REPAIR_FAILED", {
                                "layer": "planner_contract",
                                "logical_turn": state.step + 1,
                                "repair_attempts": 1,
                                "reason": "bounded_planner_contract_retry_exhausted",
                            })
                            raise
                        raise
                    except PromptBudgetExceeded as exc:
                        record_planner_request_snapshot("prompt_budget_error")
                        planner_failure = {
                            "step": state.step + 1,
                            "error_type": type(exc).__name__, "message": str(exc),
                        }
                        planner_metadata = _safe_planner_metadata_for_trace(exc)
                        if planner_metadata is not None:
                            planner_failure["planner_metadata"] = planner_metadata
                        trace.record("PLANNER_CALL_FAILED", planner_failure)
                        raise
                    except Exception as exc:
                        record_planner_request_snapshot("runtime_error")
                        planner_failure = {
                            "step": state.step + 1,
                            "error_type": type(exc).__name__, "message": str(exc),
                        }
                        trace.record("PLANNER_CALL_FAILED", planner_failure)
                        raise
                    finally:
                        add_usage(
                            planner, stage="planner",
                            prompt_breakdown=getattr(planner, "last_prompt_breakdown", None),
                        )
                if prompt_budget_retry:
                    # Re-enter the outer loop so ContextManager can compact
                    # L2/L3 while preserving Critical State and Evidence.
                    continue
                dynamic_budget.clear_context_state_override()
                trace.record("PLANNER_CALL_COMPLETED", {
                    "step": state.step + 1,
                    "prompt_breakdown": dict(planner.last_prompt_breakdown),
                    "prompt_breakdowns": list(getattr(planner, "last_prompt_breakdowns", ()) or ()),
                    "usage": result.usage_dict,
                    "physical_request_count": result.physical_request_count,
                    "provider_attempt_count": result.provider_attempt_count,
                    "contract_retry_count": result.contract_retry_count,
                    "context_resolution_rounds": result.context_resolution_rounds,
                })
                trace.record("PLANNER_RESPONSE_AUDIT", {
                    "step": state.step + 1,
                    "raw_output": dict(getattr(result.response, "raw_output", {}) or {}),
                    "parsed_output": result.response.structured,
                    "normalized_tool_count": len(result.tool_calls),
                    "tool_names": [call.name for call in result.tool_calls],
                })
                state.step += 1
                trace.record("AGENT_STEP", {
                    "step": state.step, "tool_call_count": len(result.tool_calls),
                    "planned_task_count": len(result.plan.tasks),
                })
                for audit in getattr(result, "tool_call_audits", ()):
                    trace.record("PROVIDER_TOOL_CALL_AUDIT", {
                        "step": state.step,
                        "provider": model_capability.provider or type(self.llm).__name__,
                        "model": model_capability.model or self.model,
                        **dict(audit),
                    })
                if planner_phase in {"converge", "source_verify"}:
                    phase_reasoning_limit = (
                        self.config.converge_max_reasoning_tokens
                        if planner_phase == "converge" else self.config.source_verify_max_output_tokens
                    )
                    phase_output_limit = (
                        self.config.converge_max_output_tokens
                        if planner_phase == "converge" else self.config.source_verify_max_output_tokens
                    )
                    usage = dict(result.response.usage or {})
                    completion_details = usage.get("completion_tokens_details") or {}
                    reasoning_tokens = completion_details.get("reasoning_tokens")
                    if reasoning_tokens is not None and int(reasoning_tokens) > phase_reasoning_limit:
                        trace.record("CONVERGE_REASONING_LIMIT_EXCEEDED", {
                            "step": state.step,
                            "reasoning_tokens": int(reasoning_tokens),
                            "max_reasoning_tokens": phase_reasoning_limit,
                            "provider_completion_cap_tokens": phase_output_limit,
                            "tool_call_count": len(result.tool_calls),
                            "reason": "provider usage exceeded Runtime convergence reasoning limit",
                        })
                        raise ConvergenceBudgetExhausted(
                            "convergence planner reasoning limit exceeded"
                        )
                if evidence_admission_rejections:
                    trigger_reflection(admission_rejection_count=evidence_admission_rejections)
                continuation = result.continuation
                if not isinstance(continuation, PlanAdmissionDecision):
                    raise RuntimeError("Planner result requires typed post-commit plan admission")
                if continuation.action == "finalize":
                    termination_reason = "post_commit_finalization"
                    return finalize_candidate(termination_reason)
                semantic_progress = progress_made(
                    progress_before_action, progress_marker(),
                )
                if continuation.reason == "plan_acquisition_already_acquired":
                    duplicate_calls += 1
                    state.repeated_actions += 1
                    trace.record("ACTION_REJECTED", {"step": state.step,
                        "reason": continuation.reason, "detail": continuation.detail})
                acquired_before = action_runtime.acquired_keys()
                if continuation.action == "execute":
                    execute_plan(result, turn_input, committed_hypothesis)
                progress_candidate = observe_action_progress(
                    canonical_progress=semantic_progress,
                    acquisition_progress=bool(action_runtime.acquired_keys() - acquired_before),
                )
                if progress_candidate is not None:
                    return progress_candidate
                if continuation.action != "execute":
                    trace.record("SEMANTIC_BARRIER", {
                        "step": state.step, "reason_code": continuation.reason,
                        "committed_state_version": continuation.committed_state_version,
                        "observation_ids": [],
                    })
                    runtime_action_constraint = json.dumps({
                        "reason_code": continuation.reason,
                        "admission_detail": continuation.detail,
                        "open_obligation_ids": list(canonical_rca_state.open_critical_obligation_ids),
                    }, sort_keys=True)
                    transition_phase("PLAN", continuation.reason)
                    continue

        def apply_review_rejection(
            decision: ReviewDecision,
            rejected_candidate: RootCauseCandidate,
        ) -> tuple[ProgressDecision, bool]:
            """Admit typed Review gaps through the common Obligation policy."""
            nonlocal canonical_rca_state, obligation_created_count
            current = active_hypothesis()
            causal_gap_audit = tuple(dict.fromkeys(
                " ".join(str(item).split()).strip()
                for item in decision.causal_gaps
                if " ".join(str(item).split()).strip()
            ))
            created: list[str] = []
            actionable: list[str] = []
            rejected: list[dict[str, Any]] = []
            evidence_tools = tuple(
                name for name in capabilities.planner_tool_names(tools)
                if name not in {"finalize_diagnosis", "knowledge_retrieval"}
            )
            for proposal in decision.gap_proposals:
                admission = ObligationAdmissionPolicy.admit_review_gap(
                    proposal,
                    provenance_profiles=completion_provenance_profiles(),
                    candidate=rejected_candidate,
                    reported_unsupported_claims=decision.unsupported_claims,
                    related_hypothesis=(current.claim if current else ""),
                    existing_obligations=canonical_rca_state.obligations,
                    available_tools=evidence_tools,
                )
                trace.record("REVIEW_GAP_ADMISSION", {
                    "round": review_rounds,
                    "status": admission.status,
                    "reasons": list(admission.reasons),
                    "obligation_id": admission.obligation_id,
                    "unsupported_claim": proposal.unsupported_claim,
                    "claim_kind": proposal.claim_kind,
                    "suggested_tools": list(proposal.suggested_tools),
                })
                if admission.status == "REJECTED":
                    rejected.append({
                        "claim": proposal.claim,
                        "reasons": list(admission.reasons),
                    })
                    continue
                if not admission.actionable:
                    continue
                actionable.append(admission.obligation_id)
                if admission.status == "DUPLICATE":
                    continue
                canonical_rca_state = canonical_rca_state.reduce(admission.accepted_event())
                created.append(admission.obligation_id)
                obligation_created_count += 1
                obligation = canonical_rca_state.obligation(admission.obligation_id)
                trace.record("OBLIGATION_CREATED", {
                    **obligation.model_dump(),
                    "reason": "review_gap_proposal_admitted",
                    "review_round": review_rounds,
                    "source": "rca_state_reducer",
                })

            if created:
                project_obligation_ids_to_dag("review_recovery")
                trace.record("REVIEW_RECOVERY_OBLIGATIONS", {
                    "round": review_rounds,
                    "obligation_ids": created,
                    "reason": "typed_review_gap_admitted",
                    "causal_gaps_audit_only": list(causal_gap_audit),
                    "rejected_proposals": rejected,
                })

            blocking = tuple(dict.fromkeys(
                (*decision.blocking_contradictions, *decision.contradictions)
            ))
            contradiction_requires_reflection = bool(blocking)
            recovery = ProgressPolicy.evaluate(operational_facts(
                "review_recovery",
                review_recovery_permitted=decision.recoverability.unrecoverable is False,
                actionable_obligation_ids=tuple(actionable),
                review_contradiction_ids=blocking,
            ))
            trace.record("REVIEW_RECOVERY_DECISION", {
                "round": review_rounds,
                "recoverable": recovery.action == "retry",
                "progress_action": recovery.action,
                "progress_reasons": list(recovery.reasons),
                "unrecoverable": decision.recoverability.unrecoverable,
                "gap_proposal_count": len(decision.gap_proposals),
                "actionable_gap_count": len(actionable),
                "rejected_gap_count": len(rejected),
                "missing_evidence_audit_count": len(decision.missing_evidence),
                "causal_gap_audit_count": len(causal_gap_audit),
                "blocking_contradiction_count": len(blocking),
                "targeted_followup": decision.targeted_followup or decision.suggested_investigation,
            })
            return recovery, contradiction_requires_reflection

        def review_inconclusive_report() -> None:
            """Give an INCONCLUSIVE report the same independent Review gate.

            This path is deliberately report-only: Review may assess whether
            the partial diagnosis is traceable and score its likely cause, but
            it cannot manufacture a Candidate or promote the run to PASS.
            Operational/provider failures remain explicitly UNAVAILABLE.
            """
            nonlocal decision, review_rounds, review_record_cursor
            nonlocal llm_calls, review_calls
            if not self.config.enable_review or status != "INCONCLUSIVE":
                return
            # A hard deadline or physical call exhaustion cannot be repaired by
            # another provider request.  The report still records Review as
            # unavailable through the normal terminal projection.
            if termination_reason in {
                "physical_call_budget_exhausted",
                "run_deadline_after_review",
                "provider_timeout_after_review",
            }:
                return
            if (
                termination_reason in {"planner_contract_exhausted", "reflection_contract_exhausted"}
                and not canonical_rca_state.semantic_records
                and not evidence_memory.pinned
            ):
                # A pure protocol/context failure has no diagnostic material
                # for an independent report assessment; preserve its zero
                # semantic/external-effect boundary and mark Review unavailable.
                return
            review_rounds += 1
            transition_phase("REVIEW", "inconclusive_report")
            review_record_cursor = len(provider_gateway.records)
            review_usage_accounted = False
            admitted = False
            started_at = time.monotonic()
            configured_timeout = float(self.config.review_llm_timeout_seconds)
            try:
                admit("review", count_llm=True)
                admitted = True
                review_timeout = provider_timeout(self.config.review_llm_timeout_seconds)
                trace.record("REVIEW_CALL_STARTED", {
                    "round": review_rounds,
                    "stage": "inconclusive_report",
                    "configured_timeout": configured_timeout,
                    "effective_timeout": review_timeout,
                    "remaining_run_deadline": deadline.remaining(),
                    "report_only": True,
                })
                current_hypothesis = active_hypothesis()
                blocking_contradictions = tuple(
                    item.model_dump()
                    for item in canonical_rca_state.contradictions
                    if item.blocks_finalization
                )
                source_context = self._source_mechanism_coverage(
                    case,
                    current_hypothesis,
                    source_workspace_available=source_workspace_available,
                    available_code_tools=available_code_tools,
                    source_evidence_ids=source_evidence_ids(),
                    source_relevance=evaluate_source_claims(
                        current_hypothesis.source_claims if current_hypothesis else ()
                    ),
                )
                source_context["finalization_conditions"] = (
                    finalization_decision().conditions.model_dump(mode="json")
                )
                review_evidence = incident_evidence()
                frozen_report = build_diagnostic_report(
                    case=case,
                    run_status="INCONCLUSIVE",
                    termination_reason=termination_reason,
                    candidate=None,
                    rca_state=canonical_rca_state,
                    evidence_memory=evidence_memory,
                    projected_facts=projected_facts.values(),
                )
                from debug_assistant.tools.provenance import (
                    provenance_profile_catalog,
                    registry_provenance_profiles,
                )
                review_input = dict(
                    canonical_questions=tuple(
                        question.model_dump(mode="json")
                        for question in canonical_rca_state.obligations
                    ),
                    blocking_contradictions=blocking_contradictions,
                    source_mechanism_context=source_context,
                    open_critical_obligations=tuple(
                        item.model_dump()
                        for item in canonical_rca_state.obligations
                        if item.blocks_finalization
                    ),
                    source_profiles=provenance_profile_catalog(
                        registry_provenance_profiles(
                            tools, capabilities.planner_tool_names(tools)
                        )
                    ),
                    evidence_acquisition_tools=tuple(
                        name for name in capabilities.planner_tool_names(tools)
                        if name not in {"finalize_diagnosis", "knowledge_retrieval"}
                    ),
                    diagnostic_report=frozen_report.model_copy(
                        update={"evidence_links": ()}
                    ),
                )
                requested_ids = tuple(dict.fromkeys(
                    item.evidence_id for item in review_evidence
                ))
                review_context = context_manager.build(
                    state,
                    evidence_memory,
                    observation_store,
                    max_context_chars=self.config.planner_context_chars,
                    max_context_tokens=dynamic_budget.context_token_budget(),
                    token_estimator=dynamic_budget.estimate_tokens,
                    include_agent_control_state=False,
                    rca_state=canonical_rca_state,
                    role="review",
                    max_request_chars=self.config.max_context_chars,
                    requested_ids=requested_ids,
                    request_renderer=lambda view: reviewer.render_envelope(
                        None, review_evidence, view, **review_input
                    ),
                )
                trace.record("REVIEW_CONTEXT_VIEW", {
                    "role": "review",
                    "report_only": True,
                    "visible_observation_ids": list(
                        review_context.context_view.visible_observation_ids
                    ),
                    "visible_evidence_refs": list(
                        review_context.context_view.visible_evidence_refs
                    ),
                    "body_coverage": json.loads(
                        review_context.context_view.body_coverage_json
                    ),
                })
                model_decision = reviewer.review(
                    None,
                    review_evidence,
                    **review_input,
                    context_view=review_context.context_view,
                    logical_timeout_seconds=review_timeout,
                    prompt_budget=dynamic_budget,
                    max_output_tokens=self.config.review_max_completion_tokens,
                    repair_max_output_tokens=self.config.review_repair_max_completion_tokens,
                    allow_terminal_reserve=False,
                )
                account_review_attempts(reviewer)
                add_usage(
                    reviewer,
                    stage="review",
                    prompt_breakdown=getattr(reviewer, "last_prompt_breakdown", None),
                )
                review_usage_accounted = True
                record_review_trace(
                    round_number=review_rounds,
                    configured_timeout=configured_timeout,
                    effective_timeout=review_timeout,
                    remaining_run_deadline=deadline.remaining(),
                    started_at=started_at,
                    reviewer=reviewer,
                )
                decision = enforce_review_consistency(model_decision)
                trace.record("REPORT_REVIEWED", {
                    "round": review_rounds,
                    "decision": decision.decision,
                    "confidence_score": decision.confidence_score,
                    "confidence_reason": decision.confidence_reason,
                    "report_only": True,
                    "run_status_remains": "INCONCLUSIVE",
                })
            except (RunBudgetAdmissionExceeded, PromptBudgetExceeded) as exc:
                if admitted:
                    llm_calls = max(0, llm_calls - 1)
                    review_calls = max(0, review_calls - 1)
                    dynamic_budget.set_review_calls_used(review_calls)
                    dynamic_budget.set_run_usage(
                        tokens_used=prompt_tokens + completion_tokens,
                        llm_calls_used=llm_calls,
                        cost_used=run_cost,
                    )
                trace.record("REPORT_REVIEW_UNAVAILABLE", {
                    "round": review_rounds,
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                    "report_only": True,
                })
            except Exception as exc:
                if admitted and not review_usage_accounted:
                    materialize_review_attempts(
                        reviewer,
                        max(1, int(getattr(reviewer, "last_call_count", 0) or 0)),
                        effective_timeout=max(0.0, deadline.remaining()),
                        started_at=started_at,
                        terminal_error=exc,
                    )
                    account_review_attempts(reviewer)
                    add_usage(
                        reviewer,
                        stage="review",
                        prompt_breakdown=getattr(reviewer, "last_prompt_breakdown", None),
                    )
                    review_usage_accounted = True
                trace.record("REPORT_REVIEW_UNAVAILABLE", {
                    "round": review_rounds,
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                    "report_only": True,
                })

        candidate = None
        decision = None
        report = None
        report_review = None
        status = "FAILED"
        error_type = error_message = ""
        try:
            candidate = investigate(self.config.max_steps, self.config.max_tool_calls)
            if candidate is None:
                status = "INCONCLUSIVE"
                termination_reason = termination_reason or "investigation_inconclusive"
            if not self.config.enable_review:
                # A candidate without the independent terminal critic is a
                # deliverable report, never a successful RCA result.
                status = "INCONCLUSIVE"
                termination_reason = "review_disabled"
            # A missing Candidate is a report-only terminal path.  It must not
            # enter the Candidate recovery loop or dereference candidate-owned
            # claims/evidence; the bounded report Review runs after this block.
            while self.config.enable_review and candidate is not None:
                if ProgressPolicy.evaluate(operational_facts(
                    "recovery", stage="review_round", recovery_count=review_rounds,
                    recovery_limit=self.config.max_review_rounds,
                )).action == "stop":
                    break
                review_rounds += 1
                transition_phase("REVIEW", "candidate_finalized", review_round=review_rounds)
                review_usage_accounted = False
                review_record_cursor = len(provider_gateway.records)
                try:
                    admit("review", count_llm=True)
                    review_started_at = time.monotonic()
                    review_attempt_state = {"count": 0}

                    def on_review_attempt_started(attempt: dict[str, Any]) -> None:
                        review_attempt_state["count"] = max(
                            review_attempt_state["count"],
                            int(attempt.get("call_index", 0) or 0),
                        )

                    configured_review_timeout = float(self.config.review_llm_timeout_seconds)
                    remaining_run_deadline = deadline.remaining()
                    review_timeout = provider_timeout(self.config.review_llm_timeout_seconds)
                    trace.record("REVIEW_CALL_STARTED", {
                        "round": review_rounds,
                        "stage": "first_pass",
                        "call_index": 1,
                        "schema_repair": False,
                        "configured_timeout": configured_review_timeout,
                        "effective_timeout": review_timeout,
                        "remaining_run_deadline": remaining_run_deadline,
                    })
                    trace.record("REVIEW_RESERVE_APPLIED", {
                        "round": review_rounds,
                        "reserve_tokens": self.config.review_reserve_tokens,
                        "reserve_calls": self.config.review_reserve_calls,
                        "call_index": review_calls,
                        "policy": "review_stage_owns_terminal_reserve",
                    })
                    blocking_contradictions = tuple(
                        item.model_dump()
                        for item in canonical_rca_state.contradictions
                        if item.blocks_finalization
                    )
                    current_hypothesis = active_hypothesis()
                    source_context = self._source_mechanism_coverage(
                        case,
                        current_hypothesis,
                        source_workspace_available=source_workspace_available,
                        available_code_tools=available_code_tools,
                        source_evidence_ids=source_evidence_ids(),
                        source_relevance=evaluate_source_claims(
                            current_hypothesis.source_claims if current_hypothesis else ()
                        ),
                    )
                    # Capability/status projections are context, not a second
                    # source gate. Review sees the canonical publication facts.
                    source_context["finalization_conditions"] = (
                        finalization_decision().conditions.model_dump(mode="json")
                    )
                    open_critical_obligations = tuple(
                        item.model_dump()
                        for item in (
                            canonical_rca_state.obligations
                            if canonical_rca_state is not None else ()
                        )
                        if item.blocks_finalization
                    )
                    from debug_assistant.tools.provenance import provenance_profile_catalog, registry_provenance_profiles
                    review_evidence = incident_evidence()
                    frozen_report = build_diagnostic_report(
                        case=case, run_status="PASS",
                        termination_reason=termination_reason,
                        candidate=candidate, rca_state=canonical_rca_state,
                        evidence_memory=evidence_memory,
                        projected_facts=projected_facts.values(),
                    )
                    review_input = dict(
                        canonical_questions=tuple(question.model_dump(mode="json")
                            for question in canonical_rca_state.obligations),
                        blocking_contradictions=blocking_contradictions,
                        source_mechanism_context=source_context,
                        open_critical_obligations=open_critical_obligations,
                        source_profiles=provenance_profile_catalog(registry_provenance_profiles(
                            tools, capabilities.planner_tool_names(tools))),
                        evidence_acquisition_tools=tuple(name for name in capabilities.planner_tool_names(tools)
                            if name not in {"finalize_diagnosis", "knowledge_retrieval"}),
                        # Review receives the same frozen report identity but
                        # obtains Evidence bodies through its own ContextView;
                        # do not duplicate every source excerpt in the prompt.
                        diagnostic_report=frozen_report.model_copy(update={"evidence_links": ()}),
                    )
                    review_context = context_manager.build(
                        state, evidence_memory, observation_store,
                        max_context_chars=self.config.planner_context_chars,
                        max_context_tokens=dynamic_budget.context_token_budget(),
                        token_estimator=dynamic_budget.estimate_tokens,
                        include_agent_control_state=False, rca_state=canonical_rca_state,
                        role="review", max_request_chars=self.config.max_context_chars,
                        requested_ids=candidate.evidence_ids,
                        request_renderer=lambda view: reviewer.render_envelope(
                            candidate, review_evidence, view, **review_input))
                    trace.record("REVIEW_CONTEXT_VIEW", {"role": "review",
                        "visible_observation_ids": list(review_context.context_view.visible_observation_ids),
                        "visible_evidence_refs": list(review_context.context_view.visible_evidence_refs),
                        "body_coverage": json.loads(review_context.context_view.body_coverage_json)})
                    try:
                        model_decision = reviewer.review(
                            candidate, review_evidence, **review_input,
                            context_view=review_context.context_view,
                            logical_timeout_seconds=review_timeout,
                            on_attempt_started=on_review_attempt_started,
                            prompt_budget=dynamic_budget,
                            max_output_tokens=self.config.review_max_completion_tokens,
                            repair_max_output_tokens=self.config.review_repair_max_completion_tokens,
                            # Review owns its protected allocation; the
                            # terminal safety reserve remains untouched.
                            allow_terminal_reserve=False,
                        )
                    except RunBudgetAdmissionExceeded as exc:
                        add_usage(reviewer, stage="review", call_status="failed")
                        # Review is the terminal semantic gate. A run-budget
                        # admission rejection must remain INCONCLUSIVE and
                        # must never be misreported as a Review PASS.
                        llm_calls = max(0, llm_calls - 1)
                        review_calls = max(0, review_calls - 1)
                        dynamic_budget.set_review_calls_used(review_calls)
                        dynamic_budget.set_run_usage(
                            tokens_used=prompt_tokens + completion_tokens,
                            llm_calls_used=llm_calls,
                            cost_used=run_cost,
                        )
                        review_usage_accounted = True
                        trace.record("BUDGET_ADMISSION_REJECTED", {
                            "stage": "review",
                            "round": review_rounds,
                            "reason": exc.decision.reason,
                            "budget_before_call": exc.decision.remaining_run_tokens,
                            "estimated_call_cost": exc.decision.breakdown,
                            "budget_after_call": max(
                                0,
                                exc.decision.remaining_run_tokens
                                - exc.decision.breakdown.get("required_run_tokens", 0),
                            ),
                        })
                        status = "INCONCLUSIVE"
                        termination_reason = "review_budget_admission_exceeded"
                        error_type, error_message = type(exc).__name__, str(exc)
                        break
                    except PromptBudgetExceeded as exc:
                        add_usage(reviewer, stage="review", call_status="failed")
                        # Review is the terminal semantic gate. If even its
                        # complete payload cannot fit, stop fail-closed as
                        # INCONCLUSIVE; never invoke the provider with an
                        # over-capacity prompt.
                        llm_calls = max(0, llm_calls - 1)
                        review_calls = max(0, review_calls - 1)
                        dynamic_budget.set_run_usage(
                            tokens_used=prompt_tokens + completion_tokens,
                            llm_calls_used=llm_calls,
                            cost_used=run_cost,
                        )
                        review_usage_accounted = True
                        trace.record("CONTEXT_PRESSURE", {
                            "stage": "review",
                            "context_state": exc.decision.state.value,
                            "reason": exc.decision.reason,
                            "estimated_prompt_tokens": exc.decision.estimated_prompt_tokens,
                            "input_hard_capacity": exc.decision.input_hard_capacity,
                            "breakdown": exc.decision.breakdown,
                            "compaction_applied": False,
                        })
                        trace.record("TERMINAL_RESERVE_PROTECTED", {
                            "stage": "review",
                            "reason": "final Review prompt cannot fit physical input capacity",
                        })
                        status = "INCONCLUSIVE"
                        termination_reason = "review_prompt_capacity_exceeded"
                        error_type, error_message = type(exc).__name__, str(exc)
                        break
                    except Exception as exc:
                        materialize_review_attempts(
                            reviewer, review_attempt_state["count"],
                            effective_timeout=review_timeout,
                            started_at=review_started_at,
                            terminal_error=exc,
                        )
                        account_review_attempts(reviewer)
                        add_usage(
                            reviewer, stage="review",
                            prompt_breakdown=getattr(reviewer, "last_prompt_breakdown", None),
                        )
                        review_usage_accounted = True
                        record_review_trace(
                            round_number=review_rounds,
                            configured_timeout=configured_review_timeout,
                            effective_timeout=review_timeout,
                            remaining_run_deadline=remaining_run_deadline,
                            started_at=review_started_at,
                            reviewer=reviewer,
                            terminal_error=exc,
                        )
                        raise
                    account_review_attempts(reviewer)
                    add_usage(
                        reviewer, stage="review",
                        prompt_breakdown=getattr(reviewer, "last_prompt_breakdown", None),
                    )
                    review_usage_accounted = True
                    record_review_trace(
                        round_number=review_rounds,
                        configured_timeout=configured_review_timeout,
                        effective_timeout=review_timeout,
                        remaining_run_deadline=remaining_run_deadline,
                        started_at=review_started_at,
                        reviewer=reviewer,
                    )
                    decision = enforce_review_consistency(model_decision)
                    if decision.decision != model_decision.decision:
                        trace.record("REVIEW_PASS_DOWNGRADED", {
                            "round": review_rounds,
                            "reason": "PASS response contained structured unsupported or missing evidence",
                            "unsupported_claims": list(model_decision.unsupported_claims),
                            "missing_evidence": list(model_decision.missing_evidence),
                            "contradictions": list(model_decision.contradictions),
                        })
                    if ProgressPolicy.evaluate(operational_facts("post_usage")).action == "stop":
                        raise RuntimeError("max_llm_calls exceeded during review repair")
                except ReviewSchemaError as exc:
                    if not review_usage_accounted:
                        account_review_attempts(reviewer)
                    add_usage(
                        reviewer, stage="review",
                        prompt_breakdown=getattr(reviewer, "last_prompt_breakdown", None),
                    )
                    trace.record("REVIEW_ERROR", {
                        "round": review_rounds, "error_type": type(exc).__name__,
                        "message": str(exc), "validation_error": reviewer.last_schema_error,
                    })
                    trace.record("REPAIR_FAILED", {
                        "layer": "review_contract",
                        "logical_turn": review_rounds,
                        "repair_attempts": 1,
                        "reason": "review_schema_repair_exhausted",
                    })
                    trace.record("PROVISIONAL_CANDIDATE", {
                        "candidate": candidate.model_dump() if candidate is not None else None,
                        "reason": "Final Review structured contract remained invalid after one bounded repair",
                        "review_rounds": review_rounds,
                        "hypothesis": (
                            active_hypothesis().model_dump()
                            if active_hypothesis() is not None else None
                        ),
                        "open_obligations": [
                            item.model_dump() for item in (
                                canonical_rca_state.obligations
                            ) if item.blocks_finalization
                        ],
                    })
                    trace.record("INCIDENT_INCONCLUSIVE", {
                        "error_type": type(exc).__name__,
                        "failure_category": "review_contract_exhausted",
                        "message": "Final Review response remained invalid after one schema repair attempt",
                        "review_rounds": review_rounds,
                        "step": state.step, "tool_calls": state.tool_calls,
                    })
                    status = "INCONCLUSIVE"
                    termination_reason = "review_contract_exhausted"
                    error_type = type(exc).__name__
                    error_message = "Final Review response remained invalid after one schema repair attempt"
                    break
                trace.record("FINAL_REVIEW", {
                    "round": review_rounds, "schema_repaired": reviewer.last_schema_repaired,
                    "schema_validation_error": (
                        reviewer.last_schema_error if reviewer.last_schema_repaired else ""
                    ),
                    **decision.model_dump(),
                })
                trace.record("CAUSAL_REVIEW", {
                    "round": review_rounds,
                    "decision": decision.decision,
                    "causal_chain_valid": decision.causal_chain_valid,
                    "causal_gaps": list(decision.causal_gaps),
                    "mapping_count": len(candidate.claim_evidence_mapping),
                    "causal_link_count": len(candidate.causal_chain_summary),
                })
                if decision.decision == "PASS":
                    first_pass_accepted = review_rounds == 1
                    status = "PASS"
                    termination_reason = "final_review_pass"
                    break
                recovery, contradiction_requires_reflection = apply_review_rejection(
                    decision, candidate,
                )
                trace.record("REVIEW_REJECT", {
                    "round": review_rounds,
                    "recoverable": recovery.action == "retry",
                    "contradiction_requires_reflection": contradiction_requires_reflection,
                })
                if recovery.action == "stop":
                    review_stop_reason = (
                        "review_unrecoverable" if decision.recoverability.unrecoverable
                        else "review_recovery_unavailable"
                    )
                    transition_phase("INCONCLUSIVE", review_stop_reason)
                    trace.record("REVIEW_RECOVERY_BLOCKED", {
                        "round": review_rounds,
                        "reason": review_stop_reason,
                        "progress_reasons": list(recovery.reasons),
                    })
                    status = "INCONCLUSIVE"
                    termination_reason = review_stop_reason
                    break
                if ProgressPolicy.evaluate(operational_facts(
                    "recovery", stage="review_round", recovery_count=review_rounds,
                    recovery_limit=self.config.max_review_rounds,
                )).action == "stop":
                    status = "INCONCLUSIVE"
                    termination_reason = "review_bound_exhausted"
                    break
                if ProgressPolicy.evaluate(operational_facts(
                    "recovery", stage="review_recovery", recovery_count=review_recovery_cycles,
                    recovery_limit=self.config.max_review_recovery_cycles,
                )).action == "stop":
                    transition_phase("INCONCLUSIVE", "review_recovery_bound_exhausted")
                    trace.record("REVIEW_RECOVERY_BLOCKED", {
                        "round": review_rounds,
                        "reason": "max_review_recovery_cycles",
                        "max_review_recovery_cycles": self.config.max_review_recovery_cycles,
                    })
                    status = "INCONCLUSIVE"
                    termination_reason = "review_recovery_bound_exhausted"
                    break
                review_recovery_cycles += 1
                trace.record("REVIEW_RECOVERY_CYCLE", {
                    "round": review_rounds,
                    "cycle": review_recovery_cycles,
                    "max_cycles": self.config.max_review_recovery_cycles,
                })
                dynamic_budget.begin_investigation_recovery(
                    tokens_used=prompt_tokens + completion_tokens,
                    llm_calls_used=llm_calls,
                )
                trace.record("BUDGET_RECOVERY_PHASE_STARTED", {
                    "from": "review",
                    "to": "investigation_recovery",
                    "reason": "review_reject_recoverable",
                    "round": review_rounds,
                    "cycle": review_recovery_cycles,
                    "budget_lifecycle": dynamic_budget.lifecycle_payload(
                        tokens_used=prompt_tokens + completion_tokens,
                        llm_calls_used=llm_calls,
                    ),
                })
                rejected_candidate = candidate
                rejected_evidence_ids = frozenset(
                    item.evidence_id for item in evidence_memory.pinned
                )
                rejected_hypothesis = active_hypothesis()
                review_feedback_json = json.dumps(decision.model_dump(), ensure_ascii=False)
                reflection_result = ""
                if contradiction_requires_reflection:
                    trace.record("REFLECTION_TRIGGER", {
                        "reason": "review_reasoning_contradiction", "round": review_rounds,
                    })
                    reflection_result = trigger_reflection(
                        review_recovery_required=True, review_feedback_text=review_feedback_json,
                    )
                else:
                    transition_phase("PLAN", "review_reject_recoverable")
                combined_feedback = json.dumps({
                    "review": decision.model_dump(),
                    "reflection": json.loads(reflection_result) if reflection_result else None,
                }, ensure_ascii=False)
                # Recovery turns can spend a bounded Planner turn rebinding a
                # Review-created Obligation before an executable Action is
                # admitted.  Reserve two such turns explicitly so the
                # configured post-reject action/tool bounds are not consumed
                # by contract-preserving replans.  This does not expand the
                # global token or LLM allocation.
                recovery_step_limit = self.config.max_post_reject_steps + 2
                trace.record("REVIEW_RECOVERY_BOUND_APPLIED", {
                    "round": review_rounds,
                    "cycle": review_recovery_cycles,
                    "configured_steps": self.config.max_post_reject_steps,
                    "effective_steps": recovery_step_limit,
                    "reason": "bounded obligation_rebind_turns_do_not_replace_evidence_actions",
                })
                candidate = investigate(
                    recovery_step_limit, self.config.max_post_reject_tool_calls,
                    review_feedback=combined_feedback,
                    review_baseline=rejected_candidate,
                    review_evidence_ids=rejected_evidence_ids,
                    review_hypothesis=rejected_hypothesis,
                )
        except ContextResolutionError as exc:
            status = 'INCONCLUSIVE'
            termination_reason = exc.reason_code
            error_type, error_message = type(exc).__name__, str(exc)
            trace.record('INCIDENT_INCONCLUSIVE', {'reason_code': termination_reason,
                'step': state.step, 'tool_calls': state.tool_calls})
        except FinalizationBlocked as exc:
            status = "INCONCLUSIVE"
            termination_reason = "finalization_policy_blocked"
            trace.record("INCIDENT_INCONCLUSIVE", {
                "error_type": type(exc).__name__,
                "failure_category": termination_reason,
                "message": str(exc),
                "finalization_decision": exc.decision.model_dump(mode="json"),
                "dag_metrics": dag_state.metrics() if dag_state is not None else {},
                "step": state.step,
                "tool_calls": state.tool_calls,
            })
            error_type, error_message = type(exc).__name__, str(exc)
        except RunBudgetAdmissionExceeded as exc:
            status = "INCONCLUSIVE"
            termination_reason = "pre_call_budget_admission_exceeded"
            trace.record("INCIDENT_INCONCLUSIVE", {
                "error_type": type(exc).__name__,
                "failure_category": termination_reason,
                "budget_failure_category": "run_budget_admission_exceeded",
                "message": str(exc),
                "budget_before_call": exc.decision.remaining_run_tokens,
                "estimated_call_cost": exc.decision.breakdown,
                "step": state.step,
                "tool_calls": state.tool_calls,
            })
            error_type, error_message = type(exc).__name__, str(exc)
        except ConvergenceBudgetExhausted as exc:
            status = "INCONCLUSIVE"
            termination_reason = "converge_output_budget_exhausted"
            trace.record("PROVISIONAL_CANDIDATE", {
                "candidate": candidate.model_dump() if candidate is not None else None,
                "reason": "convergence response exhausted the Runtime output envelope before producing an action",
                "hypothesis": (
                    active_hypothesis().model_dump()
                    if active_hypothesis() is not None else None
                ),
                "open_obligations": [
                    item.model_dump() for item in (
                        canonical_rca_state.obligations
                    ) if item.blocks_finalization
                ],
            })
            trace.record("INCIDENT_INCONCLUSIVE", {
                "error_type": type(exc).__name__,
                "failure_category": termination_reason,
                "message": str(exc),
                "step": state.step,
                "tool_calls": state.tool_calls,
            })
            error_type, error_message = type(exc).__name__, str(exc)
        except ReflectionContractExhausted as exc:
            status = "INCONCLUSIVE"
            termination_reason = "reflection_contract_exhausted"
            trace.record("PROVISIONAL_CANDIDATE", {
                "candidate": candidate.model_dump() if candidate is not None else None,
                "reason": "Reflection core structured contract failed after one bounded repair",
                "hypothesis": (
                    active_hypothesis().model_dump()
                    if active_hypothesis() is not None else None
                ),
                "open_obligations": [
                    item.model_dump() for item in (
                        canonical_rca_state.obligations
                    ) if item.blocks_finalization
                ],
            })
            trace.record("INCIDENT_INCONCLUSIVE", {
                "error_type": type(exc).__name__,
                "failure_category": termination_reason,
                "message": str(exc),
                "step": state.step, "tool_calls": state.tool_calls,
            })
            error_type, error_message = type(exc).__name__, str(exc)
        except PlannerContractExhausted as exc:
            # A malformed core response is never converted to a guessed action
            # or a plaintext fallback.  Preserve all state collected before the
            # failure and expose any already-built candidate as provisional.
            status = "INCONCLUSIVE"
            termination_reason = "planner_contract_exhausted"
            trace.record("PROVISIONAL_CANDIDATE", {
                "candidate": candidate.model_dump() if candidate is not None else None,
                "reason": "Planner core structured contract failed after one bounded retry",
                "hypothesis": (
                    active_hypothesis().model_dump()
                    if active_hypothesis() is not None else None
                ),
                "open_obligations": [
                    item.model_dump() for item in (
                        canonical_rca_state.obligations
                    ) if item.blocks_finalization
                ],
            })
            trace.record("INCIDENT_INCONCLUSIVE", {
                "error_type": type(exc).__name__,
                "failure_category": termination_reason,
                "message": str(exc),
                "step": state.step, "tool_calls": state.tool_calls,
            })
            error_type, error_message = type(exc).__name__, str(exc)
        except ReviewProgressRequired as exc:
            status = "INCONCLUSIVE"
            termination_reason = "review_feedback_unresolved"
            trace.record("INCIDENT_INCONCLUSIVE", {
                "error_type": type(exc).__name__, "message": str(exc),
                "step": state.step, "tool_calls": state.tool_calls,
            })
            error_type, error_message = type(exc).__name__, str(exc)
        except ProviderCallLimitExceeded as exc:
            status = "INCONCLUSIVE"
            termination_reason = "physical_call_budget_exhausted"
            trace.record("BUDGET_ADMISSION_REJECTED", {
                "boundary": "physical_api_attempt", "reason": str(exc),
                "physical_calls": sum(row.attempt_count for row in provider_gateway.records),
                "provider_calls": 0, "step": state.step})
            error_type, error_message = type(exc).__name__, str(exc)
        except LLMDeadlineExceeded as exc:
            # Once a candidate has entered the Review lifecycle, a later
            # provider timeout means that no candidate has reached a valid
            # terminal boundary. Keep the candidate as explicitly provisional
            # for trace/run diagnostics, while status remains INCONCLUSIVE and
            # the evaluator cannot accept it without a PASS Review decision.
            # Initial investigation timeouts remain execution failures.
            if review_rounds > 0:
                status = "INCONCLUSIVE"
                termination_reason = "provider_timeout_after_review"
                trace.record("PROVISIONAL_CANDIDATE", {
                    "candidate": candidate.model_dump() if candidate is not None else None,
                    "reason": "final review did not produce a decision before its deadline",
                    "review_rounds": review_rounds,
                })
                trace.record("INCIDENT_INCONCLUSIVE", {
                    "error_type": type(exc).__name__, "message": str(exc),
                    "reason": "provider_timeout_before_reviewed_candidate",
                    "review_rounds": review_rounds,
                    "step": state.step, "tool_calls": state.tool_calls,
                })
                error_type, error_message = type(exc).__name__, str(exc)
            else:
                status = "FAILED"
                termination_reason = self._failure_category(exc)
                failure_payload = {
                    "error_type": type(exc).__name__, "failure_category": termination_reason,
                    "message": str(exc),
                    "step": state.step, "tool_calls": state.tool_calls,
                }
                planner_metadata = _safe_planner_metadata_for_trace(exc)
                if planner_metadata is not None:
                    failure_payload["planner_metadata"] = planner_metadata
                trace.record("INCIDENT_FAILED", failure_payload)
                error_type, error_message = type(exc).__name__, str(exc)
        except TimeoutError as exc:
            # A run-level deadline can expire between investigation and the
            # Review provider call (for example while admitting a new round).
            # Once Review has been entered, this is still an unresolved
            # candidate, never a confident FAILED/PASS result.
            if review_rounds > 0:
                status = "INCONCLUSIVE"
                termination_reason = "run_deadline_after_review"
                trace.record("PROVISIONAL_CANDIDATE", {
                    "candidate": candidate.model_dump() if candidate is not None else None,
                    "reason": "run deadline expired before Review completed",
                    "review_rounds": review_rounds,
                })
                trace.record("INCIDENT_INCONCLUSIVE", {
                    "error_type": type(exc).__name__, "message": str(exc),
                    "reason": "run_deadline_after_review",
                    "review_rounds": review_rounds,
                    "step": state.step, "tool_calls": state.tool_calls,
                })
                error_type, error_message = type(exc).__name__, str(exc)
            else:
                status = "FAILED"
                termination_reason = self._failure_category(exc)
                failure_payload = {
                    "error_type": type(exc).__name__, "failure_category": termination_reason,
                    "message": str(exc),
                    "step": state.step, "tool_calls": state.tool_calls,
                }
                planner_metadata = _safe_planner_metadata_for_trace(exc)
                if planner_metadata is not None:
                    failure_payload["planner_metadata"] = planner_metadata
                trace.record("INCIDENT_FAILED", failure_payload)
                error_type, error_message = type(exc).__name__, str(exc)
        except Exception as exc:
            status = "FAILED"
            # Inner bounded loops may already have recorded a more specific
            # terminal cause (for example tool_contract_repair_exhausted).
            # Do not overwrite that cause with generic runtime_failure.
            if not termination_reason:
                termination_reason = self._failure_category(exc)
            failure_payload = {
                "error_type": type(exc).__name__, "failure_category": termination_reason,
                "message": str(exc),
                "step": state.step, "tool_calls": state.tool_calls,
            }
            planner_metadata = _safe_planner_metadata_for_trace(exc)
            if planner_metadata is not None:
                failure_payload["planner_metadata"] = planner_metadata
            trace.record("INCIDENT_FAILED", failure_payload)
            error_type, error_message = type(exc).__name__, str(exc)
        # Investigation failures that still produce an ordinary INCONCLUSIVE
        # terminal state receive one independent report Review.  This does not
        # alter status or admit a RootCauseCandidate.
        review_inconclusive_report()
        if status == "PASS":
            transition_phase("DONE", termination_reason or "review_pass")
        else:
            transition_phase("INCONCLUSIVE", termination_reason or "terminal_failure")
        record_terminal_convergence(
            terminal_status=status,
            reason=termination_reason or "review_pass",
        )
        terminal_outcome = self._terminal_outcome(
            status,
            termination_reason,
            convergence_status=(
                terminal_convergence_decision.status
                if terminal_convergence_decision is not None else ""
            ),
            failure_category=self._termination_failure_category(status, termination_reason),
        )
        # Output status is a projection of the canonical terminal classification.
        if terminal_outcome.outcome_class == "SEMANTIC" and terminal_outcome.status.startswith("INCONCLUSIVE"):
            status = "INCONCLUSIVE"
        report_review_status = (
            "DISABLED" if not self.config.enable_review
            else "PASS" if decision is not None and decision.decision == "PASS"
            else "REJECT" if decision is not None
            else "UNAVAILABLE"
        )
        report_review_reason = (
            decision.reason if decision is not None
            else (error_message or termination_reason or "Review did not complete")
        )
        report = build_diagnostic_report(
            case=case, run_status=status,
            termination_reason=termination_reason,
            candidate=candidate, rca_state=canonical_rca_state,
            evidence_memory=evidence_memory,
            projected_facts=projected_facts.values(),
            review_status=report_review_status,
            review_reason=report_review_reason,
            review_confidence_score=(
                decision.confidence_score if decision is not None else None
            ),
            review_confidence_reason=(
                decision.confidence_reason if decision is not None else ""
            ),
        )
        trace.record("DIAGNOSTIC_REPORT_BUILT", {
            "status": report.status,
            "review_status": report.review_status,
            "semantic_record_ids": list(report.semantic_record_ids),
            "supporting_evidence_ids": list(report.supporting_evidence_ids),
            "open_required_questions": list(report.open_required_questions),
            "open_exploratory_questions": list(report.open_exploratory_questions),
            "reason": "deterministic_projection_of_committed_state",
        })
        if decision is not None:
            report_only_review = candidate is None
            report_review = DiagnosticReportReview(
                decision=decision.decision,
                evidence_quality=(
                    "PARTIAL" if report_only_review and evidence_memory.pinned
                    else "INSUFFICIENT" if report_only_review
                    else "SUFFICIENT" if decision.decision == "PASS"
                    else "PARTIAL" if decision.missing_evidence
                    else "INSUFFICIENT"
                ),
                reasoning_quality=("COHERENT" if decision.causal_chain_valid
                                    and decision.decision == "PASS" else "PARTIAL"
                                    if decision.causal_chain_valid else "UNSUPPORTED"),
                confidence_score=decision.confidence_score,
                confidence_reason=decision.confidence_reason,
                # A PASS report review confirms delivery quality only.  An
                # INCONCLUSIVE report is never a complete RCA candidate.
                report_complete=(decision.decision == "PASS" and not report_only_review),
                unsupported_claims=decision.unsupported_claims,
                missing_evidence=decision.missing_evidence,
                contradictions=decision.blocking_contradictions or decision.contradictions,
                reason=decision.reason,
            )
            trace.record("DIAGNOSTIC_REPORT_REVIEWED", {
                "decision": report_review.decision,
                "evidence_quality": report_review.evidence_quality,
                "reasoning_quality": report_review.reasoning_quality,
                "report_complete": report_review.report_complete,
                "reason": report_review.reason,
            })
        terminal_budget = budget_snapshot("terminal")
        final_dag_metrics = dag_state.metrics() if dag_state is not None else {}
        dag_task_count = int(final_dag_metrics.get("task_count", 0) or 0)
        dag_completed_count = int(final_dag_metrics.get("completed_task_count", 0) or 0)
        trace.record("INCIDENT_COMPLETED", {
            "status": status, "termination_reason": termination_reason,
            "terminal_outcome": terminal_outcome.model_dump(mode="json"),
            "steps": state.step, "tool_calls": state.tool_calls,
            "budget": budget_payload(terminal_budget),
            "execution_mode": execution_mode,
            "context_projection_mode": self.config.context_projection_mode,
            "dag_metrics": final_dag_metrics,
        })
        trace.record("INCIDENT_FINISHED", {
            "status": status, "termination_reason": termination_reason,
            "failure_category": self._termination_failure_category(status, termination_reason),
            "terminal_outcome": terminal_outcome.model_dump(mode="json"),
            "steps": state.step, "tool_calls": state.tool_calls,
        })
        failure_category = self._termination_failure_category(status, termination_reason)
        return IncidentRunResult(
            case_id=case.case_id, status=status, candidate=candidate, review=decision,
            report=report, report_review=report_review,
            semantic_records=tuple(canonical_rca_state.semantic_records),
            evidence=incident_evidence(), hypotheses=hypothesis_history(), actions=tuple(actions),
            metrics=IncidentMetrics(
                steps=state.step, tool_calls=state.tool_calls, prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens,
                cost=run_cost if cost_measured else None,
                llm_calls=llm_calls,
                llm_attempts=provider_gateway.accounting()["provider_attempts"],
                provider_calls=provider_gateway.accounting()["physical_provider_calls"],
                provider_accounting=provider_gateway.accounting(),
                budget_rejected_attempts=budget_rejected_attempts,
                schema_repair_calls=schema_repair_calls,
                provider_failures=provider_failures,
                provider_contract_failures=provider_contract_failures,
                semantic_adjudication_failures=semantic_adjudication_failures,
                token_accounting_error_count=token_accounting_error_count,
                unattributed_prompt_tokens=unattributed_prompt_total,
                token_telemetry=tuple(token_telemetry),
                planner_calls=planner_calls,
                review_calls=review_calls,
                wall_clock_seconds=time.monotonic() - run_started,
                review_rounds=review_rounds, first_pass_accepted=first_pass_accepted,
                evidence_sufficient_step=evidence_sufficient_step,
                final_candidate_step=final_candidate_step,
                tool_calls_after_evidence_sufficient=(
                    state.tool_calls - evidence_sufficient_tool_calls
                    if evidence_sufficient_tool_calls is not None else 0
                ),
                termination_reason=termination_reason,
                reflection_calls=reflection_calls,
                reflection_delta_accepts=reflection_delta_accepts,
                reflection_no_delta_count=reflection_no_delta_count,
                duplicate_calls=duplicate_calls,
                circuit_open_count=tool_circuit.open_count,
                blocked_tool_call_count=blocked_tool_call_count,
                schema_repair_count=schema_repair_count,
                contract_repair_count=contract_repair_calls,
                review_recovery_cycles=review_recovery_cycles,
                obligation_created_count=obligation_created_count,
                obligation_blocked_count=obligation_blocked_count,
                blocking_contradiction_count=blocking_contradiction_count,
                stable_rounds=(
                    active_hypothesis().stable_rounds
                    if active_hypothesis() is not None else 0
                ),
                execution_mode=execution_mode,
                context_projection_mode=self.config.context_projection_mode,
                dag_task_count=dag_task_count,
                dag_completed_task_count=dag_completed_count,
                dag_blocked_task_count=int(final_dag_metrics.get("blocked_task_count", 0) or 0),
                dag_failed_task_count=int(
                    final_dag_metrics.get("failed_task_count", 0) or 0
                ),
                dag_local_replan_count=int(
                    final_dag_metrics.get("local_replan_count", 0) or 0
                ),
                dag_task_completion_rate=(
                    dag_completed_count / dag_task_count if dag_task_count else 0.0
                ),
            ),
            trace_path=str(trace.path),
            error_type=error_type, error_message=error_message,
            failure_category=failure_category,
            terminal_outcome=terminal_outcome,
        )

    @staticmethod
    def _target(arguments: dict[str, Any]) -> str:
        return str(arguments.get("service_name") or arguments.get("app_name") or arguments.get("name") or arguments.get("resource_type") or "")

    @staticmethod
    def _candidate_semantic_fingerprint(candidate: RootCauseCandidate) -> tuple[Any, ...]:
        """Ignore confidence-only churn when comparing Review re-submissions."""
        return (
            " ".join(candidate.component.split()).casefold(),
            " ".join(candidate.fault.split()).casefold(),
            " ".join(candidate.fault_code.split()).casefold(),
            " ".join(candidate.fault_explanation.split()).casefold(),
            " ".join(candidate.mechanism.split()).casefold(),
            tuple(sorted(set(candidate.evidence_ids))),
        )

    @staticmethod
    def _hypothesis_semantic_fingerprint(hypothesis: IncidentHypothesis) -> tuple[Any, ...]:
        normalize = lambda value: " ".join(str(value or "").split()).casefold()
        return (
            normalize(hypothesis.claim),
            normalize(hypothesis.evidence_gap),
            normalize(hypothesis.component),
            normalize(hypothesis.fault),
            normalize(hypothesis.fault_code),
            normalize(hypothesis.fault_explanation),
            normalize(hypothesis.mechanism),
            tuple(sorted(set(hypothesis.supporting_evidence_ids))),
            tuple(sorted(set(hypothesis.contradicting_evidence_ids))),
            tuple(sorted(normalize(item) for item in hypothesis.required_gaps)),
            hypothesis.evidence_sufficient,
            hypothesis.source_mechanism_status,
        )

    @staticmethod
    def _material_hypothesis_fingerprint(hypothesis: IncidentHypothesis | None) -> tuple[Any, ...] | None:
        """Identity used by stability; evidence accumulation is separate progress."""
        if hypothesis is None:
            return None
        normalize = lambda value: " ".join(str(value or "").split()).casefold()
        return (
            normalize(hypothesis.component),
            normalize(hypothesis.fault),
            normalize(hypothesis.fault_code),
            normalize(hypothesis.fault_explanation),
            normalize(hypothesis.mechanism_category or hypothesis.mechanism),
        )

    @staticmethod
    def _bound_prior_context(prior_context, token_budget: int | None, estimator) -> Any | None:
        """Project Prior into the available L3 slice without touching Evidence."""
        if prior_context is None or token_budget is None:
            return prior_context
        remaining = max(0, int(token_budget))
        selected = []
        for candidate in prior_context.candidates:
            size = max(0, int(estimator(candidate.content)))
            if size > remaining:
                continue
            selected.append(candidate)
            remaining -= size
            if remaining <= 0:
                break
        packed_chars = sum(len(item.content) for item in selected)
        return prior_context.model_copy(update={
            "candidates": tuple(selected),
            "source_types": tuple(dict.fromkeys(item.source_type for item in selected)),
            "provenance": tuple({
                (item.provenance.source, item.provenance.source_id, item.provenance.path): item.provenance
                for item in selected
            }.values()),
            "packed_chars": packed_chars,
            "packed_tokens": max(0, sum(int(estimator(item.content)) for item in selected)),
        })

    @staticmethod
    def _failure_category(exc: Exception) -> str:
        message = str(exc).lower()
        from debug_assistant.harness.dynamic_budget import RequestEnvelopeExceeded
        if isinstance(exc, RequestEnvelopeExceeded):
            return "request_envelope_exceeded"
        if isinstance(exc, ProviderCallLimitExceeded):
            return "physical_call_budget_exhausted"
        if isinstance(exc, LLMDeadlineExceeded):
            return "provider_timeout"
        if isinstance(exc, PlannerContractExhausted):
            return "planner_contract_exhausted"
        if isinstance(exc, ReflectionContractExhausted):
            return "reflection_contract_exhausted"
        if isinstance(exc, (PlannerContractError,)):
            return "planner_contract"
        if isinstance(exc, ReviewSchemaError):
            return "review_contract"
        if isinstance(exc, TimeoutError):
            return "deadline_exceeded"
        if isinstance(exc, LLMError):
            return "provider_failure"
        # An explicit finalize action can be rejected by the current
        # hypothesis gate.  If the bounded investigation then has no valid
        # fallback, the loop reports its exhaustion using this generic
        # message; retain the semantic finalization-validation category
        # instead of misclassifying the rejected candidate as a runtime bug.
        if (
            message == "incident step budget exhausted before finalization"
            or message == "finalization reserve reached without a complete hypothesis"
        ):
            return "finalization_validation"
        if "max_" in message or "budget" in message:
            return "budget_exhausted"
        if "final candidate" in message or "evidence id" in message:
            return "finalization_validation"
        if "duplicate" in message or "no-progress" in message:
            return "no_progress"
        return "runtime_failure"

    @staticmethod
    def _termination_failure_category(status: str, termination_reason: str) -> str:
        """Normalize terminal causes without collapsing Provider and Agent errors."""
        if status == "PASS" or not termination_reason:
            return ""
        reason = str(termination_reason).casefold()
        if "planner_contract" in reason:
            return "planner_contract_failure"
        if "tool_contract" in reason:
            return "tool_contract_failure"
        if "review_contract" in reason:
            return "review_provider_failure"
        if reason in {"review_unrecoverable", "review_recovery_unavailable"}:
            return "convergence_failure"
        if "review_feedback" in reason or "review_unresolved" in reason:
            return "convergence_failure"
        if "provider_timeout_after_review" in reason or (
            "review" in reason and "timeout" in reason
        ):
            return "review_provider_failure"
        if "provider_timeout" in reason or "provider_failure" in reason:
            return "provider_failure"
        if "budget" in reason:
            return "budget_failure"
        if "no_progress" in reason or "review_bound" in reason or "convergence" in reason:
            return "convergence_failure"
        if "finalization" in reason or "sufficient_hypothesis" in reason:
            return "semantic_reasoning_failure"
        if "evaluation" in reason:
            return "evaluation_failure"
        return "runtime_failure"

    @staticmethod
    def _terminal_outcome(
        status: str,
        termination_reason: str,
        *,
        convergence_status: str = "",
        failure_category: str = "",
    ) -> TerminalOutcome:
        """Return the one canonical terminal classification.

        ``status`` and ``failure_category`` are serialized output fields. This
        method is the sole terminal boundary:
        a bounded semantic stop is never reported as an execution failure,
        while provider/runtime/contract failures remain failures.
        """
        reason = str(termination_reason or "")
        reason_text = reason.casefold()
        category = str(failure_category or "").casefold()
        convergence = str(convergence_status or "").upper()
        if convergence == "CONVERGED" or (status == "PASS" and not convergence):
            return TerminalOutcome(
                outcome_class="SEMANTIC", status="CONVERGED", reason=reason,
                failure_category="",
            )
        if reason == "planner_contract_exhausted" and category in {
            "planner_contract_exhausted", "planner_contract_failure",
        }:
            return TerminalOutcome(
                outcome_class="SEMANTIC", status="INCONCLUSIVE_CONTRACT",
                reason=reason, failure_category=failure_category,
            )
        if convergence == "INCONCLUSIVE_CAPABILITY" or "capability" in reason_text:
            return TerminalOutcome(
                outcome_class="SEMANTIC", status="INCONCLUSIVE_CAPABILITY",
                reason=reason, failure_category="",
            )
        if convergence == "INCONCLUSIVE_BUDGET" or any(token in reason_text for token in (
            "budget", "review_bound", "max_steps", "max_tool_calls", "token_budget",
            "deadline_exceeded",
        )):
            # A Provider exception/timeout is still an execution failure even
            # when it happened after Review.  A bounded Runtime budget stop is
            # semantic and uses the explicit INCONCLUSIVE_BUDGET outcome.
            if any(token in reason_text or token in category for token in (
                "provider_failure", "provider_timeout", "review_provider",
                "runtime_failure", "tool_contract", "planner_contract",
                "execution_failure", "review_contract", "reflection_contract",
            )):
                return TerminalOutcome(
                    outcome_class="EXECUTION_FAILURE", status="EXECUTION_FAILURE",
                    reason=reason, failure_category=failure_category,
                )
            return TerminalOutcome(
                outcome_class="SEMANTIC", status="INCONCLUSIVE_BUDGET",
                reason=reason, failure_category="",
            )
        if reason.startswith("context_") and status == "INCONCLUSIVE":
            return TerminalOutcome(outcome_class="SEMANTIC", status="INCONCLUSIVE_NO_PROGRESS",
                                   reason=reason, failure_category="")
        if reason == "semantic_dead_end" or "no_progress" in reason_text or "no-progress" in reason_text or any(
            token in reason_text for token in (
                "review_feedback_unresolved", "convergence_failure",
                "dag_finalization_blocked", "finalization_validation",
            )
        ):
            return TerminalOutcome(
                outcome_class="SEMANTIC", status="INCONCLUSIVE_NO_PROGRESS",
                reason=reason, failure_category="",
            )
        if status == "FAILED" or any(token in reason_text or token in category for token in (
            "provider_failure", "provider_timeout", "runtime_failure",
            "tool_contract", "planner_contract", "review_contract", "reflection_contract",
            "execution_failure", "schema", "invalid",
        )):
            return TerminalOutcome(
                outcome_class="EXECUTION_FAILURE", status="EXECUTION_FAILURE",
                reason=reason, failure_category=failure_category,
            )
        return TerminalOutcome(
            outcome_class="SEMANTIC", status="INCONCLUSIVE_NO_PROGRESS",
            reason=reason, failure_category="",
        )

    @staticmethod
    def _planner_control_state(case, semantic_state: RCAState, actions, review_feedback,
                               *, tool_budget_remaining: int,
                               agent_steps_remaining: int, llm_calls_remaining: int | None = None,
                               rejected_actions=(),
                               available_evidence_sources=(),
                               source_workspace_available: bool = False,
                               available_code_tools=(), source_evidence_ids=(),
                               reflection_feedback: str = "",
                               tool_health: dict[str, Any] | None = None,
                               reflection_trigger: str = "",
                               next_action_constraint: str = "",
                               finalization_allowed: bool = False,
                               finalization_blockers=(),
                               active_action_requirement: dict[str, Any] | None = None,
                               capability_states: dict[str, str] | None = None,
                               visible_tool_names=(),
                               feasible_skill_ids=(),
                               canonical_task_context: bool = False,
                               dag_state: Any | None = None,
                               citable_evidence_ids=(),
                               evidence_verdicts=(),
                               domain_skill_id: str = "") -> dict[str, Any]:
        current = semantic_state.active_hypothesis_projection()
        rca_state = semantic_state
        canonical_task_claims: set[str] = set()
        if canonical_task_context and dag_state is not None:
            raw_tasks = (
                dag_state.get("tasks", ())
                if isinstance(dag_state, dict)
                else getattr(dag_state, "tasks", ())
            ) or ()
            for task in raw_tasks:
                status = str(getattr(task, "status", None) or (
                    task.get("status") if isinstance(task, dict) else "PENDING"
                )).upper()
                if status == "COMPLETED":
                    continue
                obligation_id = getattr(task, "obligation_id", None) if not isinstance(task, dict) else task.get("obligation_id")
                claim = ""
                if obligation_id:
                    try:
                        claim = rca_state.obligation(str(obligation_id)).claim if rca_state else ""
                    except RCAStateError:
                        claim = ""
                normalized = " ".join(str(claim or "").split()).casefold()
                if normalized:
                    canonical_task_claims.add(normalized)

        def keep_gap(value: str) -> bool:
            return " ".join(str(value or "").split()).casefold() not in canonical_task_claims

        filtered_obligations = tuple(
            item for item in (current.verification_obligations if current else ())
            if keep_gap(item.claim)
        )
        filtered_gaps = tuple(
            value for value in (current.required_gaps if current else ())
            if keep_gap(value)
        )
        current_payload = current.model_dump() if current else None
        if current_payload is not None and canonical_task_context:
            current_payload.pop("verification_obligations", None)
            current_payload.pop("required_gaps", None)
            current_payload.pop("evidence_gap", None)
        canonical_obligations = tuple(
            item for item in rca_state.obligations
            if item.blocks_finalization and item.status == "OPEN"
        )
        all_open_obligations = tuple(
            item for item in (rca_state.obligations if rca_state else ())
            if item.status == "OPEN"
        )
        if not all_open_obligations:
            all_open_obligations = tuple(
                item for item in (current.verification_obligations if current else ())
                if item.status == "OPEN"
            )
        canonical_contradictions = tuple(
            item for item in (rca_state.contradictions if rca_state else ())
            if item.blocks_finalization and item.status == "OPEN"
        )
        if not canonical_contradictions:
            canonical_contradictions = tuple(
                item for item in (current.contradictions if current else ())
                if item.blocks_finalization and item.status == "OPEN"
            )
        executed_action_items = tuple(
            item for item in actions if isinstance(item, dict)
        )
        executed_action_limit = 8
        executed_action_summary = []
        for item in executed_action_items[-executed_action_limit:]:
            executed_action_summary.append({
                key: item.get(key)
                for key in (
                    "step", "tool", "arguments", "acquisition_key", "observation_id", "outcome",
                    "target_fact_slot", "requested_for", "snapshot_identity", "acquisition_state",
                    "new_evidence_count", "new_evidence_ids",
                    "obligation_progress_count",
                    "semantic_information_gain", "action_class", "obligation_id",
                    "semantic_negative", "observation_status", "error_type",
                )
                if key in item
            })

        def compact_feedback(value: str, *, max_chars: int = 2_000) -> str:
            """Keep Planner control feedback bounded without changing semantics."""
            text = str(value or "")
            if len(text) <= max_chars:
                return text
            try:
                payload = json.loads(text)
            except (TypeError, ValueError, json.JSONDecodeError):
                return text[:max_chars] + "..."
            if isinstance(payload, dict):
                compact: dict[str, Any] = {}
                for key, item in payload.items():
                    if isinstance(item, list):
                        compact[key] = [str(entry)[:360] for entry in item[:4]]
                    elif isinstance(item, str):
                        compact[key] = item[:720]
                    else:
                        compact[key] = item
                bounded = json.dumps(compact, ensure_ascii=False)
                return bounded[:max_chars] + ("..." if len(bounded) > max_chars else "")
            return text[:max_chars] + "..."

        def compact_obligation(item: VerificationObligation) -> dict[str, Any]:
            return {
                key: getattr(item, key)
                for key in (
                    "id", "claim", "verification_question", "evidence_type",
                    "evidence_requirement", "target_component", "critical", "status",
                    "supporting_evidence_ids", "blocked_capabilities",
                )
            }
        active_task = None
        raw_tasks = (
            dag_state.get("tasks", ()) if isinstance(dag_state, dict)
            else getattr(dag_state, "tasks", ())
        ) if dag_state is not None else ()
        for task in raw_tasks or ():
            task_status = str(getattr(task, "status", None) or (
                task.get("status") if isinstance(task, dict) else "PENDING"
            )).upper()
            if task_status in {"COMPLETED", "BLOCKED", "FAILED"}:
                continue
            active_task = task.model_dump(mode="json") if hasattr(task, "model_dump") else dict(task)
            break
        if active_task is None and all_open_obligations:
            obligation = all_open_obligations[0]
            active_task = {
                "task_id": None,
                "obligation_id": obligation.id,
                "claim": obligation.claim,
                "verification_question": obligation.verification_question,
                "status": "READY",
            }
        compact_hypothesis = None
        if current is not None:
            compact_hypothesis = {
                "claim": current.claim,
                "evidence_gap": current.evidence_gap,
                "component": current.component,
                "fault": current.fault,
                "fault_code": current.fault_code,
                "mechanism": current.mechanism,
                "status": current.status,
                "supporting_evidence_ids": list(current.supporting_evidence_ids),
                "contradicting_evidence_ids": list(current.contradicting_evidence_ids),
                "evidence_sufficient": current.evidence_sufficient,
                "source_mechanism_status": current.source_mechanism_status,
                "stable_rounds": current.stable_rounds,
            }
        citable_ids = tuple(dict.fromkeys(
            str(item) for item in (
                *tuple(citable_evidence_ids),
                *(current.supporting_evidence_ids if current else ()),
                *(current.contradicting_evidence_ids if current else ()),
            ) if str(item).startswith("ev-")
        ))
        payload = {
            "INCIDENT": {"summary": case.summary, "system": case.system, "namespace": case.namespace},
            "CURRENT_HYPOTHESIS": compact_hypothesis,
            "ACTIVE_VERIFICATION_TASK": active_task,
            "OPEN_CRITICAL_OBLIGATIONS": [compact_obligation(item) for item in canonical_obligations[:6]],
            "BLOCKING_CONTRADICTIONS": [item.model_dump(mode="json") for item in canonical_contradictions],
            "FINALIZE_ALLOWED": bool(finalization_allowed),
            "FINALIZE_BLOCKERS": list(finalization_blockers)[:12],
            "ACTIVE_ACTION_REQUIREMENT": active_action_requirement or {
                "hypothesis": current.claim if current else "",
                "obligation_id": "",
                "open_obligation": "",
                "verification_question": "",
                "evidence_need": current.evidence_gap if current else "",
                "must_acquire_new_canonical_evidence": bool(current),
                "must_not_repeat_acquired_data": True,
            },
            "CITABLE_EVIDENCE": list(citable_ids),
            "EVIDENCE_SET_VERDICTS": [
                dict(item) for item in tuple(evidence_verdicts)[-12:]
                if isinstance(item, dict)
            ],
            "ACTIVE_SKILL": domain_skill_id,
            "FEASIBLE_SKILL_IDS": list(feasible_skill_ids),
            "AVAILABLE_TOOLS": list(visible_tool_names),
            "REMAINING_BUDGET": {
                "tool_calls": tool_budget_remaining,
                "agent_steps": agent_steps_remaining,
                "llm_calls": llm_calls_remaining,
            },
            # Rejected actions are Runtime facts, not Planner lifecycle
            # controls. Project a bounded recent slice so the next native
            # action can select a different evidence-changing Tool instead of
            # rediscovering the same duplicate acquisition.
            "RECENT_REJECTED_ACTIONS": [
                {
                    key: value for key, value in item.items()
                    if key in {"step", "tool", "arguments", "reason", "instruction"}
                }
                for item in tuple(rejected_actions)[-4:]
                if isinstance(item, dict)
            ],
            "EXECUTED_ACTION_SUMMARY": {
                "shown": executed_action_summary,
                "omitted_count": max(0, len(executed_action_items) - executed_action_limit),
                "acquisition_identity_rule": "tool + canonical executable arguments + Runtime snapshot/freshness; only successful ACQUIRED input is unavailable",
            },
            "NEXT_ACTION_CONSTRAINT": next_action_constraint or None,
            "REVIEW_FEEDBACK": compact_feedback(review_feedback) or None,
            "REVIEW_RECOVERY_RULES": (
                {
                    "baseline_candidate_was_rejected": True,
                    "must_not_repeat_unsupported_claims": True,
                    "every_causal_clause_must_be_supported_by_cited_evidence": True,
                    "if_requested_evidence_is_unavailable_or_not_new": (
                        "narrow the hypothesis to the strongest observed mechanism, "
                        "update supporting_evidence_ids, and finalize only after "
                        "the revised claim is Runtime-allowed"
                    ),
                    "do_not_reuse_the_rejected_candidate_verbatim": True,
                }
                if review_feedback else None
            ),
            "REFLECTION_FEEDBACK": compact_feedback(reflection_feedback) or None,
            "SOURCE_CAPABILITY_SUMMARY": {
                "source_workspace_available": bool(source_workspace_available),
                "available_code_tools": list(available_code_tools),
                "source_evidence_ids": list(source_evidence_ids),
                "status": current.source_mechanism_status if current else "unknown",
            },
            "CAPABILITY_STATES": dict(sorted((capability_states or {}).items())),
        }
        return payload

    @staticmethod
    def _source_mechanism_coverage(
        case,
        hypothesis: IncidentHypothesis | None,
        *,
        source_workspace_available: bool,
        available_code_tools=(),
        source_evidence_ids=(),
        source_relevance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Project source capability and coverage without deciding causality."""
        source_ids = tuple(str(item) for item in source_evidence_ids if str(item).startswith("ev-"))
        supporting = set(hypothesis.supporting_evidence_ids) if hypothesis else set()
        return {
            "application_code_declared": "application_code" in tuple(case.evidence_sources),
            "workspace_available": bool(source_workspace_available),
            "available_code_tools": list(available_code_tools),
            "status": hypothesis.source_mechanism_status if hypothesis else "unknown",
            "source_claims": [item.model_dump() for item in (hypothesis.source_claims if hypothesis else ())],
            "source_relevance": dict(source_relevance or {}),
            "source_backed_evidence_ids": list(source_ids),
            "cited_source_backed_evidence_ids": sorted(supporting.intersection(source_ids)),
            "open_critical_obligation_ids": [
                item.id for item in (hypothesis.verification_obligations if hypothesis else ())
                if item.blocks_finalization
            ],
        }
