from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import multiprocessing
import inspect
from hashlib import sha1
from pathlib import Path
import time
from typing import Any

from debug_assistant.agent.final_review import (
    FinalReviewAgent, ReviewSchemaError, enforce_review_consistency,
)
from debug_assistant.agent.reflection import IncidentReflectionAgent, ReflectionContractExhausted
from debug_assistant.agent.planner import (
    NativePlannerResult, NativeToolPlanner, PlannerContractError,
    PlannerContractExhausted,
)
from debug_assistant.config import ContextConfig
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
from debug_assistant.harness.tool_executor import execute_with_retry
from debug_assistant.harness.guards import LoopGuard
from debug_assistant.harness.provider_health import ToolCircuitBreaker
from debug_assistant.harness.feature_flags import FeatureFlags
from debug_assistant.incidents.contracts import (
    Contradiction, IncidentCase, IncidentEvidence, IncidentHypothesis, IncidentMetrics,
    IncidentRunResult, ReflectionFeedback, ReflectionContradictionUpdate,
    ReviewDecision, RootCauseCandidate,
    VerificationObligation,
)
from debug_assistant.models import ActionKind, ActionProposal, AgentState, TaskSpec, ToolObservation
from debug_assistant.memory.evidence_memory import EvidenceMemory
from debug_assistant.memory.observation_store import ObservationStore
from debug_assistant.knowledge import (
    CapabilitySnapshot, HybridRouter, IncidentEntityExtractor,
    KnowledgeCoordinator, KnowledgeQueryBuilder, KnowledgeRetrievalTool,
)
from debug_assistant.llm.base import LLMDeadlineExceeded, LLMError, complete_json_compat
from debug_assistant.llm.base import ModelCapability
from debug_assistant.skills.catalog import INCIDENT_SKILLS
from debug_assistant.tools.cloudops_snapshot import CloudOpsSnapshotToolRegistry


# Runtime-owned convergence controls. The provider adapter carries the
# reasoning limit as a bounded completion envelope for tool-call JSON.
CONVERGE_MAX_REASONING_TOKENS = 2_000
CONVERGE_MAX_OUTPUT_TOKENS = 3_000
CONVERGE_MAX_READ_FILE_CALLS = 3


def _safe_planner_metadata_for_trace(exc: Exception) -> dict[str, Any] | None:
    """Keep planner diagnostics useful without copying model output to Trace."""
    metadata = getattr(exc, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    safe: dict[str, Any] = {}
    for key in (
        "error_type", "index", "tool", "arguments_type", "actions_type",
        "actions_count", "child_types", "output_shape", "repair_rejection_reason",
        "raw_output", "parsed_output", "normalized_output", "validation_error",
        "repair_prompt", "repair_output", "repair_result",
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


@dataclass(frozen=True, slots=True)
class DiagnosisHarnessConfig:
    max_steps: int = 10
    max_tool_calls: int = 10
    max_review_rounds: int = 2
    max_post_reject_tool_calls: int = 3
    max_post_reject_steps: int = 4
    # Once the current hypothesis has passed the deterministic completion
    # predicate, bounded extra exploration is useful only for a small amount
    # of verification.  This prevents a model from consuming the whole run
    # budget after the answer is already reviewable.
    max_post_sufficiency_tool_calls: int = 4
    max_tool_contract_repairs: int = 1
    planner_llm_timeout_seconds: float = 60.0
    review_llm_timeout_seconds: float = 45.0
    max_llm_calls: int = 40
    max_total_tokens: int = 120_000
    terminal_reserve_tokens: int = 0
    terminal_reserve_llm_calls: int = 3
    context_pressure_ratio: float = 0.75
    hard_pressure_ratio: float = 0.95
    max_cost_per_incident: float | None = None
    max_wall_time_seconds: float = 900.0
    finalization_reserve_seconds: float = 0.0
    tool_retry_attempts: int = 1
    retry_base_delay_seconds: float = 0.0
    retry_max_delay_seconds: float = 0.0
    max_duplicate_actions: int = 2
    max_no_progress: int = 4
    enable_reflection: bool = True
    max_reflection_calls: int = 2
    max_consecutive_reflection_calls: int = 1
    enable_review: bool = True
    max_review_recovery_cycles: int = 1
    tool_circuit_failure_threshold: int = 2
    max_context_chars: int = 24_000
    context: ContextConfig = field(default_factory=ContextConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    trace_dir: str = ".debug_assistant/incident_traces"

    @classmethod
    def from_app_harness(cls, source, *, trace_dir: str | None = None,
                         planner_llm_timeout_seconds: float | None = None,
                         review_llm_timeout_seconds: float | None = None) -> "DiagnosisHarnessConfig":
        """Explicitly adapt the generic AppConfig harness to incident semantics.

        The two runtimes intentionally keep separate config types.  This adapter
        maps only fields with the same operational meaning and makes the few
        renamed controls explicit instead of relying on ``**vars(...)``.
        """
        return cls(
            max_steps=source.max_steps,
            max_tool_calls=source.max_tool_calls,
            max_llm_calls=source.max_llm_calls,
            max_total_tokens=source.max_total_tokens,
            max_cost_per_incident=getattr(source, "max_cost_per_incident", None),
            terminal_reserve_tokens=getattr(source, "terminal_reserve_tokens", 0),
            terminal_reserve_llm_calls=getattr(source, "terminal_reserve_llm_calls", 3),
            max_review_recovery_cycles=getattr(source, "max_review_recovery_cycles", 1),
            max_consecutive_reflection_calls=getattr(source, "max_consecutive_reflection_calls", 1),
            context_pressure_ratio=getattr(source, "context_pressure_ratio", 0.75),
            hard_pressure_ratio=getattr(source, "hard_pressure_ratio", 0.95),
            max_wall_time_seconds=source.max_wall_time_seconds,
            finalization_reserve_seconds=source.finalization_reserve_seconds,
            planner_llm_timeout_seconds=(
                source.planner_llm_timeout_seconds
                if planner_llm_timeout_seconds is None else planner_llm_timeout_seconds
            ),
            review_llm_timeout_seconds=(
                source.reporter_llm_timeout_seconds
                if review_llm_timeout_seconds is None else review_llm_timeout_seconds
            ),
            tool_retry_attempts=source.tool_retry_attempts,
            retry_base_delay_seconds=source.retry_base_delay_seconds,
            retry_max_delay_seconds=source.retry_max_delay_seconds,
            max_duplicate_actions=source.max_repeat_action,
            max_no_progress=source.max_no_progress_steps,
            max_context_chars=source.max_context_chars,
            context=source.context,
            features=source.features,
            trace_dir=trace_dir or source.trace_dir,
        )

    def __post_init__(self):
        if self.max_steps <= 0 or self.max_tool_calls <= 0:
            raise ValueError("step and tool budgets must be positive")
        if self.max_review_rounds != 2:
            raise ValueError("max_review_rounds is fixed at 2")
        if not 1 <= self.max_post_reject_tool_calls <= 3:
            raise ValueError("post-review tool calls must be between 1 and 3")
        if not 1 <= self.max_post_reject_steps <= 4:
            raise ValueError("post-review steps must be between 1 and 4")
        if not 1 <= self.max_post_sufficiency_tool_calls <= 8:
            raise ValueError("post-sufficiency tool calls must be between 1 and 8")
        if not 1 <= self.max_tool_contract_repairs <= 3:
            raise ValueError("tool contract repairs must be between 1 and 3")
        if self.planner_llm_timeout_seconds <= 0 or self.review_llm_timeout_seconds <= 0:
            raise ValueError("LLM timeouts must be positive")
        if self.max_context_chars <= 0:
            raise ValueError("max_context_chars must be positive")
        if self.max_llm_calls <= 0 or self.max_total_tokens <= 0 or self.max_wall_time_seconds <= 0:
            raise ValueError("LLM, token, and wall-clock budgets must be positive")
        if self.terminal_reserve_tokens < 0 or self.terminal_reserve_llm_calls <= 0:
            raise ValueError("terminal reserves must be non-negative tokens and positive calls")
        if not 0.5 <= self.context_pressure_ratio < 1.0:
            raise ValueError("context pressure ratio must be in [0.5, 1.0)")
        if not self.context_pressure_ratio <= self.hard_pressure_ratio <= 1.0:
            raise ValueError("hard pressure ratio must be >= pressure ratio and <= 1")
        if self.max_cost_per_incident is not None and self.max_cost_per_incident <= 0:
            raise ValueError("max_cost_per_incident must be positive when configured")
        if self.finalization_reserve_seconds < 0 or self.finalization_reserve_seconds >= self.max_wall_time_seconds:
            raise ValueError("finalization reserve must be smaller than the run deadline")
        if self.tool_retry_attempts <= 0 or self.max_duplicate_actions <= 0 or self.max_no_progress <= 0:
            raise ValueError("retry and progress limits must be positive")
        if self.max_reflection_calls < 0:
            raise ValueError("max_reflection_calls must be non-negative")
        if self.max_consecutive_reflection_calls <= 0:
            raise ValueError("max_consecutive_reflection_calls must be positive")
        if self.max_review_recovery_cycles < 0:
            raise ValueError("max_review_recovery_cycles must be non-negative")
        if self.tool_circuit_failure_threshold <= 0:
            raise ValueError("tool_circuit_failure_threshold must be positive")


# Kept for callers and tests written against the first Incident vertical slice.
IncidentHarnessConfig = DiagnosisHarnessConfig


_OBLIGATION_TRANSITIONS = {
    "OPEN": {
        "OPEN", "SATISFIED", "BLOCKED_BY_CAPABILITY",
        "NOT_APPLICABLE_BY_CAPABILITY",
    },
    "SATISFIED": {"SATISFIED"},
    "BLOCKED_BY_CAPABILITY": {
        "BLOCKED_BY_CAPABILITY",
        "NOT_APPLICABLE_BY_CAPABILITY",
    },
    "NOT_APPLICABLE_BY_CAPABILITY": {"NOT_APPLICABLE_BY_CAPABILITY"},
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


class ReviewProgressRequired(RuntimeError):
    """The bounded post-review loop tried to submit an unchanged candidate."""


class ConvergenceBudgetExhausted(RuntimeError):
    """A convergence response exhausted its bounded output envelope."""


class DiagnosisHarness:
    """Bounded CloudOps diagnosis runtime with a separate final-review boundary.

    This is deliberately the lightweight diagnosis core. The older generic
    AgentHarness remains the SWE compatibility path until Task B.
    """

    def __init__(self, llm, review_llm=None, *, model: str = "", review_model: str = "",
                 config: DiagnosisHarnessConfig | None = None,
                 topology_path: str | Path | None = None, search_engine=None,
                 knowledge_store=None, domain_rag=None, static_graph=None,
                 knowledge_coordinator: KnowledgeCoordinator | None = None):
        self.llm = llm
        self.review_llm = review_llm or llm
        self.model = model
        self.review_model = review_model
        self.config = config or DiagnosisHarnessConfig()
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

    def run(self, case: IncidentCase) -> IncidentRunResult:
        tools = CloudOpsSnapshotToolRegistry(
            case.runtime_data_dir, self.topology_path, search_engine=self.search_engine,
        )
        # Keep this state aligned with the provider-facing registry projection;
        # hidden repository primitives remain executable but are not advertised
        # as Planner capabilities.
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
            planner_state_envelope=self.config.features.planner_state_envelope,
        )
        reviewer = FinalReviewAgent(self.review_llm, self.review_model)
        reflection_agent = IncidentReflectionAgent(self.llm, self.model)
        trace = TraceRecorder(self.config.trace_dir, case.case_id)
        deadline = RunDeadline(self.config.max_wall_time_seconds)
        state = AgentState(TaskSpec(
            task_id=case.case_id, issue=case.summary, repo_path=case.runtime_data_dir,
            repo_name=case.system, metadata={"task_kind": "incident", "namespace": case.namespace},
        ))
        observation_store = ObservationStore()
        evidence_memory = EvidenceMemory()
        state.evidence = evidence_memory.pinned
        incident_projection = IncidentProjectionPolicy()
        context_manager = ContextManager(
            self.config.context,
            enable_catalog=self.config.features.context_manager,
            enable_model_selection=False,
            enable_budget_packing=self.config.features.context_manager,
            enable_lifecycle=self.config.features.context_manager,
            enable_projection=self.config.features.context_manager,
            projection_policy=incident_projection,
        )
        hypotheses: list[IncidentHypothesis] = []
        actions: list[dict[str, Any]] = []
        rejected_actions: list[dict[str, Any]] = []
        fingerprints: set[str] = set()
        loop_guard = LoopGuard(
            max_repeat=self.config.max_duplicate_actions,
            max_no_progress=(
                self.config.max_no_progress
                if self.config.features.no_progress_detection else 1_000_000
            ),
        )
        tool_circuit = ToolCircuitBreaker(
            failure_threshold=self.config.tool_circuit_failure_threshold,
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
        reflection_signatures: set[str] = set()
        reflection_trigger = ""
        duplicate_calls = 0
        blocked_tool_call_count = 0
        schema_repair_count = 0
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
        tool_contract_repairs = 0
        contract_repair_calls = 0
        review_calls = 0
        termination_reason = ""
        run_started = time.monotonic()
        runtime_phase = "INIT"
        state_transition_count = 0
        converge_read_file_calls = 0

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
        model_capability = resolve_model_capability(self.llm, model=self.model)
        dynamic_budget = DynamicBudgetController(
            model_capability,
            max_context_chars=self.config.max_context_chars,
            max_total_tokens=self.config.max_total_tokens,
            max_llm_calls=self.config.max_llm_calls,
            max_wall_time_seconds=self.config.max_wall_time_seconds,
            max_cost_per_incident=self.config.max_cost_per_incident,
            terminal_reserve_tokens=self.config.terminal_reserve_tokens,
            terminal_reserve_llm_calls=self.config.terminal_reserve_llm_calls,
            terminal_reserve_seconds=self.config.finalization_reserve_seconds,
            pressure_ratio=self.config.context_pressure_ratio,
            hard_pressure_ratio=self.config.hard_pressure_ratio,
            started_at=state.started_at,
        )
        trace.record("INCIDENT_STARTED", {
            "case_id": case.case_id, "summary": case.summary, "system": case.system,
            "namespace": case.namespace, "evidence_sources": case.evidence_sources,
            "model_capability": model_capability.as_dict(),
            "terminal_reserve": dynamic_budget.terminal_reserve_payload(),
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

        def add_usage(usage_source, *, stage: str = "unknown", prompt_breakdown: dict[str, Any] | None = None) -> None:
            nonlocal prompt_tokens, completion_tokens, run_cost, cost_measured
            run_tokens_before = prompt_tokens + completion_tokens
            usage = dict(getattr(usage_source, "usage", None) or {})
            if not usage and hasattr(usage_source, "response"):
                usage = dict(getattr(usage_source.response, "usage", None) or {})
            if not usage and hasattr(usage_source, "last_usage"):
                usage = dict(getattr(usage_source, "last_usage", None) or {})
            actual_prompt_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
            actual_completion_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
            raw_cost = usage.get("cost", usage.get("total_cost", usage.get("estimated_cost")))
            try:
                actual_cost = None if raw_cost is None else float(raw_cost)
            except (TypeError, ValueError):
                actual_cost = None
            prompt_tokens += actual_prompt_tokens
            completion_tokens += actual_completion_tokens
            if actual_cost is not None:
                run_cost += max(0.0, actual_cost)
                cost_measured = True
            dynamic_budget.set_run_usage(
                tokens_used=prompt_tokens + completion_tokens,
                llm_calls_used=llm_calls,
                cost_used=run_cost,
            )
            breakdown = dict(prompt_breakdown or getattr(usage_source, "last_prompt_breakdown", {}) or {})
            estimated_prompt_tokens = int(
                breakdown.get("estimated_prompt_tokens", breakdown.get("last_estimated_prompt_tokens", 0)) or 0
            )
            dynamic_budget.record_actual_usage(
                stage=stage,
                estimated_prompt_tokens=estimated_prompt_tokens,
                actual_prompt_tokens=actual_prompt_tokens if usage else None,
                actual_completion_tokens=actual_completion_tokens if usage else None,
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
                "actual_prompt_tokens": actual_prompt_tokens if usage else None,
                "actual_completion_tokens": actual_completion_tokens if usage else None,
                "actual_cost": actual_cost,
                "run_tokens_before": run_tokens_before,
                "run_tokens_after": prompt_tokens + completion_tokens,
                "run_cost_after": run_cost,
                "context_state": after_state.state.value,
                "context_breakdown": dict(breakdown.get("token_breakdown") or breakdown),
                "compaction_applied": after_state.state is not BudgetState.NORMAL,
                "prior_pruned": False,
                "rehydrated_evidence_count": 0,
                "cache_hit": False,
                "duplicate_blocked": False,
            })
            if prompt_tokens + completion_tokens > self.config.max_total_tokens:
                raise RuntimeError("max_total_tokens exceeded after LLM response")
            if (
                dynamic_budget.max_cost_per_incident is not None
                and run_cost > dynamic_budget.max_cost_per_incident
            ):
                raise RuntimeError("max_cost_per_incident exceeded after LLM response")

        def budget_snapshot(boundary: str) -> Any:
            return budget.snapshot(
                steps=state.step, tool_calls=state.tool_calls,
                llm_calls=llm_calls, tokens=prompt_tokens + completion_tokens,
            )

        def budget_payload(snapshot) -> dict[str, Any]:
            dynamic_state = dynamic_budget.decision(
                "runtime", tokens_used=prompt_tokens, llm_calls_used=llm_calls,
                cost_used=run_cost,
            )
            return {
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
            }

        def admit(stage: str, *, count_llm: bool = False,
                  allow_targeted_recovery: bool = False) -> None:
            nonlocal llm_calls, planner_calls, review_calls
            deadline.check()
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
                    if (
                        stage in {"planner", "tool", "reflection", "review", "contract_repair"}
                        and not allow_targeted_recovery
                    ):
                        raise RunBudgetAdmissionExceeded(dynamic_state)
            if snapshot.tokens_used >= self.config.max_total_tokens:
                raise RuntimeError("max_total_tokens exceeded")
            if (
                dynamic_budget.max_cost_per_incident is not None
                and run_cost >= dynamic_budget.max_cost_per_incident
            ):
                raise RuntimeError("max_cost_per_incident exceeded")
            if count_llm:
                if llm_calls >= self.config.max_llm_calls:
                    raise RuntimeError("max_llm_calls exceeded")
                llm_calls += 1
                if stage == "planner":
                    planner_calls += 1
                elif stage == "review":
                    review_calls += 1
            elif stage == "planner" and state.step >= self.config.max_steps:
                raise RuntimeError("max_steps exceeded")
            elif stage == "tool" and state.tool_calls >= self.config.max_tool_calls:
                raise RuntimeError("max_tool_calls exceeded")

        def provider_timeout(configured: float) -> float:
            remaining = deadline.remaining()
            timeout = min(float(configured), remaining)
            if timeout <= 0:
                raise LLMDeadlineExceeded("diagnosis run deadline exceeded")
            return timeout

        def account_review_attempts(reviewer: FinalReviewAgent) -> None:
            """Account every logical Review call, including a failed repair.

            The first Review call is admitted before entering FinalReviewAgent;
            subsequent calls are reported by the agent.  Counting from the
            agent's pre-call counter also works when a repair raises before it
            can return a ReviewDecision.
            """
            nonlocal llm_calls, review_calls, schema_repair_count
            call_count = max(1, int(getattr(reviewer, "last_call_count", 1) or 1))
            extra_calls = call_count - 1
            llm_calls += extra_calls
            review_calls += extra_calls
            attempts = tuple(getattr(reviewer, "last_attempts", ()) or ())
            repaired_calls = sum(1 for item in attempts if bool(item.get("schema_repair")))
            # If a hard process boundary kills the child before its state can
            # be copied back, the parent still knows that the Review object
            # entered its bounded repair path from this marker/call count.
            if repaired_calls == 0 and bool(getattr(reviewer, "last_repair_attempted", False)):
                repaired_calls = 1
            schema_repair_count += repaired_calls

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

        def execute_tool(tool, arguments: dict[str, Any], tool_name: str) -> ToolObservation:
            nonlocal blocked_tool_call_count
            if not tool_circuit.before_call(tool_name):
                blocked_tool_call_count += 1
                trace.record("TOOL_CALL_BLOCKED_BY_CIRCUIT", {
                    "tool": tool_name,
                    "health": tool_circuit.summary().get(tool_name, {}),
                })
                observation = tool_circuit.blocked_observation(tool_name)
                update_capability_state(tool_name, observation)
                return observation
            admit("tool")
            policy = RetryPolicy(
                max_attempts=self.config.tool_retry_attempts,
                base_delay_seconds=self.config.retry_base_delay_seconds,
                max_delay_seconds=self.config.retry_max_delay_seconds,
            )
            try:
                observation = execute_with_retry(
                    tool, arguments, policy=policy,
                    absolute_deadline=deadline.absolute_deadline,
                    on_retry=lambda attempt, failed: trace.record("TOOL_RETRY", {
                        "tool": tool_name, "attempt": attempt,
                        "error_type": failed.error_type,
                    }),
                )
            except Exception as exc:
                trace.record("TOOL_EXECUTION_FAILED", {
                    "tool": tool_name, "error_type": type(exc).__name__,
                    "failure_category": self._failure_category(exc),
                    "message": str(exc),
                })
                observation = ToolObservation(
                    tool_name, False, str(exc),
                    {
                        "retryable": False,
                        "failure_category": "execution_failure",
                        "execution_failure": True,
                    },
                    type(exc).__name__, 0.0,
                )
                transition = tool_circuit.observe(tool_name, observation)
                if transition == "opened":
                    trace.record("TOOL_CIRCUIT_OPEN", {
                        "tool": tool_name,
                        "health": tool_circuit.summary().get(tool_name, {}),
                    })
                update_capability_state(tool_name, observation)
                return observation
            normalization_action = (observation.metadata or {}).get("normalization_action")
            if isinstance(normalization_action, dict):
                trace.record("PATH_NORMALIZED", {
                    "tool": tool_name,
                    "normalization_action": dict(normalization_action),
                    "observation_id": observation.observation_id,
                })
            transition = tool_circuit.observe(tool_name, observation)
            if transition == "opened":
                trace.record("TOOL_CIRCUIT_OPEN", {
                    "tool": tool_name,
                    "health": tool_circuit.summary().get(tool_name, {}),
                })
            update_capability_state(tool_name, observation)
            return observation

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

        def source_evidence_ids() -> tuple[str, ...]:
            """Return canonical CODE Evidence from verified source-reading tools."""
            return evidence_memory.source_evidence_ids()

        obligation_state: dict[str, VerificationObligation] = {}

        def _obligation_id(claim: str) -> str:
            return "obl-" + sha1(" ".join(str(claim).split()).encode("utf-8")).hexdigest()[:10]

        def build_hypothesis(selection) -> IncidentHypothesis:
            """Normalize legacy planner fields into one reasoning state."""
            nonlocal previous_hypothesis, obligation_created_count
            nonlocal obligation_blocked_count, blocking_contradiction_count
            known_ids = {item.evidence_id for item in evidence_memory.pinned}
            old = hypotheses[-1] if hypotheses else None

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
            next_obligations: dict[str, VerificationObligation] = {}
            for obligation in supplied:
                if _is_source_mechanism_obligation(obligation):
                    # Providers may regenerate the source row with a fresh ID.
                    # The claim, not the model-chosen ID, is the canonical key.
                    canonical_source = next(
                        (
                            item for item in (*next_obligations.values(), *obligation_state.values())
                            if _is_source_mechanism_obligation(item)
                        ),
                        None,
                    )
                    if canonical_source is not None and obligation.id != canonical_source.id:
                        trace.record("SOURCE_OBLIGATION_ID_NORMALIZED", {
                            "proposed_id": obligation.id,
                            "canonical_id": canonical_source.id,
                            "reason": "runtime_owned_source_obligation_identity",
                        })
                        obligation = obligation.model_copy(update={"id": canonical_source.id})
                prior = obligation_state.get(obligation.id)
                if _is_source_mechanism_obligation(obligation):
                    proposed_status = obligation.status
                    if source_capability_available:
                        prior_source_ids = tuple(
                            evidence_id for evidence_id in (prior.supporting_evidence_ids if prior else ())
                            if evidence_id in set(source_ids)
                        )
                        runtime_status = "SATISFIED" if (cited_source_ids or prior_source_ids) else "OPEN"
                        runtime_support = tuple(dict.fromkeys(
                            cited_source_ids or prior_source_ids
                        ))
                        runtime_blocked = ()
                    else:
                        runtime_status = "NOT_APPLICABLE_BY_CAPABILITY"
                        runtime_support = ()
                        runtime_blocked = tuple(source_capability_gaps)
                    obligation = obligation.model_copy(update={
                        "claim": _SOURCE_MECHANISM_OBLIGATION_CLAIM,
                        "critical": True,
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
                    # normalized to the Runtime-owned OPEN/SATISFIED/N/A state.
                    if obligation.id not in obligation_state:
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
                if (
                    prior is not None
                    and prior.status == "SATISFIED"
                    and obligation.status == prior.status
                    and (
                        not obligation.supporting_evidence_ids
                        or any(
                            evidence_id not in known_ids
                            or not evidence_id.startswith("ev-")
                            for evidence_id in obligation.supporting_evidence_ids
                        )
                    )
                ):
                    # Terminal state and its Evidence are Runtime-owned. A
                    # later provider turn may repeat a stale, incomplete
                    # compatibility projection while requesting a new tool.
                    trace.record("OBLIGATION_TERMINAL_REPLAY_REJECTED", {
                        "obligation_id": obligation.id,
                        "status": obligation.status,
                        "reason": "terminal_runtime_state_preserved",
                        "invalid_supporting_evidence_ids": list(
                            obligation.supporting_evidence_ids
                        ),
                    })
                    obligation = prior
                if (
                    prior is not None
                    and prior.status == "SATISFIED"
                    and obligation.status == "OPEN"
                ):
                    # Planner outputs are a semantic proposal, but terminal
                    # obligation state is Runtime-owned. A stale or repeated
                    # structured item must not reopen a resolved obligation
                    # and crash the run on SATISFIED->OPEN.
                    trace.record("OBLIGATION_REOPEN_REJECTED", {
                        "obligation_id": obligation.id,
                        "from": prior.status,
                        "to": obligation.status,
                        "reason": "terminal_runtime_state_preserved",
                    })
                    obligation = prior
                if prior is not None and obligation.status not in _OBLIGATION_TRANSITIONS[prior.status]:
                    raise RuntimeError(
                        f"invalid obligation status transition {prior.status}->{obligation.status}"
                    )
                if obligation.status == "SATISFIED":
                    if not obligation.supporting_evidence_ids or any(
                        evidence_id not in known_ids or not evidence_id.startswith("ev-")
                        for evidence_id in obligation.supporting_evidence_ids
                    ):
                        raise RuntimeError(
                            "terminal verification obligation requires existing ev-* evidence"
                        )
                if prior is not None and prior.status != obligation.status:
                    trace.record("OBLIGATION_TRANSITION", {
                        "obligation_id": obligation.id,
                        "from": prior.status,
                        "to": obligation.status,
                        "reason": "planner_structured_state",
                    })
                if obligation.id not in obligation_state:
                    obligation_created_count += 1
                    trace.record("OBLIGATION_CREATED", obligation.model_dump())
                next_obligations[obligation.id] = obligation

            # A required_evidence_gaps response is the compatibility input.  It
            # is normalized into the obligation ledger instead of becoming a
            # second mutable source of truth.
            for gap in selection.required_evidence_gaps:
                claim = " ".join(str(gap).split()).strip()
                if not claim:
                    continue
                # VerificationObligation is canonical when the Planner emits
                # both representations. Never let the compatibility gap
                # projection overwrite a structured SATISFIED state
                # (or erase structured supporting Evidence) in this turn.
                structured_match = next(
                    (item for item in next_obligations.values()
                     if " ".join(item.claim.split()).casefold() == claim.casefold()),
                    None,
                )
                if structured_match is not None:
                    continue
                existing = next(
                    (item for item in obligation_state.values()
                     if " ".join(item.claim.split()).casefold() == claim.casefold()),
                    None,
                )
                oid = existing.id if existing is not None else _obligation_id(claim)
                prior = obligation_state.get(oid)
                if prior is not None and prior.status != "OPEN":
                    # Legacy required_gaps are an input projection, not a
                    # command to reopen a terminal obligation. A capability
                    # block must remain visible until the Planner explicitly
                    # supplies a legal Runtime transition.
                    next_obligations[oid] = prior
                    continue
                obligation = VerificationObligation(
                    id=oid,
                    claim=claim,
                    critical=True,
                    status="OPEN",
                    supporting_evidence_ids=(),
                    blocked_capabilities=prior.blocked_capabilities if prior else (),
                )
                if prior is None:
                    obligation_created_count += 1
                    trace.record("OBLIGATION_CREATED", obligation.model_dump())
                elif prior.status != "OPEN":
                    trace.record("OBLIGATION_TRANSITION", {
                        "obligation_id": oid, "from": prior.status, "to": "OPEN",
                        "reason": "planner_reopened_required_gap",
                    })
                next_obligations[oid] = obligation

            # Source coverage is a Runtime-owned capability/evidence decision.
            # The LLM field is retained for audit only and cannot close the
            # source obligation or declare it not applicable.
            proposed_source_status = getattr(
                selection, "source_mechanism_status", "unknown"
            ) or "unknown"
            source_status = (
                "not_applicable" if not source_capability_available
                else "sufficient" if cited_source_ids else "gap"
            )
            if proposed_source_status != source_status:
                trace.record("SOURCE_MECHANISM_STATUS_OVERRIDDEN", {
                    "proposed_status": proposed_source_status,
                    "runtime_status": source_status,
                    "capability_available": source_capability_available,
                    "capability_gaps": list(source_capability_gaps),
                    "cited_source_evidence_ids": list(cited_source_ids),
                    "reason": "runtime_owned_source_coverage",
                })
            source_obligation = next(
                (
                    item for item in (*next_obligations.values(), *obligation_state.values())
                    if _is_source_mechanism_obligation(item)
                ),
                None,
            )
            source_should_track = bool(
                source_obligation is not None
                or selection.candidate_mechanism.strip()
            )
            if source_should_track:
                desired_status = (
                    "SATISFIED" if source_status == "sufficient"
                    else "NOT_APPLICABLE_BY_CAPABILITY"
                    if source_status == "not_applicable" else "OPEN"
                )
                desired_support = tuple(cited_source_ids) if desired_status == "SATISFIED" else ()
                desired_blocked = (
                    tuple(source_capability_gaps)
                    if desired_status == "NOT_APPLICABLE_BY_CAPABILITY" else ()
                )
                if source_obligation is None:
                    source_obligation = VerificationObligation(
                        id=_obligation_id(_SOURCE_MECHANISM_OBLIGATION_CLAIM),
                        claim=_SOURCE_MECHANISM_OBLIGATION_CLAIM,
                        critical=True,
                        status=desired_status,
                        supporting_evidence_ids=desired_support,
                        blocked_capabilities=desired_blocked,
                    )
                    obligation_created_count += 1
                    trace.record("OBLIGATION_CREATED", source_obligation.model_dump())
                elif (
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
                next_obligations[source_obligation.id] = source_obligation

            # When the Planner explicitly closes all required gaps and has
            # evidence, close prior obligations deterministically. The Harness
            # does not decide whether one evidence source is semantically
            # equivalent to another; it only applies the Planner's completion
            # proposal to the known Evidence IDs.
            if not selection.required_evidence_gaps and selection.evidence_sufficiency == "sufficient":
                for oid, prior in obligation_state.items():
                    if oid in next_obligations or prior.status not in {"OPEN", "BLOCKED_BY_CAPABILITY"}:
                        continue
                    if _is_source_mechanism_obligation(prior):
                        # Source status is decided only by the deterministic
                        # capability/evidence branch above; generic completion
                        # must never close or waive it.
                        continue
                    if prior.status == "BLOCKED_BY_CAPABILITY":
                        continue
                    status = "SATISFIED"
                    updated = prior.model_copy(update={
                        "status": status,
                        "supporting_evidence_ids": tuple(selection.supporting_evidence_ids),
                    })
                    next_obligations[oid] = updated
                    trace.record("OBLIGATION_TRANSITION", {
                        "obligation_id": oid, "from": prior.status, "to": status,
                        "reason": "planner_declared_sufficient_with_known_evidence",
                    })
            obligation_state.clear()
            obligation_state.update(next_obligations)
            for obligation in obligation_state.values():
                if obligation.status == "BLOCKED_BY_CAPABILITY":
                    obligation_blocked_count += 1

            structured_contradictions = tuple(selection.contradictions or ())
            if structured_contradictions:
                flat_contradictions = tuple(
                    item.evidence_id for item in structured_contradictions
                    if item.blocks_finalization
                )
            else:
                # Legacy flat IDs retain their historical blocking semantics.
                flat_contradictions = tuple(selection.contradicting_evidence_ids)
                structured_contradictions = tuple(
                    Contradiction(
                        evidence_id=evidence_id,
                        claim="Evidence directly contradicts the current diagnosis.",
                        severity="BLOCKING",
                        status="OPEN",
                    )
                    for evidence_id in flat_contradictions
                )
            for contradiction in structured_contradictions:
                if (
                    not contradiction.evidence_id.startswith("ev-")
                    or contradiction.evidence_id not in known_ids
                ):
                    raise RuntimeError(
                        "contradiction must cite an existing ev-* Evidence ID"
                    )
            blocking = {
                item.evidence_id for item in structured_contradictions
                if item.blocks_finalization
            }
            prior_contradictions = {
                item.evidence_id: item for item in (old.contradictions if old else ())
            }
            for contradiction in structured_contradictions:
                prior = prior_contradictions.get(contradiction.evidence_id)
                if prior is not None and contradiction.status not in _CONTRADICTION_TRANSITIONS[prior.status]:
                    raise RuntimeError(
                        f"invalid contradiction status transition {prior.status}->{contradiction.status}"
                    )
                if prior is None:
                    trace.record("CONTRADICTION_CREATED", contradiction.model_dump())
                elif prior != contradiction:
                    trace.record("CONTRADICTION_TRANSITION", {
                        "evidence_id": contradiction.evidence_id,
                        "from": prior.model_dump(),
                        "to": contradiction.model_dump(),
                    })
            blocking_contradiction_count = len(blocking)

            required_projection = tuple(
                item.claim for item in obligation_state.values() if item.blocks_finalization
            )
            hypothesis = IncidentHypothesis(
                claim=selection.current_hypothesis,
                evidence_gap=selection.evidence_gap,
                component=selection.candidate_component,
                fault=(selection.candidate_fault
                       or selection.candidate_fault_code
                       or selection.candidate_fault_explanation),
                fault_code=selection.candidate_fault_code,
                fault_explanation=selection.candidate_fault_explanation,
                mechanism=selection.candidate_mechanism,
                supporting_evidence_ids=selection.supporting_evidence_ids,
                contradicting_evidence_ids=flat_contradictions,
                required_gaps=required_projection,
                evidence_sufficient=selection.evidence_sufficiency == "sufficient",
                mechanism_category=selection.mechanism_category,
                source_mechanism_status=source_status,
                verification_obligations=tuple(obligation_state.values()),
                contradictions=structured_contradictions,
            )
            if old is not None:
                same_identity = self._material_hypothesis_fingerprint(old) == self._material_hypothesis_fingerprint(hypothesis)
                old_blocking = {
                    item.evidence_id for item in old.contradictions if item.blocks_finalization
                }
                new_blocking = bool(blocking - old_blocking)
                stable_rounds = old.stable_rounds + 1 if same_identity and not new_blocking else 0
            else:
                stable_rounds = 0
            hypothesis = hypothesis.model_copy(update={"stable_rounds": stable_rounds})
            previous_hypothesis = old
            return hypothesis

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
            """Apply only a genuinely new Runtime-owned Reflection Delta."""
            nonlocal obligation_created_count, blocking_contradiction_count
            nonlocal reflection_constraint, reflection_delta_accepts
            current = hypotheses[-1] if hypotheses else None
            before = (
                self._hypothesis_semantic_fingerprint(current) if current else None,
                tuple(sorted((item.id, item.claim.casefold(), item.status)
                             for item in obligation_state.values())),
                tuple(sorted((item.evidence_id, item.status, item.severity)
                             for item in (current.contradictions if current else ()))),
                reflection_constraint,
            )
            constraint = " ".join(str(feedback.next_action_constraint or "").split()).casefold()
            allowed_constraints = {
                "", "targeted_verification", "no_broad_exploration",
                "finalize", "inconclusive",
            }
            if constraint not in allowed_constraints:
                trace.record("REFLECTION_DELTA_REJECTED", {
                    "trigger": trigger,
                    "reason": "unknown_next_action_constraint",
                    "constraint": feedback.next_action_constraint,
                })
                return False

            next_obligations = dict(obligation_state)
            new_obligation_ids: list[str] = []
            proposals = [
                *(item for item in feedback.proposed_evidence_gaps),
                *(item for item in feedback.proposed_obligations),
            ]
            for proposal in proposals:
                claim = " ".join(str(proposal.claim).split()).strip()
                if not claim:
                    continue
                existing = next((
                    item for item in next_obligations.values()
                    if " ".join(item.claim.split()).casefold() == claim.casefold()
                ), None)
                if existing is not None:
                    continue
                supporting = tuple(getattr(proposal, "supporting_evidence_ids", ()) or ())
                obligation = VerificationObligation(
                    id=_obligation_id(claim),
                    claim=claim,
                    critical=bool(getattr(proposal, "critical", True)),
                    status="OPEN",
                    supporting_evidence_ids=supporting,
                )
                next_obligations[obligation.id] = obligation
                new_obligation_ids.append(obligation.id)
                obligation_created_count += 1
                trace.record("OBLIGATION_CREATED", {
                    **obligation.model_dump(),
                    "reason": "reflection_delta_proposal",
                    "trigger": trigger,
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
            updated = current
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
                support_ids = tuple(dict.fromkeys(
                    (*current.supporting_evidence_ids,
                     *feedback.supporting_evidence_ids,
                     *delta.supporting_evidence_ids)
                ))
                if support_ids != current.supporting_evidence_ids:
                    changes["supporting_evidence_ids"] = support_ids
                    accepted_fields.append("supporting_evidence_ids")
                contradiction_ids = tuple(dict.fromkeys(
                    (*current.contradicting_evidence_ids,
                     *feedback.contradicting_evidence_ids,
                     *delta.contradicting_evidence_ids)
                ))
                if contradiction_ids != current.contradicting_evidence_ids:
                    changes["contradicting_evidence_ids"] = contradiction_ids
                    accepted_fields.append("contradicting_evidence_ids")
                if next_obligations != obligation_state:
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
                    updated = current.model_copy(update=changes)
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
            if before == after:
                trace.record("REFLECTION_NO_DELTA", {
                    "trigger": trigger,
                    "reason": "canonical_runtime_state_unchanged",
                })
                return False

            obligation_state.clear()
            obligation_state.update(next_obligations)
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
                if not hypotheses or hypotheses[-1] != updated:
                    hypotheses.append(updated)
                state.current_hypothesis = updated.model_dump()
            reflection_delta_accepts += 1
            trace.record("REFLECTION_DELTA_ACCEPTED", {
                "trigger": trigger,
                "accepted_fields": sorted(set(accepted_fields)),
                "new_obligation_ids": new_obligation_ids,
                "runtime_hypothesis_version": len(hypotheses),
                "next_action_constraint": reflection_constraint,
            })
            return True

        def mark_blocked_capability(tool_name: str) -> None:
            nonlocal obligation_blocked_count
            if not hypotheses:
                return
            current = hypotheses[-1]
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
            obligation_state.clear()
            obligation_state.update({item.id: item for item in updated_obligations})
            hypotheses[-1] = current
            state.current_hypothesis = current.model_dump()
            trace.record("HYPOTHESIS_UPDATED", {
                "reason": "capability_blocked",
                "hypothesis": current.model_dump(),
            })

        def targeted_contract_repair(tool_name: str, arguments: dict[str, Any],
                                     error: dict[str, Any], *,
                                     logical_repair_index: int = 1) -> tuple[dict[str, Any] | None, dict[str, Any]]:
            """Repair only missing executable fields, once per bounded budget.

            This call is deliberately not a second Planner. It receives the
            rejected argument object, the live Pydantic schema, and compact
            runtime-owned state. It may add a required value only when the
            provider can identify it from that state; it never fabricates an
            Evidence ID, permission, or benchmark label.
            """
            nonlocal llm_calls, contract_repair_calls
            spec = next((item for item in tools.specs() if item.name == tool_name), None)
            if spec is None:
                return None, {"status": "not_attempted", "reason": "unknown_tool"}
            required = tuple(
                name for name, field in spec.args_model.model_fields.items()
                if field.is_required()
            )
            missing = tuple(
                name for name in required
                if name not in arguments or arguments.get(name) is None
            )
            if not missing:
                return None, {"status": "not_attempted", "reason": "not_missing_required_field"}
            if logical_repair_index > self.config.max_tool_contract_repairs:
                return None, {"status": "not_attempted", "reason": "repair_budget_exhausted"}
            provider = self.llm
            if not hasattr(provider, "complete_json"):
                return None, {"status": "not_attempted", "reason": "provider_has_no_json_api"}

            contract_repair_calls += 1
            trace.record("TOOL_CONTRACT_REPAIR_STARTED", {
                "tool": tool_name,
                "repair_index": contract_repair_calls,
                "missing_fields": list(missing),
            })
            known_targets = sorted({
                str(item.target).strip() for item in incident_evidence()
                if str(item.target).strip()
            })
            repair_system = (
                "You are a bounded tool-argument repairer for a read-only incident Agent. "
                "Return one JSON object containing only missing required tool arguments. "
                "Use a value only when it is explicitly present and unambiguous in the "
                "provided structured state or known entity candidates. If it is ambiguous "
                "or absent, return an empty object. Do not re-plan, add optional fields, "
                "invent service names, infer benchmark labels, or emit prose."
            )
            repair_user = json.dumps({
                "tool_name": tool_name,
                "rejected_arguments": arguments,
                "validation_error": error,
                "missing_required_fields": list(missing),
                "required_schema": spec.args_model.model_json_schema(),
                "current_hypothesis": state.current_hypothesis,
                "known_entity_candidates": known_targets,
            }, ensure_ascii=False, default=str)
            try:
                admit("contract_repair", count_llm=True)
                timeout = provider_timeout(min(self.config.planner_llm_timeout_seconds, 30.0))
                repaired = self._provider_call(
                    complete_json_compat,
                    provider, repair_system, repair_user,
                    model=self.model or None,
                    logical_timeout_seconds=timeout,
                    hard_timeout_seconds=timeout,
                    state_source=provider,
                    state_attributes=("last_usage",),
                )
                add_usage(provider, stage="contract_repair")
            except Exception as exc:
                result = {
                    "status": "failed",
                    "failure_type": type(exc).__name__,
                    "message": str(exc),
                }
                trace.record("TOOL_CONTRACT_REPAIR_RESULT", {
                    "tool": tool_name, "repair_index": contract_repair_calls, **result,
                })
                return None, result
            if not isinstance(repaired, dict):
                result = {"status": "rejected", "reason": "repair_result_not_object"}
                trace.record("TOOL_CONTRACT_REPAIR_RESULT", {
                    "tool": tool_name, "repair_index": contract_repair_calls, **result,
                })
                return None, result
            repaired_fields = {
                name: repaired[name] for name in missing if name in repaired
            }
            merged = {**arguments, **repaired_fields}
            validated, repaired_error = tools.validate_arguments(tool_name, merged)
            result = {
                "status": "accepted" if not repaired_error else "rejected",
                "returned_fields": sorted(repaired_fields),
                "normalized_arguments": dict(validated or merged),
                "validation_error": repaired_error,
            }
            trace.record("TOOL_CONTRACT_REPAIR_RESULT", {
                "tool": tool_name, "repair_index": contract_repair_calls, **result,
            })
            return (validated, result) if not repaired_error else (None, result)

        def trigger_reflection(reason: str, *, review_feedback_text: str = "") -> str:
            nonlocal reflection_calls, reflection_feedback, reflection_trigger, schema_repair_count
            nonlocal llm_calls, reflection_no_delta_count, consecutive_reflection_no_delta
            if not self.config.enable_reflection or self.config.max_reflection_calls == 0:
                return ""
            if consecutive_reflection_no_delta >= self.config.max_consecutive_reflection_calls:
                trace.record("REFLECTION_BLOCKED", {
                    "trigger": reason,
                    "reason": "max_consecutive_reflection_without_runtime_delta",
                    "max_consecutive_reflection_calls": self.config.max_consecutive_reflection_calls,
                })
                transition_phase("PLAN", "reflection_blocked_without_delta")
                return ""
            # Test doubles and old compatibility providers may only implement
            # native planner calls. They simply cannot serve the optional role.
            provider = getattr(reflection_agent, "llm", None)
            if not hasattr(provider, "complete_json") and not hasattr(provider, "complete_structured"):
                trace.record("REFLECTION_SKIPPED", {
                    "reason": "provider_has_no_json_reflection_api", "trigger": reason,
                })
                return ""
            current = hypotheses[-1] if hypotheses else None
            signature = json.dumps({
                "trigger": reason,
                "hypothesis": self._material_hypothesis_fingerprint(current) if current else None,
                "evidence": sorted(item.evidence_id for item in evidence_memory.pinned),
                "review": review_feedback_text,
            }, sort_keys=True, ensure_ascii=False, default=str)
            if signature in reflection_signatures:
                trace.record("REFLECTION_SKIPPED", {
                    "reason": "duplicate_trigger_signature", "trigger": reason,
                })
                return ""
            if reflection_calls >= self.config.max_reflection_calls:
                trace.record("REFLECTION_SKIPPED", {
                    "reason": "max_reflection_calls", "trigger": reason,
                })
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
                "relevant_evidence": [item.model_dump() for item in incident_evidence()
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
            try:
                admit("reflection", count_llm=True)
                timeout = provider_timeout(self.config.planner_llm_timeout_seconds)
                feedback = self._provider_call(
                    reflection_agent.reflect, snapshot,
                    logical_timeout_seconds=timeout,
                    prompt_budget=dynamic_budget,
                    hard_timeout_seconds=timeout,
                    state_source=reflection_agent,
                    state_attributes=(
                        "last_usage", "last_call_count", "last_prompt_breakdown",
                        "last_prompt_breakdowns",
                        "last_normalization_actions", "last_metadata_drops",
                        "last_metadata_warnings", "last_failure_type",
                    ),
                )
                add_usage(
                    reflection_agent, stage="reflection",
                    prompt_breakdown=getattr(reflection_agent, "last_prompt_breakdown", None),
                )
                extra_calls = max(0, int(getattr(reflection_agent, "last_call_count", 1)) - 1)
                if extra_calls:
                    llm_calls += extra_calls
                    schema_repair_count += extra_calls
                    if llm_calls > self.config.max_llm_calls:
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
                reflection_feedback = json.dumps(feedback.model_dump(), ensure_ascii=False)
                delta_accepted = apply_reflection_delta(feedback, reason)
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
                    **feedback.model_dump(),
                })
                transition_phase("PLAN", "reflection_completed", delta_accepted=delta_accepted)
                return reflection_feedback
            except RunBudgetAdmissionExceeded as exc:
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
                transition_phase("PLAN", "reflection_rejected_schema")
                trace.record("REFLECTION_REJECTED_SCHEMA", {
                    "call": reflection_calls, "trigger": reason,
                    "error_type": type(exc).__name__, "message": str(exc),
                })
                # A previous successful reflection remains visible as control
                # context, but it is not fresh feedback for this trigger. Do
                # not let a failed/timeout call reset the progress guard.
                return ""

        def hypothesis_can_finalize(hypothesis: IncidentHypothesis | None,
                                    known_evidence_ids: set[str]) -> bool:
            """Return the run-time completion predicate for this experiment arm.

            Baseline keeps the same typed candidate and review protocol, but
            deliberately omits the optimized evidence-sufficiency gate.  It
            still requires enough known evidence IDs for the shared output
            schema and evaluator to remain meaningful.
            """
            if hypothesis is None:
                return False
            if self.config.features.finalization_gate:
                return hypothesis.can_finalize(known_evidence_ids)
            return len(set(hypothesis.supporting_evidence_ids)) >= 2

        def best_candidate_from_hypothesis() -> RootCauseCandidate | None:
            if not hypotheses:
                return None
            hypothesis = hypotheses[-1]
            known = {item.evidence_id for item in evidence_memory.pinned}
            if not hypothesis_can_finalize(hypothesis, known):
                return None
            return self._candidate_from_hypothesis(hypothesis)

        def finalize_candidate(candidate: RootCauseCandidate, source: str,
                               *, hypothesis: IncidentHypothesis | None = None) -> RootCauseCandidate:
            """The only candidate-producing boundary before Final Review."""
            transition_phase("FINALIZE", source)
            known_evidence_ids = {item.evidence_id for item in evidence_memory.pinned}
            if any(not evidence_id.startswith("ev-") for evidence_id in candidate.evidence_ids):
                raise RuntimeError("final candidate may cite only ev-* Evidence IDs")
            if len(set(candidate.evidence_ids)) < 2:
                raise RuntimeError("final candidate requires two distinct Evidence IDs")
            if not set(candidate.evidence_ids).issubset(known_evidence_ids):
                raise RuntimeError("final candidate cites unknown evidence")
            candidate_ids = set(candidate.evidence_ids)

            # Claim/causal metadata is an optional explanation of the frozen
            # Hypothesis.  Providers sometimes copy a broader working-set
            # citation list into these fields even though the Candidate core
            # cites a narrower, validated set.  Do not turn that serialization
            # drift into a runtime failure, and never promote the extra IDs:
            # retain only metadata whose citations are already in the Candidate
            # Evidence projection.  The Evidence and Review gates below remain
            # unchanged.
            dropped_mapping_count = 0
            dropped_causal_count = 0
            retained_mappings = []
            for mapping in candidate.claim_evidence_mapping:
                ids = set(mapping.evidence_ids)
                if ids and ids.issubset(candidate_ids) and ids.issubset(known_evidence_ids):
                    retained_mappings.append(mapping)
                else:
                    dropped_mapping_count += 1
            retained_causal = []
            for link in candidate.causal_chain_summary:
                ids = set(link.evidence_ids)
                if ids and ids.issubset(candidate_ids) and ids.issubset(known_evidence_ids):
                    retained_causal.append(link)
                else:
                    dropped_causal_count += 1
            if (
                dropped_mapping_count
                or dropped_causal_count
                or len(retained_mappings) != len(candidate.claim_evidence_mapping)
                or len(retained_causal) != len(candidate.causal_chain_summary)
            ):
                candidate = candidate.model_copy(update={
                    "claim_evidence_mapping": tuple(retained_mappings),
                    "causal_chain_summary": tuple(retained_causal),
                })
                trace.record("FINALIZATION_METADATA_NORMALIZED", {
                    "source": source,
                    "dropped_claim_mappings": dropped_mapping_count,
                    "dropped_causal_links": dropped_causal_count,
                    "reason": "optional structured metadata cited Evidence outside frozen Candidate",
                })
            mapping_ids = {
                evidence_id
                for mapping in candidate.claim_evidence_mapping
                for evidence_id in mapping.evidence_ids
            }
            causal_ids = {
                evidence_id
                for link in candidate.causal_chain_summary
                for evidence_id in link.evidence_ids
            }
            structured_ids = mapping_ids | causal_ids
            if any(not evidence_id.startswith("ev-") or evidence_id not in known_evidence_ids
                   for evidence_id in structured_ids):
                raise RuntimeError("final candidate structured reasoning cites unknown evidence")
            if not mapping_ids.issubset(candidate_ids) or not causal_ids.issubset(candidate_ids):
                raise RuntimeError("final candidate structured reasoning cites evidence outside candidate")
            # Older native providers do not yet emit the optional structured
            # fields. Create a minimal auditable projection at this boundary;
            # this is not a chain-of-thought copy and keeps the old contract
            # compatible while making Review see a causal structure.
            if not candidate.claim_evidence_mapping:
                candidate = RootCauseCandidate.model_validate({
                    **candidate.model_dump(),
                    "claim_evidence_mapping": (
                        {
                            "claim": f"{candidate.component}: {candidate.fault}",
                            "evidence_ids": tuple(candidate.evidence_ids),
                        },
                    ),
                })
            if not candidate.causal_chain_summary:
                candidate = RootCauseCandidate.model_validate({
                    **candidate.model_dump(),
                    "causal_chain_summary": (
                        {
                            "cause": candidate.fault,
                            "effect": candidate.mechanism,
                            "evidence_ids": tuple(candidate.evidence_ids),
                        },
                    ),
                })
            if hypothesis is not None:
                source_obligation = next(
                    (
                        item for item in hypothesis.verification_obligations
                        if _is_source_mechanism_obligation(item)
                    ),
                    None,
                )
                if source_obligation is not None:
                    source_capability_gaps = []
                    if not source_workspace_available:
                        source_capability_gaps.append("source_workspace")
                    if "application_code" not in case.evidence_sources:
                        source_capability_gaps.append("application_code")
                    if "read_file" not in available_code_tools:
                        source_capability_gaps.append("read_file")
                    source_capability_available = not source_capability_gaps
                    cited_source_ids = tuple(
                        evidence_id for evidence_id in source_evidence_ids()
                        if evidence_id in set(source_obligation.supporting_evidence_ids)
                    )
                    source_state_valid = (
                        source_obligation.status == "SATISFIED" and bool(cited_source_ids)
                        if source_capability_available
                        else source_obligation.status == "NOT_APPLICABLE_BY_CAPABILITY"
                    )
                    if not source_state_valid:
                        trace.record("FINALIZATION_REJECTED", {
                            "source": source,
                            "reason": "source_obligation_not_runtime_validated",
                            "obligation": source_obligation.model_dump(),
                            "capability_available": source_capability_available,
                            "capability_gaps": source_capability_gaps,
                            "cited_source_evidence_ids": list(cited_source_ids),
                        })
                        raise RuntimeError(
                            "source verification obligation is not Runtime-validated"
                        )
                if self.config.features.finalization_gate and not hypothesis.can_finalize(known_evidence_ids):
                    raise RuntimeError("final candidate is not supported by a complete hypothesis")
                omitted_support = tuple(
                    evidence_id for evidence_id in candidate.evidence_ids
                    if evidence_id not in hypothesis.supporting_evidence_ids
                )
                if omitted_support:
                    # Native tool calls carry two related representations of the
                    # same state: the candidate's cited Evidence and the current
                    # Hypothesis support set. Providers can omit an otherwise valid
                    # candidate citation from the latter. Reconcile only known
                    # ev-* IDs at this single deterministic boundary; unknown IDs
                    # have already failed closed above.
                    merged_support = tuple(dict.fromkeys(
                        (*hypothesis.supporting_evidence_ids, *omitted_support)
                    ))
                    reconciled = hypothesis.model_copy(update={
                        "supporting_evidence_ids": merged_support,
                    })
                    if self.config.features.finalization_gate and not reconciled.can_finalize(known_evidence_ids):
                        raise RuntimeError("final candidate is not supported by a complete hypothesis")
                    if hypotheses and hypotheses[-1] == hypothesis:
                        hypotheses[-1] = reconciled
                    state.current_hypothesis = reconciled.model_dump()
                    trace.record("FINALIZATION_HYPOTHESIS_RECONCILED", {
                        "source": source,
                        "candidate_evidence_ids": list(candidate.evidence_ids),
                        "omitted_supporting_evidence_ids": list(omitted_support),
                        "supporting_evidence_ids": list(merged_support),
                    })
            trace.record("FINALIZATION_VALIDATED", {
                "source": source,
                "candidate": candidate.model_dump(),
                "known_evidence_count": len(known_evidence_ids),
            })
            trace.record("ROOT_CAUSE_CANDIDATE", {
                **candidate.model_dump(), "source": source,
            })
            return candidate

        def investigate(step_limit: int, tool_limit: int, review_feedback: str = "",
                        review_baseline: RootCauseCandidate | None = None,
                        review_evidence_ids: frozenset[str] = frozenset(),
                        review_hypothesis: IncidentHypothesis | None = None) -> RootCauseCandidate:
            nonlocal active_skill, evidence_sufficient_step, evidence_sufficient_tool_calls
            nonlocal converge_read_file_calls
            nonlocal final_candidate_step, termination_reason
            nonlocal duplicate_calls
            nonlocal planner_contract_retry_used
            nonlocal prior_context
            nonlocal llm_calls, planner_calls
            nonlocal consecutive_reflection_no_delta
            nonlocal tool_contract_repairs
            start_steps, start_tools = state.step, state.tool_calls
            prompt_compaction_attempts = 0

            def progress_marker() -> ProgressMarker:
                current = hypotheses[-1] if hypotheses else None
                hypothesis_identity = (
                    json.dumps(current.model_dump(), ensure_ascii=False, sort_keys=True, default=str)
                    if current is not None else ""
                )
                return ProgressMarker(
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

            def visible_tool_names() -> tuple[str, ...] | None:
                """Expose a capability/stage slice without changing authority.

                The registry remains the permission and validation authority.
                Capability state is applied on the initial turn as well; a
                later Skill transition can further reduce the already-valid
                surface without changing execution authority.
                """
                allowed = set(capabilities.planner_tool_names(tools))
                if not self.config.features.lightweight_skills or not active_skill:
                    return tuple(sorted(allowed))
                skill_spec = INCIDENT_SKILLS.get(active_skill)
                names = set(skill_spec.suggested_tools if skill_spec else ())
                names.update(available_code_tools)
                names.add("finalize_diagnosis")
                if knowledge_enabled:
                    names.add("knowledge_retrieval")
                return tuple(sorted(names.intersection(allowed)))

            def ensure_review_progress(candidate: RootCauseCandidate,
                                        current_hypothesis: IncidentHypothesis | None) -> None:
                if review_baseline is None or current_hypothesis is None:
                    return
                current_evidence_ids = {
                    item.evidence_id for item in evidence_memory.pinned
                }
                if self._review_progress_made(
                        review_baseline, candidate, review_evidence_ids,
                        current_evidence_ids, review_hypothesis, current_hypothesis):
                    return
                trace.record("REVIEW_REFINALIZE_BLOCKED", {
                    "step": state.step,
                    "reason": "review_feedback_has_no_semantic_progress",
                    "baseline_candidate": review_baseline.model_dump(),
                    "candidate": candidate.model_dump(),
                    "baseline_evidence_ids": sorted(review_evidence_ids),
                    "current_evidence_ids": sorted(current_evidence_ids),
                })
                raise ReviewProgressRequired(
                    "Review feedback was not addressed before re-finalization"
                )

            while state.step - start_steps < step_limit:
                progress_before_action = progress_marker()
                tool_budget_remaining = max(0, tool_limit - (state.tool_calls - start_tools))
                agent_steps_remaining = max(0, step_limit - (state.step - start_steps))
                runtime_budget = budget_snapshot("investigation_loop")
                if runtime_budget.phase == "finalize":
                    fallback = best_candidate_from_hypothesis()
                    if fallback is None:
                        raise RuntimeError("finalization reserve reached without a complete hypothesis")
                    ensure_review_progress(fallback, hypotheses[-1] if hypotheses else None)
                    termination_reason = "finalization_reserve_sufficient_hypothesis"
                    return finalize_candidate(
                        fallback, termination_reason, hypothesis=hypotheses[-1],
                    )
                if state.tool_calls >= self.config.max_tool_calls:
                    fallback = best_candidate_from_hypothesis()
                    if fallback is None:
                        raise RuntimeError("max_tool_calls exceeded before finalization")
                    ensure_review_progress(fallback, hypotheses[-1] if hypotheses else None)
                    termination_reason = "max_tool_calls_sufficient_hypothesis"
                    return finalize_candidate(
                        fallback, termination_reason, hypothesis=hypotheses[-1],
                    )
                if (
                    evidence_sufficient_tool_calls is not None
                    and state.tool_calls - evidence_sufficient_tool_calls
                    >= self.config.max_post_sufficiency_tool_calls
                ):
                    fallback = best_candidate_from_hypothesis()
                    if fallback is not None:
                        termination_reason = "post_sufficiency_tool_budget"
                        trace.record("BUDGET_AWARE_CONVERGENCE", {
                            "step": state.step,
                            "tool_calls": state.tool_calls,
                            "evidence_sufficient_step": evidence_sufficient_step,
                            "evidence_sufficient_tool_calls": evidence_sufficient_tool_calls,
                            "post_sufficiency_tool_calls": (
                                state.tool_calls - evidence_sufficient_tool_calls
                            ),
                            "limit": self.config.max_post_sufficiency_tool_calls,
                            "reason": "bounded verification window ended",
                        })
                        ensure_review_progress(fallback, hypotheses[-1])
                        return finalize_candidate(
                            fallback, termination_reason, hypothesis=hypotheses[-1],
                        )
                dynamic_budget.set_run_usage(
                    tokens_used=prompt_tokens + completion_tokens,
                    llm_calls_used=llm_calls,
                    cost_used=run_cost,
                )
                pressure_before_planner = dynamic_budget.decision(
                    "planner_admission", tokens_used=prompt_tokens,
                    llm_calls_used=llm_calls, cost_used=run_cost,
                )
                if pressure_before_planner.state is BudgetState.HARD_PRESSURE:
                    fallback = best_candidate_from_hypothesis()
                    if fallback is not None:
                        termination_reason = "token_budget_sufficient_hypothesis"
                        trace.record("BUDGET_AWARE_CONVERGENCE", {
                            "step": state.step,
                            "tokens_used": prompt_tokens + completion_tokens,
                            "remaining_run_tokens": pressure_before_planner.remaining_run_tokens,
                            "state": pressure_before_planner.state.value,
                            "reason": pressure_before_planner.reason,
                        })
                        ensure_review_progress(fallback, hypotheses[-1])
                        return finalize_candidate(
                            fallback, termination_reason, hypothesis=hypotheses[-1],
                        )
                dynamic_budget.set_run_usage(
                    tokens_used=prompt_tokens + completion_tokens,
                    llm_calls_used=llm_calls,
                    cost_used=run_cost,
                )
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
                        "hypothesis": hypotheses[-1].model_dump() if hypotheses else None,
                    }),
                    verified_evidence_tokens=dynamic_budget.estimate_json_tokens(current_evidence_payload),
                    required_control_tokens=dynamic_budget.estimate_json_tokens({
                        "actions": actions[-8:], "rejected_actions": rejected_actions[-4:],
                        "obligations": hypotheses[-1].verification_obligations if hypotheses else (),
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
                control_state = self._planner_control_state(
                    case, hypotheses, actions, review_feedback,
                    tool_budget_remaining=tool_budget_remaining,
                    agent_steps_remaining=agent_steps_remaining,
                    llm_calls_remaining=max(0, self.config.max_llm_calls - llm_calls),
                    rejected_actions=rejected_actions,
                    available_evidence_sources=case.evidence_sources,
                    source_workspace_available=source_workspace_available,
                    available_code_tools=available_code_tools,
                    capability_states=capabilities.states,
                    visible_tool_names=visible_tool_names(),
                    source_evidence_ids=source_evidence_ids(),
                    reflection_feedback=reflection_feedback,
                    tool_health=tool_circuit.summary(),
                    reflection_trigger=reflection_trigger,
                    next_action_constraint=reflection_constraint,
                    entities=entities,
                    capabilities=capabilities,
                    router_decision=router_decision,
                    prior_context=bounded_prior,
                    compact=self.config.features.planner_state_envelope,
                )
                exposed_tools = visible_tool_names()
                if exposed_tools is not None and (active_skill or state.step > 0):
                    trace.record("TOOL_EXPOSURE_SLICE", {
                        "step": state.step + 1,
                        "active_skill": active_skill,
                        "tools": list(exposed_tools),
                        "reason": "stage_and_capability_aware_provider_catalog",
                    })
                control_prefix = (
                    ("PLANNER_STATE:\n" if self.config.features.planner_state_envelope
                     else "AGENT_CONTROL_STATE:\n")
                    + json.dumps(control_state, ensure_ascii=False)
                    + "\n\nDIAGNOSTIC_CONTEXT:\n"
                )
                dynamic_context_tokens = dynamic_budget.context_token_budget(
                    state=dynamic_state.state,
                )
                context_result = context_manager.build(
                    state, evidence_memory, observation_store,
                    max_context_chars=(
                        None if dynamic_context_tokens is not None else self.config.max_context_chars
                    ),
                    max_context_tokens=dynamic_context_tokens,
                    token_estimator=dynamic_budget.estimate_tokens,
                    max_steps=step_limit, max_tool_calls=tool_limit,
                    include_agent_control_state=False,
                    external_context_chars=len(control_prefix),
                    pressure_state=dynamic_state.state.value,
                )
                context = control_prefix + context_result.text
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
                    "control_state_chars": len(control_prefix),
                    "planner_context_chars": len(context),
                    "budget_tokens": context_result.budget_tokens,
                    "used_tokens": context_result.used_tokens,
                    "context_state": dynamic_state.state.value,
                    "input_hard_capacity": dynamic_state.input_hard_capacity,
                    "prior_capacity": prior_capacity,
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
                planner_call_context = context
                planner_phase = budget_snapshot("planner_start").phase
                converge_budget_kwargs: dict[str, Any] = {}
                if planner_phase == "converge":
                    remaining_reads = max(
                        0, CONVERGE_MAX_READ_FILE_CALLS - converge_read_file_calls
                    )
                    planner_call_context += (
                        "\n\nRUNTIME_CONVERGENCE_CONTROL: This is a Runtime-enforced convergence turn. "
                        f"Keep hidden reasoning within {CONVERGE_MAX_REASONING_TOKENS} tokens, "
                        f"use at most {remaining_reads} further read_file call(s), and close the "
                        "current evidence gap or finalize; do not open a new investigation branch."
                    )
                    converge_budget_kwargs["max_output_tokens"] = CONVERGE_MAX_OUTPUT_TOKENS
                    trace.record("CONVERGE_BUDGET_APPLIED", {
                        "step": state.step + 1,
                        "phase": planner_phase,
                        "max_reasoning_tokens": CONVERGE_MAX_REASONING_TOKENS,
                        "provider_completion_cap_tokens": CONVERGE_MAX_OUTPUT_TOKENS,
                        "max_read_file_calls": CONVERGE_MAX_READ_FILE_CALLS,
                        "read_file_calls_used": converge_read_file_calls,
                        "read_file_calls_remaining": remaining_reads,
                        "enforcement": "runtime_adapter_completion_cap_and_action_gate",
                    })
                prompt_budget_retry = False
                contract_retry_for_step = False
                while True:
                    trace.record("PLANNER_CALL_STARTED", {
                        "step": state.step + 1,
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
                        planner_kwargs = dict(converge_budget_kwargs)
                        if planner_kwargs:
                            try:
                                planner_parameters = inspect.signature(planner.propose).parameters
                                accepts_kwargs = any(
                                    item.kind is inspect.Parameter.VAR_KEYWORD
                                    for item in planner_parameters.values()
                                )
                                if (
                                    "max_output_tokens" not in planner_parameters
                                    and not accepts_kwargs
                                ):
                                    planner_kwargs.clear()
                            except (TypeError, ValueError):
                                planner_kwargs.clear()
                        result = self._provider_call(
                            planner.propose,
                            state, planner_call_context,
                            logical_timeout_seconds=planner_timeout,
                            **planner_kwargs,
                            prompt_budget=dynamic_budget,
                            visible_tool_names=exposed_tools,
                            hard_timeout_seconds=planner_timeout,
                            state_source=planner,
                            state_attributes=("last_prompt_breakdown", "last_prompt_breakdowns"),
                        )
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
                            "reason": exc.decision.reason,
                            "budget_before_call": exc.decision.remaining_run_tokens,
                            "estimated_call_cost": exc.decision.breakdown,
                            "budget_after_call": max(
                                0,
                                exc.decision.remaining_run_tokens
                                - exc.decision.breakdown.get("required_run_tokens", 0),
                            ),
                        })
                        fallback = best_candidate_from_hypothesis()
                        if fallback is None:
                            raise
                        termination_reason = "pre_call_budget_admission_sufficient_hypothesis"
                        ensure_review_progress(fallback, hypotheses[-1])
                        return finalize_candidate(
                            fallback, termination_reason, hypothesis=hypotheses[-1],
                        )
                    except PromptBudgetExceeded as exc:
                        # Admission happens before the agent can render its
                        # complete provider payload. This was a local
                        # compaction attempt, not an LLM call.
                        llm_calls = max(0, llm_calls - 1)
                        planner_calls = max(0, planner_calls - 1)
                        prompt_compaction_attempts += 1
                        decision = exc.decision
                        trace.record("CONTEXT_PRESSURE", {
                            "stage": "planner",
                            "context_state": decision.state.value,
                            "reason": decision.reason,
                            "estimated_prompt_tokens": decision.estimated_prompt_tokens,
                            "input_hard_capacity": decision.input_hard_capacity,
                            "breakdown": decision.breakdown,
                            "compaction_applied": True,
                        })
                        if prompt_compaction_attempts > 2:
                            raise RuntimeError(
                                "hard prompt capacity remains exceeded after bounded context compaction"
                            ) from exc
                        prompt_budget_retry = True
                        break
                    except LLMDeadlineExceeded as exc:
                        trace.record("PLANNER_CALL_TIMEOUT", {
                            "step": state.step + 1,
                            "error_type": type(exc).__name__, "message": str(exc),
                        })
                        fallback = best_candidate_from_hypothesis()
                        if fallback is None:
                            raise
                        final_candidate_step = state.step
                        termination_reason = "planner_timeout_sufficient_hypothesis"
                        trace.record("TIMEOUT_FINALIZATION_FALLBACK", {
                            "error_type": type(exc).__name__, "message": str(exc),
                            "hypothesis": hypotheses[-1].model_dump(),
                        })
                        ensure_review_progress(fallback, hypotheses[-1] if hypotheses else None)
                        return finalize_candidate(
                            fallback, "sufficient_hypothesis_timeout_fallback",
                            hypothesis=hypotheses[-1],
                        )
                    except PlannerContractError as exc:
                        planner_failure = {
                            "step": state.step + 1,
                            "error_type": type(exc).__name__, "message": str(exc),
                        }
                        planner_metadata = _safe_planner_metadata_for_trace(exc)
                        if planner_metadata is not None:
                            planner_failure["planner_metadata"] = planner_metadata
                        trace.record("PLANNER_CALL_FAILED", planner_failure)
                        if not contract_retry_for_step:
                            contract_retry_for_step = True
                            planner_contract_retry_used = True
                            trace.record("PLANNER_CONTRACT_RETRY", {
                                "step": state.step + 1,
                                "retry_index": 1,
                                "reason": "core structured Planner contract failure",
                                "planner_metadata": planner_metadata or {},
                            })
                            planner_call_context = (
                                context
                                + "\n\nBOUNDED_CONTRACT_RETRY: The previous Planner response violated the "
                                "structured tool contract. Return exactly one valid native tool call; preserve "
                                "the current semantic state and do not guess missing Evidence IDs. "
                                f"Contract error: {str(exc)}. Use empty strings/lists instead of null for "
                                "optional incident controls. For finalize_diagnosis, include evidence_ids and "
                                "copy the same known ev-* IDs into supporting_evidence_ids."
                            )
                            continue
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
                        raise PlannerContractExhausted(exc) from exc
                    except PromptBudgetExceeded as exc:
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
                        planner_failure = {
                            "step": state.step + 1,
                            "error_type": type(exc).__name__, "message": str(exc),
                        }
                        trace.record("PLANNER_CALL_FAILED", planner_failure)
                        raise
                if prompt_budget_retry:
                    # Re-enter the outer loop so ContextManager can compact
                    # L2/L3 while preserving Critical State and Evidence.
                    continue
                add_usage(
                    result, stage="planner",
                    prompt_breakdown=getattr(planner, "last_prompt_breakdown", None),
                )
                dynamic_budget.clear_context_state_override()
                trace.record("PLANNER_CALL_COMPLETED", {
                    "step": state.step + 1,
                    "prompt_breakdown": dict(planner.last_prompt_breakdown),
                    "prompt_breakdowns": list(getattr(planner, "last_prompt_breakdowns", ()) or ()),
                    "usage": dict(result.response.usage or {}),
                })
                trace.record("PLANNER_RESPONSE_AUDIT", {
                    "step": state.step + 1,
                    "raw_output": dict(getattr(result.response, "raw_output", {}) or {}),
                    "parsed_output": result.response.structured,
                    "normalized_tool_count": len(result.tool_calls),
                    "tool_names": [call.name for call in result.tool_calls],
                })
                state.step += 1
                trace.record("AGENT_STEP", {"step": state.step, "tool_call_count": len(result.tool_calls)})
                for audit in getattr(result, "tool_call_audits", ()):
                    trace.record("PROVIDER_TOOL_CALL_AUDIT", {
                        "step": state.step,
                        "provider": model_capability.provider or type(self.llm).__name__,
                        "model": model_capability.model or self.model,
                        **dict(audit),
                    })
                if planner_phase == "converge":
                    usage = dict(result.response.usage or {})
                    completion_details = usage.get("completion_tokens_details") or {}
                    reasoning_tokens = completion_details.get("reasoning_tokens")
                    if reasoning_tokens is not None and int(reasoning_tokens) > CONVERGE_MAX_REASONING_TOKENS:
                        trace.record("CONVERGE_REASONING_LIMIT_EXCEEDED", {
                            "step": state.step,
                            "reasoning_tokens": int(reasoning_tokens),
                            "max_reasoning_tokens": CONVERGE_MAX_REASONING_TOKENS,
                            "provider_completion_cap_tokens": CONVERGE_MAX_OUTPUT_TOKENS,
                            "tool_call_count": len(result.tool_calls),
                            "reason": "provider usage exceeded Runtime convergence reasoning limit",
                        })
                        raise ConvergenceBudgetExhausted(
                            "convergence planner reasoning limit exceeded"
                        )
                if not result.tool_calls:
                    if budget_snapshot("converge_empty_response").phase == "converge":
                        trace.record("CONVERGE_OUTPUT_CAP_EXHAUSTED", {
                            "step": state.step,
                            "provider_completion_cap_tokens": CONVERGE_MAX_OUTPUT_TOKENS,
                            "max_reasoning_tokens": CONVERGE_MAX_REASONING_TOKENS,
                            "reason": "provider returned no executable action after bounded completion envelope",
                            "hypothesis": hypotheses[-1].model_dump() if hypotheses else None,
                        })
                        raise ConvergenceBudgetExhausted(
                            "convergence planner response contained no executable action"
                        )
                    raise RuntimeError("Diagnosis Agent returned no action")
                if len(result.tool_calls) != len(result.skill_selections):
                    raise RuntimeError("every incident action must carry a Skill selection")
                tool_repairs_this_turn = 0
                tool_contract_replan_requested = False
                for call, selection in zip(result.tool_calls, result.skill_selections):
                    previous_skill = active_skill
                    active_skill = selection.skill
                    trace.record("SKILL_SELECTED", {
                        "step": state.step, "skill": selection.skill, "reason": selection.reason,
                        "current_hypothesis": selection.current_hypothesis,
                        "evidence_gap": selection.evidence_gap, "tool": call.name,
                        "source_mechanism_status": selection.source_mechanism_status,
                        "obligation_id": selection.obligation_id,
                        "expected_information_gain": selection.expected_information_gain,
                    })
                    if selection.compatibility_normalizations:
                        trace.record("PLANNER_COMPATIBILITY_NORMALIZED", {
                            "step": state.step,
                            "tool": call.name,
                            "actions": list(selection.compatibility_normalizations),
                            "reason": "provider-native incident argument compatibility",
                        })
                    if selection.reasoning_metadata_normalizations:
                        trace.record("PLANNER_REASONING_METADATA_NORMALIZED", {
                            "step": state.step,
                            "tool": call.name,
                            "actions": list(selection.reasoning_metadata_normalizations),
                        })
                    if selection.reasoning_metadata_drops or selection.reasoning_metadata_warnings:
                        trace.record("PLANNER_REASONING_METADATA_DROPPED", {
                            "step": state.step,
                            "tool": call.name,
                            "dropped_items": list(selection.reasoning_metadata_drops),
                            "warnings": list(selection.reasoning_metadata_warnings),
                            "reason": "optional structured metadata was not used; flat projections remain authoritative",
                        })
                    suggested_tools = INCIDENT_SKILLS[selection.skill].suggested_tools
                    if call.name != "finalize_diagnosis" and call.name not in suggested_tools:
                        trace.record("SKILL_TOOL_DEVIATION", {
                            "step": state.step, "skill": selection.skill, "tool": call.name,
                            "reason": "tool is outside the Skill's suggested tools; allowed because suggestions are guidance",
                        })
                    if previous_skill != active_skill:
                        trace.record("SKILL_TRANSITION", {
                            "step": state.step, "from": previous_skill or None, "to": active_skill,
                            "reason": selection.reason,
                        })
                    known_evidence_ids = {item.evidence_id for item in evidence_memory.pinned}
                    hypothesis = build_hypothesis(selection)
                    finalization_diagnostics = hypothesis.finalization_diagnostics(known_evidence_ids)
                    if hypothesis.evidence_sufficient and finalization_diagnostics:
                        hypothesis = hypothesis.model_copy(update={"evidence_sufficient": False})
                        trace.record("EVIDENCE_SUFFICIENCY_DOWNGRADED", {
                            "step": state.step,
                            "reason": "structured completion criteria not met",
                            "diagnostics": finalization_diagnostics,
                        })
                    if not hypotheses or hypotheses[-1] != hypothesis:
                        hypotheses.append(hypothesis)
                        trace.record("HYPOTHESIS_STABILITY", {
                            "step": state.step,
                            "stable_rounds": hypothesis.stable_rounds,
                            "material_fingerprint": self._material_hypothesis_fingerprint(hypothesis),
                        })
                    state.current_hypothesis = hypothesis.model_dump()
                    if (
                        hypothesis.contradictions
                        and any(item.blocks_finalization for item in hypothesis.contradictions)
                    ):
                        trigger_reflection("blocking_contradiction")
                    if (
                        len(hypotheses) > 2
                        and self._has_material_hypothesis_instability(
                            previous_hypothesis, hypothesis,
                        )
                    ):
                        trigger_reflection("hypothesis_instability")
                    deterministic_sufficiency = hypothesis_can_finalize(
                        hypothesis, known_evidence_ids,
                    )
                    trace.record("EVIDENCE_SUFFICIENCY_EVALUATED", {
                        "step": state.step,
                        "tool": call.name,
                        "planner_declared": selection.evidence_sufficiency == "sufficient",
                        "evidence_sufficient": bool(deterministic_sufficiency),
                        "supporting_evidence_count": len(hypothesis.supporting_evidence_ids),
                        "required_gap_count": len(hypothesis.required_gap_projection()),
                        "open_critical_obligation_count": sum(
                            1 for item in hypothesis.verification_obligations
                            if item.blocks_finalization and item.status == "OPEN"
                        ),
                        "blocking_contradiction_count": sum(
                            1 for item in hypothesis.contradictions
                            if item.blocks_finalization and item.status == "OPEN"
                        ),
                        "diagnostics": finalization_diagnostics,
                    })
                    # Positive edge only: old traces could contain a false
                    # EVIDENCE_SUFFICIENT payload when the Planner declared
                    # sufficient but the deterministic predicate rejected it.
                    if deterministic_sufficiency and evidence_sufficient_step is None:
                        evidence_sufficient_step = state.step
                        evidence_sufficient_tool_calls = state.tool_calls
                        trace.record("EVIDENCE_SUFFICIENT", {
                            "step": state.step,
                            "evidence_sufficient": True,
                            "planner_declared": selection.evidence_sufficiency == "sufficient",
                            "hypothesis": hypothesis.model_dump(),
                        })
                    validated, error = tools.validate_arguments(call.name, call.arguments)
                    repair_attempted = False
                    repair_result_for_admission = None
                    if error:
                        repair_attempted = True
                        rejection = {
                            "step": state.step,
                            "tool": call.name,
                            "arguments": dict(call.arguments),
                            "reason": "tool_contract_validation",
                            "validation_error": error,
                            "instruction": (
                                "Repair only the rejected tool arguments on the next Planner turn; "
                                "do not invent values or repeat the invalid fingerprint."
                            ),
                        }
                        tool_repairs_this_turn += 1
                        repaired, repair_result = targeted_contract_repair(
                            call.name, dict(call.arguments), error,
                            logical_repair_index=tool_repairs_this_turn,
                        )
                        if repaired is not None:
                            validated = repaired
                            call = type(call)(call.id, call.name, dict(repaired))
                            rejection["repair_status"] = "accepted"
                            repair_result_for_admission = repair_result
                        else:
                            tool_contract_repairs += 1
                            rejection["repair_status"] = repair_result.get("status", "rejected")
                            rejection["repair_index"] = tool_contract_repairs
                            rejected_actions.append(rejection)
                            trace.record("TOOL_CONTRACT_REJECTED", rejection)
                            trace.record("TOOL_CONTRACT_REPAIR_REQUESTED", {
                                "step": state.step,
                                "tool": call.name,
                                "repair_index": tool_contract_repairs,
                                "max_repairs": self.config.max_tool_contract_repairs,
                                "missing_or_invalid_fields": repair_result.get("missing_fields", []),
                                "repair_result": repair_result,
                            })
                            trace.record("REPAIR_FAILED", {
                                "layer": "tool_contract",
                                "logical_turn": state.step,
                                "tool": call.name,
                                "repair_attempts": tool_contract_repairs,
                                "reason": "targeted_tool_contract_repair_rejected",
                            })
                            trace.record("TOOL_CONTRACT_REPAIR_EXHAUSTED", {
                                "step": state.step,
                                "tool": call.name,
                                "repair_index": tool_repairs_this_turn,
                                "max_repairs": self.config.max_tool_contract_repairs,
                            })
                            # A failed repair exits this logical Planner turn.
                            # The next Planner turn may choose a different
                            # action, but it cannot spend a second repair in
                            # this turn.
                            tool_contract_replan_requested = True
                            break
                    trace.record("TOOL_CONTRACT_ADMITTED", {
                        "step": state.step,
                        "tool": call.name,
                        "validation_result": "valid",
                        "validated_arguments": dict(validated),
                        "repair_attempted": repair_attempted,
                        "repair": repair_result_for_admission,
                    })
                    if (
                        budget_snapshot("converge_read_gate").phase == "converge"
                        and call.name == "read_file"
                        and converge_read_file_calls >= CONVERGE_MAX_READ_FILE_CALLS
                    ):
                        rejection = {
                            "step": state.step,
                            "tool": call.name,
                            "reason": "converge_read_file_budget_exhausted",
                            "limit": CONVERGE_MAX_READ_FILE_CALLS,
                            "calls_used": converge_read_file_calls,
                        }
                        trace.record("CONVERGE_READ_FILE_REJECTED", rejection)
                        fallback = best_candidate_from_hypothesis()
                        if fallback is not None:
                            termination_reason = "converge_read_file_budget_sufficient_hypothesis"
                            ensure_review_progress(fallback, hypotheses[-1])
                            return finalize_candidate(
                                fallback, termination_reason, hypothesis=hypotheses[-1],
                            )
                        raise ConvergenceBudgetExhausted(
                            "convergence read_file budget exhausted before a complete hypothesis"
                        )
                    if call.name != "finalize_diagnosis" and state.tool_calls - start_tools >= tool_limit:
                        trace.record("ACTION_REJECTED", {
                            "reason": "tool_budget_exhausted", "tool": call.name,
                            "step": state.step, "tool_calls": state.tool_calls,
                        })
                        continue
                    final_candidate = None
                    re_finalization = call.name == "finalize_diagnosis" and review_baseline is not None
                    if call.name == "finalize_diagnosis":
                        if self.config.features.finalization_gate and not hypothesis.can_finalize(known_evidence_ids):
                            trace.record("REFLECTION_TRIGGER", {
                                "reason": "premature_finalize", "step": state.step,
                            })
                            # Reflection is an optional recovery aid.  Its
                            # absence must never turn an invalid finalization
                            # into an accepted candidate.
                            trigger_reflection("premature_finalize")
                            rejection = {
                                "step": state.step,
                                "tool": call.name,
                                "reason": "premature_finalize",
                                "instruction": "Continue investigation or close the missing critical hypothesis state before finalizing.",
                            }
                            rejected_actions.append(rejection)
                            trace.record("ACTION_REJECTED", rejection)
                            continue
                        final_candidate = self._candidate_from_final_hypothesis(
                            hypothesis, validated,
                            known_evidence_ids=known_evidence_ids,
                        )
                        ensure_review_progress(final_candidate, hypothesis)
                    if (call.name != "finalize_diagnosis"
                            and hypothesis_can_finalize(hypothesis, known_evidence_ids)
                            and not selection.remaining_evidence_need):
                        candidate = self._candidate_from_hypothesis(hypothesis)
                        final_candidate_step = state.step
                        termination_reason = "sufficient_hypothesis"
                        ensure_review_progress(candidate, hypothesis)
                        trace.record("EVIDENCE_SUFFICIENT_FINALIZATION", {
                            "step": state.step, "rejected_tool": call.name,
                            "reason": "no remaining critical evidence need",
                        })
                        return finalize_candidate(
                            candidate, "structured_sufficient_hypothesis", hypothesis=hypothesis,
                        )
                    fingerprint = f"{call.name}:{json.dumps(validated, sort_keys=True, ensure_ascii=False)}"
                    action_probe = ActionProposal(
                        kind=ActionKind.TOOL, skill=selection.skill,
                        reason=selection.reason, tool=call.name, arguments=validated,
                    )
                    allowed, guard_reason = loop_guard.observe_action(action_probe, state)
                    if re_finalization:
                        # The semantic Review guard above has already decided whether
                        # this is a meaningful re-submission. Exact action fingerprints
                        # alone cannot reject the valid case where new Evidence was
                        # acquired while the candidate text stayed unchanged.
                        allowed, guard_reason = True, "review_semantic_progress"
                    if not allowed:
                        trace.record("PROGRESS_GUARD_REJECTED", {
                            "step": state.step, "tool": call.name,
                            "fingerprint": fingerprint, "reason": guard_reason,
                        })
                        raise RuntimeError("duplicate action limit exceeded")
                    if fingerprint in fingerprints and not re_finalization:
                        duplicate_calls += 1
                        state.repeated_actions += 1
                        prior_action = next(
                            (item for item in reversed(actions)
                             if item.get("fingerprint") == fingerprint), None,
                        )
                        prior_observation_id = (
                            prior_action.get("observation_id") if prior_action else None
                        )
                        if prior_observation_id and observation_store.get(prior_observation_id):
                            context_manager.rehydrate(prior_observation_id)
                            state.rehydration_count += 1
                            state.observation_reuse_count += 1
                            trace.record("OBSERVATION_REHYDRATED", {
                                "observation_id": prior_observation_id,
                                "fingerprint": fingerprint,
                                "reason": "duplicate_action_reused_shared_observation",
                            })
                        rejection = {
                            "step": state.step,
                            "tool": call.name,
                            "arguments": validated,
                            "fingerprint": fingerprint,
                            "reason": "duplicate_fingerprint",
                            "action_class": "duplicate",
                            "instruction": "Do not request this action again; use its existing Evidence or choose a different critical check.",
                        }
                        rejected_actions.append(rejection)
                        trace.record("ACTION_REJECTED", rejection)
                        if self.config.features.no_progress_detection:
                            loop_guard.observe_semantic_progress(
                                state,
                                evidence_ids=(item.evidence_id for item in evidence_memory.pinned),
                                hypothesis=hypotheses[-1] if hypotheses else None,
                                obligations=(hypotheses[-1].verification_obligations if hypotheses else ()),
                                contradictions=(hypotheses[-1].contradictions if hypotheses else ()),
                                review_feedback=review_feedback,
                                meaningful_progress=False,
                            )
                        if self.config.features.no_progress_detection and state.no_progress_count >= self.config.max_no_progress:
                            reflected = trigger_reflection("semantic_no_progress")
                            if reflected:
                                loop_guard.no_progress = 0
                                state.no_progress_count = 0
                                continue
                            fallback = best_candidate_from_hypothesis()
                            if fallback is None:
                                raise RuntimeError("semantic no-progress limit exceeded")
                            ensure_review_progress(fallback, hypotheses[-1] if hypotheses else None)
                            termination_reason = "semantic_no_progress_sufficient_hypothesis"
                            return finalize_candidate(
                                fallback, termination_reason, hypothesis=hypotheses[-1],
                            )
                        continue
                    fingerprints.add(fingerprint)
                    action = {
                        "step": state.step, "skill": selection.skill, "skill_reason": selection.reason,
                        "tool": call.name, "arguments": validated, "fingerprint": fingerprint,
                        "obligation_id": selection.obligation_id,
                        "expected_information_gain": selection.expected_information_gain,
                    }
                    actions.append(action)
                    if call.name == "finalize_diagnosis":
                        trace.record("FINALIZATION_PROPOSED", {
                            "step": state.step,
                            "hypothesis_valid": hypothesis_can_finalize(hypothesis, known_evidence_ids),
                        })
                        candidate = final_candidate or RootCauseCandidate.model_validate(validated)
                        final_candidate_step = state.step
                        termination_reason = "explicit_finalize"
                        return finalize_candidate(
                            candidate, "explicit_finalize", hypothesis=hypothesis,
                        )
                    tool = tools.get(call.name)
                    if (
                        budget_snapshot("converge_read_file_execute").phase == "converge"
                        and call.name == "read_file"
                    ):
                        converge_read_file_calls += 1
                    observation = execute_tool(tool, validated, call.name)
                    state.tool_calls += 1
                    state.observations.append(observation)
                    observation_store.add(observation)
                    if observation.error_type == "tool_circuit_open":
                        mark_blocked_capability(call.name)
                    action["observation_id"] = observation.observation_id
                    action["outcome"] = "ok" if observation.ok else "error"
                    action["error_type"] = observation.error_type
                    action["observation_status"] = observation.metadata.get("status")
                    action["semantic_negative"] = observation.metadata.get("semantic_negative")
                    trace.record("TOOL_OBSERVATION", observation)
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
                    if observation.ok:
                        target = self._target(validated)
                        projected = incident_projection.compact_content(observation)
                        shared_evidence = evidence_memory.add_observation(
                            observation,
                            evidence_id=f"ev-{len(evidence_memory.pinned) + 1:03d}",
                            kind=str(observation.metadata.get("context_kind") or "OBSERVATION"),
                            source=call.name, summary=projected, excerpt=projected,
                            target=target,
                            tags=["task_kind:incident"],
                            enforce_admission=self.config.features.evidence_lifecycle,
                            deduplicate=self.config.features.evidence_lifecycle,
                        )
                        if shared_evidence is not None:
                            state.no_progress_count = 0
                            consecutive_reflection_no_delta = 0
                            item = IncidentEvidence(
                                evidence_id=shared_evidence.evidence_id, source=shared_evidence.source,
                                target=shared_evidence.target or target,
                                summary=shared_evidence.summary,
                                observation_id=shared_evidence.raw_observation_id or observation.observation_id,
                            )
                            trace.record("EVIDENCE_ADDED", item)
                    else:
                        trace.record("REFLECTION_TRIGGER", {"reason": "no_progress", "step": state.step})
                    progress_after_action = progress_marker()
                    dynamic_progress = progress_made(
                        progress_before_action, progress_after_action,
                    )
                    action["new_evidence_count"] = len(
                        progress_after_action.evidence_ids - progress_before_action.evidence_ids
                    )
                    action["obligation_progress_count"] = max(
                        0,
                        len(progress_before_action.open_obligations)
                        - len(progress_after_action.open_obligations),
                    )
                    action["hypothesis_changed"] = (
                        progress_before_action.hypothesis != progress_after_action.hypothesis
                    )
                    action["information_gain"] = (
                        "high" if action["new_evidence_count"] > 0
                        or action["obligation_progress_count"] > 0
                        or action["hypothesis_changed"] else "none"
                    )
                    # Gold required-group matching is evaluator-only. Keep a
                    # clearly marked obligation proxy in the runtime artifact
                    # instead of leaking evaluator facts into planning.
                    action["required_evidence_group_progress"] = {
                        "status": "obligation_proxy",
                        "progress_count": action["obligation_progress_count"],
                        "new_evidence_count": action["new_evidence_count"],
                        "note": "Gold group matching is evaluator-only",
                    }
                    if not observation.ok:
                        action["action_class"] = "failed"
                    elif action["new_evidence_count"] > 0 or action["obligation_progress_count"] > 0:
                        action["action_class"] = (
                            "necessary" if selection.obligation_id
                            or action["obligation_progress_count"] > 0
                            else "useful_noncritical"
                        )
                    else:
                        action["action_class"] = "zero_information_gain"
                    dynamic_state_after_action = dynamic_budget.decision(
                        "post_tool_progress",
                        tokens_used=prompt_tokens,
                        llm_calls_used=llm_calls,
                        cost_used=run_cost,
                    )
                    trace.record("DYNAMIC_PROGRESS_CHECK", {
                        "step": state.step,
                        "tool": call.name,
                        "progress": dynamic_progress,
                        "context_state": dynamic_state_after_action.state.value,
                        "evidence_count": len(progress_after_action.evidence_ids),
                        "open_obligation_count": len(progress_after_action.open_obligations),
                        "blocking_contradiction_count": len(progress_after_action.blocking_contradictions),
                        "component_count": len(progress_after_action.components),
                        "obligation_id": selection.obligation_id,
                        "expected_information_gain": selection.expected_information_gain,
                        "new_evidence_count": action["new_evidence_count"],
                        "obligation_progress_count": action["obligation_progress_count"],
                        "hypothesis_changed": action["hypothesis_changed"],
                        "information_gain": action["information_gain"],
                        "action_class": action["action_class"],
                        "required_evidence_group_progress": action[
                            "required_evidence_group_progress"
                        ],
                    })
                    if self.config.features.no_progress_detection:
                        loop_guard.observe_semantic_progress(
                            state,
                            evidence_ids=(item.evidence_id for item in evidence_memory.pinned),
                            hypothesis=hypotheses[-1] if hypotheses else None,
                            obligations=(hypotheses[-1].verification_obligations if hypotheses else ()),
                            contradictions=(hypotheses[-1].contradictions if hypotheses else ()),
                            review_feedback=review_feedback,
                            meaningful_progress=action["action_class"] not in {
                                "zero_information_gain", "failed",
                            },
                        )
                    if self.config.features.no_progress_detection and state.no_progress_count >= self.config.max_no_progress:
                        reflected = trigger_reflection("semantic_no_progress")
                        if reflected:
                            loop_guard.no_progress = 0
                            state.no_progress_count = 0
                            continue
                        fallback = best_candidate_from_hypothesis()
                        if fallback is None:
                            raise RuntimeError("semantic no-progress limit exceeded")
                        ensure_review_progress(fallback, hypotheses[-1] if hypotheses else None)
                        termination_reason = "semantic_no_progress_sufficient_hypothesis"
                        return finalize_candidate(
                            fallback, "semantic_no_progress_sufficient_hypothesis",
                            hypothesis=hypotheses[-1],
                        )
                if tool_contract_replan_requested:
                    trace.record("REPAIR_FAILED_REPLAN", {
                        "logical_turn": state.step,
                        "reason": "tool_contract_repair_failed; return to PLAN",
                    })
                    transition_phase("PLAN", "tool_contract_repair_failed")
                    continue
            fallback = best_candidate_from_hypothesis()
            if fallback is not None:
                termination_reason = "step_budget_sufficient_hypothesis"
                ensure_review_progress(fallback, hypotheses[-1] if hypotheses else None)
                return finalize_candidate(
                    fallback, termination_reason, hypothesis=hypotheses[-1],
                )
            raise RuntimeError("incident step budget exhausted before finalization")

        def apply_review_rejection(decision: ReviewDecision) -> tuple[bool, bool]:
            """Convert Review findings into a bounded recovery decision.

            Returns ``(recoverable, contradiction_requires_reflection)``. Review
            never chooses tools or re-investigates; it only supplies grounding
            findings for this deterministic Runtime transition.
            """
            nonlocal obligation_created_count, blocking_contradiction_count
            current = hypotheses[-1] if hypotheses else None
            claims = tuple(dict.fromkeys(
                " ".join(str(item).split()).strip()
                for item in (*decision.missing_evidence, *decision.causal_gaps)
                if " ".join(str(item).split()).strip()
            ))
            next_obligations = dict(obligation_state)
            created: list[str] = []
            for claim in claims:
                existing = next((
                    item for item in next_obligations.values()
                    if " ".join(item.claim.split()).casefold() == claim.casefold()
                ), None)
                if existing is not None:
                    continue
                obligation = VerificationObligation(
                    id=_obligation_id(claim),
                    claim=claim,
                    critical=True,
                    status="OPEN",
                    supporting_evidence_ids=(),
                )
                next_obligations[obligation.id] = obligation
                created.append(obligation.id)
                obligation_created_count += 1
                trace.record("OBLIGATION_CREATED", {
                    **obligation.model_dump(),
                    "reason": "review_missing_evidence",
                    "review_round": review_rounds,
                })

            if next_obligations != obligation_state:
                obligation_state.clear()
                obligation_state.update(next_obligations)
                if current is not None:
                    updated = current.model_copy(update={
                        "verification_obligations": tuple(next_obligations.values()),
                        "required_gaps": tuple(
                            item.claim for item in next_obligations.values()
                            if item.blocks_finalization
                        ),
                        "evidence_sufficient": False,
                    })
                    hypotheses[-1] = updated
                    state.current_hypothesis = updated.model_dump()
                trace.record("REVIEW_RECOVERY_OBLIGATIONS", {
                    "round": review_rounds,
                    "obligation_ids": created,
                    "reason": "recoverable_missing_evidence",
                })

            blocking = tuple(dict.fromkeys(
                (*decision.blocking_contradictions, *decision.contradictions)
            ))
            contradiction_requires_reflection = bool(blocking)
            recoverability = decision.recoverability
            recoverable = bool(
                not recoverability.unrecoverable
                and (
                    recoverability.recoverable
                    or claims
                    or decision.targeted_followup
                    or decision.suggested_investigation
                    or contradiction_requires_reflection
                )
            )
            trace.record("REVIEW_RECOVERY_DECISION", {
                "round": review_rounds,
                "recoverable": recoverable,
                "unrecoverable": recoverability.unrecoverable,
                "missing_evidence_count": len(claims),
                "blocking_contradiction_count": len(blocking),
                "targeted_followup": decision.targeted_followup or decision.suggested_investigation,
            })
            return recoverable, contradiction_requires_reflection

        candidate = None
        decision = None
        status = "FAILED"
        error_type = error_message = ""
        try:
            candidate = investigate(self.config.max_steps, self.config.max_tool_calls)
            if not self.config.enable_review:
                status = "PASS"
                termination_reason = "review_disabled"
            while self.config.enable_review and review_rounds < self.config.max_review_rounds:
                review_rounds += 1
                transition_phase("REVIEW", "candidate_finalized", review_round=review_rounds)
                review_usage_accounted = False
                try:
                    admit("review", count_llm=True)
                    review_started_at = time.monotonic()
                    review_attempt_state = multiprocessing.Value("i", 0)

                    def on_review_attempt_started(attempt: dict[str, Any]) -> None:
                        review_attempt_state.value = max(
                            review_attempt_state.value,
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
                    blocking_contradictions = tuple(
                        item.model_dump()
                        for item in (hypotheses[-1].contradictions if hypotheses else ())
                        if item.blocks_finalization
                    )
                    current_hypothesis = hypotheses[-1] if hypotheses else None
                    source_context = self._source_mechanism_coverage(
                        case,
                        current_hypothesis,
                        source_workspace_available=source_workspace_available,
                        available_code_tools=available_code_tools,
                        source_evidence_ids=source_evidence_ids(),
                    )
                    open_critical_obligations = tuple(
                        item.model_dump()
                        for item in (
                            current_hypothesis.verification_obligations
                            if current_hypothesis else ()
                        )
                        if item.blocks_finalization
                    )
                    try:
                        model_decision = self._provider_call(
                            reviewer.review,
                            candidate, incident_evidence(),
                            blocking_contradictions=blocking_contradictions,
                            source_mechanism_context=source_context,
                            open_critical_obligations=open_critical_obligations,
                            logical_timeout_seconds=review_timeout,
                            on_attempt_started=on_review_attempt_started,
                            hard_timeout_seconds=review_timeout,
                            state_source=reviewer,
                            state_attributes=(
                                "last_usage", "last_schema_repaired", "last_repair_attempted",
                                "last_deterministic_normalized", "last_schema_error", "last_call_count",
                                "last_attempts", "last_failure_type",
                                "last_prompt_breakdown",
                                "last_prompt_breakdowns",
                            ),
                            prompt_budget=dynamic_budget,
                        )
                    except RunBudgetAdmissionExceeded as exc:
                        # Review is the terminal semantic gate. A run-budget
                        # admission rejection must remain INCONCLUSIVE and
                        # must never be misreported as a Review PASS.
                        llm_calls = max(0, llm_calls - 1)
                        review_calls = max(0, review_calls - 1)
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
                            reviewer, review_attempt_state.value,
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
                    review_gate_gaps = []
                    if open_critical_obligations:
                        review_gate_gaps.append(
                            "A critical verification obligation remains open."
                        )
                    if (
                        source_context.get("application_code_declared")
                        and source_context.get("workspace_available")
                        and source_context.get("status") in {"unknown", "gap"}
                        and not source_context.get("cited_source_backed_evidence_ids")
                    ):
                        review_gate_gaps.append(
                            "The application-level mechanism is not supported by cited source Evidence."
                        )
                    if decision.decision == "PASS" and review_gate_gaps:
                        decision = decision.model_copy(update={
                            "decision": "REJECT",
                            "causal_gaps": tuple(dict.fromkeys(
                                (*decision.causal_gaps, *review_gate_gaps)
                            )),
                            "reason": (
                                "Deterministic Review gate rejected PASS while critical source "
                                "coverage remains unresolved. " + decision.reason
                            ),
                        })
                        trace.record("REVIEW_PASS_DOWNGRADED", {
                            "round": review_rounds,
                            "reason": "critical source verification remains unresolved",
                            "causal_gaps": review_gate_gaps,
                        })
                    if decision.decision != model_decision.decision:
                        trace.record("REVIEW_PASS_DOWNGRADED", {
                            "round": review_rounds,
                            "reason": "PASS response contained structured unsupported or missing evidence",
                            "unsupported_claims": list(model_decision.unsupported_claims),
                            "missing_evidence": list(model_decision.missing_evidence),
                            "contradictions": list(model_decision.contradictions),
                        })
                    if llm_calls > self.config.max_llm_calls:
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
                        "hypothesis": hypotheses[-1].model_dump() if hypotheses else None,
                        "open_obligations": [
                            item.model_dump() for item in (
                                hypotheses[-1].verification_obligations if hypotheses else ()
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
                recoverable, contradiction_requires_reflection = apply_review_rejection(decision)
                trace.record("REVIEW_REJECT", {
                    "round": review_rounds,
                    "recoverable": recoverable,
                    "contradiction_requires_reflection": contradiction_requires_reflection,
                })
                if not recoverable:
                    transition_phase("INCONCLUSIVE", "review_unrecoverable")
                    trace.record("REVIEW_RECOVERY_BLOCKED", {
                        "round": review_rounds,
                        "reason": "review_declared_unrecoverable_or_no_targeted_path",
                    })
                    status = "INCONCLUSIVE"
                    termination_reason = "review_unrecoverable"
                    break
                if review_rounds == self.config.max_review_rounds:
                    status = "INCONCLUSIVE"
                    termination_reason = "review_bound_exhausted"
                    break
                if review_recovery_cycles >= self.config.max_review_recovery_cycles:
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
                rejected_candidate = candidate
                rejected_evidence_ids = frozenset(
                    item.evidence_id for item in evidence_memory.pinned
                )
                rejected_hypothesis = hypotheses[-1] if hypotheses else None
                review_feedback_json = json.dumps(decision.model_dump(), ensure_ascii=False)
                reflection_result = ""
                if contradiction_requires_reflection:
                    trace.record("REFLECTION_TRIGGER", {
                        "reason": "review_reasoning_contradiction", "round": review_rounds,
                    })
                    reflection_result = trigger_reflection(
                        "review_reasoning_contradiction", review_feedback_text=review_feedback_json,
                    )
                else:
                    transition_phase("PLAN", "review_reject_recoverable")
                combined_feedback = json.dumps({
                    "review": decision.model_dump(),
                    "reflection": json.loads(reflection_result) if reflection_result else None,
                }, ensure_ascii=False)
                candidate = investigate(
                    self.config.max_post_reject_steps, self.config.max_post_reject_tool_calls,
                    review_feedback=combined_feedback,
                    review_baseline=rejected_candidate,
                    review_evidence_ids=rejected_evidence_ids,
                    review_hypothesis=rejected_hypothesis,
                )
        except RunBudgetAdmissionExceeded as exc:
            status = "INCONCLUSIVE"
            termination_reason = "pre_call_budget_admission_exceeded"
            trace.record("INCIDENT_INCONCLUSIVE", {
                "error_type": type(exc).__name__,
                "failure_category": termination_reason,
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
                "hypothesis": hypotheses[-1].model_dump() if hypotheses else None,
                "open_obligations": [
                    item.model_dump() for item in (
                        hypotheses[-1].verification_obligations if hypotheses else ()
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
                "hypothesis": hypotheses[-1].model_dump() if hypotheses else None,
                "open_obligations": [
                    item.model_dump() for item in (
                        hypotheses[-1].verification_obligations if hypotheses else ()
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
                "hypothesis": hypotheses[-1].model_dump() if hypotheses else None,
                "open_obligations": [
                    item.model_dump() for item in (
                        hypotheses[-1].verification_obligations if hypotheses else ()
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
        except LLMDeadlineExceeded as exc:
            # Once a candidate has entered the Review lifecycle, a later
            # provider timeout means that no candidate has reached a valid
            # terminal boundary. Keep the candidate as explicitly provisional
            # for trace/run diagnostics, while status remains INCONCLUSIVE and
            # the evaluator cannot accept it without a PASS Review decision.
            # Initial investigation timeouts retain the historical FAILED
            # semantics.
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
        if status == "PASS":
            transition_phase("DONE", termination_reason or "review_pass")
        else:
            transition_phase("INCONCLUSIVE", termination_reason or "terminal_failure")
        terminal_budget = budget_snapshot("terminal")
        trace.record("INCIDENT_COMPLETED", {
            "status": status, "termination_reason": termination_reason,
            "steps": state.step, "tool_calls": state.tool_calls,
            "budget": budget_payload(terminal_budget),
        })
        trace.record("INCIDENT_FINISHED", {
            "status": status, "termination_reason": termination_reason,
            "failure_category": self._termination_failure_category(status, termination_reason),
            "steps": state.step, "tool_calls": state.tool_calls,
        })
        failure_category = self._termination_failure_category(status, termination_reason)
        return IncidentRunResult(
            case_id=case.case_id, status=status, candidate=candidate, review=decision,
            evidence=incident_evidence(), hypotheses=tuple(hypotheses), actions=tuple(actions),
            metrics=IncidentMetrics(
                steps=state.step, tool_calls=state.tool_calls, prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens,
                cost=run_cost if cost_measured else None,
                llm_calls=llm_calls,
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
                stable_rounds=(hypotheses[-1].stable_rounds if hypotheses else 0),
            ),
            trace_path=str(trace.path),
            error_type=error_type, error_message=error_message,
            failure_category=failure_category,
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

    @classmethod
    def _has_material_hypothesis_instability(
        cls,
        previous: IncidentHypothesis | None,
        current: IncidentHypothesis | None,
    ) -> bool:
        """Detect reasoning instability without penalizing topology traversal.

        A Planner commonly moves through several candidate services while the
        fault and mechanism are still unknown.  Component-only changes are
        therefore normal exploration, not a reason to spend Reflection budget.
        Compare only the fault/mechanism portion of the material fingerprint;
        a non-empty reasoning identity must be present on at least one side and
        the reasoning tuple must actually change.
        """
        previous_fingerprint = cls._material_hypothesis_fingerprint(previous)
        current_fingerprint = cls._material_hypothesis_fingerprint(current)
        if previous_fingerprint is None or current_fingerprint is None:
            return False
        previous_reasoning = previous_fingerprint[1:]
        current_reasoning = current_fingerprint[1:]
        return bool(
            previous_reasoning != current_reasoning
            and (any(previous_reasoning) or any(current_reasoning))
        )

    @classmethod
    def _review_progress_made(cls, baseline: RootCauseCandidate,
                              candidate: RootCauseCandidate,
                              baseline_evidence_ids: frozenset[str],
                              current_evidence_ids: set[str],
                              baseline_hypothesis: IncidentHypothesis | None,
                              current_hypothesis: IncidentHypothesis) -> bool:
        if current_evidence_ids - baseline_evidence_ids:
            return True
        if cls._candidate_semantic_fingerprint(baseline) != cls._candidate_semantic_fingerprint(candidate):
            return True
        if baseline_hypothesis is None:
            return True
        return cls._hypothesis_semantic_fingerprint(baseline_hypothesis) != cls._hypothesis_semantic_fingerprint(current_hypothesis)

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
    def _planner_control_state(case, hypotheses, actions, review_feedback,
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
                               entities: Any | None = None,
                               capabilities: Any | None = None,
                               capability_states: dict[str, str] | None = None,
                               visible_tool_names=(),
                               router_decision: Any | None = None,
                               prior_context: Any | None = None,
                               compact: bool = False) -> dict[str, Any]:
        current = hypotheses[-1] if hypotheses else None
        visible_actions = [
            {key: value for key, value in action.items() if key != "observation_id"}
            for action in actions[-8:]
        ]
        payload = {
            "INCIDENT": {"summary": case.summary, "system": case.system, "namespace": case.namespace},
            "ENTITIES": entities.model_dump(mode="json") if hasattr(entities, "model_dump") else {},
            "CAPABILITIES": capabilities.model_dump(mode="json") if hasattr(capabilities, "model_dump") else {},
            "CAPABILITY_STATES": dict(capability_states or {}),
            "VISIBLE_TOOL_NAMES": list(visible_tool_names),
            "ROUTER_DECISION": router_decision.model_dump(mode="json") if hasattr(router_decision, "model_dump") else {},
            "PRIOR_KNOWLEDGE": (
                prior_context.model_dump(mode="json") if hasattr(prior_context, "model_dump") else None
            ),
            "AVAILABLE_EVIDENCE_SOURCES": list(available_evidence_sources),
            "SOURCE_WORKSPACE": {
                "available": bool(source_workspace_available),
                "root": "code" if source_workspace_available else None,
            },
            "AVAILABLE_CODE_TOOLS": list(available_code_tools),
            "CURRENT_HYPOTHESIS": current.model_dump() if current else None,
            "EVIDENCE_GAP": current.evidence_gap if current else "Identify the affected request path and component.",
            "SUPPORTING_EVIDENCE_IDS": list(current.supporting_evidence_ids) if current else [],
            "CONTRADICTING_EVIDENCE_IDS": list(current.contradicting_evidence_ids) if current else [],
            "REQUIRED_EVIDENCE_GAPS": list(current.required_gaps) if current else [],
            "VERIFICATION_OBLIGATIONS": [item.model_dump() for item in (current.verification_obligations if current else ())],
            "OBLIGATION_SUMMARY": [
                {
                    "id": item.id, "claim": item.claim, "critical": item.critical,
                    "status": item.status, "blocks_finalization": item.blocks_finalization,
                }
                for item in (current.verification_obligations if current else ())
            ],
            "CONTRADICTIONS": [item.model_dump() for item in (current.contradictions if current else ())],
            "CONTRADICTION_SUMMARY": [
                {
                    "evidence_id": item.evidence_id, "severity": item.severity,
                    "status": item.status, "blocks_finalization": item.blocks_finalization,
                }
                for item in (current.contradictions if current else ())
            ],
            "STABLE_ROUNDS": current.stable_rounds if current else 0,
            "EVIDENCE_SUFFICIENCY": "sufficient" if current and current.evidence_sufficient else "insufficient",
            "PREVIOUS_ACTIONS": visible_actions,
            "RECENT_ACTIONS": visible_actions,
            "REJECTED_ACTIONS": rejected_actions[-4:],
            "TOOL_BUDGET_REMAINING": tool_budget_remaining,
            "AGENT_STEPS_REMAINING": agent_steps_remaining,
            "REMAINING_STEPS": agent_steps_remaining,
            "REMAINING_LLM_CALLS": llm_calls_remaining,
            "REVIEW_FEEDBACK": review_feedback or None,
            "REFLECTION_FEEDBACK": reflection_feedback or None,
            "REFLECTION_TRIGGER": reflection_trigger or None,
            "NEXT_ACTION_CONSTRAINT": next_action_constraint or None,
            "TOOL_HEALTH": tool_health or {},
            "SOURCE_CAPABILITIES": {
                "source_workspace_available": bool(source_workspace_available),
                "available_code_tools": list(available_code_tools),
            },
            "SOURCE_MECHANISM_COVERAGE": DiagnosisHarness._source_mechanism_coverage(
                case,
                current,
                source_workspace_available=source_workspace_available,
                available_code_tools=available_code_tools,
                source_evidence_ids=source_evidence_ids,
            ),
        }
        if not compact:
            return payload
        # The envelope protocol already carries Planner controls once. Keep
        # one compact state projection here as well: old payloads repeated
        # PREVIOUS_ACTIONS/RECENT_ACTIONS, full hypothesis plus projections,
        # and obligations/contradictions plus summaries.
        compact_hypothesis = None
        if current is not None:
            compact_hypothesis = {
                "claim": current.claim,
                "evidence_gap": current.evidence_gap,
                "component": current.component,
                "fault": current.fault,
                "fault_code": current.fault_code,
                "mechanism": current.mechanism,
                "supporting_evidence_ids": list(current.supporting_evidence_ids),
                "contradicting_evidence_ids": list(current.contradicting_evidence_ids),
                "required_gaps": list(current.required_gaps),
                "evidence_sufficient": current.evidence_sufficient,
                "source_mechanism_status": current.source_mechanism_status,
                "stable_rounds": current.stable_rounds,
            }
        compact_payload = dict(payload)
        compact_payload["CURRENT_HYPOTHESIS"] = compact_hypothesis
        for key in (
            "VERIFICATION_OBLIGATIONS", "CONTRADICTIONS", "PREVIOUS_ACTIONS",
            "SOURCE_CAPABILITIES",
        ):
            compact_payload.pop(key, None)
        return compact_payload

    @staticmethod
    def _source_mechanism_coverage(
        case,
        hypothesis: IncidentHypothesis | None,
        *,
        source_workspace_available: bool,
        available_code_tools=(),
        source_evidence_ids=(),
    ) -> dict[str, Any]:
        """Project source capability and coverage without deciding causality."""
        source_ids = tuple(str(item) for item in source_evidence_ids if str(item).startswith("ev-"))
        supporting = set(hypothesis.supporting_evidence_ids) if hypothesis else set()
        return {
            "application_code_declared": "application_code" in tuple(case.evidence_sources),
            "workspace_available": bool(source_workspace_available),
            "available_code_tools": list(available_code_tools),
            "status": hypothesis.source_mechanism_status if hypothesis else "unknown",
            "source_backed_evidence_ids": list(source_ids),
            "cited_source_backed_evidence_ids": sorted(supporting.intersection(source_ids)),
            "open_critical_obligation_ids": [
                item.id for item in (hypothesis.verification_obligations if hypothesis else ())
                if item.blocks_finalization
            ],
        }

    @staticmethod
    def _candidate_from_hypothesis(hypothesis: IncidentHypothesis) -> RootCauseCandidate:
        return RootCauseCandidate(
            component=hypothesis.component,
            fault=hypothesis.fault,
            fault_code=hypothesis.fault_code,
            fault_explanation=hypothesis.fault_explanation,
            mechanism=hypothesis.mechanism,
            evidence_ids=hypothesis.supporting_evidence_ids,
            confidence=0.7,
            claim_evidence_mapping=(
                {
                    "claim": f"{hypothesis.component}: {hypothesis.fault}",
                    "evidence_ids": hypothesis.supporting_evidence_ids,
                },
            ),
            causal_chain_summary=(
                {
                    "cause": hypothesis.fault,
                    "effect": hypothesis.mechanism,
                    "evidence_ids": hypothesis.supporting_evidence_ids,
                },
            ),
        )

    @staticmethod
    def _candidate_from_final_hypothesis(
        hypothesis: IncidentHypothesis,
        proposal: dict[str, Any],
        *,
        known_evidence_ids: set[str],
    ) -> RootCauseCandidate:
        """Freeze Candidate core fields from the already validated Hypothesis.

        ``finalize_diagnosis`` keeps its legacy component/fault/mechanism and
        evidence fields so older providers still satisfy their argument schema.
        Those fields are deliberately not copied into the Candidate.  The
        Planner may still submit the optional claim/evidence mapping and causal
        chain that explain the frozen diagnosis, but neither metadata block may
        introduce a new Evidence ID or a second root-cause authority.
        """
        submitted_evidence = tuple(proposal.get("evidence_ids") or ())
        invalid_submitted = [
            evidence_id for evidence_id in submitted_evidence
            if not evidence_id.startswith("ev-") or evidence_id not in known_evidence_ids
        ]
        if invalid_submitted:
            raise RuntimeError(
                "finalize proposal cites unknown ev-* Evidence IDs: "
                + ", ".join(sorted(set(invalid_submitted)))
            )

        supporting = tuple(hypothesis.supporting_evidence_ids)
        return RootCauseCandidate(
            # These are the only final root-cause fields used by Runtime.
            component=hypothesis.component,
            fault=hypothesis.fault,
            fault_code=hypothesis.fault_code,
            fault_explanation=hypothesis.fault_explanation,
            mechanism=hypothesis.mechanism,
            evidence_ids=supporting,
            confidence=float(proposal.get("confidence", 0.7)),
            # These fields remain Planner-authored argumentation metadata.
            claim_evidence_mapping=tuple(proposal.get("claim_evidence_mapping") or ()),
            causal_chain_summary=tuple(proposal.get("causal_chain_summary") or ()),
        )

    @classmethod
    def _candidate_from_sufficient_hypothesis(cls, hypotheses, evidence):
        if not hypotheses:
            return None
        known = {item.evidence_id for item in evidence}
        hypothesis = hypotheses[-1]
        return cls._candidate_from_hypothesis(hypothesis) if hypothesis.can_finalize(known) else None

    def _provider_call(self, method, *args, hard_timeout_seconds: float,
                       state_source, state_attributes: tuple[str, ...], **kwargs):
        if hard_timeout_seconds <= 0:
            raise LLMDeadlineExceeded("diagnosis provider call has no remaining deadline")
        provider = getattr(state_source, "llm", self.llm)
        if not getattr(provider, "requires_process_timeout", False):
            return method(*args, **kwargs)
        return _call_in_terminable_process(
            method, args, kwargs, hard_timeout_seconds,
            state_source=state_source, state_attributes=state_attributes,
        )


# Compatibility import for the original Incident vertical-slice API. The
# CloudOps entry point uses DiagnosisHarness; there is no second lifecycle.
IncidentHarness = DiagnosisHarness


def _provider_process_worker(send_connection, method, args, kwargs,
                             state_source, state_attributes) -> None:
    try:
        result = method(*args, **kwargs)
        state = {name: getattr(state_source, name) for name in state_attributes}
        send_connection.send(("ok", result, state))
    except BaseException as exc:
        state = {name: getattr(state_source, name) for name in state_attributes}
        send_connection.send((
            "error", type(exc).__name__, str(exc), state,
            getattr(exc, "metadata", None),
        ))
    finally:
        send_connection.close()


def _call_in_terminable_process(method, args, kwargs, timeout_seconds: float,
                                *, state_source, state_attributes):
    """Bound a provider sync boundary even if async transport cleanup cannot return."""
    if "fork" not in multiprocessing.get_all_start_methods():
        return method(*args, **kwargs)
    context = multiprocessing.get_context("fork")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_provider_process_worker,
        args=(send_connection, method, args, kwargs, state_source, state_attributes),
        daemon=True,
    )
    process.start()
    send_connection.close()
    try:
        if not receive_connection.poll(timeout_seconds):
            _terminate_process(process)
            raise LLMDeadlineExceeded(
                f"provider call exceeded hard wall-clock deadline of {timeout_seconds:.2f}s"
            )
        packet = receive_connection.recv()
    finally:
        receive_connection.close()
    process.join(timeout=0.5)
    if process.is_alive():
        _terminate_process(process)
    if packet[0] == "ok":
        _, result, state = packet
        for name, value in state.items():
            setattr(state_source, name, value)
        return result
    _, error_type, message, state, metadata = (
        (*packet, None) if len(packet) == 4 else packet
    )
    for name, value in state.items():
        setattr(state_source, name, value)
    if error_type == "LLMDeadlineExceeded":
        raise LLMDeadlineExceeded(message)
    if error_type in {"LLMError", "LLMTransportTimeout"}:
        raise LLMError(message)
    if error_type in {"PlannerContractError", "NativePlannerContractError"}:
        # The provider runs in a terminable child process.  Reconstruct the
        # contract exception with its bounded metadata so raw provider-visible
        # output and normalized arguments survive the process boundary.
        restored = PlannerContractError(
            message,
            validation_errors=(metadata or {}).get("validation_errors", [])
            if isinstance(metadata, dict) else [],
            output=(metadata or {}) if isinstance(metadata, dict) else {},
        )
        if isinstance(metadata, dict):
            restored.metadata.update(metadata)
        raise restored
    if error_type == "ReflectionContractExhausted":
        raise ReflectionContractExhausted(message)
    if error_type == "ReviewSchemaError":
        raise ReviewSchemaError(message)
    if error_type in {"PromptBudgetExceeded", "RunBudgetAdmissionExceeded"}:
        from debug_assistant.harness.dynamic_budget import PromptBudgetDecision
        metadata = metadata if isinstance(metadata, dict) else {}
        decision = PromptBudgetDecision(
            stage=str(metadata.get("stage", "unknown")),
            estimated_prompt_tokens=int(metadata.get("estimated_prompt_tokens", 0) or 0),
            input_hard_capacity=metadata.get("input_hard_capacity"),
            state=BudgetState(str(metadata.get("state", "HARD_PRESSURE"))),
            remaining_run_tokens=int(metadata.get("remaining_run_tokens", 0) or 0),
            terminal_reserve_tokens=0,
            terminal_reserve_llm_calls=0,
            breakdown=dict(metadata.get("breakdown") or {}),
            compaction_required=True,
            reason=str(metadata.get("reason") or "prompt is at the physical input boundary"),
        )
        if error_type == "RunBudgetAdmissionExceeded":
            raise RunBudgetAdmissionExceeded(decision)
        raise PromptBudgetExceeded(decision)
    raise RuntimeError(f"provider child failed with {error_type}: {message}")


def _terminate_process(process) -> None:
    if not process.is_alive():
        process.join(timeout=0.1)
        return
    process.terminate()
    process.join(timeout=0.5)
    if process.is_alive():
        process.kill()
        process.join(timeout=0.5)
