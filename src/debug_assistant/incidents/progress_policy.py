"""Operational continuation decisions over canonical projections, never RCA meaning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from .convergence import ConvergenceDecision
from .finalization import FinalizationDecision


class ReflectionTrigger(str, Enum):
    NO_CANONICAL_PROGRESS = "NO_CANONICAL_PROGRESS"
    REPEATED_ACTION = "REPEATED_ACTION"
    UNRESOLVED_CONTRADICTION = "UNRESOLVED_CONTRADICTION"
    ADMISSION_REJECTION_LOOP = "ADMISSION_REJECTION_LOOP"
    REVIEW_RECOVERY_REQUIRED = "REVIEW_RECOVERY_REQUIRED"


@dataclass(frozen=True, slots=True)
class ProgressFacts:
    boundary: Literal["admission", "post_usage", "progress", "loop", "action", "publication", "salvage", "review", "review_recovery", "convergence", "terminal", "recovery", "phase", "reflection_trigger"]
    stage: str = ""
    tokens: int = 0
    token_limit: int = 0
    llm_calls: int = 0
    llm_limit: int = 0
    tool_calls: int = 0
    tool_limit: int = 0
    steps: int = 0
    step_limit: int = 0
    cost: float = 0.0
    cost_limit: float | None = None
    count_llm: bool = False
    review_calls: int = 0
    review_limit: int = 0
    deadline_available: bool = True
    terminal_reserve_available: bool = True
    targeted_recovery_allowed: bool = False
    phase: str = "explore"
    hard_pressure: bool = False
    consolidation_available: bool = False
    canonical_progress: bool = False
    acquisition_progress: bool = False
    no_progress_count: int = 0
    no_progress_limit: int = 0
    repeated_count: int = 0
    repeat_limit: int = 0
    finalization: FinalizationDecision | None = None
    hypothesis_stable: bool = False
    open_obligation_ids: tuple[str, ...] = ()
    contradiction_ids: tuple[str, ...] = ()
    capability_blocked: bool = False
    terminal_reason: str = ""
    terminal_status: str = ""
    previous_convergence: ConvergenceDecision | None = None
    recovery_available: bool = True
    automatic_finalization: bool = True
    post_sufficiency_calls: int | None = None
    post_sufficiency_limit: int = 0
    recovery_count: int = 0
    recovery_limit: int = 0
    recovery_enabled: bool = True
    recovery_phase_restricted: bool = False
    recovery_duplicate: bool = False
    recovery_capability_available: bool = True
    consecutive_recovery_count: int = 0
    consecutive_recovery_limit: int | None = None
    phase_start_step: int = 0
    verification_ready: bool = True
    baseline_candidate_fingerprint: tuple[object, ...] | None = None
    candidate_fingerprint: tuple[object, ...] | None = None
    baseline_hypothesis_fingerprint: tuple[object, ...] | None = None
    hypothesis_fingerprint: tuple[object, ...] | None = None
    baseline_evidence_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    review_recovery_permitted: bool = True
    actionable_obligation_ids: tuple[str, ...] = ()
    review_contradiction_ids: tuple[str, ...] = ()
    admission_rejection_count: int = 0
    admission_rejection_limit: int = 2
    review_recovery_required: bool = False


@dataclass(frozen=True, slots=True)
class ProgressDecision:
    action: Literal["continue", "stop", "retry", "finalize"]
    reasons: tuple[str, ...] = ()
    no_progress_count: int = 0
    convergence: ConvergenceDecision | None = None
    phase: str = ""
    reflection_trigger: ReflectionTrigger | None = None


class ProgressPolicy:
    """One stateless entry; callers own counters, events and phase transitions."""

    @staticmethod
    def evaluate(facts: ProgressFacts) -> ProgressDecision:
        ready = bool(facts.finalization is not None and facts.finalization.allowed)
        if facts.boundary == "reflection_trigger":
            trigger = None
            if facts.review_recovery_required:
                trigger = ReflectionTrigger.REVIEW_RECOVERY_REQUIRED
            elif facts.contradiction_ids:
                trigger = ReflectionTrigger.UNRESOLVED_CONTRADICTION
            elif facts.repeated_count:
                trigger = ReflectionTrigger.REPEATED_ACTION
            elif facts.admission_rejection_count >= facts.admission_rejection_limit:
                trigger = ReflectionTrigger.ADMISSION_REJECTION_LOOP
            elif facts.no_progress_limit and facts.no_progress_count >= facts.no_progress_limit:
                trigger = ReflectionTrigger.NO_CANONICAL_PROGRESS
            return ProgressDecision(
                "retry" if trigger else "continue",
                (trigger.value,) if trigger else ("normal_investigation",),
                reflection_trigger=trigger,
            )
        if facts.boundary == "review":
            initial_submission = (facts.baseline_candidate_fingerprint is None
                                  or facts.hypothesis_fingerprint is None)
            progress = (bool(set(facts.evidence_ids) - set(facts.baseline_evidence_ids))
                        or facts.candidate_fingerprint != facts.baseline_candidate_fingerprint
                        or facts.hypothesis_fingerprint != facts.baseline_hypothesis_fingerprint)
            allowed = initial_submission or progress
            return ProgressDecision("continue" if allowed else "stop",
                                    ("review_progress_observed" if allowed else "review_progress_required",))
        if facts.boundary == "review_recovery":
            executable_recovery = bool(facts.actionable_obligation_ids or facts.review_contradiction_ids)
            recover = facts.review_recovery_permitted and executable_recovery
            return ProgressDecision("retry" if recover else "stop",
                                    ("review_recovery_available" if recover else "review_recovery_unavailable",))
        if facts.boundary in {"action", "publication", "salvage"}:
            publish = ready and facts.automatic_finalization
            return ProgressDecision("finalize" if publish else "stop" if facts.boundary in {"publication", "salvage"}
                                    else "continue",
                                    ("canonical_finalization_allowed" if publish
                                     else "finalization_blocked" if facts.boundary in {"publication", "salvage"}
                                     else "investigation_capacity_available",))
        if facts.boundary == "phase":
            phase = facts.phase
            if phase == "explore" and facts.steps + 1 >= facts.phase_start_step:
                phase = "converge" if facts.verification_ready else "source_verify"
            return ProgressDecision("continue", ("operational_phase_selected",), phase=phase)
        if facts.boundary == "recovery":
            if facts.stage == "reflection":
                reason = ""
                if (facts.recovery_phase_restricted
                        and facts.phase in {"converge", "source_verify", "verify_only", "finalize"}
                        and (facts.phase == "finalize" or ready or not facts.targeted_recovery_allowed)):
                    reason = "converge_hard_call_budget"
                elif not facts.recovery_enabled or facts.recovery_limit == 0:
                    reason = "reflection_disabled"
                elif (facts.consecutive_recovery_limit is not None
                      and facts.consecutive_recovery_count >= facts.consecutive_recovery_limit):
                    reason = "max_consecutive_reflection_without_runtime_delta"
                elif not facts.recovery_capability_available:
                    reason = "provider_has_no_json_reflection_api"
                elif facts.recovery_duplicate:
                    reason = "duplicate_trigger_signature"
                elif facts.recovery_count >= facts.recovery_limit:
                    reason = "max_reflection_calls"
                if reason:
                    return ProgressDecision("stop", (reason,))
            available = facts.recovery_count < facts.recovery_limit
            return ProgressDecision("retry" if available else "stop",
                                    ("recovery_capacity_available" if available else "retry_exhausted",))
        if (facts.boundary == "terminal" and facts.terminal_status == "PASS" and ready
                and facts.previous_convergence is not None
                and facts.previous_convergence.status == "CONVERGED"):
            return ProgressDecision("finalize", convergence=facts.previous_convergence)
        if facts.boundary in {"admission", "post_usage"}:
            post = facts.boundary == "post_usage"
            limits = (
                (facts.tokens, facts.token_limit, "max_total_tokens"),
                (facts.cost, facts.cost_limit, "max_cost_per_incident"),
                (facts.llm_calls, facts.llm_limit, "max_llm_calls"),
            )
            if not facts.deadline_available:
                return ProgressDecision("stop", ("global_deadline",))
            if (not facts.terminal_reserve_available
                    and facts.stage in {"planner", "tool", "reflection", "contract_repair"}
                    and not facts.targeted_recovery_allowed):
                return ProgressDecision("stop", ("terminal_reserve_protected",))
            for used, limit, reason in limits:
                if reason == "max_llm_calls" and not (post or facts.count_llm):
                    continue
                if limit is not None and limit > 0 and (used > limit if post else used >= limit):
                    return ProgressDecision("stop", (reason,))
            if facts.count_llm:
                if facts.stage == "review" and facts.review_calls >= facts.review_limit:
                    return ProgressDecision("stop", ("review_call_reserve_exhausted",))
            elif facts.stage == "planner" and facts.steps >= facts.step_limit:
                return ProgressDecision("stop", ("max_steps",))
            elif facts.stage == "tool" and facts.tool_calls >= facts.tool_limit:
                return ProgressDecision("stop", ("max_tool_calls",))
            return ProgressDecision("continue", ("operational_capacity_available",))

        count = (0 if facts.canonical_progress or facts.acquisition_progress or not facts.no_progress_limit
                 else facts.no_progress_count + 1)
        if facts.boundary == "progress":
            if facts.repeat_limit and facts.repeated_count > facts.repeat_limit:
                return ProgressDecision("stop", ("duplicate_action_limit",), count)
            if facts.no_progress_limit and count >= facts.no_progress_limit:
                action = "retry" if facts.recovery_available else "finalize" if ready else "stop"
                return ProgressDecision(action, ("no_progress_limit",), count)
            return ProgressDecision("continue", ("progress_observed" if facts.canonical_progress
                                                  else "acquisition_progress_observed" if facts.acquisition_progress
                                                  else "investigation_capacity_available",), count)

        if facts.boundary == "loop":
            reason = ""
            if not facts.deadline_available:
                reason = "global_deadline"
            elif facts.phase == "finalize" and not facts.consolidation_available:
                reason = "finalization_reserve"
            elif facts.tool_calls >= facts.tool_limit and not facts.consolidation_available:
                reason = "max_tool_calls"
            elif facts.steps >= facts.step_limit:
                reason = "max_steps"
            elif facts.hard_pressure and ready:
                reason = "token_budget"
            elif (facts.post_sufficiency_calls is not None
                  and facts.post_sufficiency_calls >= facts.post_sufficiency_limit and ready):
                reason = "post_sufficiency_tool_budget"
            if reason:
                return ProgressDecision("finalize" if ready else "stop", (reason,))
            if ready and facts.automatic_finalization:
                return ProgressDecision("finalize", ("canonical_finalization_allowed",))
            return ProgressDecision("continue", ("investigation_capacity_available",))

        # FinalizationPolicy supplies semantic permission. This is its operational
        # convergence projection, not a second Evidence/completion predicate.
        hard_budget = facts.hard_pressure or not facts.deadline_available
        if facts.boundary == "terminal":
            hard_budget = hard_budget or any(marker in facts.terminal_reason.casefold()
                                            for marker in ("budget", "timeout", "deadline", "max_", "token", "review_bound"))
        reasons: list[str] = []
        no_progress = bool(facts.no_progress_limit and facts.no_progress_count >= facts.no_progress_limit)
        if hard_budget:
            status = "INCONCLUSIVE_BUDGET"
            reasons.append("hard_budget")
        elif facts.capability_blocked:
            status = "INCONCLUSIVE_CAPABILITY"
            reasons.append("capability_blocked")
        elif facts.contradiction_ids:
            status = "REPLAN"
            reasons.append("blocking_contradiction")
        elif ready and facts.hypothesis_stable:
            status = "CONVERGED"
            if no_progress:
                reasons.append("no_progress_after_sufficient_stable_state")
        else:
            status = "CONTINUE"
            if not ready:
                reasons.append("finalization_blocked")
            if not facts.hypothesis_stable:
                reasons.append("hypothesis_unstable")
            if facts.open_obligation_ids:
                reasons.append("critical_obligation_open")
            if no_progress:
                reasons.append("no_progress_without_sufficiency")
        convergence = ConvergenceDecision(
            status=status, evidence_sufficient=ready, hypothesis_stable=facts.hypothesis_stable,
            open_critical_obligations=facts.open_obligation_ids,
            blocking_contradictions=facts.contradiction_ids, no_progress=no_progress,
            hard_budget=hard_budget, capability_blocked=facts.capability_blocked, reasons=tuple(reasons),
        )
        return ProgressDecision("finalize" if status == "CONVERGED" else "stop" if hard_budget
                                or facts.capability_blocked else "retry" if status == "REPLAN"
                                else "continue", tuple(reasons), facts.no_progress_count, convergence)
