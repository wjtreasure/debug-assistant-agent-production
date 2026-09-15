from __future__ import annotations

"""Model-independent context and run-budget control.

This module owns deterministic admission decisions only.  It does not select
Evidence, infer a diagnosis, or replace the existing run ``BudgetController``.
The ContextManager remains responsible for selecting content; this controller
only supplies the physical ceiling, pressure state, and continuation signal.
"""

from dataclasses import dataclass, field, replace
from enum import Enum
import json
import math
import time
from typing import Any, Callable, Iterable

from debug_assistant.llm.base import ModelCapability, estimate_tokens_char4


class BudgetState(str, Enum):
    NORMAL = "NORMAL"
    PRESSURE = "PRESSURE"
    HARD_PRESSURE = "HARD_PRESSURE"


@dataclass(frozen=True, slots=True)
class PromptBudgetDecision:
    stage: str
    estimated_prompt_tokens: int
    input_hard_capacity: int | None
    state: BudgetState
    remaining_run_tokens: int
    terminal_reserve_tokens: int
    terminal_reserve_llm_calls: int
    breakdown: dict[str, int] = field(default_factory=dict)
    compaction_required: bool = False
    reason: str = ""
    remaining_run_cost: float | None = None


class PromptBudgetExceeded(RuntimeError):
    """Raised before a provider request when the physical input ceiling is exceeded."""

    def __init__(self, decision: PromptBudgetDecision):
        self.decision = decision
        self.metadata = {
            "stage": decision.stage,
            "estimated_prompt_tokens": decision.estimated_prompt_tokens,
            "input_hard_capacity": decision.input_hard_capacity,
            "state": decision.state.value,
            "over_budget_tokens": max(
                0,
                decision.estimated_prompt_tokens
                - int(decision.input_hard_capacity or 0),
            ),
            "breakdown": dict(decision.breakdown),
        }
        super().__init__(
            f"{decision.stage} prompt estimated at "
            f"{decision.estimated_prompt_tokens} tokens exceeds input hard capacity "
            f"{decision.input_hard_capacity}"
        )


class RunBudgetAdmissionExceeded(PromptBudgetExceeded):
    """Raised before a provider request cannot preserve the run terminal reserve."""

    def __init__(self, decision: PromptBudgetDecision):
        super().__init__(decision)
        self.metadata["kind"] = "run_budget_admission"
        self.metadata["remaining_run_tokens"] = decision.remaining_run_tokens
        self.metadata["required_run_tokens"] = decision.breakdown.get(
            "required_run_tokens", 0
        )


@dataclass(frozen=True, slots=True)
class ProgressMarker:
    """Minimal semantic state used by the deterministic continuation predicate."""

    evidence_ids: frozenset[str] = frozenset()
    hypothesis: tuple[Any, ...] = ()
    open_obligations: frozenset[str] = frozenset()
    blocking_contradictions: frozenset[str] = frozenset()
    components: frozenset[str] = frozenset()


def progress_made(before: ProgressMarker | None, after: ProgressMarker) -> bool:
    """Return true only for an observable diagnosis-state improvement."""
    if before is None:
        return True
    return bool(
        after.evidence_ids - before.evidence_ids
        or after.open_obligations < before.open_obligations
        or after.blocking_contradictions < before.blocking_contradictions
        or after.components - before.components
        or after.hypothesis != before.hypothesis
    )


class DynamicBudgetController:
    """Compute physical prompt capacity and bounded run pressure.

    If a provider does not expose a context window, ``input_hard_capacity`` is
    ``None`` and the existing ``max_context_chars`` policy remains the
    compatibility fallback.  It is never presented as a physical model limit.
    """

    def __init__(
        self,
        capability: ModelCapability | None = None,
        *,
        max_context_chars: int = 50_000,
        max_total_tokens: int = 180_000,
        max_llm_calls: int = 40,
        max_wall_time_seconds: float = 900.0,
        max_cost_per_incident: float | None = None,
        terminal_reserve_tokens: int = 0,
        terminal_reserve_llm_calls: int = 3,
        terminal_reserve_seconds: float = 0.0,
        pressure_ratio: float = 0.75,
        hard_pressure_ratio: float = 0.95,
        started_at: float | None = None,
    ) -> None:
        self.capability = capability or ModelCapability()
        self.max_context_chars = max(1, int(max_context_chars))
        self.max_total_tokens = max(1, int(max_total_tokens))
        self.max_llm_calls = max(1, int(max_llm_calls))
        self.max_wall_time_seconds = max(1.0, float(max_wall_time_seconds))
        self.max_cost_per_incident = (
            None if max_cost_per_incident is None else max(0.0, float(max_cost_per_incident))
        )
        derived_reserve = max(1, math.ceil(self.max_total_tokens * 0.15))
        self.terminal_reserve_tokens = max(
            1, int(terminal_reserve_tokens) or derived_reserve
        )
        self.terminal_reserve_llm_calls = max(1, int(terminal_reserve_llm_calls))
        self.terminal_reserve_seconds = max(0.0, float(terminal_reserve_seconds))
        self.pressure_ratio = min(0.99, max(0.50, float(pressure_ratio)))
        self.hard_pressure_ratio = min(
            1.0, max(self.pressure_ratio, float(hard_pressure_ratio))
        )
        self.started_at = started_at or time.time()
        self.last_decision: PromptBudgetDecision | None = None
        self.actual_usage: list[dict[str, Any]] = []
        self.tokens_used = 0
        self.llm_calls_used = 0
        self.cost_used = 0.0
        self.context_state_override = BudgetState.NORMAL

    @property
    def input_hard_capacity(self) -> int | None:
        return self.capability.input_hard_capacity

    @property
    def fallback_context_tokens(self) -> int:
        return max(1, self.estimate_tokens("x" * self.max_context_chars))

    def estimate_tokens(self, text: str) -> int:
        return self.capability.estimate_tokens(text)

    def estimate_json_tokens(self, value: Any) -> int:
        return self.estimate_tokens(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))

    def estimate_prompt(
        self,
        system: str,
        user: str,
        *,
        tools: Iterable[dict[str, Any]] | None = None,
        breakdown: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, int]]:
        parts = {
            "system": self.estimate_tokens(system),
            "tools": self.estimate_json_tokens(list(tools or [])) if tools is not None else 0,
            "user": self.estimate_tokens(user),
        }
        if breakdown:
            for key, value in breakdown.items():
                parts[key] = self.estimate_tokens(value)
        # The textual prompt is the physical payload. Breakdown fields are
        # diagnostics only and must not be summed on top of system/user.
        return parts["system"] + parts["tools"] + parts["user"], parts

    def _run_pressure(
        self,
        *,
        tokens_used: int,
        llm_calls_used: int,
        cost_used: float = 0.0,
        now: float | None = None,
    ) -> tuple[BudgetState, str, int]:
        now = time.time() if now is None else float(now)
        remaining_tokens = max(0, self.max_total_tokens - int(tokens_used))
        remaining_calls = max(0, self.max_llm_calls - int(llm_calls_used))
        elapsed = max(0.0, now - self.started_at)
        remaining_time = max(0.0, self.max_wall_time_seconds - elapsed)
        remaining_cost = (
            None if self.max_cost_per_incident is None
            else max(0.0, self.max_cost_per_incident - float(cost_used))
        )
        if (
            remaining_tokens <= 0 or remaining_calls <= 0 or remaining_time <= 0
            or remaining_cost is not None and remaining_cost <= 0
        ):
            return BudgetState.HARD_PRESSURE, "run budget exhausted", remaining_tokens
        if (
            remaining_tokens <= self.terminal_reserve_tokens
            or remaining_calls <= self.terminal_reserve_llm_calls
            or remaining_time <= self.terminal_reserve_seconds
            or remaining_cost is not None
            and remaining_cost <= self.max_cost_per_incident * 0.10
        ):
            return BudgetState.PRESSURE, "terminal reserve must be protected", remaining_tokens
        return BudgetState.NORMAL, "", remaining_tokens

    def decision(
        self,
        stage: str,
        *,
        estimated_prompt_tokens: int = 0,
        tokens_used: int = 0,
        llm_calls_used: int = 0,
        cost_used: float = 0.0,
        breakdown: dict[str, int] | None = None,
        now: float | None = None,
    ) -> PromptBudgetDecision:
        run_state, run_reason, remaining = self._run_pressure(
            tokens_used=tokens_used, llm_calls_used=llm_calls_used,
            cost_used=cost_used, now=now,
        )
        physical_state = BudgetState.NORMAL
        physical_reason = ""
        capacity = self.input_hard_capacity
        if capacity is not None:
            if capacity <= 0 or estimated_prompt_tokens > capacity:
                physical_state = BudgetState.HARD_PRESSURE
                physical_reason = "prompt is at the physical input boundary"
            elif capacity > 0:
                ratio = float(estimated_prompt_tokens) / float(capacity)
                if ratio >= self.hard_pressure_ratio:
                    physical_state = BudgetState.HARD_PRESSURE
                    physical_reason = "prompt is at the physical input boundary"
                elif ratio >= self.pressure_ratio:
                    physical_state = BudgetState.PRESSURE
                    physical_reason = "prompt is near the physical input boundary"
        state = max(
            (run_state, physical_state, self.context_state_override),
            key=lambda item: (
                item is BudgetState.HARD_PRESSURE,
                item is BudgetState.PRESSURE,
            ),
        )
        reason = "; ".join(item for item in (physical_reason, run_reason) if item)
        result = PromptBudgetDecision(
            stage=stage,
            estimated_prompt_tokens=max(0, int(estimated_prompt_tokens)),
            input_hard_capacity=capacity,
            state=state,
            remaining_run_tokens=remaining,
            terminal_reserve_tokens=self.terminal_reserve_tokens,
            terminal_reserve_llm_calls=self.terminal_reserve_llm_calls,
            breakdown=dict(breakdown or {}),
            compaction_required=state in {BudgetState.PRESSURE, BudgetState.HARD_PRESSURE},
            reason=reason,
            remaining_run_cost=(
                None if self.max_cost_per_incident is None
                else max(0.0, self.max_cost_per_incident - float(cost_used))
            ),
        )
        self.last_decision = result
        return result

    def check_prompt(
        self,
        stage: str,
        system: str,
        user: str,
        *,
        tools: Iterable[dict[str, Any]] | None = None,
        breakdown: dict[str, str] | None = None,
        tokens_used: int | None = None,
        llm_calls_used: int | None = None,
        cost_used: float | None = None,
    ) -> PromptBudgetDecision:
        if tokens_used is None:
            tokens_used = self.tokens_used
        if llm_calls_used is None:
            llm_calls_used = self.llm_calls_used
        if cost_used is None:
            cost_used = self.cost_used
        estimate, parts = self.estimate_prompt(system, user, tools=tools, breakdown=breakdown)
        result = self.decision(
            stage,
            estimated_prompt_tokens=estimate,
            tokens_used=tokens_used,
            llm_calls_used=llm_calls_used,
            cost_used=cost_used,
            breakdown=parts,
        )
        if result.input_hard_capacity is not None and estimate > result.input_hard_capacity:
            self.context_state_override = BudgetState.HARD_PRESSURE
            raise PromptBudgetExceeded(result)
        # A prompt can fit the model's context window while still consuming
        # the run's finalization reserve.  Reject the provider call before it
        # starts when the estimated prompt plus declared completion reserve
        # would leave no room for the terminal path.
        completion_reserve = max(
            int(self.capability.reserved_output_tokens),
            int(self.capability.max_output_tokens or 0),
        )
        remaining_calls = max(0, self.max_llm_calls - int(llm_calls_used))
        required_run_tokens = (
            estimate + completion_reserve + self.terminal_reserve_tokens
        )
        if (
            required_run_tokens > result.remaining_run_tokens
            or (
                self.terminal_reserve_seconds > 0
                and self.max_wall_time_seconds - (time.time() - self.started_at)
                <= self.terminal_reserve_seconds
            )
        ):
            reason = "pre-call run budget admission would consume terminal reserve"
            if (
                self.terminal_reserve_seconds > 0
                and self.max_wall_time_seconds - (time.time() - self.started_at)
                <= self.terminal_reserve_seconds
            ):
                reason = "pre-call wall-clock admission would consume terminal reserve"
            admission = replace(
                result,
                state=BudgetState.HARD_PRESSURE,
                reason=reason,
                breakdown={
                    **result.breakdown,
                    "estimated_prompt_tokens": estimate,
                    "reserved_output_tokens": completion_reserve,
                    "terminal_reserve_tokens": self.terminal_reserve_tokens,
                    "required_run_tokens": required_run_tokens,
                    "remaining_run_tokens": result.remaining_run_tokens,
                    "remaining_llm_calls": remaining_calls,
                },
                compaction_required=True,
            )
            self.last_decision = admission
            raise RunBudgetAdmissionExceeded(admission)
        return result

    def force_context_state(self, state: BudgetState) -> None:
        self.context_state_override = state

    def clear_context_state_override(self) -> None:
        self.context_state_override = BudgetState.NORMAL

    def set_run_usage(
        self, *, tokens_used: int, llm_calls_used: int, cost_used: float | None = None
    ) -> None:
        self.tokens_used = max(0, int(tokens_used))
        self.llm_calls_used = max(0, int(llm_calls_used))
        if cost_used is not None:
            self.cost_used = max(0.0, float(cost_used))

    def context_token_budget(self, *, state: BudgetState = BudgetState.NORMAL) -> int | None:
        capacity = self.input_hard_capacity
        if capacity is None:
            return None
        # Content selection is intentionally below the physical ceiling. The
        # remaining headroom is for the full prompt's system/tool protocol,
        # which is checked again after rendering.
        if state is BudgetState.HARD_PRESSURE:
            return max(1, math.floor(capacity * 0.45))
        if state is BudgetState.PRESSURE:
            return max(1, math.floor(capacity * 0.65))
        return max(1, math.floor(capacity * 0.80))

    def available_prior_capacity(
        self,
        *,
        active_input_capacity: int | None = None,
        critical_state_tokens: int = 0,
        verified_evidence_tokens: int = 0,
        required_control_tokens: int = 0,
        output_protocol_reserve: int = 0,
    ) -> int | None:
        capacity = self.input_hard_capacity if active_input_capacity is None else active_input_capacity
        if capacity is None:
            return None
        return max(
            0,
            int(capacity)
            - max(0, int(critical_state_tokens))
            - max(0, int(verified_evidence_tokens))
            - max(0, int(required_control_tokens))
            - max(0, int(output_protocol_reserve))
        )

    def record_actual_usage(
        self,
        *,
        stage: str,
        estimated_prompt_tokens: int,
        actual_prompt_tokens: int | None,
        actual_completion_tokens: int | None,
        actual_cost: float | None = None,
    ) -> None:
        self.actual_usage.append({
            "stage": stage,
            "estimated_prompt_tokens": max(0, int(estimated_prompt_tokens)),
            "actual_prompt_tokens": None if actual_prompt_tokens is None else max(0, int(actual_prompt_tokens)),
            "actual_completion_tokens": None if actual_completion_tokens is None else max(0, int(actual_completion_tokens)),
            "actual_cost": None if actual_cost is None else max(0.0, float(actual_cost)),
        })

    def terminal_reserve_payload(self) -> dict[str, int | float]:
        return {
            "terminal_reserve_tokens": self.terminal_reserve_tokens,
            "terminal_reserve_llm_calls": self.terminal_reserve_llm_calls,
            "terminal_reserve_seconds": self.terminal_reserve_seconds,
        }


def resolve_model_capability(client: Any, *, model: str = "") -> ModelCapability:
    """Read one provider/model capability snapshot without provider-specific branching."""
    candidate = getattr(client, "model_capability", None)
    if isinstance(candidate, ModelCapability):
        if candidate.provider or candidate.model or candidate.context_window is not None:
            return candidate
    features = getattr(client, "capabilities", None)
    converter = getattr(features, "model_capability", None)
    if callable(converter):
        converted = converter()
        if isinstance(converted, ModelCapability):
            return converted
    return ModelCapability(
        provider=type(client).__name__,
        model=model,
    )
