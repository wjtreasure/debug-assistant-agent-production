from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from debug_assistant.agent.planner import PlannerContractError, PlannerContractExhausted
from debug_assistant.incidents.provider_gateway import (
    PlannerFailureLayer,
    ProviderCallRecord,
    ProviderOperationalGateway,
    classify_planner_failure,
    aggregate_provider_usage,
)
from debug_assistant.llm.base import LLMResponse
from debug_assistant.context.requests import ContextResolutionError


@dataclass(frozen=True, slots=True)
class PlanningSessionResult:
    planner_result: Any
    turn_input: Any
    request_snapshots: tuple[dict[str, Any], ...]
    provider_calls: tuple[ProviderCallRecord, ...]
    usage: tuple[tuple[str, int | float], ...]
    contract_retry_count: int = 0
    context_resolution_rounds: int = 0

    @property
    def response(self) -> LLMResponse:
        return self.planner_result.response

    @property
    def physical_request_count(self) -> int:
        return len(self.provider_calls)

    @property
    def provider_attempt_count(self) -> int:
        return sum(item.attempt_count for item in self.provider_calls)

    @property
    def usage_dict(self) -> dict[str, int | float]:
        return dict(self.usage)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.planner_result, name)


@dataclass(slots=True)
class LogicalTurnRetryBudget:
    limit: int = 1
    used: int = 0

    def consume(self, exc):
        if self.used >= self.limit:
            raise PlannerContractExhausted(exc) from exc
        self.used += 1


class PlanningSession:
    """Run the fixed Planner transport over one immutable turn input."""

    CONTEXT_ROUND_LIMIT = 2

    def __init__(self, planner: Any, gateway: ProviderOperationalGateway):
        self.planner = planner
        self.gateway = gateway
        self.last_failure_layer: PlannerFailureLayer | None = None
        self.last_request_snapshots: tuple[dict[str, Any], ...] = ()
        self.last_provider_calls: tuple[ProviderCallRecord, ...] = ()

    def run(
        self,
        turn_input: Any,
        *,
        on_contract_retry: Callable[[BaseException], None] | None = None,
        **kwargs: Any,
    ) -> PlanningSessionResult:
        record_cursor = len(self.gateway.records)
        self.last_failure_layer = None
        self.last_request_snapshots = ()
        self.last_provider_calls = ()
        snapshots: list[dict[str, Any]] = []
        retry_count = 0
        contract_feedback = None
        commit = kwargs.pop("on_interpretation_commit", None)
        retry_budget = kwargs.pop("retry_budget", None) or LogicalTurnRetryBudget()
        committed = False
        resolve_context = kwargs.pop('on_context_request', None)
        on_context_feedback = kwargs.pop('on_context_feedback', None)
        context_rounds = 0
        # Local resolution is legal output, with its own narrow fixed bound.
        # This never expands Provider or contract-retry allowance.
        context_round_limit = self.CONTEXT_ROUND_LIMIT
        seen_context_requests = set()

        def commit_once(selection, response):
            nonlocal committed
            if committed:
                raise RuntimeError("semantic commit may occur only once per PlanningSession")
            result = commit(selection, response)
            committed = True
            return result

        while True:
            try:
                result = self.planner.propose(
                    turn_input,
                    provider_gateway=self.gateway,
                    contract_retry=contract_feedback is not None,
                    contract_feedback=contract_feedback,
                    context_resolution_round=context_rounds,
                    context_resolution_remaining=context_round_limit - context_rounds,
                    **kwargs,
                )
                if result.context_requests is not None:
                    for snapshot in self.planner.last_request_snapshots:
                        snapshot['context_resolution_round'] = context_rounds
                    snapshots.extend(self.planner.last_request_snapshots)
                    self.planner.last_request_snapshots = []
                    self.last_request_snapshots = tuple(snapshots)
                    self.last_provider_calls = tuple(self.gateway.records[record_cursor:])
                    signature = (turn_input.state_version, turn_input.semantic_projection_json,
                                 turn_input.context_view, result.context_requests.model_dump_json())
                    if signature in seen_context_requests:
                        raise ContextResolutionError('context_request_repeated')
                    if context_rounds >= context_round_limit:
                        raise ContextResolutionError('context_resolution_limit')
                    if resolve_context is None:
                        raise ContextResolutionError('context_resolution_unavailable')
                    seen_context_requests.add(signature)
                    context_rounds += 1  # Failed packing consumes the same two-interaction allowance.
                    try:
                        updated = resolve_context(result.context_requests, turn_input, context_rounds)
                    except ContextResolutionError as exc:
                        if (exc.reason_code != 'context_packing_insufficient' or not exc.feedback
                                or context_rounds >= context_round_limit):
                            raise
                        updated = self.planner.with_context_view(turn_input, turn_input.context_view,
                                                                 local_feedback=exc.feedback)
                        if on_context_feedback is not None:
                            on_context_feedback(exc, context_rounds)
                    if (updated.state_version != turn_input.state_version
                            or updated.semantic_projection_json != turn_input.semantic_projection_json):
                        raise RuntimeError('context resolution must preserve canonical semantic snapshot')
                    turn_input = updated
                    # Local body recovery preserves the same semantic contract and retry budget.
                    continue
                if commit is not None:
                    commit_once(result.skill_selections[0], result.response)
                snapshots.extend(self.planner.last_request_snapshots)
                records = tuple(self.gateway.records[record_cursor:])
                self.last_request_snapshots = tuple(snapshots)
                self.last_provider_calls = records
                return PlanningSessionResult(
                    planner_result=result,
                    turn_input=turn_input,
                    request_snapshots=self.last_request_snapshots,
                    provider_calls=records,
                    usage=tuple(sorted(aggregate_provider_usage(records).items())),
                    contract_retry_count=retry_count,
                    context_resolution_rounds=context_rounds,
                )
            except PlannerContractError as exc:
                snapshots.extend(self.planner.last_request_snapshots)
                self.last_failure_layer = classify_planner_failure(exc)
                if committed:
                    self.last_request_snapshots = tuple(snapshots)
                    self.last_provider_calls = tuple(self.gateway.records[record_cursor:])
                    raise
                self.last_request_snapshots = tuple(snapshots)
                self.last_provider_calls = tuple(self.gateway.records[record_cursor:])
                retry_budget.consume(exc)
                retry_count += 1
                metadata = exc.metadata
                contract_feedback = {
                    "reason_code": metadata.get("error_type", "planner_contract_invalid"),
                    "fields": metadata.get("validation_errors", []),
                    **{key: metadata[key] for key in (
                        "operation", "unsupported_constraint_keys", "allowed_constraint_keys",
                        "validation_details",
                        "choice_feedback", "target_feedback", "evidence_feedback", "requirement_feedback", "capability_feedback",
                    ) if key in metadata},
                }
                self.last_request_snapshots = tuple(snapshots)
                self.last_provider_calls = tuple(self.gateway.records[record_cursor:])
                if on_contract_retry is not None:
                    on_contract_retry(exc)
            except BaseException as exc:
                snapshots.extend(self.planner.last_request_snapshots)
                self.last_failure_layer = classify_planner_failure(exc)
                self.last_request_snapshots = tuple(snapshots)
                self.last_provider_calls = tuple(self.gateway.records[record_cursor:])
                raise


__all__ = [
    "PlannerFailureLayer",
    "PlanningSession",
    "PlanningSessionResult",
    "ProviderCallRecord",
    "ProviderOperationalGateway",
    "classify_planner_failure",
]
