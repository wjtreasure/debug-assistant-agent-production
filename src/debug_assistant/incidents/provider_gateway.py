from __future__ import annotations

import inspect
import multiprocessing
import time
import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any, Callable

from debug_assistant.llm.base import LLMDeadlineExceeded, LLMError, LLMOutputError, LLMResponse, strict_response_format
from debug_assistant.harness.dynamic_budget import validate_request_envelope, resolve_model_capability


class ProviderCallLimitExceeded(LLMError):
    """No physical API attempt may start beyond the configured total."""


class PlannerFailureLayer(str, Enum):
    BUDGET_ADMISSION = "budget_admission"
    TRANSPORT_PROVIDER = "transport_provider"
    PARSE = "parse"
    CONTRACT_SCHEMA = "contract_schema"
    TOOL_ACTION_VALIDATION = "tool_action_validation"
    SEMANTIC_REJECTION = "semantic_rejection"


@dataclass(frozen=True, slots=True)
class ProviderAttemptRecord:
    role: str
    request_kind: str
    attempt_index_within_request: int
    outcome: str
    retry_reason: str
    retry_category: str
    transport_error: str
    http_status: int | None
    latency_seconds: float | None
    usage: tuple[tuple[str, Any], ...]
    provider: str
    model: str
    transport_profile: str
    response_format: str
    same_provider_model: bool
    provider_fallback: bool
    format_fallback: bool
    logical_turn_id: str = ""
    attempt_index_within_turn: int | None = None
    reason_code: str = ""
    attempt_kind: str = "first_attempt"


@dataclass(frozen=True, slots=True)
class ProviderCallRecord:
    request_kind: str
    turn_id: str
    state_snapshot_hash: str
    status: str
    failure_layer: str = ""
    error_type: str = ""
    attempt_count: int = 0
    elapsed_seconds: float = 0.0
    usage: tuple[tuple[str, Any], ...] = ()
    role: str = "planner"
    request_sha256: str = ""
    attempts: tuple[ProviderAttemptRecord, ...] = ()

    @property
    def attempt_details(self) -> list[dict[str, Any]]:
        return [{**asdict(item), "usage": dict(item.usage)} for item in self.attempts]

    @property
    def usage_dict(self) -> dict[str, Any]:
        return dict(self.usage)


def classify_planner_failure(exc: BaseException) -> PlannerFailureLayer:
    """Classify an observed failure without importing semantic owners."""
    if isinstance(exc, ProviderCallLimitExceeded):
        return PlannerFailureLayer.BUDGET_ADMISSION
    if isinstance(exc, LLMOutputError):
        return PlannerFailureLayer.PARSE
    if isinstance(exc, LLMError):
        return PlannerFailureLayer.TRANSPORT_PROVIDER
    metadata = getattr(exc, "metadata", {}) or {}
    error_type = str(metadata.get("error_type", "")).casefold()
    error_layer = str(metadata.get("error_layer", "")).casefold()
    if "parse" in error_type or error_layer == "parser":
        return PlannerFailureLayer.PARSE
    if error_layer == "tool" or any(
        token in error_type for token in ("tool", "action", "executable")
    ):
        return PlannerFailureLayer.TOOL_ACTION_VALIDATION
    if any(token in error_type for token in ("infeasible_skill", "evidence_id")):
        return PlannerFailureLayer.SEMANTIC_REJECTION
    if metadata or isinstance(exc, ValueError):
        return PlannerFailureLayer.CONTRACT_SCHEMA
    return PlannerFailureLayer.SEMANTIC_REJECTION


class ProviderOperationalGateway:
    """Operational boundary shared by all diagnosis Provider roles."""

    def __init__(self, provider: Any, *, transport_profile: str = "role_json", max_context_chars: int = 50_000,
                 max_request_input_tokens: int | None = None, max_llm_calls: int | None = None):
        self.provider = provider
        self.max_context_chars = max_context_chars
        self.max_request_input_tokens = max_request_input_tokens
        self.transport_profile = transport_profile
        self.records: list[ProviderCallRecord] = []
        self.max_llm_calls = max_llm_calls
        self.attempt_admission: Callable[[str, int, int], str] | None = None

    def bind(self, provider: Any, role: str) -> RoleProviderGateway:
        return RoleProviderGateway(self, provider, role)

    def accounting(self, records: Iterable[ProviderCallRecord] | None = None) -> dict[str, Any]:
        """One ledger for artifacts and runtime; admissions are not attempts."""
        records = self.records if records is None else tuple(records)
        attempts = sum(item.attempt_count for item in records)
        kinds = {kind: sum(item.attempt_count for item in records if item.request_kind == kind)
                 for kind in sorted({item.request_kind for item in records})}
        usage = aggregate_provider_usage(records)
        return {
            "physical_provider_calls": attempts, "provider_attempts": attempts,
            "request_records": len(records),
            "same_provider_retries": sum(max(0, item.attempt_count - 1) for item in records),
            "transport_retries": sum(attempt.retry_category == "transport_retry"
                                     for item in records for attempt in item.attempts),
            "schema_repairs": sum(attempt.retry_category == "schema_repair"
                                  for item in records for attempt in item.attempts),
            "unknown_retry_causes": sum(attempt.retry_category == "UNKNOWN"
                                        for item in records for attempt in item.attempts)
                                    + sum(max(0, item.attempt_count - 1)
                                          for item in records if not item.attempts),
            "attempt_details": [attempt for item in records for attempt in item.attempt_details],
            "provider_fallbacks": sum(attempt.provider_fallback for item in records for attempt in item.attempts),
            "format_fallbacks": sum(attempt.format_fallback for item in records for attempt in item.attempts),
            "requests_by_kind": kinds,
            "requests_by_role": {role: sum(item.attempt_count for item in records if item.role == role)
                                 for role in sorted({item.role for item in records})},
            "logical_planner_turns": len({item.turn_id for item in records
                                          if item.role == "planner" and item.turn_id}),
            "provider_latency_seconds": sum(item.elapsed_seconds for item in records),
            "total_tokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
            "token_usage_complete": bool(records) and all(
                any(key in item.usage_dict for key in ("prompt_tokens", "input_tokens"))
                and any(key in item.usage_dict for key in ("completion_tokens", "output_tokens"))
                for item in records
            ),
            **usage,
        }

    def complete_json(self, system: str, user: str, **kwargs: Any) -> Any:
        return self._invoke("complete_json", system, user, **kwargs)

    def complete_structured(self, system: str, user: str, **kwargs: Any) -> Any:
        return self._invoke("complete_structured", system, user, **kwargs)

    def complete_with_tools(self, system: str, user: str, **kwargs: Any) -> Any:
        return self._invoke("complete_with_tools", system, user, **kwargs)

    def _invoke(self, method_name: str, system: str, user: str, **kwargs: Any) -> Any:
        request_kind = str(kwargs.pop("request_kind", method_name))
        turn_id = str(kwargs.pop("turn_id", ""))
        snapshot_hash = str(kwargs.pop("state_snapshot_hash", ""))
        transport_profile = str(kwargs.pop("transport_profile", self.transport_profile))
        contract_retry_reason = str(kwargs.pop("contract_retry_reason", ""))
        hard_timeout = kwargs.pop("hard_timeout_seconds", None)
        provider = kwargs.pop("_provider", self.provider)
        role = str(kwargs.pop("_role", "planner"))
        if hard_timeout is None:
            hard_timeout = kwargs.get("logical_timeout_seconds")
        external_attempt_callback = kwargs.get("on_attempt_started")
        attempt_count = 0
        admission_rejected = False
        process_timeout = bool(
            getattr(provider, "requires_process_timeout", False)
            and hard_timeout is not None
            and "fork" in multiprocessing.get_all_start_methods()
        )
        shared_attempts = multiprocessing.get_context("fork").Value("i", 0) if process_timeout else None

        def on_attempt_started(*args: Any, **callback_kwargs: Any) -> None:
            nonlocal attempt_count, admission_rejected
            used = sum(record.attempt_count for record in self.records) + attempt_count
            if self.max_llm_calls is not None and used >= self.max_llm_calls:
                admission_rejected = True
                raise ProviderCallLimitExceeded("physical API call budget exhausted")
            if self.attempt_admission is not None:
                reviewed = sum(record.attempt_count for record in self.records if record.role == "review")
                reason = self.attempt_admission(role, used, reviewed + (attempt_count if role == "review" else 0))
                if reason:
                    admission_rejected = True
                    raise ProviderCallLimitExceeded(reason)
            # Role-specific admission must succeed before recording an API start.
            if callable(external_attempt_callback):
                external_attempt_callback(*args, **callback_kwargs)
            attempt_count += 1
            if shared_attempts is not None:
                with shared_attempts.get_lock():
                    shared_attempts.value += 1

        method = getattr(provider, method_name, None)
        if not callable(method):
            raise LLMError(f"provider has no {method_name} API")
        call_kwargs = self._supported_kwargs(method, kwargs)
        supports_callback = self._accepts_keyword(method, "on_attempt_started")
        if supports_callback:
            call_kwargs["on_attempt_started"] = on_attempt_started

        event_start = len(getattr(provider, "events", ()) or ())
        if hasattr(provider, "last_usage"):
            provider.last_usage = {}
        previous_usage = getattr(provider, "last_usage", None)
        response_format = strict_response_format(kwargs['schema']) if method_name == 'complete_structured' and kwargs.get('schema') is not None else None
        request_hash = hashlib.sha256(json.dumps({
            "system": system, "user": user,
            "tools": kwargs.get("tools", ()), "tool_choice": kwargs.get("tool_choice"), "model": kwargs.get("model"),
            "response_format": response_format,
            "max_output_tokens": kwargs.get("max_output_tokens"),
            "role": role,
        }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        started = time.monotonic()
        def attempt_details(count: int, status: str):
            preceding = sum(record.attempt_count for record in self.records
                            if turn_id and record.turn_id == turn_id and record.role == role)
            details = _attempt_records(provider, event_start, count, role=role,
                request_kind=request_kind, model=str(kwargs.get("model") or getattr(provider, "model", "")),
                transport_profile=transport_profile, status=status, contract_retry_reason=contract_retry_reason)
            return tuple(replace(item, logical_turn_id=turn_id,
                attempt_index_within_turn=(preceding + item.attempt_index_within_request if turn_id else None),
                reason_code=contract_retry_reason or item.retry_reason,
                attempt_kind=("semantic_contract_repair" if contract_retry_reason and request_kind == "semantic_interpretation"
                              else item.retry_category)) for item in details)
        invoked = False
        try:
            if hard_timeout is not None and float(hard_timeout) <= 0:
                raise LLMDeadlineExceeded("Provider call has no remaining deadline")
            validate_request_envelope(system, user, tools=kwargs.get("tools", ()),
                response_format=response_format,
                capability=resolve_model_capability(provider, model=str(kwargs.get("model") or "")),
                max_input_tokens=self.max_request_input_tokens,
                max_context_chars=self.max_context_chars)
            if self.max_llm_calls is not None and sum(record.attempt_count for record in self.records) >= self.max_llm_calls:
                admission_rejected = True
                raise ProviderCallLimitExceeded("physical API call budget exhausted")
            if self.attempt_admission is not None:
                reason = self.attempt_admission(role, sum(row.attempt_count for row in self.records),
                    sum(row.attempt_count for row in self.records if row.role == "review"))
                if reason:
                    admission_rejected = True
                    raise ProviderCallLimitExceeded(reason)
            if not supports_callback:
                on_attempt_started()
            invoked = True
            if process_timeout:
                result = _call_in_terminable_process(
                    method, (system, user), call_kwargs, float(hard_timeout), provider,
                )
            else:
                result = method(system, user, **call_kwargs)
        except BaseException as exc:
            usage = getattr(provider, "last_usage", None)
            count = self._attempts_since(provider, event_start,
                shared_attempts.value if shared_attempts is not None else attempt_count) or int(invoked and not admission_rejected)
            self.records.append(ProviderCallRecord(
                request_kind=request_kind, turn_id=turn_id,
                state_snapshot_hash=snapshot_hash, status="failed",
                failure_layer=classify_planner_failure(exc).value,
                error_type=type(exc).__name__, role=role,
                attempt_count=count,
                elapsed_seconds=max(0.0, time.monotonic() - started),
                usage=tuple(sorted(dict(usage or {}).items())) if usage is not previous_usage else (),
                request_sha256=request_hash,
                attempts=attempt_details(count, "failed"),
            ))
            raise
        usage = (
            dict(result.usage or {})
            if isinstance(result, LLMResponse)
            else dict(getattr(provider, "last_usage", {}) or {})
        )
        count = self._attempts_since(provider, event_start,
            shared_attempts.value if shared_attempts is not None else attempt_count) or 1
        self.records.append(ProviderCallRecord(
            request_kind=request_kind,
            turn_id=turn_id,
            state_snapshot_hash=snapshot_hash,
            status="completed",
            attempt_count=count,
            elapsed_seconds=max(0.0, time.monotonic() - started),
            usage=tuple(sorted(usage.items())),
            role=role,
            request_sha256=request_hash,
            attempts=attempt_details(count, "completed"),
        ))
        return result

    @staticmethod
    def _signature(method: Callable[..., Any]) -> dict[str, inspect.Parameter]:
        try:
            return dict(inspect.signature(method).parameters)
        except (TypeError, ValueError):
            return {}

    @classmethod
    def _accepts_keyword(cls, method: Callable[..., Any], keyword: str) -> bool:
        parameters = cls._signature(method)
        return keyword in parameters or any(
            value.kind is inspect.Parameter.VAR_KEYWORD for value in parameters.values()
        )

    @classmethod
    def _supported_kwargs(cls, method: Callable[..., Any], values: dict[str, Any]) -> dict[str, Any]:
        parameters = cls._signature(method)
        accepts_kwargs = any(
            value.kind is inspect.Parameter.VAR_KEYWORD for value in parameters.values()
        )
        return {
            key: value for key, value in values.items()
            if key in parameters or accepts_kwargs
        }

    @staticmethod
    def _attempts_since(provider: Any, cursor: int, callback_count: int) -> int:
        events = list(getattr(provider, "events", ()) or ())[cursor:]
        event_count = sum(
            1 for event in events if event.get("type") == "LLM_ATTEMPT_STARTED"
        )
        return max(event_count, callback_count)


def _attempt_records(provider, cursor, count, *, role, request_kind, model, transport_profile, status,
                     contract_retry_reason=""):
    """Use allowlisted wire events; absent historical causes remain UNKNOWN."""
    events = [{**event.get("payload", {}), **{key: value for key, value in event.items() if key != "payload"}}
              for event in list(getattr(provider, "events", ()) or ())[cursor:]]
    starts = [event for event in events if event.get("type") == "LLM_ATTEMPT_STARTED"]
    records = []
    previous_error = ""
    initial_format = starts[0].get("response_format", "UNKNOWN") if starts else "UNKNOWN"
    provider_name = str(getattr(getattr(provider, "capabilities", None), "provider", "") or type(provider).__name__)
    for index in range(1, count + 1):
        start = starts[index - 1] if index <= len(starts) else {}
        finish = next((event for event in reversed(events)
                       if event.get("attempt_index") == start.get("attempt_index", index)
                       and event.get("type") in {"LLM_ATTEMPT_FINISHED", "LLM_ATTEMPT_FAILED"}), {})
        error = str(finish.get("error_type", ""))
        elapsed = finish.get("attempt_elapsed_ms")
        wire_format = str(start.get("response_format", "UNKNOWN"))
        records.append(ProviderAttemptRecord(
            role=role, request_kind=request_kind, attempt_index_within_request=index,
            outcome=("failed" if finish.get("type") == "LLM_ATTEMPT_FAILED" else
                     "completed" if finish.get("success") else status if index == count else "UNKNOWN"),
            retry_reason=previous_error if index > 1 else contract_retry_reason or "first_attempt",
            retry_category=("format_fallback" if index > 1 and previous_error == "json_mode_unsupported"
                            else "transport_retry" if index > 1 and previous_error != "UNKNOWN"
                            else "UNKNOWN" if index > 1
                            else "schema_repair" if contract_retry_reason or request_kind.endswith("schema_repair")
                            else "first_attempt"),
            transport_error=error, http_status=finish.get("status_code"),
            latency_seconds=float(elapsed) / 1000 if elapsed is not None else None,
            usage=tuple(sorted(dict(finish.get("usage") or {}).items())),
            provider=provider_name, model=str(start.get("model", model)),
            transport_profile=transport_profile or "role_json",
            response_format=wire_format, same_provider_model=True, provider_fallback=False,
            format_fallback=wire_format != initial_format and "UNKNOWN" not in (wire_format, initial_format),
        ))
        previous_error = error or "UNKNOWN"
    return tuple(records)


class RoleProviderGateway:
    """A role/provider binding; invocation and records belong to one gateway."""

    def __init__(self, gateway: ProviderOperationalGateway, provider: Any, role: str):
        self.gateway, self.provider, self.role = gateway, provider, role

    @property
    def records(self) -> list[ProviderCallRecord]:
        return [record for record in self.gateway.records if record.role == self.role]

    def complete_json(self, system: str, user: str, **kwargs: Any) -> Any:
        return self.gateway._invoke(
            "complete_json", system, user, _provider=self.provider, _role=self.role, **kwargs,
        )

    def complete_structured(self, system: str, user: str, **kwargs: Any) -> Any:
        return self.gateway._invoke(
            "complete_structured", system, user, _provider=self.provider, _role=self.role, **kwargs,
        )

    def complete_with_tools(self, system: str, user: str, **kwargs: Any) -> Any:
        return self.gateway._invoke(
            "complete_with_tools", system, user, _provider=self.provider, _role=self.role, **kwargs,
        )


def aggregate_provider_usage(records) -> dict[str, int | float]:
    totals: dict[str, int | float] = {}
    for record in records:
        usage = record.usage_dict
        for canonical, aliases in (
            ("prompt_tokens", ("prompt_tokens", "input_tokens")),
            ("completion_tokens", ("completion_tokens", "output_tokens")),
            ("cost", ("cost", "total_cost", "estimated_cost")),
            ("cached_tokens", ("cached_tokens", "cache_read_input_tokens")),
            ("provider_overhead_tokens", ("provider_overhead_tokens",)),
        ):
            value = next((usage[key] for key in aliases if key in usage), None)
            if value is None and canonical == "cached_tokens":
                value = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[canonical] = totals.get(canonical, 0) + value
    return totals


def _provider_worker(connection: Any, method: Callable[..., Any], args: tuple[Any, ...],
                     kwargs: dict[str, Any], provider: Any) -> None:
    try:
        connection.send(("ok", method(*args, **kwargs), _provider_state(provider)))
    except BaseException as exc:
        connection.send(("error", exc, _provider_state(provider)))
    finally:
        connection.close()


def _provider_state(provider: Any) -> dict[str, Any]:
    state = {
        "events": list(getattr(provider, "events", ()) or ()),
        "calls": list(getattr(provider, "calls", ()) or ()),
        "last_usage": dict(getattr(provider, "last_usage", {}) or {}),
    }
    for name in ("last_raw_content", "last_raw_output"):
        if hasattr(provider, name):
            state[name] = getattr(provider, name)
    return state


def _call_in_terminable_process(method: Callable[..., Any], args: tuple[Any, ...],
                                kwargs: dict[str, Any], timeout: float, provider: Any) -> Any:
    if "fork" not in multiprocessing.get_all_start_methods():
        return method(*args, **kwargs)
    context = multiprocessing.get_context("fork")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_provider_worker,
        args=(send, method, args, kwargs, provider),
        daemon=True,
    )
    process.start()
    send.close()
    try:
        if not receive.poll(timeout):
            process.terminate()
            process.join(timeout=1.0)
            raise LLMDeadlineExceeded(
                f"Provider call exceeded hard deadline of {timeout:.2f}s"
            )
        status, payload, state = receive.recv()
    finally:
        receive.close()
    process.join(timeout=0.5)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
    for name, value in state.items():
        if hasattr(provider, name):
            setattr(provider, name, value)
    if status == "ok":
        return payload
    raise payload
