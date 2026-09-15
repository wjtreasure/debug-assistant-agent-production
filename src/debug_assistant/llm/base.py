from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping
import math
import inspect
import json
import re


class LLMError(RuntimeError):
    pass


class LLMTransportTimeout(LLMError):
    """One provider transport attempt timed out before the logical deadline."""


class LLMDeadlineExceeded(LLMError):
    """The total logical LLM-call budget was exhausted."""


class LLMClientUsageError(LLMError):
    """The caller used the sync/async client API in an unsupported context."""


class LLMOutputError(LLMError):
    """The provider responded successfully but returned unusable model output."""


class LLMInvalidJSON(LLMOutputError):
    pass


class LLMToolArgumentsError(LLMOutputError):
    """A provider-native tool call contains arguments that cannot be decoded."""

    error_type = "tool_arguments_invalid_json"

    def __init__(self, message: str, *, tool: str | None = None, index: int | None = None):
        self.tool = tool
        self.index = index
        super().__init__(message)


def estimate_tokens_char4(text: str) -> int:
    """Conservative, provider-neutral fallback when no tokenizer is exposed."""
    if not text:
        return 0
    return max(1, math.ceil(len(str(text)) / 4))


@dataclass(frozen=True, slots=True)
class ModelCapability:
    """Physical model limits and token-estimation capability.

    Feature negotiation remains in :class:`ProviderCapabilities`.  This
    object describes the model's context boundary and is intentionally
    provider/model agnostic so the runtime can use the same budget policy for
    OpenAI-compatible, mock, or future adapters.
    """

    provider: str = ""
    model: str = ""
    context_window: int | None = None
    max_output_tokens: int | None = None
    tokenizer: str = "char4"
    token_estimator: Callable[[str], int] | None = field(
        default=None, compare=False, repr=False
    )
    reserved_output_tokens: int = 0
    protocol_safety_reserve_tokens: int = 0

    def __post_init__(self) -> None:
        if self.context_window is not None and int(self.context_window) <= 0:
            raise ValueError("context_window must be positive when configured")
        if self.max_output_tokens is not None and int(self.max_output_tokens) <= 0:
            raise ValueError("max_output_tokens must be positive when configured")
        if self.reserved_output_tokens < 0 or self.protocol_safety_reserve_tokens < 0:
            raise ValueError("model token reserves must be non-negative")

    @property
    def context_window_tokens(self) -> int | None:
        """Compatibility spelling used by configuration and benchmark code."""
        return self.context_window

    @property
    def input_hard_capacity(self) -> int | None:
        if self.context_window is None:
            return None
        output_reserve = self.reserved_output_tokens
        if output_reserve <= 0 and self.max_output_tokens is not None:
            output_reserve = self.max_output_tokens
        return max(
            0,
            int(self.context_window)
            - max(0, int(output_reserve))
            - max(0, int(self.protocol_safety_reserve_tokens)),
        )

    def estimate_tokens(self, text: str) -> int:
        estimator = self.token_estimator or estimate_tokens_char4
        try:
            return max(0, int(estimator(text)))
        except (TypeError, ValueError):
            return estimate_tokens_char4(text)

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "tokenizer": self.tokenizer,
            "reserved_output_tokens": self.reserved_output_tokens,
            "protocol_safety_reserve_tokens": self.protocol_safety_reserve_tokens,
            "input_hard_capacity": self.input_hard_capacity,
        }


# Singular spelling used by the Dynamic Budget specification.  Keep the
# existing plural feature-capability type below for backward compatibility.
ProviderCapability = ModelCapability


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """Capabilities negotiated by a provider adapter, never inferred by the agent."""

    json_object: bool = True
    json_schema: bool = False
    tool_calling: bool = False
    parallel_tool_calls: bool = False
    # Optional physical-limit fields are additive. Existing providers that do
    # not advertise a model window continue to use the legacy context fallback.
    provider: str = ""
    model: str = ""
    context_window: int | None = None
    max_output_tokens: int | None = None
    tokenizer: str = "char4"
    token_estimator: Callable[[str], int] | None = field(
        default=None, compare=False, repr=False
    )
    reserved_output_tokens: int = 0
    protocol_safety_reserve_tokens: int = 0

    def model_capability(self) -> ModelCapability:
        return ModelCapability(
            provider=self.provider,
            model=self.model,
            context_window=self.context_window,
            max_output_tokens=self.max_output_tokens,
            tokenizer=self.tokenizer,
            token_estimator=self.token_estimator,
            reserved_output_tokens=self.reserved_output_tokens,
            protocol_safety_reserve_tokens=self.protocol_safety_reserve_tokens,
        )


@dataclass(frozen=True, slots=True)
class LLMToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Provider-neutral response envelope used by typed runtime paths."""

    content: str | None = None
    structured: Any = None
    tool_calls: tuple[LLMToolCall, ...] = field(default_factory=tuple)
    usage: Mapping[str, Any] = field(default_factory=dict)
    # Bounded provider-safe material for contract audits. Adapters must omit
    # hidden reasoning fields and credentials.
    raw_output: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class LLMClient(ABC):
    capabilities = ProviderCapabilities()
    model_capability = ModelCapability()

    @abstractmethod
    def complete_json(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        logical_timeout_seconds: float | None = None,
        on_attempt_started: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]: ...

    def complete_structured(self, system: str, user: str, *, schema=None, model=None,
                            logical_timeout_seconds=None, on_attempt_started=None) -> LLMResponse:
        data = complete_json_compat(self, system, user, model=model,
                                    logical_timeout_seconds=logical_timeout_seconds,
                                    on_attempt_started=on_attempt_started)
        return LLMResponse(content=json.dumps(data, ensure_ascii=False), structured=data,
                           usage=getattr(self, "last_usage", {}) or {})

    def complete_with_tools(self, system: str, user: str, *, tools: list[dict[str, Any]],
                            model=None, logical_timeout_seconds=None,
                            on_attempt_started=None) -> LLMResponse:
        """Safe default for providers without native tools: use structured JSON fallback."""
        data = self.complete_structured(system, user, schema=None, model=model,
                                        logical_timeout_seconds=logical_timeout_seconds,
                                        on_attempt_started=on_attempt_started)
        return data


def complete_json_compat(
    client,
    system: str,
    user: str,
    *,
    model: str | None = None,
    logical_timeout_seconds: float | None = None,
    on_attempt_started: Callable[[dict[str, Any]], None] | None = None,
):
    """Pass V1.4.5 deadline metadata without breaking legacy test/provider doubles."""
    fn = client.complete_json
    try:
        sig = inspect.signature(fn)
        accepts_timeout = (
            "logical_timeout_seconds" in sig.parameters
            or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        )
    except (TypeError, ValueError):
        accepts_timeout = False
    kwargs={"model":model}
    if accepts_timeout:
        kwargs["logical_timeout_seconds"]=logical_timeout_seconds
    try:
        sig = inspect.signature(fn)
        accepts_callback = (
            "on_attempt_started" in sig.parameters
            or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        )
    except (TypeError, ValueError):
        accepts_callback = False
    if accepts_callback:
        kwargs["on_attempt_started"]=on_attempt_started
    return fn(system, user, **kwargs)


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{"); end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise LLMInvalidJSON(f"Model did not return valid JSON: {text[:500]}")


def parse_tool_calls(message: Mapping[str, Any]) -> tuple[LLMToolCall, ...]:
    """Parse OpenAI-compatible function calls without trusting them as a safety boundary."""
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list):
        raise LLMOutputError("Provider tool_calls must be a list")
    parsed = []
    for index, row in enumerate(calls):
        if not isinstance(row, Mapping):
            error = LLMOutputError("Provider returned a malformed tool call")
            error.error_type = "malformed_tool_call"
            raise error
        function = row.get("function") or {}
        name = function.get("name") if isinstance(function, Mapping) else None
        raw_args = function.get("arguments", "{}") if isinstance(function, Mapping) else None
        if not isinstance(name, str) or not name.strip():
            error = LLMOutputError("Provider tool call has no function name")
            error.error_type = "malformed_tool_call"
            raise error
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                raise LLMToolArgumentsError(f"Malformed arguments for tool {name}", tool=name, index=index) from exc
        if not isinstance(raw_args, dict):
            raise LLMToolArgumentsError(f"Arguments for tool {name} must be an object", tool=name, index=index)
        parsed.append(LLMToolCall(str(row.get("id") or f"tool-call-{index + 1}"), name, raw_args))
    return tuple(parsed)
