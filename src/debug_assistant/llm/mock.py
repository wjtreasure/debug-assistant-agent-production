from __future__ import annotations
from typing import Any, Iterable
from .base import LLMClient, LLMResponse, LLMToolCall, ProviderCapabilities, ModelCapability

class MockLLMClient(LLMClient):
    """Deterministic smoke-test provider. Production diagnosis must use a real model."""
    def __init__(self, *, tool_calls: Iterable[LLMToolCall | dict[str, Any]] | None = None,
                 tool_calling: bool = False, responses: Iterable[Any] | None = None,
                 native_responses: Iterable[LLMResponse | dict[str, Any]] | None = None,
                 model_capability: ModelCapability | None = None):
        self.n=0; self.calls=[]; self.last_raw_content=None; self.last_usage={}
        self.capabilities=ProviderCapabilities(
            tool_calling=tool_calling, parallel_tool_calls=tool_calling,
            provider="mock", model="mock",
        )
        self.model_capability = model_capability or self.capabilities.model_capability()
        self._tool_calls=list(tool_calls or []); self._responses=list(responses or [])
        self._native_responses=list(native_responses or [])
    def complete_json(self, system: str, user: str, *, model: str | None=None, logical_timeout_seconds: float | None=None, max_output_tokens: int | None=None) -> dict[str, Any]:
        self.n+=1
        self.calls.append({
            "model":"mock","prompt_tokens":0,"completion_tokens":0,"total_tokens":0,
            "input_tokens":0,"output_tokens":0,"cached_tokens":None,"reasoning_tokens":None,
            "prompt_chars":len(system)+len(user),"completion_chars":0,"latency_ms":0.0,
        })
        self.last_usage=self.calls[-1]
        if self._responses:
            value=self._responses.pop(0)
            if isinstance(value, dict): return value
        # Keep the deterministic adapter aligned with the active CloudOps
        # contracts.  It is a transport fixture, not a second Planner or a
        # source of generic/SWE Skill names.  Callers that need a full
        # lifecycle should provide explicit ``responses``/``tool_calls`` or
        # use the manifest-backed final smoke provider.
        return {
            "decision": "REJECT",
            "unsupported_claims": [],
            "missing_evidence": ["A real Provider response is required for semantic diagnosis."],
            "blocking_contradictions": [],
            "contradictions": [],
            "recoverability": {"recoverable": True, "unrecoverable": False},
            "targeted_followup": "Use an OpenAI-compatible Provider or an explicit deterministic fixture.",
            "suggested_investigation": "Use an OpenAI-compatible Provider or an explicit deterministic fixture.",
            "causal_chain_valid": False,
            "causal_gaps": ["semantic Provider response unavailable"],
            "reason": "The mock adapter does not perform semantic reasoning.",
        }

    def complete_with_tools(self, system: str, user: str, *, tools: list[dict[str, Any]],
                            model: str | None = None, logical_timeout_seconds: float | None = None,
                            on_attempt_started=None, max_output_tokens: int | None = None) -> LLMResponse:
        if not self.capabilities.tool_calling:
            return super().complete_with_tools(
                system, user, tools=tools, model=model,
                logical_timeout_seconds=logical_timeout_seconds,
                on_attempt_started=on_attempt_started,
                max_output_tokens=max_output_tokens,
            )
        self.n += 1
        self.calls.append({"model":"mock","prompt_tokens":0,"completion_tokens":0,"total_tokens":0,
                           "input_tokens":0,"output_tokens":0,"prompt_chars":len(system)+len(user),
                           "completion_chars":0,"latency_ms":0.0})
        self.last_usage=self.calls[-1]
        if self._native_responses:
            raw=self._native_responses.pop(0)
            if isinstance(raw, LLMResponse):
                return raw
            if isinstance(raw, dict):
                raw_calls=[]
                for index, item in enumerate(raw.get("tool_calls") or []):
                    if isinstance(item, LLMToolCall):
                        raw_calls.append(item)
                    elif isinstance(item, dict):
                        raw_calls.append(LLMToolCall(
                            str(item.get("id", f"mock-call-{index + 1}")),
                            str(item["name"]), dict(item.get("arguments") or {}),
                        ))
                return LLMResponse(
                    content=raw.get("content"), structured=raw.get("structured"),
                    tool_calls=tuple(raw_calls), usage=self.last_usage,
                    raw_output=dict(raw.get("raw_output") or {}),
                )
        calls=[]
        for index, raw in enumerate(self._tool_calls):
            if isinstance(raw, LLMToolCall):
                calls.append(raw)
            else:
                calls.append(LLMToolCall(str(raw.get("id", f"mock-call-{index + 1}")),
                                         str(raw["name"]), dict(raw.get("arguments") or {})))
        return LLMResponse(
            content="", structured=None, tool_calls=tuple(calls), usage=self.calls[-1],
            raw_output={
                "content": "",
                "tool_calls": [
                    {"id": call.id, "type": "function", "function": {
                        "name": call.name, "arguments": dict(call.arguments),
                    }}
                    for call in calls
                ],
            },
        )
