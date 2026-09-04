from __future__ import annotations
from typing import Any, Iterable
import json
from .base import LLMClient, LLMResponse, LLMToolCall, ProviderCapabilities

class MockLLMClient(LLMClient):
    """Deterministic smoke-test provider. Production diagnosis must use a real model."""
    def __init__(self, *, tool_calls: Iterable[LLMToolCall | dict[str, Any]] | None = None,
                 tool_calling: bool = False, responses: Iterable[Any] | None = None,
                 native_responses: Iterable[LLMResponse | dict[str, Any]] | None = None):
        self.n=0; self.calls=[]; self.last_raw_content=None; self.last_usage={}
        self.capabilities=ProviderCapabilities(tool_calling=tool_calling, parallel_tool_calls=tool_calling)
        self._tool_calls=list(tool_calls or []); self._responses=list(responses or [])
        self._native_responses=list(native_responses or [])
    def complete_json(self, system: str, user: str, *, model: str | None=None, logical_timeout_seconds: float | None=None) -> dict[str, Any]:
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
        low=user.lower()
        if 'final_report_schema' in low:
            return {"summary":"Smoke-test diagnosis", "root_cause":"Evidence suggests the fixture parser boundary check is the likely defect.",
                    "likely_files":["src/parser.py"], "likely_symbols":["parse_value"], "impact_scope":["parser callers"],
                    "recommended_change_points":[{"file":"src/parser.py","symbol":"parse_value","reason":"boundary handling"}],
                    "uncertainties":["Mock provider does not perform semantic reasoning"], "next_checks":["Review boundary tests"], "confidence":0.62}
        if 'reflection_schema' in low:
            return {"decision":"continue","reason":"collect one more targeted source excerpt","current_diagnosis":"fixture parser boundary validation",
                    "evidence_sufficient":False,"supporting_evidence_ids":[],"contradicting_evidence_ids":[],
                    "required_missing_evidence":[{"target":"target implementation","location":"src/parser.py","reason":"source is required to ground the mechanism"}],
                    "optional_validation":[],"recommended_next_goal":"inspect target implementation","confidence":0.6,"hypothesis_changed":None}
        if self.n == 1:
            return {"kind":"tool","skill":"repository_exploration","reason":"find issue keywords","confidence":0.8,
                    "tool":"grep","arguments":{"query":"parse_value|boundary|invalid","glob":"*.py","max_results":20},"expected_evidence":"candidate implementation"}
        if self.n == 2:
            return {"kind":"tool","skill":"hypothesis_validation","reason":"read candidate source","confidence":0.9,
                    "tool":"read_file","arguments":{"path":"src/parser.py","start_line":1,"end_line":120},"expected_evidence":"implementation details"}
        return {"kind":"finish","skill":"report_synthesis","reason":"enough evidence for smoke test","confidence":0.75,"tool":None,"arguments":{},"expected_evidence":""}

    def complete_with_tools(self, system: str, user: str, *, tools: list[dict[str, Any]],
                            model: str | None = None, logical_timeout_seconds: float | None = None,
                            on_attempt_started=None) -> LLMResponse:
        if not self.capabilities.tool_calling:
            return super().complete_with_tools(
                system, user, tools=tools, model=model,
                logical_timeout_seconds=logical_timeout_seconds,
                on_attempt_started=on_attempt_started,
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
                )
        calls=[]
        for index, raw in enumerate(self._tool_calls):
            if isinstance(raw, LLMToolCall):
                calls.append(raw)
            else:
                calls.append(LLMToolCall(str(raw.get("id", f"mock-call-{index + 1}")),
                                         str(raw["name"]), dict(raw.get("arguments") or {})))
        return LLMResponse(content="", structured=None, tool_calls=tuple(calls), usage=self.calls[-1])
