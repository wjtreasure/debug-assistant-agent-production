import json

import pytest
import httpx

from debug_assistant.agent.planner import NativePlannerContractError, NativeToolPlanner, PlannerContractError
from debug_assistant.contracts import ReflectionDecision
from debug_assistant.harness.obligations import EvidenceObligationTracker
from debug_assistant.harness.semantic_reducer import SemanticReducer
from debug_assistant.harness.tool_orchestrator import RequestedToolCall, ToolOrchestrator, ToolPlanningError
from debug_assistant.llm.base import LLMInvalidJSON, LLMOutputError, LLMResponse, LLMToolCall, ProviderCapabilities, parse_tool_calls
from debug_assistant.llm.mock import MockLLMClient
from debug_assistant.memory.hypothesis import HypothesisManager
from debug_assistant.models import AgentState, Evidence, TaskSpec
from debug_assistant.tools.registry import ToolRegistry


def test_provider_parses_single_and_multiple_tool_calls():
    calls = parse_tool_calls({"tool_calls": [
        {"id": "a", "function": {"name": "grep", "arguments": '{"query":"needle"}'}},
        {"id": "b", "function": {"name": "read_file", "arguments": {"path": "a.py"}}},
    ]})
    assert [x.name for x in calls] == ["grep", "read_file"]
    assert calls[0].arguments == {"query": "needle"}


def test_provider_rejects_malformed_tool_arguments():
    with pytest.raises(LLMOutputError):
        parse_tool_calls({"tool_calls": [{"function": {"name": "grep", "arguments": "["}}]})


def test_mock_tool_calling_and_capabilities():
    client = MockLLMClient(tool_calling=True, tool_calls=[
        {"id": "m1", "name": "grep", "arguments": {"query": "x"}},
    ])
    response = client.complete_with_tools("s", "u", tools=[])
    assert client.capabilities.tool_calling
    assert response.tool_calls == (LLMToolCall("m1", "grep", {"query": "x"}),)
    assert isinstance(response, LLMResponse)


def test_tool_function_schema_is_derived_from_pydantic_schema(tmp_path):
    registry = ToolRegistry(tmp_path)
    read = next(x for x in registry.function_schemas() if x["function"]["name"] == "read_file")
    assert read["function"]["parameters"] == registry.get("read_file").spec.json_schema()


def test_native_planner_returns_tool_calls_without_execution_shape(tmp_path):
    client = MockLLMClient(tool_calling=True, tool_calls=[
        {"id": "m1", "name": "grep", "arguments": {"query": "needle"}},
    ])
    planner = NativeToolPlanner(client, ToolRegistry(tmp_path))
    result = planner.propose(AgentState(TaskSpec("t", "find needle", str(tmp_path))), "context")
    assert result.tool_calls[0].name == "grep"
    assert not hasattr(result, "kind")


def test_native_planner_unknown_tool_fails_closed(tmp_path):
    client = MockLLMClient(tool_calling=True, tool_calls=[
        {"id": "m1", "name": "noop", "arguments": {}},
    ])
    with pytest.raises(PlannerContractError):
        NativeToolPlanner(client, ToolRegistry(tmp_path)).propose(
            AgentState(TaskSpec("t", "issue", str(tmp_path))), "context"
        )


def test_orchestrator_splits_read_file_and_preserves_metadata(tmp_path):
    orchestrator = ToolOrchestrator(ToolRegistry(tmp_path), max_tool_calls=4)
    plan = orchestrator.build_plan([RequestedToolCall("r", "read_file", {"path": "a.py", "start_line": 200, "end_line": 460}, "N1", ("O1",))])
    assert [(x.arguments["start_line"], x.arguments["end_line"]) for x in plan.calls] == [(200, 399), (400, 460)]
    assert all(x.requested_range["end_line"] == 460 for x in plan.calls)
    assert all(x.request.information_need_id == "N1" and x.request.obligation_ids == ("O1",) for x in plan.calls)


def test_orchestrator_can_pad_short_source_reads_without_changing_requested_range(tmp_path):
    orchestrator = ToolOrchestrator(ToolRegistry(tmp_path), max_tool_calls=4,
                                    read_context_padding=60)
    plan = orchestrator.build_plan([
        RequestedToolCall("r", "read_file", {"path": "a.py", "start_line": 300, "end_line": 360})
    ])
    assert len(plan.calls) == 1
    call = plan.calls[0]
    assert (call.arguments["start_line"], call.arguments["end_line"]) == (240, 420)
    assert call.requested_range == {"path": "a.py", "start_line": 300, "end_line": 360}


def test_orchestrator_execution_keeps_split_linkage_on_each_observation(tmp_path):
    (tmp_path / "a.py").write_text("value = 1\n" * 401, encoding="utf-8")
    orchestrator = ToolOrchestrator(ToolRegistry(tmp_path), max_tool_calls=3)
    plan = orchestrator.build_plan([
        RequestedToolCall("r", "read_file", {"path": "a.py", "start_line": 1, "end_line": 401}, "N1", ("O1",))
    ])
    observations = orchestrator.execute(plan)
    assert len(observations) == 3
    assert all(observation.ok for observation in observations)
    assert all(observation.metadata["information_need_id"] == "N1" for observation in observations)
    assert all(observation.metadata["obligation_ids"] == ["O1"] for observation in observations)


def test_orchestrator_groups_safe_and_serial_tools_deterministically(tmp_path):
    orchestrator = ToolOrchestrator(ToolRegistry(tmp_path), max_parallel_actions=2)
    plan = orchestrator.build_plan([
        RequestedToolCall("0", "repo_tree", {}),
        RequestedToolCall("1", "grep", {"query": "x"}),
        RequestedToolCall("2", "read_file", {"path": "a.py"}),
    ])
    assert [[x.request.name for x in group] for group in plan.parallel_groups] == [["grep", "read_file"]]
    assert [x.request.name for x in plan.serial_calls] == ["repo_tree"]
    assert [x.request.name for x in plan.calls] == ["repo_tree", "grep", "read_file"]


def test_orchestrator_unknown_tool_and_expansion_budget_fail_closed(tmp_path):
    registry = ToolRegistry(tmp_path)
    with pytest.raises(ToolPlanningError) as unknown:
        ToolOrchestrator(registry).build_plan([RequestedToolCall("x", "noop")])
    assert unknown.value.error_type == "unknown_tool"
    with pytest.raises(ToolPlanningError) as budget:
        ToolOrchestrator(registry, max_tool_calls=1).build_plan([
            RequestedToolCall("x", "read_file", {"path": "a.py", "start_line": 1, "end_line": 201})
        ])
    assert budget.value.error_type == "tool_budget_preflight"


def test_reflection_decision_does_not_accept_derived_state_fields():
    with pytest.raises(Exception):
        ReflectionDecision.model_validate({"diagnosis": "x", "evidence_sufficient": True})


def test_semantic_reducer_never_supports_with_required_gap(tmp_path):
    tracker = EvidenceObligationTracker(repo_root=tmp_path)
    manager = HypothesisManager(tmp_path)
    evidence = Evidence("ev-1", "read_file", "read_file", "source", file="a.py", source_start_line=1, source_end_line=2)
    reducer = SemanticReducer(tracker, manager, [evidence])
    candidate = reducer.reduce({"diagnosis": "specific", "root_cause_target": "foo", "root_cause_mechanism": "bad branch", "supporting_evidence_ids": ["ev-1"], "new_requirements": [{"target": "other source", "file": "b.py", "goal_type": "behavior", "reason": "need source"}]})
    assert candidate.hypothesis.status == "partial"
    assert candidate.hypothesis.evidence_sufficient is False
    assert candidate.hypothesis.required_missing_evidence


def test_semantic_reducer_drops_unknown_evidence_ids(tmp_path):
    tracker = EvidenceObligationTracker(repo_root=tmp_path)
    manager = HypothesisManager(tmp_path)
    reducer = SemanticReducer(tracker, manager, [])
    candidate = reducer.reduce({"diagnosis": "partial", "supporting_evidence_ids": ["missing"]})
    assert candidate.dropped_evidence_ids == ("missing",)
    assert candidate.hypothesis.supporting_evidence_ids == []


def test_native_runtime_enters_orchestrator_without_parallel_json(monkeypatch, tmp_path):
    from debug_assistant.config import AppConfig, HarnessConfig, ModelConfig
    from debug_assistant.harness.runtime import AgentHarness

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("value = 1\n" * 10)

    class NativeLLM:
        capabilities = ProviderCapabilities(tool_calling=True, parallel_tool_calls=True)

        def __init__(self):
            self.calls = []
            self.native_calls = 0

        def _usage(self, system, user):
            self.calls.append({"model": "fake", "prompt_tokens": 1, "completion_tokens": 1,
                               "total_tokens": 2, "input_tokens": 1, "output_tokens": 1,
                               "prompt_chars": len(system) + len(user), "completion_chars": 1})

        def complete_with_tools(self, system, user, *, tools, model=None, logical_timeout_seconds=None,
                                on_attempt_started=None):
            self._usage(system, user)
            self.native_calls += 1
            calls = () if self.native_calls > 1 else (LLMToolCall("r1", "read_file", {"path": "a.py", "start_line": 1, "end_line": 3}),)
            return LLMResponse(
                content="I'll inspect the file." if calls else None,
                tool_calls=calls, usage=self.calls[-1]
            )

        def complete_json(self, system, user, model=None, logical_timeout_seconds=None):
            self._usage(system, user)
            if "FINAL_REPORT_SCHEMA" in user:
                return {"summary": "done", "root_cause": "uncertain", "likely_files": ["a.py"],
                        "likely_symbols": [], "impact_scope": [], "recommended_change_points": [],
                        "uncertainties": [], "next_checks": [], "evidence_ids": [], "confidence": .3}
            return {"decision": "continue", "reason": "more evidence", "diagnosis": "partial",
                    "supporting_evidence_ids": [], "contradicting_evidence_ids": [],
                    "obligation_reviews": [], "new_requirements": [], "optional_validation": [],
                    "recommended_next_goal": "", "confidence": .2}

    fake = NativeLLM()
    monkeypatch.setattr("debug_assistant.harness.runtime.build_llm", lambda cfg: fake)
    cfg = AppConfig(model=ModelConfig(provider="mock"), harness=HarnessConfig(
        build_task_index=False, max_steps=2, reflect_every=1, finalization_reserve_seconds=0,
        planner_start_guard_seconds=0, reflection_start_guard_seconds=0,
        trace_dir=str(tmp_path / "traces")))
    result = AgentHarness(cfg).run(TaskSpec("native-runtime", "inspect source", str(repo)))
    events = [json.loads(line) for line in open(result["trace"]["trace_path"], encoding="utf-8")]
    types = [event["type"] for event in events]
    assert result["state"]["tool_calls"] == 1
    assert "NATIVE_TOOL_CALLS_RECEIVED" in types
    assert "TOOL_REQUEST_EXPANDED" in types
    assert "TOOL_EXECUTION_PLAN_BUILT" in types
    assert "SEMANTIC_REDUCER_RESULT" in types
    assert "DERIVED_HYPOTHESIS_STATE" in types
    assert "ACTION_PROPOSED" not in types


def test_native_no_tool_turn_is_not_planner_failure(monkeypatch, tmp_path):
    from debug_assistant.config import AppConfig, HarnessConfig, ModelConfig
    from debug_assistant.harness.runtime import AgentHarness

    repo = tmp_path / "repo"
    repo.mkdir()

    class NoToolLLM:
        capabilities = ProviderCapabilities(tool_calling=True, parallel_tool_calls=True)

        def __init__(self):
            self.calls = []

        def complete_with_tools(self, system, user, *, tools, **kwargs):
            self.calls.append({"model": "fake", "prompt_tokens": 1, "completion_tokens": 1,
                               "total_tokens": 2, "input_tokens": 1, "output_tokens": 1})
            return LLMResponse(content="I need more reasoning.", usage=self.calls[-1])

        def complete_json(self, system, user, model=None, logical_timeout_seconds=None):
            self.calls.append({"model": "fake", "prompt_tokens": 1, "completion_tokens": 1,
                               "total_tokens": 2, "input_tokens": 1, "output_tokens": 1})
            if "FINAL_REPORT_SCHEMA" in user:
                return {"summary": "no-tool", "root_cause": "uncertain", "likely_files": [],
                        "likely_symbols": [], "impact_scope": [], "recommended_change_points": [],
                        "uncertainties": [], "next_checks": [], "evidence_ids": [], "confidence": .2}
            return {"decision": "continue", "reason": "reasoning remains incomplete", "diagnosis": "partial",
                    "supporting_evidence_ids": [], "contradicting_evidence_ids": [],
                    "obligation_reviews": [], "new_requirements": [], "optional_validation": [],
                    "confidence": .2}

    fake = NoToolLLM()
    monkeypatch.setattr("debug_assistant.harness.runtime.build_llm", lambda cfg: fake)
    cfg = AppConfig(model=ModelConfig(provider="mock"), harness=HarnessConfig(
        build_task_index=False, max_steps=3, reflect_every=1, finalization_reserve_seconds=0,
        planner_start_guard_seconds=0, reflection_start_guard_seconds=0,
        trace_dir=str(tmp_path / "traces")))
    result = AgentHarness(cfg).run(TaskSpec("native-no-tool", "inspect source", str(repo)))
    events = [json.loads(line) for line in open(result["trace"]["trace_path"], encoding="utf-8")]
    types = [event["type"] for event in events]
    assert "NATIVE_PLANNER_NO_TOOL_TURN" in types
    assert "PLANNER_FAILED" not in types
    assert "LLMInvalidJSON" not in " ".join(result["state"]["errors"])
    assert result["state"]["planner_contract_failure_count"] == 0


@pytest.mark.asyncio
async def test_openai_compatible_provider_returns_native_tool_calls():
    class Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            body = json.loads(request.content)
            assert body["tools"][0]["function"]["name"] == "grep"
            return httpx.Response(200, request=request, json={
                "choices": [{"message": {"content": "I'll inspect the file.", "tool_calls": [
                    {"id": "call-1", "function": {"name": "grep", "arguments": '{"query":"needle"}'}}
                ]}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            })
    from debug_assistant.llm.openai_compatible import OpenAICompatibleClient
    client = OpenAICompatibleClient("https://example.invalid/v1", "key", "model", async_transport=Transport())
    response = await client.acomplete_json("s", "u", tools=[{"type": "function", "function": {"name": "grep"}}], return_response=True, logical_timeout_seconds=1)
    assert response.tool_calls[0].arguments == {"query": "needle"}
    assert response.content == "I'll inspect the file."
    assert response.structured is None
    assert response.usage["total_tokens"] == 5


def test_native_no_tool_turn_keeps_natural_language_typed(tmp_path):
    class NoToolLLM:
        capabilities = ProviderCapabilities(tool_calling=True)

        def complete_with_tools(self, system, user, *, tools, **kwargs):
            return LLMResponse(content="I need more reasoning.")

    result = NativeToolPlanner(NoToolLLM(), ToolRegistry(tmp_path)).propose(
        AgentState(TaskSpec("t", "inspect source", str(tmp_path))), "context"
    )
    assert result.tool_calls == ()
    assert result.assistant_text == "I need more reasoning."
    assert result.intent.information_need == "I need more reasoning."


def test_native_empty_content_no_tool_turn_is_typed(tmp_path):
    class EmptyNativeLLM:
        capabilities = ProviderCapabilities(tool_calling=True)

        def complete_with_tools(self, system, user, *, tools, **kwargs):
            return LLMResponse(content=None, tool_calls=())

    result = NativeToolPlanner(EmptyNativeLLM(), ToolRegistry(tmp_path)).propose(
        AgentState(TaskSpec("t", "inspect source", str(tmp_path))), "context"
    )
    assert result.tool_calls == ()
    assert result.assistant_text is None
    assert result.intent.information_need is None


def test_native_invalid_intent_metadata_does_not_block_valid_tool(tmp_path):
    class InvalidIntentLLM:
        capabilities = ProviderCapabilities(tool_calling=True)

        def complete_with_tools(self, system, user, *, tools, **kwargs):
            return LLMResponse(
                content="I'll inspect the file.",
                structured={"question_type": {"not": "a scalar"}},
                tool_calls=(LLMToolCall("r", "read_file", {"path": "a.py"}),),
            )

    result = NativeToolPlanner(InvalidIntentLLM(), ToolRegistry(tmp_path)).propose(
        AgentState(TaskSpec("t", "inspect source", str(tmp_path))), "context"
    )
    assert [call.name for call in result.tool_calls] == ["read_file"]
    assert result.assistant_text == "I'll inspect the file."


def test_native_invalid_tool_arguments_have_native_error_type(tmp_path):
    class Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(200, request=request, json={
                "choices": [{"message": {"content": "I will inspect it.", "tool_calls": [
                    {"id": "bad", "function": {"name": "read_file", "arguments": "{bad json"}}
                ]}}],
            })

    from debug_assistant.llm.openai_compatible import OpenAICompatibleClient
    client = OpenAICompatibleClient(
        "https://example.invalid/v1", "key", "model", async_transport=Transport()
    )
    with pytest.raises(NativePlannerContractError) as exc_info:
        NativeToolPlanner(client, ToolRegistry(tmp_path)).propose(
            AgentState(TaskSpec("t", "inspect source", str(tmp_path))), "context",
            logical_timeout_seconds=1,
        )
    assert exc_info.value.metadata["error_type"] == "tool_arguments_invalid_json"
    assert "LLMInvalidJSON" not in type(exc_info.value).__name__


@pytest.mark.asyncio
async def test_legacy_content_still_requires_json():
    class Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(200, request=request, json={
                "choices": [{"message": {"content": "plain assistant prose"}}],
            })

    from debug_assistant.llm.openai_compatible import OpenAICompatibleClient
    client = OpenAICompatibleClient("https://example.invalid/v1", "key", "model", async_transport=Transport())
    with pytest.raises(LLMInvalidJSON):
        await client.acomplete_json("s", "u", logical_timeout_seconds=1)
