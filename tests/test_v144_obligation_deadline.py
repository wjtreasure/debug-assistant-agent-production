import asyncio
import json
import time

import httpx
import pytest

from debug_assistant.contracts import ReflectionContract
from debug_assistant.harness.obligations import EvidenceObligationTracker, ObligationStatus
from debug_assistant.llm.base import LLMClientUsageError, LLMDeadlineExceeded
from debug_assistant.llm.openai_compatible import OpenAICompatibleClient
from debug_assistant.models import Evidence


def _source(eid, path, start, end):
    return Evidence(
        eid, "read_file", "read_file", "source",
        file=path,
        source_start_line=start, source_end_line=end,
        line_start=start, line_end=end,
    )


def _symbol_lookup(query, limit=60):
    rows = {
        "get_source_file": [
            {"path": "astroid/modutils.py", "name": "get_source_file", "qualified_name": "get_source_file", "kind": "FunctionDef", "start_line": 499, "end_line": 521},
        ],
        "file_from_modpath": [
            {"path": "astroid/modutils.py", "name": "file_from_modpath", "qualified_name": "file_from_modpath", "kind": "FunctionDef", "start_line": 330, "end_line": 365},
        ],
        "modpath_from_file": [
            {"path": "astroid/modutils.py", "name": "modpath_from_file", "qualified_name": "modpath_from_file", "kind": "FunctionDef", "start_line": 250, "end_line": 286},
        ],
        "AstroidBuilder.file_build": [
            {"path": "astroid/builder.py", "name": "file_build", "qualified_name": "AstroidBuilder.file_build", "kind": "FunctionDef", "start_line": 115, "end_line": 151},
        ],
    }
    return rows.get(query, [])[:limit]


def test_reflection_contract_prefers_structured_goal_type_but_accepts_legacy_items():
    structured = ReflectionContract.model_validate({
        "decision": "continue",
        "reason": "need source",
        "required_missing_evidence": [{
            "target": "get_source_file behavior",
            "location": "astroid/modutils.py",
            "goal_type": "behavior",
            "reason": "confirm regression mechanism",
        }],
    })
    assert structured.required_missing_evidence[0].goal_type == "behavior"

    legacy = ReflectionContract.model_validate({
        "decision": "continue",
        "reason": "need source",
        "required_missing_evidence": [{
            "target": "get_source_file behavior",
            "location": "astroid/modutils.py",
            "reason": "confirm regression mechanism",
        }],
    })
    assert legacy.required_missing_evidence[0].goal_type is None


def test_explicit_goal_type_wins_over_legacy_regression_keyword(tmp_path):
    tracker = EvidenceObligationTracker(repo_root=tmp_path, symbol_lookup=_symbol_lookup)
    tracker.sync([{
        "target": "get_source_file resolution with include_no_ext=True",
        "location": "astroid/modutils.py",
        "goal_type": "behavior",
        "reason": "confirm the regression mechanism",
    }])
    obj = next(iter(tracker.items.values()))
    assert obj.goal_type == "behavior"
    assert obj.goal_type_source == "structured"
    assert "get_source_file" in obj.canonical_symbols


def test_composite_behavior_obligation_fails_closed_and_atomic_scopes_close_independently(tmp_path):
    tracker = EvidenceObligationTracker(repo_root=tmp_path, symbol_lookup=_symbol_lookup)
    tracker.sync([{
        "target": "modutils.file_from_modpath / modpath_from_file behavior for a directory without __init__.py",
        "location": "astroid/modutils.py",
        "goal_type": "causality",
        "reason": "identify how the bad path is produced",
    }])
    composite = next(iter(tracker.items.values()))
    assert composite.scope_valid is False
    assert composite.scope_error == "composite_scope_requires_atomization"
    assert tracker.note_evidence(_source("ev-both", "astroid/modutils.py", 240, 370), "definitions") == []
    assert composite.evidence_ready is False

    # V1.4.6 requires one verifiable source scope per Obligation.  Multiple files or
    # symbols may still be reviewed together later in one EvidenceBundle.
    tracker = EvidenceObligationTracker(repo_root=tmp_path, symbol_lookup=_symbol_lookup)
    tracker.sync([
        {"target":"file_from_modpath behavior","location":"astroid/modutils.py","file":"astroid/modutils.py","symbol":"file_from_modpath","goal_type":"causality","reason":"first atomic scope"},
        {"target":"modpath_from_file behavior","location":"astroid/modutils.py","file":"astroid/modutils.py","symbol":"modpath_from_file","goal_type":"causality","reason":"second atomic scope"},
    ])
    objs=sorted(tracker.items.values(),key=lambda x:x.canonical_symbols[0])
    assert all(o.scope_valid for o in objs)
    tracker.note_evidence(_source("ev-both", "astroid/modutils.py", 240, 370), "definitions")
    assert all(o.evidence_ready for o in objs)
    for i,obj in enumerate(objs,1):
        fp=tracker.evidence_fingerprint(obj)
        tracker.mark_presented(obj.obligation_id,reflection_id="R1",projection_id=f"P{i}",evidence_fingerprint=fp)
        ok,_=tracker.apply_explicit_review({"obligation_id":obj.obligation_id,"decision":"resolved","reason":"definition proves behavior"},reflection_id="R1")
        assert ok is True
        assert obj.status is ObligationStatus.SATISFIED


def test_behavior_without_symbol_or_range_is_attempted_not_auto_satisfied(tmp_path):
    tracker = EvidenceObligationTracker(repo_root=tmp_path, symbol_lookup=lambda q, limit=60: [])
    tracker.sync([{
        "target": "package resolution behavior",
        "location": "astroid/modutils.py",
        "goal_type": "behavior",
        "reason": "understand namespace handling",
    }])
    obj = next(iter(tracker.items.values()))
    assert tracker.note_evidence(_source("ev", "astroid/modutils.py", 1, 200), "some relevant source") == []
    assert obj.status is ObligationStatus.ATTEMPTED
    assert obj.evidence_ids == []


def test_qualified_method_target_resolves_leaf_method_not_whole_class(tmp_path):
    tracker = EvidenceObligationTracker(repo_root=tmp_path, symbol_lookup=_symbol_lookup)
    tracker.sync([{
        "target": "AstroidBuilder.file_build error path",
        "location": "astroid/builder.py",
        "goal_type": "causality",
        "reason": "confirm error path",
    }])
    obj = next(iter(tracker.items.values()))
    assert obj.canonical_symbols == ("file_build",)
    assert obj.symbol_ranges == (("astroid/builder.py", "file_build", 115, 151),)


class _Slow503Transport(httpx.AsyncBaseTransport):
    def __init__(self):
        self.attempts = 0

    async def handle_async_request(self, request):
        self.attempts += 1
        await asyncio.sleep(0.2)
        return httpx.Response(503, request=request, text="busy")


@pytest.mark.asyncio
async def test_async_llm_logical_deadline_caps_all_retries():
    transport = _Slow503Transport()
    client = OpenAICompatibleClient(
        "https://example.invalid/v1", "test-key", "fake",
        timeout=1.0, async_transport=transport, max_attempts=3, min_retry_budget=0.01,
    )
    started = time.monotonic()
    with pytest.raises(LLMDeadlineExceeded):
        await client.acomplete_json("system", "user", logical_timeout_seconds=0.08)
    elapsed = time.monotonic() - started
    assert elapsed < 0.4
    assert transport.attempts == 1
    assert any(e["type"] == "LLM_DEADLINE_EXCEEDED" for e in client.events)
    assert any(e["type"] == "LLM_LOGICAL_CALL_FINISHED" and e["payload"].get("success") is False for e in client.events)


@pytest.mark.asyncio
async def test_sync_bridge_rejects_nested_running_event_loop():
    client = OpenAICompatibleClient("https://example.invalid/v1", "test-key", "fake", timeout=0.1)
    with pytest.raises(LLMClientUsageError):
        client.complete_json("system", "user", logical_timeout_seconds=0.1)

class _SuccessTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request):
        return httpx.Response(
            200,
            request=request,
            json={
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        )


def test_sync_llm_bridge_uses_async_transport_and_records_attempt_telemetry():
    client = OpenAICompatibleClient(
        "https://example.invalid/v1", "test-key", "fake",
        timeout=1.0, async_transport=_SuccessTransport(),
    )
    assert client.complete_json("system", "user", logical_timeout_seconds=0.5) == {"ok": True}
    assert client.calls[-1]["provider_attempts"] == 1
    types = [e["type"] for e in client.events]
    assert types.count("LLM_LOGICAL_CALL_STARTED") == 1
    assert types.count("LLM_ATTEMPT_STARTED") == 1
    assert types.count("LLM_LOGICAL_CALL_FINISHED") == 1


def test_reflection_schema_repair_shares_one_stage_deadline():
    from debug_assistant.agent.reflection import Reflector

    class RepairLLM:
        def __init__(self):
            self.n = 0
            self.timeouts = []
        def complete_json(self, system, user, model=None, logical_timeout_seconds=None):
            self.n += 1
            self.timeouts.append(logical_timeout_seconds)
            if self.n == 1:
                time.sleep(0.01)
                return {
                    "decision": "continue",
                    "reason": "need more",
                    "current_diagnosis": "partial",
                    "root_cause_target": {"bad": "shape"},
                    "evidence_sufficient": False,
                    "supporting_evidence_ids": [],
                    "contradicting_evidence_ids": [],
                    "required_missing_evidence": [],
                    "optional_validation": [],
                    "recommended_next_goal": "inspect",
                    "confidence": 0.2,
                    "hypothesis_changed": False,
                }
            return {
                "decision": "continue",
                "reason": "need more",
                "current_diagnosis": "partial",
                "root_cause_target": None,
                "root_cause_location": None,
                "root_cause_mechanism": None,
                "evidence_sufficient": False,
                "supporting_evidence_ids": [],
                "contradicting_evidence_ids": [],
                "required_missing_evidence": [],
                "optional_validation": [],
                "recommended_next_goal": "inspect",
                "confidence": 0.2,
                "hypothesis_changed": False,
            }

    llm = RepairLLM()
    review = Reflector(llm, compact_prompt=True).review("context", logical_timeout_seconds=0.5)
    assert review["current_diagnosis"] == "partial"
    assert len(llm.timeouts) == 2
    assert 0 < llm.timeouts[1] < llm.timeouts[0] <= 0.5
