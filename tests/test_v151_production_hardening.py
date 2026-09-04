import json

from debug_assistant.evaluation.localization import evaluate_one
from debug_assistant.evaluation.trace_metrics import summarize_trace
from debug_assistant.harness.obligations import EvidenceObligationTracker, ObligationStatus
from debug_assistant.models import Evidence


def _source(eid, path, start, end):
    return Evidence(
        eid, "read_file", "read_file", "source", file=path,
        source_start_line=start, source_end_line=end,
        line_start=start, line_end=end,
    )


def _lookup(query, limit=40):
    rows = {
        "foo": [{"path": "a.py", "name": "foo", "qualified_name": "foo",
                 "start_line": 1, "end_line": 3}],
        "bar": [{"path": "b.py", "name": "bar", "qualified_name": "bar",
                 "start_line": 1, "end_line": 3}],
    }
    return rows.get(query, [])


def test_satisfied_requirement_is_not_reactivated_by_equivalent_sync(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    tracker = EvidenceObligationTracker(repo_root=tmp_path, symbol_lookup=_lookup)
    tracker.sync([{
        "target": "foo behavior",
        "file": "a.py", "symbol": "foo", "goal_type": "behavior",
        "reason": "confirm foo behavior",
    }])
    obj = next(iter(tracker.items.values()))
    evidence = _source("ev-foo", "a.py", 1, 3)
    tracker.note_evidence(evidence, "def foo(): return 1")
    tracker.mark_presented(obj.obligation_id, reflection_id="R1", projection_id="P1",
                           evidence_fingerprint=tracker.evidence_fingerprint(obj))
    ok, _ = tracker.apply_explicit_review(
        {"obligation_id": obj.obligation_id, "decision": "resolved", "reason": "shown"},
        reflection_id="R1",
    )
    assert ok and obj.status is ObligationStatus.SATISFIED

    tracker.sync([{
        "target": "foo behavior for the boundary case",
        "file": "a.py", "symbol": "foo", "goal_type": "behavior",
        "reason": "same source scope, restated requirement",
    }])
    assert len(tracker.items) == 1
    assert obj.status is ObligationStatus.SATISFIED
    assert obj.active_required is False
    assert obj.critical is False
    ok, _ = tracker.apply_explicit_review(
        {"obligation_id": obj.obligation_id, "decision": "still_open", "reason": "repeat"},
        reflection_id="R2",
    )
    assert not ok
    assert obj.status is ObligationStatus.SATISFIED
    assert obj.active_required is False


def test_new_requirement_gets_new_active_obligation_while_old_stays_terminal(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def bar():\n    return 2\n", encoding="utf-8")
    tracker = EvidenceObligationTracker(repo_root=tmp_path, symbol_lookup=_lookup)
    tracker.sync([{
        "target": "foo behavior", "file": "a.py", "symbol": "foo",
        "goal_type": "behavior", "reason": "confirm foo",
    }])
    old = next(iter(tracker.items.values()))
    evidence = _source("ev-foo", "a.py", 1, 3)
    tracker.note_evidence(evidence, "def foo(): return 1")
    tracker.mark_presented(old.obligation_id, reflection_id="R1", projection_id="P1",
                           evidence_fingerprint=tracker.evidence_fingerprint(old))
    tracker.apply_explicit_review(
        {"obligation_id": old.obligation_id, "decision": "resolved", "reason": "shown"},
        reflection_id="R1",
    )

    tracker.sync([{
        "target": "bar behavior", "file": "b.py", "symbol": "bar",
        "goal_type": "behavior", "reason": "new independent requirement",
    }])
    assert old.status is ObligationStatus.SATISFIED
    assert old.active_required is False
    active = [x for x in tracker.open_critical()]
    assert len(active) == 1 and active[0].target == "bar behavior"


def test_terminal_goal_type_is_not_mutated_into_unreviewed_semantic_state(tmp_path):
    (tmp_path / "a.py").write_text("def foobar():\n    return 1\n", encoding="utf-8")
    tracker = EvidenceObligationTracker(repo_root=tmp_path)
    tracker.sync([{
        "target": "foobar", "file": "a.py", "goal_type": "location",
        "reason": "locate foobar",
    }])
    old = next(iter(tracker.items.values()))
    evidence = _source("ev-foobar", "a.py", 1, 3)
    assert tracker.note_evidence(evidence, "def foobar(): return 1") == [old.obligation_id]
    assert old.status is ObligationStatus.SATISFIED

    tracker.sync([{
        "target": "foobar", "file": "a.py", "goal_type": "behavior",
        "reason": "review foobar behavior",
    }])

    assert old.status is ObligationStatus.SATISFIED
    assert old.goal_type == "location"
    active = [item for item in tracker.open_critical()]
    assert len(active) == 1
    assert active[0].goal_type == "behavior"
    assert active[0].obligation_id != old.obligation_id


def test_precise_required_scope_rejects_unmotivated_range_drift_but_allows_supporting_action(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("\n".join(f"line_{i}" for i in range(1, 501)), encoding="utf-8")
    tracker = EvidenceObligationTracker(repo_root=tmp_path)
    tracker.sync([{
        "target": "module.py behavior at lines 240-300",
        "location": "module.py:240-300",
        "goal_type": "behavior", "reason": "confirm behavior",
    }])

    assert tracker.action_scope("read_file", {
        "path": "module.py", "start_line": 220, "end_line": 320,
    }, "inspect required behavior")["allowed"]
    drift = tracker.action_scope("read_file", {
        "path": "module.py", "start_line": 440, "end_line": 520,
    }, "read source")
    assert drift["allowed"] is False
    support = tracker.action_scope("read_file", {
        "path": "module.py", "start_line": 440, "end_line": 520,
    }, "inspect the direct caller")
    assert support["allowed"] is True


def test_evidence_binding_requires_exact_file_and_full_range(tmp_path):
    (tmp_path / "module.py").write_text("\n".join(f"line_{i}" for i in range(1, 301)), encoding="utf-8")
    tracker = EvidenceObligationTracker(repo_root=tmp_path)
    tracker.sync([{
        "target": "module behavior at lines 240-300",
        "location": "module.py:240-300",
        "goal_type": "behavior", "reason": "confirm behavior",
    }])
    obj = next(iter(tracker.items.values()))
    assert tracker.note_evidence(_source("ev-wrong", "other.py", 1, 400), "module") == []
    assert tracker.note_evidence(_source("ev-short", "module.py", 240, 299), "module") == []
    assert obj.evidence_ids == []
    assert tracker.note_evidence(_source("ev-right", "module.py", 220, 300), "module") == []
    assert obj.evidence_ids == ["ev-right"]
    assert obj.status is ObligationStatus.ATTEMPTED


def test_localization_reports_file_at_1_3_5_and_symbol_ranks():
    metrics = evaluate_one(
        {"files": ["src/a.py"], "symbols": [{"file": "src/a.py", "symbol": "target"}]},
        {
            "likely_files": ["other.py", "src/a.py"],
            "likely_symbols": ["other", "target"],
            "recommended_change_points": [],
        },
    )
    assert metrics["file_hit1"] == 0
    assert metrics["file_hit3"] == 1
    assert metrics["file_hit5"] == 1
    assert metrics["gold_file_rank"] == 2
    assert metrics["gold_symbol_rank"] == 2


def test_trace_metrics_use_typed_events_without_double_counting_timeout(tmp_path):
    events = [
        {"type": "ACTION_PROPOSED", "payload": {}},
        {"type": "NATIVE_PLANNER_RESULT", "payload": {}},
        {"type": "TOOL_OBSERVATION", "payload": {"ok": True}},
        {"type": "TOOL_OBSERVATION", "payload": {"ok": False}},
        {"type": "TOOL_RETRY", "payload": {}},
        {"type": "REFLECTION", "payload": {}},
        {"type": "REFLECTION_FAILED", "payload": {"error_type": "llm_deadline_exceeded"}},
        {"type": "LLM_DEADLINE_EXCEEDED", "payload": {"stage": "reflection"}},
        {"type": "REPORTER_CONTEXT_BUILT", "payload": {}},
        {"type": "EVIDENCE_ADDED", "payload": {}},
        {"type": "HYPOTHESIS_UPDATED", "payload": {}},
        {"type": "LLM_USAGE", "payload": {"totals": {"calls": 3, "tokens": 30}}},
        {"type": "RUN_END", "payload": {"summary": {
            "status": "partial_success", "reflection_count": 2,
            "hypotheses": 1,
        }}},
    ]
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps({"type": x["type"], "payload": x["payload"]}) for x in events), encoding="utf-8")
    metrics = summarize_trace(path)
    assert metrics["planner_steps"] == 2
    assert metrics["tool_calls"] == 2
    assert metrics["tool_failures"] == 1
    assert metrics["tool_retries"] == 1
    assert metrics["evidence_count"] == 1
    assert metrics["hypothesis_count"] == 1
    assert metrics["reflection_count"] == 2
    assert metrics["reflection_timeout_count"] == 1
    assert metrics["reporter_count"] == 1
    assert metrics["partial_success"] is True
    assert metrics["failure"] is False


def test_trace_metrics_count_obligation_scope_rejections(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({"type": "OBLIGATION_ACTION_REJECTED", "payload": {}}),
            json.dumps({"type": "RUN_END", "payload": {"summary": {"status": "partial_success"}}}),
        ]),
        encoding="utf-8",
    )
    metrics = summarize_trace(path)
    assert metrics["route_rejections"] == 1
    assert metrics["obligation_scope_rejections"] == 1
