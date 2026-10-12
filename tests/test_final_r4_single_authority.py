from __future__ import annotations
from completion_fixtures import completion

import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from debug_assistant.incidents import RCAState, RCAStateError
from debug_assistant.incidents.contracts import (
    Contradiction,
    IncidentHypothesis,
    VerificationObligation,
)
from debug_assistant.incidents.verification_dag import (
    VerificationDAGHarnessAdapter,
    VerificationTask,
    VerificationTaskExecution,
)
from debug_assistant.models import AgentState


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "src/debug_assistant/incidents/runtime.py"
RCA_STATE = ROOT / "src/debug_assistant/incidents/rca_state.py"


def _semantic_state() -> RCAState:
    state = RCAState(incident_id="r4")
    state = state.reduce({
        "kind": "ADD_OBLIGATION",
        "payload": {"id": "obl-r4", "claim": "verify the causal fact", "completion_contract": completion("fact").model_dump(mode="json")},
    })
    state = state.reduce({
        "kind": "LINK_EVIDENCE",
        "payload": {"evidence_id": "ev-r4"},
    })
    return state.reduce({
        "kind": "SET_HYPOTHESIS",
        "payload": {"hypothesis": IncidentHypothesis(
            claim="service routing is broken",
            evidence_gap="verify the causal fact",
            component="checkoutservice",
            fault="selector mismatch",
            fault_code="service_selector_mismatch",
            mechanism="the Service selects no backend Pod",
            supporting_evidence_ids=("ev-r4",),
            verification_obligations=(state.obligation("obl-r4"),),
        ).model_dump()},
    })


def test_hypothesis_is_committed_only_by_reducer() -> None:
    state = _semantic_state()
    with pytest.raises(ValidationError):
        state.hypotheses[-1].claim = "mutated"
    revised = state.active_hypothesis_projection().model_copy(update={
        "claim": "revised causal claim",
    })
    changed = state.reduce({
        "kind": "SET_HYPOTHESIS",
        "payload": {"hypothesis": revised.model_dump()},
    })
    assert state.hypotheses[-1].claim == "service routing is broken"
    assert changed.hypotheses[-1].claim == "revised causal claim"


def test_obligation_is_committed_only_by_reducer() -> None:
    state = _semantic_state()
    with pytest.raises(ValidationError):
        state.obligations[0].status = "SATISFIED"
    changed = state.reduce({
        "kind": "UPDATE_OBLIGATION",
        "payload": {
            "obligation_id": "obl-r4",
            "status": "SATISFIED",
            "supporting_evidence_ids": ["ev-r4"],
        },
    })
    assert state.obligation("obl-r4").status == "OPEN"
    assert changed.obligation("obl-r4").status == "SATISFIED"


def test_contradiction_is_committed_and_resolved_only_by_reducer() -> None:
    state = _semantic_state().reduce({
        "kind": "SET_CONTRADICTIONS",
        "payload": {"contradictions": [{
            "evidence_id": "ev-r4",
            "claim": "the selector matches",
        }]},
    })
    assert state.contradictions[0].status == "OPEN"
    resolved = state.reduce({
        "kind": "SET_CONTRADICTIONS",
        "payload": {"contradictions": [{
            "evidence_id": "ev-r4",
            "claim": "the selector matches",
            "status": "RESOLVED",
        }]},
    })
    assert resolved.contradictions[0].status == "RESOLVED"
    with pytest.raises(RCAStateError):
        resolved.reduce({
            "kind": "SET_CONTRADICTIONS",
            "payload": {"contradictions": [{
                "evidence_id": "ev-r4",
                "claim": "reopened",
                "status": "OPEN",
            }]},
        })


def test_runtime_has_no_direct_rca_model_copy_or_semantic_mirror_assignment() -> None:
    tree = ast.parse(RUNTIME.read_text())
    assigned_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else (node.target,)
        )
        if isinstance(target, ast.Name)
    }
    assert "hypotheses" not in assigned_names
    assert "obligation_state" not in assigned_names
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        assert not (
            node.func.attr == "model_copy"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "canonical_rca_state"
        )


def test_every_runtime_rca_assignment_is_initialization_or_reducer_result() -> None:
    tree = ast.parse(RUNTIME.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "canonical_rca_state"
            for target in node.targets
        ):
            continue
        value = node.value
        initialized = isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "RCAState"
        reduced = isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "reduce"
        published = isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name) and value.value.id == "staged"
        assert initialized or reduced or published
        if published:
            assert value.slice.value == 0  # validated transaction RCAState, after staging finally restores live owners


def test_agent_state_has_no_semantic_hypothesis_store() -> None:
    assert "current_hypothesis" not in AgentState.__dataclass_fields__


def test_task_completion_changes_only_dag_execution_state() -> None:
    semantic = _semantic_state()
    before = semantic.model_dump(mode="json")
    adapter = VerificationDAGHarnessAdapter()
    dag = adapter.initialize(
        [VerificationTask(task_id="task-r4", obligation_id="obl-r4")],
        obligation_ids=("obl-r4",),
    )
    dag, selected = adapter.select_ready(dag)
    assert selected is not None
    completed = adapter.apply_execution(dag, VerificationTaskExecution(
        task_id=selected.task_id,
        status="COMPLETED",
        result_ref="obs-r4",
    ))
    assert completed.task("task-r4").status == "COMPLETED"
    assert semantic.model_dump(mode="json") == before
    assert semantic.obligation("obl-r4").status == "OPEN"


def test_dag_contract_contains_no_semantic_fulfillment_fields() -> None:
    assert set(VerificationTaskExecution.model_fields) == {
        "task_id", "status", "result_ref", "reason",
    }
    assert "tasks" not in RCAState.model_fields


def test_consumer_projections_rebuild_from_canonical_state() -> None:
    state = _semantic_state()
    projection = state.context_projection()
    active = state.active_hypothesis_projection()
    assert active is not None
    assert projection["active_hypothesis"] == active.model_dump(mode="json")
    assert projection["obligations"] == [
        {**item.model_dump(mode="json"), "required_fact_slots": list(item.required_fact_slots)}
        for item in state.obligations
    ]
    assert "tasks" not in projection


def test_projection_changes_do_not_mutate_canonical_state() -> None:
    state = _semantic_state()
    projection = state.context_projection()
    projection["active_hypothesis"]["claim"] = "local display change"
    projection["obligations"][0]["status"] = "SATISFIED"
    detached = state.active_hypothesis_projection().model_copy(update={
        "claim": "another local display change",
    })
    assert detached.claim != state.hypotheses[-1].claim
    assert state.hypotheses[-1].claim == "service routing is broken"
    assert state.obligation("obl-r4").status == "OPEN"


def test_serialized_resume_needs_only_canonical_semantic_state() -> None:
    state = _semantic_state()
    resumed = RCAState.model_validate_json(state.model_dump_json())
    assert resumed == state
    assert resumed.context_projection() == state.context_projection()


def test_hypothesis_history_does_not_embed_mutable_lifecycle_mirrors() -> None:
    state = _semantic_state()
    stored = state.hypotheses[-1]
    projected = state.active_hypothesis_projection()
    assert stored.verification_obligations == ()
    assert stored.contradictions == ()
    assert projected is not None
    assert projected.verification_obligations == state.obligations


def test_reducer_is_the_only_rca_state_model_copy_boundary() -> None:
    source = RCA_STATE.read_text()
    tree = ast.parse(source)
    model_copy_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "model_copy"
    ]
    assert model_copy_calls
    assert "canonical_rca_state.model_copy" not in RUNTIME.read_text()
