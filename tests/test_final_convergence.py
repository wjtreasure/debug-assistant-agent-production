from completion_fixtures import completion
from debug_assistant.incidents import ProgressFacts, ProgressPolicy
from debug_assistant.incidents.finalization import FinalizationConditions, FinalizationDecision
from debug_assistant.agent.final_review import _normalize_review_payload
from debug_assistant.incidents.contracts import VerificationObligation
from debug_assistant.incidents.runtime import (
    _normalize_obligation_capability_status,
)
from debug_assistant.knowledge.router import CapabilitySnapshot


def converge(*, evidence_sufficient, hypothesis_stable, open_critical_obligations=(),
             blocking_contradictions=(), no_progress=False, hard_budget=False):
    conditions = FinalizationConditions(
        candidate_fields_complete=evidence_sufficient, direct_causal_support_present=evidence_sufficient,
        critical_obligations_closed=not open_critical_obligations, required_fact_slots_covered=evidence_sufficient,
        contradictions_resolved=not blocking_contradictions, source_coverage_complete=True,
        taxonomy_code_valid=True,
    )
    permission = FinalizationDecision(allowed=all(conditions.model_dump().values()), conditions=conditions)
    return ProgressPolicy.evaluate(ProgressFacts(
        boundary="convergence", finalization=permission, hypothesis_stable=hypothesis_stable,
        open_obligation_ids=open_critical_obligations, contradiction_ids=blocking_contradictions,
        no_progress_count=int(no_progress), no_progress_limit=1, hard_pressure=hard_budget,
    )).convergence


def test_convergence_requires_evidence_stability_and_closed_obligations() -> None:
    decision = converge(
        evidence_sufficient=True,
        hypothesis_stable=True,
        open_critical_obligations=("obl-1",),
    )
    assert decision.status == "CONTINUE"
    assert "critical_obligation_open" in decision.reasons


def test_contradiction_replans_and_budget_is_inconclusive() -> None:
    assert converge(
        evidence_sufficient=True, hypothesis_stable=True,
        blocking_contradictions=("ev-1",),
    ).status == "REPLAN"
    assert converge(
        evidence_sufficient=False, hypothesis_stable=False, hard_budget=True,
    ).status == "INCONCLUSIVE_BUDGET"


def test_complete_stable_state_converges_even_when_no_progress_is_observed() -> None:
    decision = converge(
        evidence_sufficient=True, hypothesis_stable=True, no_progress=True,
    )
    assert decision.status == "CONVERGED"
    assert "no_progress_after_sufficient_stable_state" in decision.reasons


def test_unknown_reflection_control_is_rejected_without_runtime_guessing() -> None:
    import pytest
    from pydantic import ValidationError
    from debug_assistant.incidents.contracts import ReflectionFeedback
    with pytest.raises(ValidationError):
        ReflectionFeedback(reason="control proposal", next_action_constraint="prefer endpoints and pod labels")
    assert ReflectionFeedback(reason="control proposal", next_action_constraint="finalize").next_action_constraint == "finalize"


def test_rehydrated_log_obligation_respects_new_empty_capability() -> None:
    obligation = VerificationObligation(
        id="obl-log",
        claim="verify pod health and captured application error pattern",
        evidence_requirement="pod state and error logs",
        target_component="service-a",
        verification_question="Does the captured pod state and log pattern explain the symptom?",
        evidence_type="pod_health_and_error_log", completion_contract=completion("fact", role="MECHANISM"))
    capabilities = CapabilitySnapshot(
        states={"logs": "EMPTY", "kubernetes_resources": "AVAILABLE"},
    )

    normalized, gaps = _normalize_obligation_capability_status(
        obligation, capabilities,
    )

    assert gaps == ("logs",)
    assert normalized.status == "NOT_APPLICABLE_BY_CAPABILITY"
    assert normalized.blocks_finalization is False
    assert normalized.supporting_evidence_ids == ()


def test_review_shape_normalization_coerces_unambiguous_boolean_strings() -> None:
    actions = []
    payload = _normalize_review_payload({
        "decision": "REJECT",
        "recoverability": {"recoverable": "true", "unrecoverable": "false"},
        "causal_chain_valid": "false",
        "reason": "missing evidence",
    }, actions=actions)

    assert payload["recoverability"] == {
        "recoverable": True,
        "unrecoverable": False,
    }
    assert payload["causal_chain_valid"] is False
    assert "recoverability.unrecoverable:string_to_false" in actions
