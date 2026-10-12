from __future__ import annotations
from completion_fixtures import completion

import pytest
from pydantic import ValidationError

from debug_assistant.incidents import EvidencePipeline, EvidenceSet
from debug_assistant.incidents.contracts import IncidentHypothesis, VerificationObligation
from debug_assistant.incidents.runtime import (
    DiagnosisHarness,
)
from debug_assistant.memory.evidence_projector import (
    EvidenceProposal,
    EvidenceProjector, EvidenceProposalError,
    ProjectedEvidenceFact,
    evidence_relationship_key,
)
from debug_assistant.memory.evidence_memory import EvidenceMemory
from debug_assistant.memory.observation_store import ObservationStore
from debug_assistant.models import ToolObservation


class Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)




def test_reflection_refinement_requires_an_operational_trigger() -> None:
    from debug_assistant.incidents.progress_policy import ProgressFacts, ProgressPolicy

    decision = ProgressPolicy.evaluate(ProgressFacts(
        boundary="reflection_trigger", canonical_progress=True,
    ))
    assert decision.reflection_trigger is None
    assert decision.action == "continue"


def fact(
    evidence_id: str,
    *,
    fulfillment: str,
    relation: str = "NEUTRAL",
    obligation_id: str = "obl-1",
    hypothesis_fingerprint: str = "scope-a",
    hypothesis_relation: str | None = None,
    covered_fact_slots: tuple[str, ...] | None = None,
) -> ProjectedEvidenceFact:
    observation_id = "obs-" + evidence_id
    return ProjectedEvidenceFact(
        evidence_id=evidence_id,
        observation_id=observation_id,
        fact="grounded observation",
        relation=relation,
        hypothesis_relation=hypothesis_relation or relation,
        fulfillment=fulfillment,
        covered_fact_slots=(
            covered_fact_slots
            if covered_fact_slots is not None
            else ("fact",) if fulfillment == "SATISFIES" else ()
        ),
        obligation_id=obligation_id,
        hypothesis_fingerprint=hypothesis_fingerprint,
        provenance={"observation_refs": [observation_id]},
    )


def qualify_set(
    *,
    obligation: VerificationObligation | Obj,
    evidence: list[ProjectedEvidenceFact] | tuple[ProjectedEvidenceFact, ...],
    hypothesis_scopes: tuple[str, ...] | None = None,
) -> EvidenceSet:
    store = ObservationStore()
    memory = EvidenceMemory()
    seen: set[str] = set()
    for item in evidence:
        if item.evidence_id in seen:
            continue
        seen.add(item.evidence_id)
        observation = ToolObservation(
            tool="fixture",
            ok=True,
            content=item.fact,
            metadata={},
            observation_id=item.observation_id,
        )
        store.add(observation)
        memory.add_observation(
            observation,
            evidence_id=item.evidence_id,
            deduplicate=False,
        )
    pipeline = EvidencePipeline(
        observation_store=store,
        evidence_memory=memory,
        projector=EvidenceProjector(),
    )
    return pipeline.qualify(
        obligation=obligation,
        evidence=evidence,
        hypothesis_scopes=hypothesis_scopes,
    )


def test_one_immutable_evidence_retains_multiple_relationships() -> None:
    shared = fact(
        "ev-shared", fulfillment="SATISFIES", relation="SUPPORTS",
        obligation_id="obl-a", hypothesis_fingerprint="scope-a",
    )
    second_obligation = shared.model_copy(update={"obligation_id": "obl-b"})
    revised_branch = shared.model_copy(update={
        "obligation_id": "obl-a",
        "hypothesis_fingerprint": "scope-b",
        "relation": "CONTRADICTS",
        "hypothesis_relation": "CONTRADICTS",
        "fulfillment": "INSUFFICIENT",
    })
    relationships = {
        evidence_relationship_key(item): item
        for item in (shared, second_obligation, revised_branch)
    }

    assert len(relationships) == 3
    assert {item.obligation_id for item in relationships.values()} == {"obl-a", "obl-b"}
    assert {item.hypothesis_fingerprint for item in relationships.values()} == {"scope-a", "scope-b"}
    assert relationships[evidence_relationship_key(shared)] == shared
    assert relationships[evidence_relationship_key(revised_branch)] == revised_branch


def test_evidence_set_preserves_contradiction_across_same_evidence_id() -> None:
    result = qualify_set(
        obligation=VerificationObligation(id="obl-1", claim="verify fact", completion_contract=completion("fact")),
        evidence=[
            fact("ev-shared", fulfillment="SATISFIES", relation="SUPPORTS"),
            fact(
                "ev-shared", fulfillment="INSUFFICIENT", relation="CONTRADICTS",
                hypothesis_fingerprint="scope-b",
            ),
        ],
    )
    assert result.evidence_ids == ("ev-shared",)
    assert result.supporting_evidence_ids == ("ev-shared",)
    assert result.contradicting_evidence_ids == ("ev-shared",)
    assert result.status == "DOES_NOT_SATISFY"
    assert result.semantic_status == "CONTRADICTS"


def test_evidence_set_direct_satisfy_closes_obligation() -> None:
    result = qualify_set(
        obligation=VerificationObligation(id="obl-1", claim="verify fact", completion_contract=completion("fact")),
        evidence=[fact("ev-1", fulfillment="SATISFIES", relation="SUPPORTS")],
    )
    assert isinstance(result, EvidenceSet)
    assert result.status == "SATISFIES"
    assert result.supporting_evidence_ids == ("ev-1",)


def test_evidence_set_filters_stale_hypothesis_scopes() -> None:
    result = qualify_set(
        obligation=VerificationObligation(id="obl-1", claim="verify fact", completion_contract=completion("fact")),
        evidence=[
            fact("ev-current", fulfillment="SATISFIES", relation="SUPPORTS", hypothesis_fingerprint="scope-a"),
            fact("ev-stale", fulfillment="SATISFIES", relation="SUPPORTS", hypothesis_fingerprint="scope-b"),
        ],
        hypothesis_scopes=("scope-a",),
    )

    assert result.evidence_ids == ("ev-current",)
    assert result.supporting_evidence_ids == ("ev-current",)
    assert result.hypothesis_scope_fingerprints == ("scope-a",)


def test_empty_hypothesis_scope_fails_closed() -> None:
    result = qualify_set(
        obligation=VerificationObligation(id="obl-1", claim="verify fact", completion_contract=completion("fact")),
        evidence=[fact("ev-1", fulfillment="SATISFIES", relation="SUPPORTS")],
        hypothesis_scopes=(),
    )

    assert result.status == "INSUFFICIENT"
    assert result.evidence_ids == ()


def test_evidence_set_rejects_inconsistent_satisfying_state() -> None:
    with pytest.raises(ValidationError, match="semantic_status=SATISFIES"):
        EvidenceSet(
            obligation_id="obl-1",
            evidence_ids=("ev-1",),
            supporting_evidence_ids=("ev-1",),
            coverage={"fact": ("ev-1",)},
            fulfillment="SATISFIES",
            status="SATISFIES",
            semantic_status="INSUFFICIENT",
            reason="invalid fixture",
        )


def test_evidence_set_rejects_evidence_ids_outside_aggregate() -> None:
    with pytest.raises(ValidationError, match="unknown Evidence IDs"):
        EvidenceSet(
            obligation_id="obl-1",
            evidence_ids=("ev-1",),
            supporting_evidence_ids=("ev-2",),
            reason="invalid fixture",
        )


def test_partial_evidence_stays_open_until_semantic_coverage_is_proposed() -> None:
    obligation = VerificationObligation(id="obl-1", claim="verify fact", completion_contract=completion("fact"))
    first = qualify_set(
        obligation=obligation,
        evidence=[fact("ev-1", fulfillment="PARTIAL")],
    )
    second = qualify_set(
        obligation=obligation,
        evidence=[
            fact("ev-1", fulfillment="PARTIAL"),
            fact("ev-2", fulfillment="PARTIAL"),
        ],
    )
    assert first.status == "INSUFFICIENT"
    assert first.fulfillment == "PARTIAL"
    assert second.status == "INSUFFICIENT"
    assert second.fulfillment == "PARTIAL"
    assert second.partial_evidence_ids == ("ev-1", "ev-2")


def test_unrelated_evidence_never_counts_toward_completion() -> None:
    result = qualify_set(
        obligation=VerificationObligation(id="obl-1", claim="verify fact", completion_contract=completion("fact")),
        evidence=[fact("ev-1", fulfillment="UNRELATED")],
    )
    assert result.status == "DOES_NOT_SATISFY"
    assert result.fulfillment == "UNRELATED"


def test_contradiction_blocks_even_when_support_exists() -> None:
    result = qualify_set(
        obligation=VerificationObligation(id="obl-1", claim="verify fact", completion_contract=completion("fact")),
        evidence=[
            fact("ev-1", fulfillment="SATISFIES", relation="SUPPORTS"),
            fact("ev-2", fulfillment="INSUFFICIENT", relation="CONTRADICTS"),
        ],
    )
    assert result.status == "DOES_NOT_SATISFY"
    assert result.semantic_status == "CONTRADICTS"
    assert result.contradicting_evidence_ids == ("ev-2",)


def test_satisfying_evidence_set_cannot_retain_contradiction() -> None:
    with pytest.raises(ValueError, match="contradictory Evidence"):
        EvidenceSet(
            obligation_id="obl-1",
            evidence_ids=("ev-1", "ev-2"),
            supporting_evidence_ids=("ev-1",),
            contradicting_evidence_ids=("ev-2",),
            coverage={"fact": ("ev-1",)},
            fulfillment="SATISFIES",
            status="SATISFIES",
            semantic_status="SATISFIES",
            reason="invalid satisfying aggregate",
        )


def test_obligation_support_and_hypothesis_contradiction_are_independent() -> None:
    observation = ToolObservation(
        "get_resources", True, "all pods are Running and Ready", {}, "", 0.0,
        observation_id="obs-pods",
    )
    obligation = VerificationObligation(id="obl-pod-health", claim="verify pod health", completion_contract=completion("fact"))
    hypothesis = Obj(
        claim="pods are unavailable because they are not Ready",
        component="boutique workloads",
        fault_code="",
        mechanism="unready pods remove healthy backends",
    )
    hypothesis_id = EvidenceProjector.hypothesis_id(hypothesis)
    projector = EvidenceProjector()
    proposal = EvidenceProposal.model_validate({
            "relation": "SUPPORTS",
            "hypothesis_relation": "CONTRADICTS",
            "covered_fact_slots": ["fact"],
            "observation_refs": ["obs-pods"],
            "obligation_id": "obl-pod-health",
            "hypothesis_id": hypothesis_id,
            "semantic_claim": "The captured pod inventory reports all pods Running and Ready.",
            "rationale": "The observation directly answers the pod health question.", "reason": "visible Observation supports the proposed fact", "method": "llm", "incident_id": "incident-a"})
    projected = projector.project(
        observation,
        evidence_id="ev-pods",
        obligation_id="obl-pod-health",
        hypothesis=hypothesis,
        proposal=proposal,
    )
    assert projected is not None
    assert projected.relation == "SUPPORTS"
    assert projected.hypothesis_relation == "CONTRADICTS"
    result = qualify_set(
        obligation=obligation,
        evidence=[projected],
    )
    assert result.status == "SATISFIES"


def test_taxonomy_refinement_requires_explicit_new_revision_scope() -> None:
    pre_taxonomy = Obj(
        hypothesis_id="hyp-branch", revision=1,
        claim="checkoutservice traffic is routed to the wrong target port",
        component="",
        fault="",
        fault_code="",
        mechanism="Service targetPort differs from containerPort",
    )
    completed = Obj(
        hypothesis_id="hyp-branch", revision=2,
        claim=pre_taxonomy.claim,
        component="checkoutservice",
        fault="service port mapping mismatch",
        fault_code="service_port_mapping_mismatch",
        mechanism="Service targetPort 5051 routes to a listener on 5050",
    )
    early_scopes = set(EvidenceProjector.hypothesis_scope_fingerprints(pre_taxonomy))
    completed_scopes = set(EvidenceProjector.hypothesis_scope_fingerprints(completed))
    assert early_scopes and completed_scopes and not early_scopes & completed_scopes


def test_empty_evidence_set_is_insufficient() -> None:
    result = qualify_set(
        obligation=VerificationObligation(id="obl-1", claim="verify fact", completion_contract=completion("fact")),
        evidence=(),
    )
    assert result.status == "INSUFFICIENT"
    assert result.evidence_ids == ()


def test_provenance_validation_rejects_unknown_and_cross_incident_targets() -> None:
    projector = EvidenceProjector()
    proposal = EvidenceProposal(
        semantic_claim="endpoint refused",
        relation="NEUTRAL",
        observation_refs=("obs-1",),
        reason="Planner rationale",
        obligation_id="obl-1",
        incident_id="incident-a",
    )
    with pytest.raises(ValueError, match="unknown Observations"):
        projector.validate(
            proposal,
            observation_ids=set(),
            incident_id="incident-a",
            obligation_id="obl-1",
        )
    with pytest.raises(ValueError, match="different incident"):
        projector.validate(
            proposal,
            observation_ids={"obs-1"},
            incident_id="incident-b",
            obligation_id="obl-1",
        )


def test_deterministic_source_proposal_uses_the_shared_admission_contract() -> None:
    observation = ToolObservation(
        tool="read_file",
        ok=True,
        content="1 | return value",
        metadata={
            "information_source": "source_read",
            "context_kind": "CODE",
            "path": "src/app.py",
            "start_line": 1,
            "end_line": 1,
        },
        observation_id="obs-source-1",
    )
    projector = EvidenceProjector()
    proposal = projector.propose_deterministic(
        observation,
        semantic_claim="Source range was read successfully",
        obligation=VerificationObligation(id="obl-source", claim="verify source", completion_contract=completion("fact")),
        hypothesis=Obj(claim="verify source mechanism", component="app", mechanism="return"),
        reason="deterministic source-read coverage fact",
        incident_id="incident-source",
    )
    admitted = projector.validate(
        proposal,
        observation_ids={observation.observation_id},
        incident_id="incident-source",
        obligation_id="obl-source",
    )
    fact_value = projector.project(
        observation,
        evidence_id="ev-source-1",
        obligation_id="obl-source",
        proposal=admitted,
    )
    assert admitted.method == "deterministic"
    assert admitted.observation_refs == ("obs-source-1",)
    assert fact_value is not None
    assert fact_value.provenance["projection_method"] == "deterministic"
    assert fact_value.fulfillment == "INSUFFICIENT"
    result = qualify_set(
        obligation=VerificationObligation(id="obl-source", claim="verify source", completion_contract=completion("fact")), evidence=[fact_value],
    )
    assert result.status == "DOES_NOT_SATISFY"
    assert result.semantic_status == "INSUFFICIENT"


def test_source_semantic_claim_cannot_fall_back_to_raw_observation_text() -> None:
    observation = ToolObservation(
        tool="read_file",
        ok=True,
        content="1 | return value",
        metadata={
            "information_source": "source_read",
            "context_kind": "CODE",
            "path": "src/app.py",
            "start_line": 1,
            "end_line": 1,
        },
        observation_id="obs-source-2",
    )
    with pytest.raises(EvidenceProposalError, match="requires an explicit"):
        EvidenceProjector().project(
            observation,
            obligation_id="obl-source",
            hypothesis=Obj(claim="verify source mechanism"),
        )


def test_planner_evidence_proposal_contract_preserves_semantic_fields_and_provenance() -> None:
    observation = Obj(
        observation_id="obs-12",
        ok=True,
        content="checkoutservice is listening on port 5050",
        metadata={},
    )
    obligation = Obj(id="obl-verify-selector")
    hypothesis = Obj(claim="checkoutservice endpoint is misconfigured")
    hypothesis_id = EvidenceProjector.hypothesis_id(hypothesis)

    projector = EvidenceProjector()
    proposal = EvidenceProposal.model_validate({
            "relation": "SUPPORTS",
            "covered_fact_slots": ["fact"],
            "observation_refs": ["obs-12"],
            "obligation_id": "obl-verify-selector",
            "hypothesis_id": hypothesis_id,
            "semantic_claim": "The observed listener port matches the selector target.",
            "rationale": "The command output directly reports the endpoint port.",
            "confidence": 0.91, "reason": "visible Observation supports the proposed fact", "method": "llm", "incident_id": "incident-a"})
    assert proposal is not None
    assert proposal.method == "llm"
    assert proposal.relation == "SUPPORTS"
    assert proposal.covered_fact_slots == ("fact",)
    assert proposal.observation_refs == ("obs-12",)
    assert proposal.semantic_claim == "The observed listener port matches the selector target."
    assert proposal.rationale.startswith("The command output")
    assert proposal.confidence == 0.91
    # Projection preserves the explicit semantic claim rather than using
    # the raw Observation body.
    assert proposal.semantic_claim != observation.content


def test_planner_evidence_proposal_rejects_missing_observation_provenance() -> None:
    observation = Obj(
        observation_id="obs-12", ok=True, content="endpoint output", metadata={}
    )
    obligation = VerificationObligation(id="obl-1", claim="verify fact", completion_contract=completion("fact"))
    hypothesis = Obj(claim="endpoint is reachable")
    projector = EvidenceProjector()
    proposal = EvidenceProposal.model_validate({
            "relation": "SUPPORTS",
            "covered_fact_slots": ["fact"],
            "observation_refs": ["obs-other"],
            "obligation_id": "obl-1",
            "hypothesis_id": EvidenceProjector.hypothesis_id(hypothesis),
            "semantic_claim": "The endpoint is reachable.",
            "rationale": "The observation reports a successful connection.", "reason": "visible Observation supports the proposed fact", "method": "llm", "incident_id": "incident-a"})
    with pytest.raises(ValueError, match="unknown Observations"):
        projector.validate(proposal, observation_ids={"obs-12"})


def test_structured_verification_key_distinguishes_questions() -> None:
    from debug_assistant.incidents.contracts import build_verification_key

    left = build_verification_key(
        target_component="checkoutservice",
        verification_question="does selector target port 5050?",
        evidence_type="service endpoint observation",
        related_hypothesis="selector mismatch",
    )
    right = build_verification_key(
        target_component="checkoutservice",
        verification_question="does selector target port 5051?",
        evidence_type="service endpoint observation",
        related_hypothesis="selector mismatch",
    )
    assert left and right and left != right


def test_verification_key_ignores_hypothesis_wording() -> None:
    from debug_assistant.incidents.contracts import build_verification_key

    left = build_verification_key(
        target_component="checkoutservice",
        verification_question="verify targetPort and containerPort",
        evidence_type="service configuration",
        related_hypothesis="the service may route to the wrong port",
    )
    right = build_verification_key(
        target_component="checkoutservice",
        verification_question="verify targetPort and containerPort",
        evidence_type="service configuration",
        related_hypothesis="checkoutservice has a bad endpoint mapping",
    )
    assert left == right


def test_evidence_scope_uses_stable_fault_code_across_hypothesis_rewording() -> None:
    first = Obj(
        claim="probe failure wording one",
        mechanism="kubelet restarts the container",
        fault_code="liveness_probe_incorrect_port",
    )
    revised = Obj(
        claim="probe failure wording two",
        mechanism="configured probe port mismatches serving port",
        fault_code="liveness_probe_incorrect_port",
    )
    projector = EvidenceProjector()
    assert projector.hypothesis_fingerprint(first) == projector.hypothesis_fingerprint(revised)


def test_fault_family_does_not_override_canonical_revision_scope() -> None:
    before_code = Obj(
        hypothesis_id="hyp-branch", revision=1,
        claim="the pod is pending",
        fault="pending or unschedulable pod",
        mechanism="the scheduler cannot place the pod",
    )
    after_code = Obj(
        hypothesis_id="hyp-branch", revision=2,
        claim="the pod has no eligible node",
        fault="pending or unschedulable pod",
        fault_code="node_affinity_mismatch",
        mechanism="required node affinity matches no node",
    )
    projector = EvidenceProjector()
    assert projector.hypothesis_fingerprint(before_code) not in projector.hypothesis_scope_fingerprints(after_code)
