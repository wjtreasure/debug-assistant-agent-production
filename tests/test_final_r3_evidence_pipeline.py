from __future__ import annotations
from completion_fixtures import completion

import pytest

from debug_assistant.incidents.contracts import VerificationObligation
from debug_assistant.incidents.evidence_pipeline import EvidencePipeline, EvidencePipelineError
from debug_assistant.incidents.rca_state import RCAState
from debug_assistant.memory.evidence_memory import EvidenceMemory
from debug_assistant.memory.evidence_projector import (
    EvidenceProjector,
    EvidenceProposal,
    ProjectedEvidenceFact,
)
from debug_assistant.memory.observation_store import ObservationStore
from debug_assistant.models import ToolObservation


class Obj:
    def __init__(self, **values):
        self.revision = 1
        self.__dict__.update(values)


def observation(observation_id: str, content: str = "captured fact") -> ToolObservation:
    return ToolObservation(
        tool="get_resources",
        ok=True,
        content=content,
        metadata={},
        observation_id=observation_id,
    )


def proposal(
    observation_id: str,
    obligation_id: str,
    hypothesis_id: str,
    *slots: str,
    relation: str = "SUPPORTS",
) -> EvidenceProposal:
    return EvidenceProposal(
        semantic_claim=f"fact from {observation_id}",
        relation=relation,
        hypothesis_relation="NEUTRAL",
        covered_fact_slots=slots,
        observation_refs=(observation_id,),
        reason="Planner semantic proposal",
        method="llm",
        obligation_id=obligation_id,
        hypothesis_id=hypothesis_id,
        incident_id="incident-r3",
    )


def pipeline_with(*observations: ToolObservation) -> EvidencePipeline:
    store = ObservationStore()
    for item in observations:
        store.add(item)
    return EvidencePipeline(
        observation_store=store,
        evidence_memory=EvidenceMemory(),
        projector=EvidenceProjector(),
    )


def test_single_slot_admission_produces_coverage_and_rca_events() -> None:
    obs = observation("obs-1")
    pipeline = pipeline_with(obs)
    obligation = VerificationObligation(id="obl-1", claim="verify fact", completion_contract=completion("fact"))
    hypothesis = Obj(id="hyp-1", claim="candidate cause")
    state = RCAState(incident_id="incident-r3", obligations=(obligation,))

    result = pipeline.admit(
        obs,
        proposal=proposal("obs-1", "obl-1", "hyp-1", "fact"),
        obligation=obligation,
        hypothesis=hypothesis,
        incident_id="incident-r3",
        evidence_id="ev-1",
        rca_state=state,
        hypothesis_scopes=EvidenceProjector.hypothesis_scope_fingerprints(hypothesis),
    )

    assert result.evidence_set is not None
    assert result.evidence_set.status == "SATISFIES"
    assert result.evidence_set.coverage == {"fact": ("ev-1",)}
    assert [event.kind for event in result.rca_events] == [
        "LINK_EVIDENCE", "UPDATE_OBLIGATION",
    ]


def test_multi_slot_coverage_requires_all_named_facts() -> None:
    first = observation("obs-config", "configured value is 1")
    second = observation("obs-runtime", "observed value is 2")
    pipeline = pipeline_with(first, second)
    obligation = VerificationObligation(
        id="obl-compare",
        claim="compare configured and observed values",
        completion_contract=completion(*("configured_value", "observed_value")),
    )
    hypothesis = Obj(id="hyp-1", claim="values differ")
    scopes = EvidenceProjector.hypothesis_scope_fingerprints(hypothesis)
    first_result = pipeline.admit(
        first,
        proposal=proposal(
            "obs-config", "obl-compare", "hyp-1", "configured_value",
        ),
        obligation=obligation,
        hypothesis=hypothesis,
        incident_id="incident-r3",
        evidence_id="ev-config",
        hypothesis_scopes=scopes,
    )
    assert first_result.evidence_set is not None
    assert first_result.evidence_set.status == "INSUFFICIENT"

    second_result = pipeline.admit(
        second,
        proposal=proposal(
            "obs-runtime", "obl-compare", "hyp-1", "observed_value",
        ),
        obligation=obligation,
        hypothesis=hypothesis,
        incident_id="incident-r3",
        evidence_id="ev-runtime",
        existing_facts=(first_result.projected_fact,),
        hypothesis_scopes=scopes,
    )
    assert second_result.evidence_set is not None
    assert second_result.evidence_set.status == "SATISFIES"
    assert second_result.evidence_set.coverage == {
        "configured_value": ("ev-config",),
        "observed_value": ("ev-runtime",),
    }


def test_redundant_partial_does_not_block_complete_coverage() -> None:
    first = observation("obs-complete")
    redundant = observation("obs-partial", "related context")
    pipeline = pipeline_with(first, redundant)
    obligation = VerificationObligation(id="obl-1", claim="verify fact", completion_contract=completion("fact"))
    hypothesis = Obj(id="hyp-1", claim="candidate")
    scopes = EvidenceProjector.hypothesis_scope_fingerprints(hypothesis)
    complete = pipeline.admit(
        first,
        proposal=proposal("obs-complete", "obl-1", "hyp-1", "fact"),
        obligation=obligation,
        hypothesis=hypothesis,
        incident_id="incident-r3",
        evidence_id="ev-complete",
        hypothesis_scopes=scopes,
    )
    result = pipeline.admit(
        redundant,
        proposal=proposal("obs-partial", "obl-1", "hyp-1"),
        obligation=obligation,
        hypothesis=hypothesis,
        incident_id="incident-r3",
        evidence_id="ev-partial",
        existing_facts=(complete.projected_fact,),
        hypothesis_scopes=scopes,
    )
    assert result.evidence_set is not None
    assert result.evidence_set.status == "SATISFIES"
    assert result.evidence_set.coverage == {"fact": ("ev-complete",)}


def test_unknown_slot_fails_closed_before_evidence_admission() -> None:
    obs = observation("obs-unknown")
    pipeline = pipeline_with(obs)
    obligation = VerificationObligation(id="obl-1", claim="verify fact", completion_contract=completion("fact"))
    with pytest.raises(EvidencePipelineError, match="unknown_fact_slot"):
        pipeline.admit(
            obs,
            proposal=proposal("obs-unknown", "obl-1", "hyp-1", "other"),
            obligation=obligation,
            hypothesis=Obj(id="hyp-1", claim="candidate"),
            incident_id="incident-r3",
            evidence_id="ev-unknown",
        )
    assert pipeline.evidence_memory.pinned == []


def test_wrong_obligation_binding_fails_closed_before_evidence_admission() -> None:
    obs = observation("obs-wrong-obligation")
    pipeline = pipeline_with(obs)
    obligation = VerificationObligation(id="obl-1", claim="verify fact", completion_contract=completion("fact"))
    with pytest.raises(ValueError, match="different obligation"):
        pipeline.admit(
            obs,
            proposal=proposal("obs-wrong-obligation", "obl-other", "hyp-1", "fact"),
            obligation=obligation,
            hypothesis=Obj(id="hyp-1", claim="candidate"),
            incident_id="incident-r3",
            evidence_id="ev-wrong-obligation",
        )
    assert pipeline.evidence_memory.pinned == []


def test_relation_without_slot_coverage_cannot_complete() -> None:
    obs = observation("obs-label-only")
    pipeline = pipeline_with(obs)
    obligation = VerificationObligation(id="obl-1", claim="verify fact", completion_contract=completion("fact"))
    result = pipeline.admit(
        obs,
        proposal=proposal(
            "obs-label-only", "obl-1", "hyp-1",
        ),
        obligation=obligation,
        hypothesis=Obj(id="hyp-1", claim="candidate"),
        incident_id="incident-r3",
        evidence_id="ev-label-only",
    )
    assert result.evidence_set is not None
    assert result.evidence_set.status == "DOES_NOT_SATISFY"
    assert result.evidence_set.coverage == {"fact": ()}


def test_deterministic_coverage_requires_an_allowlisted_adapter() -> None:
    obs = observation("obs-deterministic")
    store = ObservationStore()
    store.add(obs)
    pipeline = EvidencePipeline(
        observation_store=store,
        evidence_memory=EvidenceMemory(),
        projector=EvidenceProjector(),
        deterministic_fact_adapters=("verified_adapter",),
    )
    obligation = VerificationObligation(id="obl-1", claim="verify fact", completion_contract=completion("fact"))
    semantic = proposal("obs-deterministic", "obl-1", "hyp-1", "fact").model_copy(
        update={"method": "deterministic"}
    )
    with pytest.raises(EvidencePipelineError, match="deterministic_adapter_not_allowed"):
        pipeline.admit(
            obs,
            proposal=semantic,
            obligation=obligation,
            hypothesis=Obj(id="hyp-1", claim="candidate"),
            incident_id="incident-r3",
            evidence_id="ev-deterministic",
        )

    accepted = pipeline.admit(
        obs,
        proposal=semantic,
        obligation=obligation,
        hypothesis=Obj(id="hyp-1", claim="candidate"),
        incident_id="incident-r3",
        evidence_id="ev-deterministic",
        deterministic_adapter_id="verified_adapter",
    )
    assert accepted.evidence_set is not None
    assert accepted.evidence_set.status == "SATISFIES"


def test_unknown_evidence_id_fails_closed() -> None:
    pipeline = pipeline_with()
    obligation = VerificationObligation(id="obl-1", claim="verify fact", completion_contract=completion("fact"))
    fact = ProjectedEvidenceFact(
        evidence_id="ev-missing",
        observation_id="obs-missing",
        fact="missing fact",
        relation="SUPPORTS",
        fulfillment="PARTIAL",
        covered_fact_slots=("fact",),
        obligation_id="obl-1",
        hypothesis_fingerprint="scope-a",
        provenance={"observation_refs": ["obs-missing"]},
    )
    with pytest.raises(EvidencePipelineError, match="unknown_evidence_id"):
        pipeline.qualify(obligation=obligation, evidence=(fact,))


def test_stale_hypothesis_scope_cannot_complete() -> None:
    obs = observation("obs-stale")
    pipeline = pipeline_with(obs)
    pipeline.evidence_memory.add_observation(obs, evidence_id="ev-stale")
    obligation = VerificationObligation(id="obl-1", claim="verify fact", completion_contract=completion("fact"))
    fact = ProjectedEvidenceFact(
        evidence_id="ev-stale",
        observation_id="obs-stale",
        fact="stale fact",
        relation="SUPPORTS",
        fulfillment="SATISFIES",
        covered_fact_slots=("fact",),
        obligation_id="obl-1",
        hypothesis_fingerprint="scope-old",
        provenance={"observation_refs": ["obs-stale"]},
    )
    result = pipeline.qualify(
        obligation=obligation,
        evidence=(fact,),
        hypothesis_scopes=("scope-current",),
    )
    assert result.status == "INSUFFICIENT"
    assert result.evidence_ids == ()


def test_invalid_provenance_fails_closed() -> None:
    obs = observation("obs-origin")
    pipeline = pipeline_with(obs)
    pipeline.evidence_memory.add_observation(obs, evidence_id="ev-1")
    obligation = VerificationObligation(id="obl-1", claim="verify fact", completion_contract=completion("fact"))
    fact = ProjectedEvidenceFact(
        evidence_id="ev-1",
        observation_id="obs-other",
        fact="misbound fact",
        relation="SUPPORTS",
        fulfillment="SATISFIES",
        covered_fact_slots=("fact",),
        obligation_id="obl-1",
        provenance={"observation_refs": ["obs-other"]},
    )
    with pytest.raises(EvidencePipelineError, match="invalid_provenance"):
        pipeline.qualify(obligation=obligation, evidence=(fact,))


def test_contradiction_blocks_complete_coverage() -> None:
    support_obs = observation("obs-support")
    contradiction_obs = observation("obs-contradiction", "captured contradiction")
    pipeline = pipeline_with(support_obs, contradiction_obs)
    pipeline.evidence_memory.add_observation(support_obs, evidence_id="ev-support")
    pipeline.evidence_memory.add_observation(
        contradiction_obs, evidence_id="ev-contradiction",
    )
    obligation = VerificationObligation(id="obl-1", claim="verify fact", completion_contract=completion("fact"))
    support = ProjectedEvidenceFact(
        evidence_id="ev-support", observation_id="obs-support", fact="support",
        relation="SUPPORTS", fulfillment="SATISFIES", covered_fact_slots=("fact",),
        obligation_id="obl-1", provenance={"observation_refs": ["obs-support"]},
    )
    contradiction = ProjectedEvidenceFact(
        evidence_id="ev-contradiction", observation_id="obs-contradiction",
        fact="contradiction", relation="CONTRADICTS", fulfillment="INSUFFICIENT",
        obligation_id="obl-1",
        provenance={"observation_refs": ["obs-contradiction"]},
    )
    result = pipeline.qualify(
        obligation=obligation,
        evidence=(support, contradiction),
    )
    assert result.semantic_status == "CONTRADICTS"
    assert result.status == "DOES_NOT_SATISFY"
    assert all(event.kind != "UPDATE_OBLIGATION" for event in pipeline.propose_rca_events(
        result, facts=(support, contradiction), state=None,
    ))


def test_tool_success_metadata_cannot_propose_semantic_coverage() -> None:
    obs = ToolObservation(
        tool="get_resources",
        ok=True,
        content="tool execution succeeded",
        metadata={
            "supports_obligation": True,
            "obligation_fulfillment": "SATISFIES",
        },
        observation_id="obs-tool-success",
    )
    obligation = VerificationObligation(id="obl-1", claim="verify root cause", completion_contract=completion("fact"))
    from debug_assistant.memory.evidence_projector import EvidenceProposalError
    with pytest.raises(EvidenceProposalError, match="requires an explicit"):
        EvidenceProjector().project(obs, obligation_id=obligation.id)
