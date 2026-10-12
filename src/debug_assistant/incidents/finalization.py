"""Canonical finalization decision and Candidate projection."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from debug_assistant.memory.evidence_memory import EvidenceMemory
from debug_assistant.memory.evidence_projector import ProjectedEvidenceFact

from .contracts import RootCauseCandidate, is_canonical_fault_code
from .evidence_pipeline import EvidenceSet
from .rca_state import RCAState


FinalizationConditionName = Literal[
    "candidate_fields_complete",
    "direct_causal_support_present",
    "critical_obligations_closed",
    "required_fact_slots_covered",
    "contradictions_resolved",
    "source_coverage_complete",
    "taxonomy_code_valid",
]


class FinalizationConditions(BaseModel):
    """Affirmative predicates required to publish a Candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_fields_complete: bool
    direct_causal_support_present: bool
    critical_obligations_closed: bool
    required_fact_slots_covered: bool
    contradictions_resolved: bool
    source_coverage_complete: bool
    taxonomy_code_valid: bool


class FinalizationReason(BaseModel):
    """One typed explanation for an unmet positive condition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    condition: FinalizationConditionName
    details: dict[str, Any] = Field(default_factory=dict)


class FinalizationFacts(BaseModel):
    """Deterministic capability, source, taxonomy, and scope facts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis_scope_fingerprints: tuple[str, ...] = ()
    source_coverage_required: bool = False
    source_coverage_complete: bool = True
    taxonomy_required: bool = False


class FinalizationDecision(BaseModel):
    """The sole semantic permission and Candidate projection result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed: bool
    conditions: FinalizationConditions
    reasons: tuple[FinalizationReason, ...] = ()
    candidate: RootCauseCandidate | None = None
    direct_evidence_ids: tuple[str, ...] = ()


class FinalizationPolicy:
    """Evaluate canonical state without mutating lifecycle or Evidence."""

    @staticmethod
    def evaluate(
        state: RCAState,
        *,
        evidence_memory: EvidenceMemory,
        evidence_sets: Iterable[EvidenceSet],
        projected_facts: Iterable[ProjectedEvidenceFact],
        facts: FinalizationFacts,
    ) -> FinalizationDecision:
        hypothesis = state.active_hypothesis_projection()
        known_evidence = {
            item.evidence_id: item for item in evidence_memory.pinned
        }
        scopes = set(facts.hypothesis_scope_fingerprints)
        obligations = {
            item.id: item
            for item in state.obligations
            if item.status != "SUPERSEDED_BY_REPLAN"
        }
        sets_by_obligation: dict[str, list[EvidenceSet]] = {}
        for evidence_set in evidence_sets:
            if evidence_set.obligation_id in obligations:
                sets_by_obligation.setdefault(evidence_set.obligation_id, []).append(
                    evidence_set
                )

        covered_relationships: set[tuple[str, str, str]] = set()
        missing_slots: dict[str, tuple[str, ...]] = {}
        for obligation in obligations.values():
            if obligation.status != "SATISFIED":
                continue
            matching_sets = tuple(
                item
                for item in sets_by_obligation.get(obligation.id, ())
                if item.semantic_status != "CONTRADICTS"
                and (
                    not scopes
                    or bool(scopes.intersection(item.hypothesis_scope_fingerprints))
                )
            )
            slot_ids = {
                slot: tuple(dict.fromkeys(
                    evidence_id
                    for item in matching_sets
                    for evidence_id in item.coverage.get(slot, ())
                ))
                for slot in obligation.required_fact_slots
            }
            absent = tuple(slot for slot, ids in slot_ids.items() if not ids)
            if absent or not any(
                item.semantic_status == "SATISFIES" for item in matching_sets
            ):
                missing_slots[obligation.id] = (
                    absent or obligation.required_fact_slots
                )
            if obligation.id not in missing_slots:
                causal_slots = {requirement.slot for requirement in
                                obligation.completion_contract.required_facts
                                if requirement.role in {"MECHANISM", "CAUSAL_LINK"}}
                covered_relationships.update(
                    (obligation.id, slot, evidence_id)
                    for slot in causal_slots for evidence_id in slot_ids[slot]
                )

        direct_ids: list[str] = []
        invalid_direct_ids: set[str] = set()
        for fact in projected_facts:
            if fact.relation == "CONTRADICTS" or not any(
                (fact.obligation_id, slot, fact.evidence_id) in covered_relationships
                for slot in fact.covered_fact_slots
            ):
                continue
            if fact.obligation_id not in obligations:
                continue
            if fact.evidence_id not in obligations[fact.obligation_id].supporting_evidence_ids:
                invalid_direct_ids.add(fact.evidence_id)
                continue
            if fact.hypothesis_fingerprint not in scopes:
                invalid_direct_ids.add(fact.evidence_id)
                continue
            if fact.hypothesis_relation != "SUPPORTS":
                continue
            if fact.fulfillment not in {"PARTIAL", "SATISFIES"}:
                continue
            evidence = known_evidence.get(fact.evidence_id)
            observation_refs = set(fact.provenance.get("observation_refs", ()))
            if (
                evidence is None
                or fact.evidence_id not in state.evidence_ids
                or evidence.raw_observation_id != fact.observation_id
                or fact.observation_id not in observation_refs
            ):
                invalid_direct_ids.add(fact.evidence_id)
                continue
            direct_ids.append(fact.evidence_id)
        direct_evidence_ids = tuple(dict.fromkeys(direct_ids))

        candidate_fields_complete = bool(
            hypothesis
            and hypothesis.component.strip()
            and hypothesis.fault.strip()
            and hypothesis.mechanism.strip()
            and (hypothesis.fault_code.strip() or hypothesis.fault_explanation.strip())
        )
        direct_causal_support_present = bool(direct_evidence_ids)
        open_critical = tuple(
            item.id for item in obligations.values() if item.blocks_finalization
        )
        critical_obligations_closed = not open_critical
        satisfied_obligations = tuple(
            item.id for item in obligations.values() if item.status == "SATISFIED"
        )
        required_fact_slots_covered = bool(satisfied_obligations) and not missing_slots
        open_contradictions = tuple(
            item.evidence_id for item in state.contradictions if item.blocks_finalization
        )
        contradictions_resolved = not open_contradictions
        source_coverage_complete = (
            facts.source_coverage_complete
            if facts.source_coverage_required
            else True
        )
        taxonomy_code_valid = bool(
            hypothesis
            and (
                not facts.taxonomy_required
                or (
                    hypothesis.fault_code.strip()
                    and is_canonical_fault_code(hypothesis.fault_code)
                )
            )
        )
        conditions = FinalizationConditions(
            candidate_fields_complete=candidate_fields_complete,
            direct_causal_support_present=direct_causal_support_present,
            critical_obligations_closed=critical_obligations_closed,
            required_fact_slots_covered=required_fact_slots_covered,
            contradictions_resolved=contradictions_resolved,
            source_coverage_complete=source_coverage_complete,
            taxonomy_code_valid=taxonomy_code_valid,
        )

        reasons: list[FinalizationReason] = []
        if not candidate_fields_complete:
            reasons.append(FinalizationReason(
                code="candidate_fields_incomplete",
                condition="candidate_fields_complete",
            ))
        if not direct_causal_support_present:
            reasons.append(FinalizationReason(
                code="direct_causal_support_missing",
                condition="direct_causal_support_present",
                details={"invalid_evidence_ids": sorted(invalid_direct_ids)},
            ))
        if not critical_obligations_closed:
            reasons.append(FinalizationReason(
                code="critical_obligations_open",
                condition="critical_obligations_closed",
                details={"obligation_ids": list(open_critical)},
            ))
        if not required_fact_slots_covered:
            reasons.append(FinalizationReason(
                code="required_fact_slots_missing",
                condition="required_fact_slots_covered",
                details={
                    "missing_slots": {
                        obligation_id: list(slots)
                        for obligation_id, slots in missing_slots.items()
                    },
                    "satisfied_obligation_ids": list(satisfied_obligations),
                },
            ))
        if not contradictions_resolved:
            reasons.append(FinalizationReason(
                code="blocking_contradictions_open",
                condition="contradictions_resolved",
                details={"evidence_ids": list(open_contradictions)},
            ))
        if not source_coverage_complete:
            reasons.append(FinalizationReason(
                code="source_coverage_incomplete",
                condition="source_coverage_complete",
            ))
        if not taxonomy_code_valid:
            reasons.append(FinalizationReason(
                code=(
                    "taxonomy_code_missing"
                    if hypothesis is None or not hypothesis.fault_code.strip()
                    else "taxonomy_code_invalid"
                ),
                condition="taxonomy_code_valid",
                details={"fault_code": hypothesis.fault_code if hypothesis else ""},
            ))

        allowed = not reasons
        candidate = None
        if allowed and hypothesis is not None:
            candidate = RootCauseCandidate(
                component=hypothesis.component,
                fault=hypothesis.fault,
                fault_code=hypothesis.fault_code,
                fault_explanation=hypothesis.fault_explanation,
                mechanism=hypothesis.mechanism,
                evidence_ids=direct_evidence_ids,
                confidence=None,
                claim_evidence_mapping=(
                    {
                        "claim": f"{hypothesis.component}: {hypothesis.fault}",
                        "evidence_ids": direct_evidence_ids,
                    },
                ),
                causal_chain_summary=(
                    {
                        "cause": hypothesis.fault,
                        "effect": hypothesis.mechanism,
                        "evidence_ids": direct_evidence_ids,
                    },
                ),
                source_claims=hypothesis.source_claims,
            )
        return FinalizationDecision(
            allowed=allowed,
            conditions=conditions,
            reasons=tuple(reasons),
            candidate=candidate,
            direct_evidence_ids=direct_evidence_ids,
        )
