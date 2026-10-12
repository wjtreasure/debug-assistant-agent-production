"""Canonical Observation-to-Evidence admission and completion pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from debug_assistant.memory.evidence_memory import EvidenceMemory
from debug_assistant.memory.evidence_projector import (
    EvidenceProjector,
    EvidenceProposal,
    ProjectedEvidenceFact,
    evidence_relationship_key,
)
from debug_assistant.memory.observation_store import ObservationStore

from .contracts import ObligationFulfillment, VerificationObligation
from .rca_state import RCAEvent, RCAState


QualificationStatus = Literal["SATISFIES", "DOES_NOT_SATISFY", "INSUFFICIENT"]
EvidenceSetStatus = Literal["SATISFIES", "PARTIAL", "CONTRADICTS", "INSUFFICIENT"]


class EvidencePipelineError(ValueError):
    """A deterministic Evidence admission invariant failed closed."""

    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(f"{code}: {detail}")


class EvidenceSet(BaseModel):
    """Validated Evidence coverage for one Obligation and hypothesis scope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    obligation_id: str = Field(min_length=1)
    hypothesis_id: str = ""
    evidence_ids: tuple[str, ...] = ()
    supporting_evidence_ids: tuple[str, ...] = ()
    contradicting_evidence_ids: tuple[str, ...] = ()
    partial_evidence_ids: tuple[str, ...] = ()
    hypothesis_scope_fingerprints: tuple[str, ...] = ()
    coverage: Mapping[str, tuple[str, ...]] = Field(default_factory=dict)
    fulfillment: ObligationFulfillment = "INSUFFICIENT"
    status: QualificationStatus = "INSUFFICIENT"
    semantic_status: EvidenceSetStatus = "INSUFFICIENT"
    reason: str = Field(min_length=1)

    @property
    def is_satisfying(self) -> bool:
        return self.status == "SATISFIES"

    @property
    def gate_outcome(self) -> str:
        return "CONTRADICTED" if self.semantic_status == "CONTRADICTS" else self.semantic_status

    @model_validator(mode="after")
    def validate_invariants(self) -> "EvidenceSet":
        evidence_ids = set(self.evidence_ids)
        if len(evidence_ids) != len(self.evidence_ids):
            raise ValueError("EvidenceSet evidence_ids must be unique")
        for field_name in (
            "supporting_evidence_ids",
            "contradicting_evidence_ids",
            "partial_evidence_ids",
        ):
            values = tuple(getattr(self, field_name))
            if len(set(values)) != len(values):
                raise ValueError(f"EvidenceSet {field_name} must be unique")
            unknown = set(values) - evidence_ids
            if unknown:
                raise ValueError(
                    f"EvidenceSet {field_name} contains unknown Evidence IDs: "
                    + ", ".join(sorted(unknown))
                )
        for slot, values in self.coverage.items():
            if not str(slot).strip():
                raise ValueError("EvidenceSet coverage slot must be non-empty")
            if len(set(values)) != len(values):
                raise ValueError(f"EvidenceSet coverage for {slot} must be unique")
            unknown = set(values) - evidence_ids
            if unknown:
                raise ValueError(
                    f"EvidenceSet coverage contains unknown Evidence IDs: "
                    + ", ".join(sorted(unknown))
                )
        if self.semantic_status == "SATISFIES":
            if self.status != "SATISFIES" or self.fulfillment != "SATISFIES":
                raise ValueError("SATISFIES EvidenceSet requires matching status and fulfillment")
            if not self.supporting_evidence_ids:
                raise ValueError("SATISFIES EvidenceSet requires supporting Evidence")
            if not self.coverage or not all(self.coverage.values()):
                raise ValueError("SATISFIES EvidenceSet requires complete slot coverage")
            if self.contradicting_evidence_ids:
                raise ValueError("SATISFIES EvidenceSet cannot contain contradictory Evidence")
        if self.status == "SATISFIES" and self.semantic_status != "SATISFIES":
            raise ValueError("status=SATISFIES requires semantic_status=SATISFIES")
        if self.semantic_status == "CONTRADICTS":
            if self.status == "SATISFIES" or self.fulfillment == "SATISFIES":
                raise ValueError("CONTRADICTS EvidenceSet cannot satisfy an Obligation")
            if not self.contradicting_evidence_ids:
                raise ValueError("CONTRADICTS EvidenceSet requires contradictory Evidence")
        return self


class EvidencePipelineResult(BaseModel):
    """One pipeline outcome; Runtime applies the proposed reducer events."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    evidence_id: str = ""
    projected_fact: ProjectedEvidenceFact | None = None
    evidence_set: EvidenceSet | None = None
    rca_events: tuple[RCAEvent, ...] = ()
    duplicate: bool = False


class EvidencePipeline:
    """Single production path for Evidence admission and Obligation coverage."""

    def __init__(
        self,
        *,
        observation_store: ObservationStore,
        evidence_memory: EvidenceMemory,
        projector: EvidenceProjector,
        deterministic_fact_adapters: Iterable[str] = (),
    ):
        self.observation_store = observation_store
        self.evidence_memory = evidence_memory
        self.projector = projector
        self.deterministic_fact_adapters = frozenset(
            str(item).strip() for item in deterministic_fact_adapters if str(item).strip()
        )

    @staticmethod
    def _obligation_id(obligation: Any) -> str:
        return str(
            getattr(obligation, "id", None)
            or getattr(obligation, "obligation_id", "")
            or ""
        )

    @staticmethod
    def _required_slots(obligation: Any) -> tuple[str, ...]:
        contract = getattr(obligation, "completion_contract", None)
        if contract is None:
            raise EvidencePipelineError("completion_contract_missing", "explicit contract required")
        return contract.required_fact_slots

    @staticmethod
    def _validate_constraints(obligation: VerificationObligation, slots: Iterable[str],
                              observation: Any) -> None:
        metadata = observation.metadata or {}
        actual = {"context_kinds": metadata.get("context_kind"),
                  "information_sources": metadata.get("information_source"),
                  "source_tools": observation.tool}
        requirements = {fact.slot: fact for fact in obligation.completion_contract.required_facts}
        for slot in slots:
            if slot not in requirements:
                raise EvidencePipelineError("unknown_fact_slot", slot)
            constraint = requirements[slot].evidence_constraints
            for category, value in actual.items():
                allowed = getattr(constraint, category)
                if allowed and value not in allowed:
                    raise EvidencePipelineError("evidence_constraint_mismatch", f"{slot}: {category}")

    def _validate_coverage_proposal(
        self,
        proposal: EvidenceProposal,
        *,
        obligation: Any,
        deterministic_adapter_id: str,
    ) -> None:
        required = set(self._required_slots(obligation))
        proposed = tuple(proposal.covered_fact_slots)
        if len(set(proposed)) != len(proposed):
            raise EvidencePipelineError("duplicate_fact_slot", "covered_fact_slots must be unique")
        unknown = sorted(set(proposed) - required)
        if unknown:
            raise EvidencePipelineError(
                "unknown_fact_slot", ", ".join(unknown),
            )
        if (
            proposal.method == "deterministic"
            and proposed
            and deterministic_adapter_id not in self.deterministic_fact_adapters
        ):
            raise EvidencePipelineError(
                "deterministic_adapter_not_allowed",
                deterministic_adapter_id or "missing adapter identity",
            )

    def admit(
        self,
        observation: Any,
        *,
        proposal: EvidenceProposal,
        obligation: VerificationObligation | None,
        hypothesis: Any | None,
        incident_id: str,
        evidence_id: str,
        existing_facts: Iterable[ProjectedEvidenceFact] = (),
        rca_state: RCAState | None = None,
        hypothesis_scopes: Iterable[str] | None = None,
        completion_prerequisites_met: bool = True,
        completion_blocker: str = "",
        resolved_contradiction_ids: Iterable[str] = (),
        deterministic_adapter_id: str = "",
        evidence_kwargs: Mapping[str, Any] | None = None,
        visible_observation_ids: Iterable[str] | None = None,
    ) -> EvidencePipelineResult:
        """Admit one proposal, materialize Evidence, and propose RCA events."""
        if proposal.evidence_ref is not None:
            raise EvidencePipelineError("new_evidence_source_required", "use reuse for an Evidence reference")
        stored = self.observation_store.get(str(getattr(observation, "observation_id", "") or ""))
        if stored is None:
            raise EvidencePipelineError("unknown_observation", "Observation is not in ObservationStore")
        if not self.projector.incident_observation_available(stored):
            raise EvidencePipelineError("observation_not_admissible", stored.observation_id)
        if stored.observation_id not in proposal.observation_refs:
            raise EvidencePipelineError("observation_binding_invalid", stored.observation_id)
        existing_facts = tuple(existing_facts)
        if visible_observation_ids is not None:
            visible = set(visible_observation_ids)
            if len(proposal.observation_refs) != 1 or proposal.observation_refs[0] not in visible:
                raise EvidencePipelineError("observation_not_visible", "proposal requires one visible Observation")
        if obligation is not None and rca_state is not None:
            try:
                canonical = rca_state.obligation(obligation.id)
            except ValueError as exc:
                raise EvidencePipelineError("unknown_obligation", obligation.id) from exc
            if canonical != obligation:
                raise EvidencePipelineError("noncanonical_obligation", obligation.id)
        self.projector.validate(
            proposal,
            observation_ids={item.observation_id for item in self.observation_store.all()},
            incident_id=incident_id,
            obligation_id=obligation.id if obligation is not None else None,
            hypothesis_id=self.projector.hypothesis_id(hypothesis),
        )
        if obligation is not None:
            self._validate_coverage_proposal(
                proposal,
                obligation=obligation,
                deterministic_adapter_id=deterministic_adapter_id,
            )
            if proposal.covered_fact_slots and proposal.observation_refs != (
                stored.observation_id,
            ):
                raise EvidencePipelineError(
                    "coverage_provenance_not_atomic",
                    "slot coverage must reference only the current Observation",
                )
        elif proposal.covered_fact_slots:
            raise EvidencePipelineError(
                "missing_obligation_binding",
                "coverage requires a canonical VerificationObligation",
            )
        if obligation is not None:
            self._validate_constraints(obligation, proposal.covered_fact_slots, stored)
        evidence = self.evidence_memory.add_observation(
            stored,
            evidence_id=evidence_id,
            **dict(evidence_kwargs or {}),
        )
        evidence = evidence or self.evidence_memory.evidence_for_observation(stored.observation_id)
        if evidence is None:
            raise EvidencePipelineError("evidence_admission_rejected", stored.observation_id)
        # Content dedup may bind a second acquisition to the original Evidence.
        # Relations always retain the canonical Evidence's original provenance.
        stored = self.observation_store.get(evidence.raw_observation_id)
        if stored is None:
            raise EvidencePipelineError("invalid_provenance", evidence.evidence_id)
        if obligation is not None:
            self._validate_constraints(obligation, proposal.covered_fact_slots, stored)
        proposal = proposal.model_copy(update={"observation_refs": (stored.observation_id,)})
        return self._relate(
            stored, evidence_id=evidence.evidence_id, proposal=proposal,
            obligation=obligation, hypothesis=hypothesis, existing_facts=existing_facts,
            rca_state=rca_state, hypothesis_scopes=hypothesis_scopes,
            completion_prerequisites_met=completion_prerequisites_met,
            completion_blocker=completion_blocker,
            resolved_contradiction_ids=resolved_contradiction_ids,
        )

    def reuse(
        self, *, proposal: EvidenceProposal, obligation: VerificationObligation,
        hypothesis: Any, incident_id: str, rca_state: RCAState,
        existing_facts: Iterable[ProjectedEvidenceFact] = (),
        reusable_evidence_ids: Iterable[str],
        hypothesis_scopes: Iterable[str] | None = None,
        resolved_contradiction_ids: Iterable[str] = (),
    ) -> EvidencePipelineResult:
        """Validate an explicit new relationship to already admitted Evidence.

        Acquisition scope stays immutable. The LLM explicitly proposes the new
        semantic scope; no raw-window dependency, Tool call or body copy occurs.
        """
        if proposal.evidence_ref is None or proposal.observation_refs or proposal.source_spans:
            raise EvidencePipelineError("evidence_reuse_source_invalid", "reuse requires only an Evidence reference")
        evidence = self.evidence_memory.get(proposal.evidence_ref)
        if evidence is None or evidence.evidence_id not in rca_state.evidence_ids:
            raise EvidencePipelineError("unknown_evidence_id", str(proposal.evidence_ref))
        if evidence.evidence_id not in set(reusable_evidence_ids):
            raise EvidencePipelineError("evidence_not_visible", evidence.evidence_id)
        if rca_state.incident_id != incident_id or rca_state.obligation(obligation.id) != obligation:
            raise EvidencePipelineError("noncanonical_scope", obligation.id)
        stored = self.observation_store.get(evidence.raw_observation_id)
        if stored is None or not self.projector.incident_observation_available(stored):
            raise EvidencePipelineError("invalid_provenance", evidence.evidence_id)
        if self.evidence_memory.evidence_for_observation(stored.observation_id) != evidence:
            raise EvidencePipelineError("invalid_provenance", evidence.evidence_id)
        # Materialize the immutable provenance only internally. Model output
        # cannot replace it or cite a stale raw Observation as a new source.
        grounded = EvidenceProposal.model_validate({
            **proposal.model_dump(), "evidence_ref": None,
            "observation_refs": (stored.observation_id,),
        })
        self.projector.validate(grounded, observation_ids={stored.observation_id},
                                incident_id=incident_id, obligation_id=obligation.id,
                                hypothesis_id=self.projector.hypothesis_id(hypothesis))
        self._validate_coverage_proposal(grounded, obligation=obligation, deterministic_adapter_id="")
        self._validate_constraints(obligation, proposal.covered_fact_slots, stored)
        return self._relate(
            stored, evidence_id=evidence.evidence_id, proposal=grounded,
            obligation=obligation, hypothesis=hypothesis, existing_facts=tuple(existing_facts),
            rca_state=rca_state, hypothesis_scopes=hypothesis_scopes,
            resolved_contradiction_ids=resolved_contradiction_ids,
        )

    def _relate(
        self, stored: Any, *, evidence_id: str, proposal: EvidenceProposal,
        obligation: VerificationObligation | None, hypothesis: Any | None,
        existing_facts: tuple[ProjectedEvidenceFact, ...], rca_state: RCAState | None,
        hypothesis_scopes: Iterable[str] | None,
        completion_prerequisites_met: bool = True, completion_blocker: str = "",
        resolved_contradiction_ids: Iterable[str] = (),
    ) -> EvidencePipelineResult:
        """One projection/coverage/event path shared by admission and reuse."""
        if obligation is not None:
            proposal = proposal.model_copy(update={"covered_fact_slots": tuple(
                slot for slot in self._required_slots(obligation) if slot in proposal.covered_fact_slots
            )})
        fact = self.projector.project(
            stored,
            evidence_id=evidence_id,
            obligation_id=obligation.id if obligation is not None else "",
            hypothesis=hypothesis,
            proposal=proposal,
        )
        if fact is None:
            raise EvidencePipelineError("evidence_projection_rejected", stored.observation_id)
        # A fact may contribute coverage; only qualify() can close the complete set.
        fact = fact.model_copy(update={
            "fulfillment": "PARTIAL" if fact.covered_fact_slots and fact.relation != "CONTRADICTS"
            else "INSUFFICIENT",
        })
        if obligation is None:
            return EvidencePipelineResult(
                evidence_id=evidence_id,
                projected_fact=fact,
            )
        duplicate = any(evidence_relationship_key(item) == evidence_relationship_key(fact)
                        for item in existing_facts)
        facts = existing_facts if duplicate else existing_facts + (fact,)
        evidence_set = self.qualify(
            obligation=obligation,
            evidence=facts,
            hypothesis_id=self.projector.hypothesis_id(hypothesis),
            hypothesis_scopes=hypothesis_scopes,
            completion_prerequisites_met=completion_prerequisites_met,
            completion_blocker=completion_blocker,
            resolved_contradiction_ids=resolved_contradiction_ids,
        )
        return EvidencePipelineResult(
            evidence_id=evidence_id,
            projected_fact=fact,
            evidence_set=evidence_set,
            rca_events=() if duplicate else self.propose_rca_events(evidence_set, facts=facts, state=rca_state),
            duplicate=duplicate,
        )

    def qualify(
        self,
        *,
        obligation: VerificationObligation,
        evidence: Iterable[ProjectedEvidenceFact],
        hypothesis_id: str = "",
        hypothesis_scopes: Iterable[str] | None = None,
        completion_prerequisites_met: bool = True,
        completion_blocker: str = "",
        resolved_contradiction_ids: Iterable[str] = (),
    ) -> EvidenceSet:
        """Validate admitted relationships and compute exact required-slot coverage."""
        obligation_id = self._obligation_id(obligation)
        if not obligation_id:
            raise EvidencePipelineError("missing_obligation_id", "canonical ID is required")
        required_slots = self._required_slots(obligation)
        effective_scopes = (
            None
            if hypothesis_scopes is None
            else tuple(dict.fromkeys(str(item) for item in hypothesis_scopes if str(item)))
        )
        allowed_scopes = None if effective_scopes is None else set(effective_scopes)
        known_evidence = {item.evidence_id: item for item in self.evidence_memory.pinned}
        selected: list[ProjectedEvidenceFact] = []
        for fact in evidence:
            if fact.obligation_id != obligation_id:
                continue
            if allowed_scopes is not None and fact.hypothesis_fingerprint not in allowed_scopes:
                continue
            stored = known_evidence.get(fact.evidence_id)
            if stored is None:
                raise EvidencePipelineError("unknown_evidence_id", fact.evidence_id)
            if stored.raw_observation_id != fact.observation_id:
                raise EvidencePipelineError(
                    "invalid_provenance",
                    f"{fact.evidence_id} does not originate from {fact.observation_id}",
                )
            refs = set(fact.provenance.get("observation_refs", ()))
            if fact.observation_id not in refs:
                raise EvidencePipelineError(
                    "invalid_provenance",
                    f"{fact.evidence_id} omits its Observation reference",
                )
            unknown_slots = sorted(set(fact.covered_fact_slots) - set(required_slots))
            if unknown_slots:
                raise EvidencePipelineError("unknown_fact_slot", ", ".join(unknown_slots))
            if len(set(fact.covered_fact_slots)) != len(fact.covered_fact_slots):
                raise EvidencePipelineError(
                    "duplicate_fact_slot", f"{fact.evidence_id} repeats coverage slots",
                )
            observation = self.observation_store.get(stored.raw_observation_id)
            if observation is None:
                raise EvidencePipelineError("invalid_provenance", fact.evidence_id)
            self._validate_constraints(obligation, fact.covered_fact_slots, observation)
            selected.append(fact)

        evidence_ids = tuple(dict.fromkeys(item.evidence_id for item in selected))
        resolved = set(resolved_contradiction_ids)
        contradicting = tuple(dict.fromkeys(
            item.evidence_id
            for item in selected
            if item.relation == "CONTRADICTS" and item.evidence_id not in resolved
        ))
        coverage = {
            slot: tuple(dict.fromkeys(
                item.evidence_id
                for item in selected
                if slot in item.covered_fact_slots and item.relation != "CONTRADICTS"
            ))
            for slot in required_slots
        }
        supporting = tuple(dict.fromkeys(
            evidence_id for slot in required_slots for evidence_id in coverage[slot]
        ))
        partial = tuple(dict.fromkeys(
            item.evidence_id
            for item in selected
            if item.fulfillment == "PARTIAL"
        ))
        complete = all(coverage[slot] for slot in required_slots)
        if contradicting:
            status: QualificationStatus = "DOES_NOT_SATISFY"
            fulfillment: ObligationFulfillment = "INSUFFICIENT"
            semantic_status: EvidenceSetStatus = "CONTRADICTS"
            reason = "unresolved contradictory Evidence blocks Obligation closure"
        elif complete and completion_prerequisites_met:
            status = "SATISFIES"
            fulfillment = "SATISFIES"
            semantic_status = "SATISFIES"
            reason = "all required fact slots have admitted scoped Evidence"
        elif any(item.fulfillment == "PARTIAL" or item.covered_fact_slots for item in selected):
            status = "INSUFFICIENT"
            fulfillment = "PARTIAL"
            semantic_status = "PARTIAL"
            reason = completion_blocker or "required fact slot coverage is incomplete"
        elif selected:
            status = "DOES_NOT_SATISFY"
            fulfillment = "UNRELATED"
            semantic_status = "INSUFFICIENT"
            reason = "admitted Evidence does not cover a required fact slot"
        else:
            status = "INSUFFICIENT"
            fulfillment = "INSUFFICIENT"
            semantic_status = "INSUFFICIENT"
            reason = "EvidenceSet is empty for the active hypothesis scope"
        return EvidenceSet(
            obligation_id=obligation_id,
            hypothesis_id=hypothesis_id,
            evidence_ids=evidence_ids,
            supporting_evidence_ids=supporting,
            contradicting_evidence_ids=contradicting,
            partial_evidence_ids=partial,
            hypothesis_scope_fingerprints=effective_scopes or (),
            coverage=coverage,
            fulfillment=fulfillment,
            status=status,
            semantic_status=semantic_status,
            reason=reason,
        )

    @staticmethod
    def propose_rca_events(
        evidence_set: EvidenceSet,
        *,
        facts: Iterable[ProjectedEvidenceFact],
        state: RCAState | None,
    ) -> tuple[RCAEvent, ...]:
        """Translate one qualified set into reducer proposals without applying them."""
        known_ids = set(state.evidence_ids) if state is not None else set()
        known_contradictions = (
            {item.evidence_id for item in state.contradictions} if state is not None else set()
        )
        events: list[RCAEvent] = []
        facts_by_id = {item.evidence_id: item for item in facts}
        for evidence_id in evidence_set.evidence_ids:
            if evidence_id not in known_ids:
                events.append(RCAEvent(
                    kind="LINK_EVIDENCE", payload={"evidence_id": evidence_id},
                ))
                known_ids.add(evidence_id)
        for evidence_id in evidence_set.contradicting_evidence_ids:
            if evidence_id not in known_contradictions:
                fact = facts_by_id.get(evidence_id)
                events.append(RCAEvent(kind="CONTRADICTION", payload={
                    "evidence_id": evidence_id,
                    "claim": str(
                        getattr(fact, "fact", "")
                        or "Evidence contradicts the active Obligation"
                    ),
                }))
                known_contradictions.add(evidence_id)
        if evidence_set.status == "SATISFIES":
            current = None
            if state is not None:
                current = next(
                    (item for item in state.obligations if item.id == evidence_set.obligation_id),
                    None,
                )
            if state is None or (current is not None and current.status == "OPEN"):
                events.append(RCAEvent(kind="UPDATE_OBLIGATION", payload={
                    "obligation_id": evidence_set.obligation_id,
                    "status": "SATISFIED",
                    "supporting_evidence_ids": list(evidence_set.supporting_evidence_ids),
                }))
        return tuple(events)
