"""Canonical admission for semantic Obligation proposals.

Agents describe an atomic evidence need.  Runtime policy validates the shape,
capability path, identity and criticality, then emits one RCAState event.  This
module never mutates RCAState or an operational projection.
"""

from __future__ import annotations

from hashlib import sha1
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from .contracts import (
    CompletionContract,
    ReviewGapProposal,
    RootCauseCandidate,
    VerificationObligation,
    build_verification_key,
    candidate_claim_inventory,
)
from .rca_state import RCAEvent


AdmissionStatus = Literal["ADMITTED", "DUPLICATE", "REJECTED"]
AdmissionReason = Literal[
    "completion_contract_conflict",
    "completion_contract_provenance_unrealizable",
    "completion_contract_missing_or_invalid",
    "admitted",
    "duplicate_obligation",
    "non_atomic_obligation",
    "capability_unavailable",
    "unsupported_claim_not_reported",
    "claim_not_in_candidate",
    "claim_not_required_for_candidate",
    "no_executable_evidence_path",
]


class ObligationProposalError(ValueError):
    """A semantic proposal is not sufficient for Runtime admission."""

    def __init__(self, message: str, *, missing_fields: tuple[str, ...] = ()):
        self.missing_fields = tuple(missing_fields)
        super().__init__(message)


class CanonicalObligationProposal(BaseModel):
    """Normalized semantic inputs without identity or lifecycle fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim: str
    evidence_requirement: str
    target_component: str
    verification_question: str
    evidence_type: str
    completion_contract: CompletionContract
    related_hypothesis: str = ""
    verification_role: Literal["NECESSARY", "EXPLORATORY"] = "NECESSARY"


class ObligationAdmissionDecision(BaseModel):
    """Typed result of deterministic proposal admission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: AdmissionStatus
    reasons: tuple[AdmissionReason, ...]
    canonical_proposal: CanonicalObligationProposal | None = None
    obligation_id: str = ""
    event: RCAEvent | None = None
    blocking_path_available: bool = False

    @property
    def admitted(self) -> bool:
        return self.status == "ADMITTED" and self.event is not None

    @property
    def actionable(self) -> bool:
        return self.admitted or (
            self.status == "DUPLICATE" and self.blocking_path_available
        )

    def accepted_event(self) -> RCAEvent:
        """Return the accepted admission result, never a caller-built ADD event.

        Runtime consumes this typed decision before reducer submission. Existing
        Obligation updates do not need, and cannot use, a new-admission result.
        """
        if not self.admitted or self.canonical_proposal is None:
            raise ObligationProposalError("an accepted admission decision is required")
        obligation = VerificationObligation.model_validate(self.event.payload)
        canonical = self.canonical_proposal
        fields = ("claim", "evidence_requirement", "target_component",
                  "verification_question", "evidence_type", "related_hypothesis",
                  "completion_contract", "verification_role")
        consistent = (
            self.event.kind == "ADD_OBLIGATION"
            and obligation.id == self.obligation_id
            and obligation.status == "OPEN"
            and obligation.supporting_evidence_ids == ()
            and obligation.blocked_capabilities == ()
            and obligation.blocks_finalization == self.blocking_path_available
            and all(getattr(obligation, name) == getattr(canonical, name) for name in fields)
            and obligation.verification_key == build_verification_key(
                target_component=canonical.target_component,
                verification_question=canonical.verification_question,
                evidence_type=canonical.evidence_type,
            )
        )
        if not consistent:
            raise ObligationProposalError("admission event must match its canonical proposal")
        return self.event.model_copy(deep=True)


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def obligation_id_for_key(verification_key: str) -> str:
    normalized = _clean(verification_key)
    if not normalized:
        raise ValueError("obligation identity must not be empty")
    return "obl-" + sha1(normalized.encode("utf-8")).hexdigest()[:10]


def unrealizable_provenance_requirements(contract: CompletionContract, profiles) -> tuple[str, ...]:
    """Each fact needs one possible source tuple; this never qualifies Evidence."""
    from debug_assistant.tools.provenance import profile_matches_constraints
    return tuple(fact.slot for fact in contract.required_facts if not any(
        profile_matches_constraints(profile, fact.evidence_constraints.model_dump())
        for profile in profiles))


class ObligationAdmissionPolicy:
    """One canonicalizer and admission interface for all proposal sources."""

    @staticmethod
    def canonicalize(
        proposal: Any,
        *,
        related_hypothesis: str,
    ) -> CanonicalObligationProposal:
        values = {
            "claim": _clean(getattr(proposal, "claim", "")),
            "target_component": _clean(getattr(proposal, "target_component", "")),
            "verification_question": _clean(
                getattr(proposal, "verification_question", "")
            ),
            "evidence_type": _clean(getattr(proposal, "evidence_type", "")),
            "evidence_requirement": _clean(
                getattr(proposal, "evidence_requirement", "")
            ),
            "related_hypothesis": _clean(related_hypothesis),
            "verification_role": str(
                getattr(proposal, "verification_role", "NECESSARY") or "NECESSARY"
            ).strip().upper(),
        }
        required = (
            "claim",
            "target_component",
            "verification_question",
            "evidence_type",
            "evidence_requirement",
            "verification_role",
        )
        missing = tuple(name for name in required if not values[name])
        if missing:
            raise ObligationProposalError(
                "obligation proposal is not atomic",
                missing_fields=missing,
            )
        try:
            values["completion_contract"] = CompletionContract.model_validate(
                getattr(proposal, "completion_contract", None)
            )
        except ValidationError as exc:
            raise ObligationProposalError("completion contract missing or invalid",
                                          missing_fields=("completion_contract",)) from exc
        return CanonicalObligationProposal.model_validate(values)

    @classmethod
    def admit(
        cls,
        proposal: Any,
        *,
        related_hypothesis: str,
        existing_obligations: tuple[VerificationObligation, ...] = (),
        critical: bool = True,
        required_tools: tuple[str, ...] = (),
        available_tools: tuple[str, ...] = (),
        capability_gaps: tuple[str, ...] = (),
        identity_disambiguator: str = "",
        provenance_profiles: tuple[dict, ...] | None = None,
    ) -> ObligationAdmissionDecision:
        try:
            canonical = cls.canonicalize(
                proposal,
                related_hypothesis=related_hypothesis,
            )
        except ObligationProposalError as exc:
            return ObligationAdmissionDecision(
                status="REJECTED",
                reasons=(("completion_contract_missing_or_invalid"
                          if "completion_contract" in exc.missing_fields else "non_atomic_obligation"),),
            )

        verification_key = build_verification_key(
            target_component=canonical.target_component,
            verification_question=canonical.verification_question,
            evidence_type=canonical.evidence_type,
            related_hypothesis=canonical.related_hypothesis,
        )
        duplicates = tuple(item for item in existing_obligations
                           if item.canonical_verification_key == verification_key)
        if duplicates:
            conflicting = next((item for item in duplicates
                                if (
                                    not canonical.completion_contract.equivalent(item.completion_contract)
                                    or canonical.verification_role != (
                                        "NECESSARY" if item.critical else "EXPLORATORY"
                                    )
                                )), None)
            if conflicting is not None:
                return ObligationAdmissionDecision(
                    status="REJECTED", reasons=("completion_contract_conflict",),
                    canonical_proposal=canonical, obligation_id=conflicting.id,
                )
        # Contract conflicts take precedence over operational capability failures.
        available = set(available_tools)
        executable_path = not required_tools or bool(available.intersection(required_tools))
        if capability_gaps or not executable_path:
            return ObligationAdmissionDecision(
                status="REJECTED",
                reasons=(
                    "capability_unavailable" if capability_gaps
                    else "no_executable_evidence_path",
                ),
                canonical_proposal=canonical,
            )

        if provenance_profiles is not None and unrealizable_provenance_requirements(
                canonical.completion_contract, provenance_profiles):
            return ObligationAdmissionDecision(status="REJECTED",
                reasons=("completion_contract_provenance_unrealizable",), canonical_proposal=canonical)

        if duplicates:
            duplicate = duplicates[0]
            return ObligationAdmissionDecision(
                status="DUPLICATE",
                reasons=("duplicate_obligation",),
                canonical_proposal=canonical,
                obligation_id=duplicate.id,
                blocking_path_available=duplicate.blocks_finalization,
            )

        effective_critical = bool(
            critical and canonical.verification_role == "NECESSARY"
        )
        identity = verification_key
        if identity_disambiguator:
            identity = f"{identity}|{_clean(identity_disambiguator)}"
        obligation = VerificationObligation(
            id=obligation_id_for_key(identity),
            claim=canonical.claim,
            evidence_requirement=canonical.evidence_requirement,
            target_component=canonical.target_component,
            verification_question=canonical.verification_question,
            evidence_type=canonical.evidence_type,
            related_hypothesis=canonical.related_hypothesis,
            completion_contract=canonical.completion_contract,
            verification_key=verification_key,
            critical=effective_critical,
            verification_role=canonical.verification_role,
            status="OPEN",
            supporting_evidence_ids=(),
        )
        event = RCAEvent(kind="ADD_OBLIGATION", payload=obligation.model_dump())
        return ObligationAdmissionDecision(
            status="ADMITTED",
            reasons=("admitted",),
            canonical_proposal=canonical,
            obligation_id=obligation.id,
            event=event,
            blocking_path_available=obligation.blocks_finalization,
        )

    @classmethod
    def admit_review_gap(
        cls,
        proposal: ReviewGapProposal,
        *,
        candidate: RootCauseCandidate,
        reported_unsupported_claims: tuple[str, ...],
        related_hypothesis: str,
        existing_obligations: tuple[VerificationObligation, ...],
        available_tools: tuple[str, ...],
        provenance_profiles: tuple[dict, ...] | None = None,
    ) -> ObligationAdmissionDecision:
        unsupported = _clean(proposal.unsupported_claim).casefold()
        reported = {_clean(item).casefold() for item in reported_unsupported_claims}
        if unsupported not in reported:
            return ObligationAdmissionDecision(
                status="REJECTED",
                reasons=("unsupported_claim_not_reported",),
            )

        inventory = candidate_claim_inventory(candidate)
        if proposal.claim_kind not in inventory:
            return ObligationAdmissionDecision(
                status="REJECTED",
                reasons=("claim_not_required_for_candidate",),
            )
        required_claims = {_clean(item).casefold() for item in inventory[proposal.claim_kind]}
        if unsupported not in required_claims:
            return ObligationAdmissionDecision(
                status="REJECTED",
                reasons=("claim_not_in_candidate",),
            )

        evidence_tools = tuple(
            tool for tool in available_tools
            if tool not in {"finalize_diagnosis", "knowledge_retrieval"}
        )
        return cls.admit(
            proposal,
            related_hypothesis=related_hypothesis,
            existing_obligations=existing_obligations,
            critical=True,
            required_tools=proposal.suggested_tools,
            available_tools=evidence_tools,
            provenance_profiles=provenance_profiles,
        )
