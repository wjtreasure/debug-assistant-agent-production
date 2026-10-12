"""Deterministic projection of explicit, provenance-bound Evidence proposals."""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


EvidenceRelation = Literal["SUPPORTS", "CONTRADICTS", "NEUTRAL"]
ObligationFulfillment = Literal[
    "SATISFIES", "PARTIAL", "INSUFFICIENT", "UNRELATED", "CAPABILITY_BLOCKED"
]


class EvidenceProposalError(ValueError):
    """A semantic Evidence proposal failed its explicit contract."""


class _EvidenceFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, json_schema_extra={"oneOf": [
        {"required": ["observation_refs"], "properties": {
            "observation_refs": {"type": "array", "minItems": 1, "maxItems": 1},
            "evidence_ref": {"type": "null"},
        }},
        {"required": ["evidence_ref"], "properties": {
            "evidence_ref": {"type": "string", "pattern": "^ev-"},
            "observation_refs": {"type": "array", "maxItems": 0},
            "source_spans": {"type": "array", "maxItems": 0},
        }},
    ]})

    semantic_claim: str = Field(min_length=1, max_length=1000)
    hypothesis_relation: EvidenceRelation = Field(default="NEUTRAL", description="Fact relation to the explicitly selected hypothesis_target scope.")
    relation: EvidenceRelation = Field(default="NEUTRAL", description="Fact relation to the target Obligation, independently of hypothesis_relation.")
    observation_refs: tuple[str, ...] = ()
    evidence_ref: str | None = Field(default=None, pattern=r"^ev-")
    source_spans: tuple[str, ...] = ()
    reason: str = Field(min_length=1, max_length=1000)
    rationale: str = Field(default="", max_length=1000)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    obligation_id: str = ""

    @model_validator(mode="after")
    def validate_source(self):
        if self.evidence_ref is not None:
            if self.observation_refs or self.source_spans:
                raise ValueError("Evidence reuse references one Evidence; provenance stays canonical")
        elif len(self.observation_refs) != 1:
            raise ValueError("new Evidence requires exactly one visible Observation")
        return self


class EvidenceFactProposal(_EvidenceFact):
    """Provider selects references; only Runtime derives canonical slot names."""

    hypothesis_target: Literal["CURRENT", "PROPOSED"]
    obligation_id: str = Field(min_length=1, description="Required exact canonical target for every coverage_action, including NO_RELEVANT_COVERAGE. The fact does not cover a requirement when requirement_refs is empty.")
    coverage_action: Literal["COVER_EXISTING_REQUIREMENT", "NO_RELEVANT_COVERAGE"]
    requirement_refs: tuple[str, ...] = ()


class EvidenceProposal(_EvidenceFact):
    """Semantic fact enriched with Runtime-resolved canonical revision scope."""

    hypothesis_target: Literal["CURRENT", "PROPOSED"] = "CURRENT"
    covered_fact_slots: tuple[str, ...] = ()
    method: Literal["deterministic", "llm"] = "llm"
    hypothesis_id: str = ""
    incident_id: str = ""

class ProjectedEvidenceFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(pattern=r"^ev-")
    observation_id: str = Field(min_length=1)
    fact: str = Field(min_length=1)
    # ``relation`` is the relation to the VerificationObligation.  Keep the
    # hypothesis relation separate: one observation may satisfy a question
    # while disproving the exploratory hypothesis that motivated it.
    relation: EvidenceRelation = "NEUTRAL"
    hypothesis_relation: EvidenceRelation = "NEUTRAL"
    fulfillment: ObligationFulfillment = "INSUFFICIENT"
    covered_fact_slots: tuple[str, ...] = ()
    # Relationship-scoped semantic interpretation.  These fields are
    # provider meaning preserved after deterministic Runtime admission.
    reason: str = ""
    rationale: str = ""
    source_spans: tuple[str, ...] = ()
    obligation_id: str = ""
    hypothesis_fingerprint: str = ""
    hypothesis_id: str = ""
    provenance: dict[str, Any] = Field(default_factory=dict)


def evidence_relationship_key(
    fact: ProjectedEvidenceFact,
) -> tuple[Any, ...]:
    """Return the immutable identity of one Evidence semantic relationship.

    The Evidence node is immutable and shared.  Its relationship to an
    Obligation is independently versioned by hypothesis scope and semantic
    outcome, so a later branch or contradiction cannot replace an earlier
    relationship record.
    """
    return (
        fact.evidence_id,
        fact.obligation_id,
        fact.hypothesis_fingerprint,
        fact.relation,
        fact.hypothesis_relation,
        fact.fulfillment,
        fact.covered_fact_slots,
        fact.hypothesis_id,
    )


class EvidenceProjector:
    _NON_INCIDENT_SOURCES = {
        "prior", "candidate_retrieval", "knowledge_candidate", "incident_memory",
        "domain_rag", "static_kg",
    }


    @staticmethod
    def hypothesis_id(hypothesis: Any | None) -> str:
        if hypothesis is None:
            return ""
        explicit = str(
            getattr(hypothesis, "hypothesis_id", None)
            or getattr(hypothesis, "id", None)
            or ""
        )
        if explicit:
            return explicit
        return ""

    @staticmethod
    def hypothesis_fingerprint(hypothesis: Any | None) -> str:
        identity = EvidenceProjector.hypothesis_id(hypothesis)
        revision = getattr(hypothesis, "revision", 0) if hypothesis else 0
        return f"{identity}:r{revision}" if identity and revision else ""

    @staticmethod
    def hypothesis_scope_fingerprints(hypothesis: Any | None) -> tuple[str, ...]:
        scope = EvidenceProjector.hypothesis_fingerprint(hypothesis)
        return (scope,) if scope else ()

    @staticmethod
    def _fields(observation: Any) -> tuple[str, dict[str, Any], str, bool]:
        if isinstance(observation, dict):
            return (
                str(observation.get("observation_id", "")),
                dict(observation.get("metadata", {}) or {}),
                str(observation.get("content", "")),
                bool(observation.get("ok", False)),
            )
        return (
            str(getattr(observation, "observation_id", "") or ""),
            dict(getattr(observation, "metadata", None) or {}),
            str(getattr(observation, "content", "") or ""),
            bool(getattr(observation, "ok", False)),
        )

    @classmethod
    def incident_observation_available(cls, observation: Any) -> bool:
        observation_id, metadata, content, ok = cls._fields(observation)
        return bool(observation_id and ok and content.strip()
                    and metadata.get("semantic_negative") is not True
                    and metadata.get("result_kind") != "OPERATIONAL_ONLY"
                    and metadata.get("information_source") not in cls._NON_INCIDENT_SOURCES)

    def propose_deterministic(
        self,
        observation: Any,
        *,
        semantic_claim: str,
        obligation: Any | None = None,
        hypothesis: Any | None = None,
        relation: EvidenceRelation = "NEUTRAL",
        reason: str = "runtime deterministic verification",
        source_spans: tuple[str, ...] = (),
        covered_fact_slots: tuple[str, ...] = (),
        incident_id: str = "",
    ) -> EvidenceProposal:
        """Build a provenance-bound proposal for a computable fact.

        This producer describes a computed fact: it does
        not interpret source semantics or infer a root cause.  Its result still
        uses the normal EvidenceProposal contract and must pass the same
        Runtime admission and EvidenceSet path as an LLM proposal.
        """
        observation_id, _, content, ok = self._fields(observation)
        if not observation_id or not ok or not content.strip():
            raise EvidenceProposalError(
                "deterministic EvidenceProposal requires a successful Observation"
            )
        claim = " ".join(str(semantic_claim or "").split()).strip()
        if not claim:
            raise EvidenceProposalError(
                "deterministic EvidenceProposal requires a semantic_claim"
            )
        obligation_id = str(
            getattr(obligation, "id", None)
            or getattr(obligation, "obligation_id", "")
            or ""
        )
        return EvidenceProposal(
            semantic_claim=claim[:1000],
            hypothesis_relation=(
                relation if relation in {"SUPPORTS", "CONTRADICTS"} else "NEUTRAL"
            ),
            relation=relation,
            observation_refs=(observation_id,),
            source_spans=tuple(str(item) for item in source_spans),
            covered_fact_slots=tuple(str(item) for item in covered_fact_slots),
            reason=reason,
            rationale=reason,
            method="deterministic",
            obligation_id=obligation_id,
            hypothesis_id=self.hypothesis_id(hypothesis),
            incident_id=str(incident_id or ""),
        )

    def validate(
        self,
        proposal: EvidenceProposal,
        *,
        observation_ids: set[str],
        incident_id: str | None = None,
        obligation_id: str | None = None,
        hypothesis_id: str | None = None,
    ) -> EvidenceProposal:
        missing = sorted(set(proposal.observation_refs) - set(observation_ids))
        if missing:
            raise ValueError(f"EvidenceProposal references unknown Observations: {missing}")
        if not proposal.semantic_claim.strip() or not proposal.reason.strip():
            raise ValueError("EvidenceProposal requires grounded claim and reason")
        if incident_id and proposal.incident_id != incident_id:
            raise ValueError("EvidenceProposal belongs to a different incident")
        if obligation_id and proposal.obligation_id != obligation_id:
            raise ValueError("EvidenceProposal targets a different obligation")
        if hypothesis_id and proposal.hypothesis_id != hypothesis_id:
            raise ValueError("EvidenceProposal targets a different hypothesis")
        return proposal

    def project(
        self,
        observation: Any,
        *,
        evidence_id: str | None = None,
        obligation_id: str = "",
        hypothesis: Any | None = None,
        proposal: EvidenceProposal | None = None,
    ) -> ProjectedEvidenceFact | None:
        observation_id, metadata, content, ok = self._fields(observation)
        if not self.incident_observation_available(observation):
            return None
        if proposal is None:
            raise EvidenceProposalError("projection requires an explicit EvidenceProposal")
        if evidence_id is None:
            evidence_id = "ev-" + hashlib.sha1(observation_id.encode()).hexdigest()[:12]
        fingerprint = self.hypothesis_fingerprint(hypothesis)
        provenance = {
            "tool": str(getattr(observation, "tool", "") or metadata.get("tool", "")),
            "context_kind": metadata.get("context_kind", ""),
            "source": metadata.get("information_source", "incident_observation"),
            "observation_refs": list(proposal.observation_refs),
            "source_spans": list(proposal.source_spans),
            "projection_method": proposal.method,
            # Preserve the semantic scope at admission time.  A diagnosis may
            # acquire its public fault taxonomy only after the first causal
            # Observation; finalization may audit that narrowing, but must not
            # infer it from the Evidence text alone.
            "hypothesis_fault_code": str(
                getattr(hypothesis, "fault_code", "")
                or getattr(hypothesis, "candidate_fault_code", "")
                or ""
            ),
            "hypothesis_fault": str(
                getattr(hypothesis, "fault", "")
                or getattr(hypothesis, "candidate_fault", "")
                or ""
            ),
            "hypothesis_component": str(
                getattr(hypothesis, "component", "") or ""
            ),
        }
        return ProjectedEvidenceFact(
            evidence_id=evidence_id,
            observation_id=observation_id,
            fact=proposal.semantic_claim,
            relation=proposal.relation,
            hypothesis_relation=proposal.hypothesis_relation,
            covered_fact_slots=proposal.covered_fact_slots,
            reason=proposal.reason,
            rationale=proposal.rationale,
            source_spans=tuple(proposal.source_spans),
            obligation_id=obligation_id,
            hypothesis_fingerprint=fingerprint,
            hypothesis_id=self.hypothesis_id(hypothesis),
            provenance=provenance,
        )
