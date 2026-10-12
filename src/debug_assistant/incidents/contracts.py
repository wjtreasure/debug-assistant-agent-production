from __future__ import annotations

import re
from typing import Annotated, Any, Literal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator


# Evidence IDs are a runtime-owned namespace.  The pattern is deliberately
# narrower than "known Evidence": the Harness still checks that an ev-* ID is
# present in the current EvidenceMemory.  This type only prevents model output
# from smuggling observation IDs or arbitrary strings into semantic contracts.
EvidenceId = Annotated[str, StringConstraints(pattern=r"^ev-")]


_FAULT_CODE_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")
from .fault_taxonomy import is_canonical_fault_code, ProposedFaultCode
from debug_assistant.symbols import CanonicalOutputModel


def _normalize_verification_key_part(value: Any) -> str:
    """Normalize a structured obligation key without inferring semantics.

    The Runtime may canonicalize provider-supplied identity fields, but it
    must not invent a semantic target or question from a case label.  The
    resulting key is a tuple-like string, not a hash of ``claim`` alone.
    """
    return " ".join(str(value or "").casefold().split())


def build_verification_key(
    *,
    target_component: str = "",
    verification_question: str = "",
    evidence_type: str = "",
    related_hypothesis: str = "",
) -> str:
    """Build the canonical identity for one semantic verification question.

    ``related_hypothesis`` remains an accepted compatibility argument for
    older callers and trace payloads, but natural-language hypothesis prose is
    deliberately excluded from identity.  A wording change must not create a
    second Obligation for the same target/question/evidence type.
    """
    parts = (
        _normalize_verification_key_part(target_component),
        _normalize_verification_key_part(verification_question),
        _normalize_verification_key_part(evidence_type),
    )
    if not any(parts):
        return ""
    return "|".join(parts)


def _normalize_fault_fields(value: Any) -> Any:
    """Normalize the provider's structured fault projection."""
    if not isinstance(value, dict):
        return value
    result = dict(value)
    display_fault = str(result.get("fault") or "").strip()
    code = str(result.get("fault_code") or "").strip()
    explanation = str(result.get("fault_explanation") or "").strip()
    # ``fault`` is a display projection and may contain a DomainSkill ID or
    # another free-form label. Promote it only when it is already in the
    # published taxonomy; otherwise a Planner label becomes a false
    # ``invalid_fault_code`` finalization blocker.
    if display_fault and not code and is_canonical_fault_code(display_fault):
        result["fault_code"] = display_fault
        code = display_fault
    if display_fault and not explanation:
        result["fault_explanation"] = display_fault
        explanation = display_fault
    if (code or explanation) and not display_fault:
        # Keep the concise display field populated; structured fields remain
        # authoritative for evaluation.
        result["fault"] = code or explanation
    return result


ObligationStatus = Literal[
    "OPEN", "SATISFIED", "BLOCKED_BY_CAPABILITY",
    "NOT_APPLICABLE_BY_CAPABILITY", "SUPERSEDED_BY_REPLAN",
]
EvidenceSemanticRelation = Literal["SUPPORTS", "CONTRADICTS", "NEUTRAL"]
ObligationFulfillment = Literal[
    "SATISFIES", "PARTIAL", "INSUFFICIENT", "UNRELATED", "CAPABILITY_BLOCKED"
]
ContradictionSeverity = Literal["WEAK", "BLOCKING"]
ContradictionStatus = Literal["OPEN", "RESOLVED", "EXPLAINED"]
SourceMechanismStatus = Literal[
    "unknown", "gap", "sufficient", "not_applicable", "blocked",
]


class EvidenceConstraint(BaseModel):
    """AND between categories, OR between values; empty categories are unrestricted."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    context_kinds: tuple[str, ...] = ()
    information_sources: tuple[str, ...] = ()
    source_tools: tuple[str, ...] = ()

    @field_validator("context_kinds", "information_sources", "source_tools", mode="before")
    @classmethod
    def normalize_values(cls, value: Any) -> tuple[str, ...]:
        if not isinstance(value, (tuple, list)) or any(not isinstance(item, str) for item in value):
            raise ValueError("constraint values must be a sequence of strings")
        values = tuple(" ".join(item.split()) for item in value)
        if any(not item for item in values):
            raise ValueError("constraint values must be non-empty")
        return tuple(sorted(set(values)))


FactRole = Literal["OBSERVATION", "LOCALIZATION", "MECHANISM", "CAUSAL_LINK", "EXCLUSION"]


class FactRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    slot: str = Field(min_length=1)
    description: str = Field(min_length=1)
    role: FactRole
    evidence_constraints: EvidenceConstraint = Field(default_factory=EvidenceConstraint)

    @field_validator("slot", "description")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("requirement text must be non-empty")
        return value


class CompletionContract(BaseModel):
    """The sole source of an Obligation's completion requirements."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    required_facts: tuple[FactRequirement, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_slots(self) -> "CompletionContract":
        slots = [fact.slot for fact in self.required_facts]
        if len(slots) != len(set(slots)):
            raise ValueError("completion requirement slots must be unique")
        return self

    @property
    def required_fact_slots(self) -> tuple[str, ...]:
        return tuple(fact.slot for fact in self.required_facts)

    def equivalent(self, other: "CompletionContract") -> bool:
        return {fact.slot: fact for fact in self.required_facts} == {
            fact.slot: fact for fact in other.required_facts
        }


class VerificationObligation(BaseModel):
    """Small, incident-scoped verification contract.

    This is intentionally not the generic SWE obligation state machine.  The
    Harness validates IDs and legal status values.  Source-mechanism
    obligations are Runtime-owned: ``NOT_APPLICABLE_BY_CAPABILITY`` is only
    assigned by deterministic capability checks, never by Planner/Reflection.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    evidence_requirement: str = ""
    # Structured semantic identity. These fields are Runtime-owned after
    # admission of a Planner proposal and separate from the display claim.
    target_component: str = ""
    verification_question: str = ""
    evidence_type: str = ""
    related_hypothesis: str = ""
    verification_key: str = ""
    completion_contract: CompletionContract
    critical: bool = True
    verification_role: Literal["NECESSARY", "EXPLORATORY"] = "NECESSARY"
    status: ObligationStatus = "OPEN"
    supporting_evidence_ids: tuple[EvidenceId, ...] = ()
    blocked_capabilities: tuple[str, ...] = ()

    @property
    def required_fact_slots(self) -> tuple[str, ...]:
        return self.completion_contract.required_fact_slots

    @property
    def blocks_finalization(self) -> bool:
        return self.critical and self.verification_role == "NECESSARY" and self.status in {"OPEN", "BLOCKED_BY_CAPABILITY"}

    @property
    def canonical_verification_key(self) -> str:
        if self.verification_key:
            return self.verification_key
        # A canonical obligation carries a structured identity. A bare claim
        # is deliberately not promoted to a key: that would merge unrelated
        # verification questions.
        if not any((self.target_component, self.verification_question,
                    self.evidence_type, self.related_hypothesis)):
            return ""
        return build_verification_key(
            target_component=self.target_component,
            verification_question=self.verification_question or self.claim,
            evidence_type=self.evidence_type or self.evidence_requirement,
            related_hypothesis=self.related_hypothesis,
        )


class Contradiction(BaseModel):
    """Structured contradiction metadata; classification remains model-owned."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_id: EvidenceId
    claim: str = Field(min_length=1)
    severity: ContradictionSeverity = "BLOCKING"
    status: ContradictionStatus = "OPEN"

    @property
    def blocks_finalization(self) -> bool:
        return self.severity == "BLOCKING" and self.status == "OPEN"


class ClaimEvidenceMapping(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    claim: str = Field(min_length=1)
    evidence_ids: tuple[EvidenceId, ...] = Field(min_length=1)


class SourceClaim(BaseModel):
    """Planner-declared source claim whose coverage is Runtime-checkable.

    Runtime validates only that the declared file/range is fully covered by
    cited read_file CODE Evidence.  It deliberately does not decide whether
    the claim is true or matches evaluator Gold.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    file: str = Field(min_length=1)
    symbol: str = ""
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    claim: str = Field(min_length=1)
    evidence_ids: tuple[EvidenceId, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_range(self) -> "SourceClaim":
        if self.end_line < self.start_line:
            raise ValueError("source claim end_line must be >= start_line")
        if self.end_line - self.start_line + 1 > 200:
            raise ValueError("source claim range must not exceed 200 lines")
        return self


class CausalChainLink(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    cause: str = Field(min_length=1)
    effect: str = Field(min_length=1)
    evidence_ids: tuple[EvidenceId, ...] = Field(min_length=1)


class ReflectionObligationReview(CanonicalOutputModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    obligation_id: str = Field(min_length=1)
    status: ObligationStatus
    reason: str = Field(min_length=1)
    supporting_evidence_ids: tuple[EvidenceId, ...] = ()


class ReflectionContradictionReview(CanonicalOutputModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_id: EvidenceId
    status: ContradictionStatus
    severity: ContradictionSeverity = "BLOCKING"
    claim: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ObligationDisposition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    obligation_id: str = Field(min_length=1)
    action: Literal["RETAIN", "SUPERSEDE"]
    reason: str = Field(min_length=1)

    @field_validator("obligation_id", "reason")
    @classmethod
    def nonempty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("disposition requires a nonempty target and reason")
        return normalized


class HypothesisTransition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["INITIAL", "KEEP_CURRENT", "REFINE", "REPLACE"]
    obligation_dispositions: tuple[ObligationDisposition, ...] = ()


class SemanticEvidenceRecord(BaseModel):
    """One committed, relationship-scoped interpretation of admitted Evidence.

    The model supplies the meaning; Runtime supplies the canonical IDs and
    provenance links.  Keeping this record in ``RCAState`` makes the meaning
    recoverable on the next turn instead of leaving it only in the trace.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: EvidenceId
    observation_id: str = Field(min_length=1)
    semantic_claim: str = Field(min_length=1)
    hypothesis_relation: EvidenceSemanticRelation
    obligation_relation: EvidenceSemanticRelation
    covered_fact_slots: tuple[str, ...] = ()
    reason: str = Field(min_length=1)
    rationale: str = ""
    source_spans: tuple[str, ...] = ()


class SemanticDecisionRecord(BaseModel):
    """Immutable semantic journal entry projected across diagnostic turns."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str = Field(min_length=1)
    state_version: int = Field(ge=0)
    transition_kind: Literal["INITIAL", "KEEP_CURRENT", "REFINE", "REPLACE"]
    hypothesis_id: str = ""
    hypothesis_revision: int = Field(default=0, ge=0)
    hypothesis_claim: str = ""
    change_reason: str = Field(min_length=1)
    evidence: tuple[SemanticEvidenceRecord, ...] = ()
    changed_obligation_ids: tuple[str, ...] = ()


class ReflectionHypothesisDelta(BaseModel):
    """Provider proposal for a semantic hypothesis change.

    The proposal contains no runtime-owned version or identifier.  The
    Shared SemanticTransition validates its explicit lifecycle and allocates
    canonical branch identity/revision before staged publication.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    transition: HypothesisTransition | None = None
    claim: str = ""
    component: str = ""
    fault: str = ""
    fault_code: ProposedFaultCode = ""
    fault_explanation: str = ""
    mechanism: str = ""
    source_mechanism_status: SourceMechanismStatus | None = None
    supporting_evidence_ids: tuple[EvidenceId, ...] = ()
    contradicting_evidence_ids: tuple[EvidenceId, ...] = ()


    @model_validator(mode="after")
    def validate_keep_current(self):
        if self.transition is not None and self.transition.kind == "KEEP_CURRENT":
            if any(getattr(self, key) for key in ("claim", "component", "fault", "fault_code",
                    "fault_explanation", "mechanism", "source_mechanism_status")):
                raise ValueError("KEEP_CURRENT forbids causal field updates")
        return self


class ReflectionEvidenceGapProposal(BaseModel):
    """A gap proposal without a model-controlled gap ID."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    claim: str = Field(min_length=1)


class ReflectionObligationProposal(BaseModel):
    """A new obligation proposal; ID and lifecycle status belong to Runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    completion_contract: CompletionContract
    claim: str = Field(min_length=1)
    evidence_requirement: str = Field(min_length=1)
    target_component: str = Field(min_length=1)
    verification_question: str = Field(min_length=1)
    evidence_type: str = Field(min_length=1)
    suggested_tools: tuple[str, ...] = Field(min_length=1)
    supporting_evidence_ids: tuple[EvidenceId, ...] = ()
    verification_role: Literal["NECESSARY", "EXPLORATORY"] = "NECESSARY"


class ReflectionContradictionUpdate(CanonicalOutputModel):
    """A bounded update to a contradiction keyed by a canonical Evidence ID."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_id: EvidenceId
    claim: str = Field(min_length=1)
    severity: ContradictionSeverity = "BLOCKING"
    status: ContradictionStatus = "OPEN"
    reason: str = Field(min_length=1)


class ReflectionFeedback(CanonicalOutputModel):
    """Structured output of the incident Reflection Agent.

    It is feedback, not a replacement hypothesis.  Runtime state is changed
    only after the Harness validates the referenced canonical Evidence IDs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    supported_claims: tuple[str, ...] = ()
    unsupported_claims: tuple[str, ...] = ()
    remaining_gaps: tuple[str, ...] = ()
    supporting_evidence_ids: tuple[EvidenceId, ...] = ()
    contradicting_evidence_ids: tuple[EvidenceId, ...] = ()
    obligation_reviews: tuple[ReflectionObligationReview, ...] = ()
    contradiction_reviews: tuple[ReflectionContradictionReview, ...] = ()
    # Structured Delta is additive to the provider feedback fields. Runtime
    # accepts only a real canonical change and owns IDs/status/versioning.
    hypothesis_delta: ReflectionHypothesisDelta = Field(default_factory=ReflectionHypothesisDelta)
    proposed_evidence_gaps: tuple[ReflectionEvidenceGapProposal, ...] = ()
    proposed_obligations: tuple[ReflectionObligationProposal, ...] = ()
    contradiction_updates: tuple[ReflectionContradictionUpdate, ...] = ()
    next_action_constraint: Literal[
        "", "targeted_verification", "no_broad_exploration", "finalize", "inconclusive",
    ] = ""
    hypothesis_stable: bool = False
    highest_information_gain_direction: str = ""
    skill_reselection: str = ""
    reason: str = Field(min_length=1)


class IncidentCase(BaseModel):
    """Runtime-visible incident input. Ground truth has no field in this model."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str
    summary: str
    system: str
    namespace: str
    evidence_sources: tuple[str, ...]
    runtime_data_dir: str


class IncidentEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_id: str
    source: str
    target: str
    summary: str
    observation_id: str
    excerpt: str = ""
    file: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    raw_observation_id: str | None = None
    truncation: bool = False
    tags: tuple[str, ...] = ()
    provenance: dict[str, Any] = Field(default_factory=dict)


class IncidentHypothesis(BaseModel):
    hypothesis_id: str = ""
    revision: int = Field(default=0, ge=0)
    model_config = ConfigDict(extra="forbid", frozen=True)
    claim: str
    evidence_gap: str
    status: Literal["open", "supported", "rejected"] = "open"
    component: str = ""
    fault: str = ""
    fault_code: str = ""
    fault_explanation: str = ""
    mechanism: str = ""
    supporting_evidence_ids: tuple[str, ...] = ()
    contradicting_evidence_ids: tuple[str, ...] = ()
    required_gaps: tuple[str, ...] = ()
    evidence_sufficient: bool = False
    mechanism_category: str = ""
    # Semantic status emitted by Planner/Reflection.  The Harness only applies
    # deterministic source-capability checks to this status; it does not infer
    # whether a runtime symptom is caused by application code.
    source_mechanism_status: SourceMechanismStatus = "unknown"
    source_claims: tuple[SourceClaim, ...] = ()
    verification_obligations: tuple[VerificationObligation, ...] = ()
    contradictions: tuple[Contradiction, ...] = ()
    stable_rounds: int = Field(default=0, ge=0)

    @model_validator(mode="before")
    @classmethod
    def migrate_fault_contract(cls, value: Any) -> Any:
        return _normalize_fault_fields(value)

    def required_gap_projection(self) -> tuple[str, ...]:
        """Derive the display gap view from canonical obligations."""
        if self.verification_obligations:
            return tuple(
                obligation.claim
                for obligation in self.verification_obligations
                if obligation.blocks_finalization
            )
        return self.required_gaps

    def contradicting_evidence_projection(self) -> tuple[str, ...]:
        """Derive the display contradiction view from structured contradictions."""
        if self.contradictions:
            return tuple(
                item.evidence_id
                for item in self.contradictions
                if item.blocks_finalization
            )
        return self.contradicting_evidence_ids

class RootCauseCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    component: str = Field(min_length=1)
    # ``fault`` is the concise display field; structured code/explanation are
    # the evaluation fields.
    fault: str = Field(min_length=1)
    fault_code: str = ""
    fault_explanation: str = ""
    mechanism: str = Field(min_length=1)
    evidence_ids: tuple[EvidenceId, ...] = Field(min_length=1)
    # Finalization publishes a candidate before independent Review; the
    # confidence score is therefore intentionally absent until Review writes
    # its evidence-grounded assessment.
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    claim_evidence_mapping: tuple[ClaimEvidenceMapping, ...] = ()
    causal_chain_summary: tuple[CausalChainLink, ...] = ()
    source_claims: tuple[SourceClaim, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def migrate_fault_contract(cls, value: Any) -> Any:
        return _normalize_fault_fields(value)

    @model_validator(mode="after")
    def validate_fault_code(self) -> "RootCauseCandidate":
        if self.fault_code and not _FAULT_CODE_PATTERN.fullmatch(self.fault_code):
            raise ValueError(
                "fault_code must be a lowercase snake_case taxonomy identifier"
            )
        if not (self.fault_code.strip() or self.fault_explanation.strip()):
            raise ValueError(
                "Candidate requires fault_code or fault_explanation"
            )
        return self


def candidate_claim_inventory(
    candidate: RootCauseCandidate,
) -> dict[str, tuple[str, ...]]:
    """Expose exact Candidate claims that Final Review may challenge."""

    def clean(value: Any) -> str:
        return " ".join(str(value or "").split()).strip()

    fault_claims = tuple(dict.fromkeys(filter(None, (
        clean(candidate.fault),
        clean(candidate.fault_code),
        clean(candidate.fault_explanation),
    ))))
    return {
        "component": (clean(candidate.component),),
        "fault": fault_claims,
        "mechanism": (clean(candidate.mechanism),),
        "causal_link": tuple(
            f"{clean(link.cause)} -> {clean(link.effect)}"
            for link in candidate.causal_chain_summary
        ),
        "source_claim": tuple(clean(item.claim) for item in candidate.source_claims),
    }


class ReviewRecoverability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    recoverable: bool = False
    unrecoverable: bool = False


class ReviewGapProposal(BaseModel):
    """Final Review proposal for one missing fact about an existing Candidate.

    The proposal carries semantic meaning and a suggested executable evidence
    path.  Runtime owns identity, criticality and lifecycle after admission.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_kind: Literal[
        "component", "fault", "mechanism", "causal_link", "source_claim",
    ]
    unsupported_claim: str = Field(min_length=1)
    completion_contract: CompletionContract
    claim: str = Field(min_length=1)
    evidence_requirement: str = Field(min_length=1)
    target_component: str = Field(min_length=1)
    verification_question: str = Field(min_length=1)
    evidence_type: str = Field(min_length=1)
    suggested_tools: tuple[str, ...] = Field(min_length=1)
    verification_role: Literal["NECESSARY", "EXPLORATORY"] = "NECESSARY"


class ReviewDecision(CanonicalOutputModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def migrate_review_contract(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        result = dict(value)
        # Symbol resolution belongs to CanonicalOutputModel. This projection
        # derives optional feedback fields without translating decision aliases.
        decision = str(result.get("decision") or "").strip().upper()
        if "blocking_contradictions" not in result and "contradictions" in result:
            result["blocking_contradictions"] = result["contradictions"]
        if "contradictions" not in result and "blocking_contradictions" in result:
            result["contradictions"] = result["blocking_contradictions"]
        if "targeted_followup" not in result and "suggested_investigation" in result:
            result["targeted_followup"] = result["suggested_investigation"]
        if "suggested_investigation" not in result and "targeted_followup" in result:
            result["suggested_investigation"] = result["targeted_followup"]
        if "recoverability" not in result:
            has_followup = decision != "PASS" and bool(
                result.get("gap_proposals")
                or result.get("missing_evidence")
                or result.get("blocking_contradictions")
                or result.get("targeted_followup")
                or result.get("suggested_investigation")
            )
            result["recoverability"] = {
                "recoverable": has_followup,
                "unrecoverable": False,
            }
        return result

    decision: Literal["PASS", "REJECT"]
    # Review owns the only calibrated confidence assessment.  Keep the
    # existing 0..1 convention used by the incident contracts; presentation
    # may render this as a percentage.  Runtime never supplies a default.
    confidence_score: float = Field(ge=0.0, le=1.0)
    confidence_reason: str = Field(min_length=1)
    unsupported_claims: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    blocking_contradictions: tuple[str, ...] = ()
    suggested_investigation: str = ""
    targeted_followup: str = ""
    recoverability: ReviewRecoverability = Field(default_factory=ReviewRecoverability)
    reason: str = Field(min_length=1)
    causal_chain_valid: bool = True
    causal_gaps: tuple[str, ...] = ()
    gap_proposals: tuple[ReviewGapProposal, ...] = ()


class DiagnosticEvidenceLink(BaseModel):
    """Traceable Evidence → Observation/source projection in a report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: EvidenceId
    observation_id: str = ""
    source: str = ""
    file: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    relations: tuple[dict[str, Any], ...] = ()


class DiagnosticReport(BaseModel):
    """Deterministic terminal view of canonical RCA state.

    This is a delivery artifact, not another hypothesis/evidence owner.  The
    Runtime may expose what is known and what remains open, but it never
    invents a causal claim or confidence value here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    report_version: str = "diagnostic-report-v1"
    case_id: str = Field(min_length=1)
    status: Literal["SUPPORTED_ROOT_CAUSE", "INCONCLUSIVE", "FAILED"]
    incident_summary: str = ""
    termination_reason: str = ""
    hypothesis_id: str = ""
    hypothesis_revision: int = Field(default=0, ge=0)
    hypothesis_claim: str = ""
    hypothesis_status: Literal["open", "supported", "rejected", "none"] = "none"
    root_cause_component: str = ""
    root_cause_fault: str = ""
    root_cause_fault_code: str = ""
    root_cause_mechanism: str = ""
    root_cause_evidence_ids: tuple[EvidenceId, ...] = ()
    # None means Review has not supplied a score.  There is no deterministic
    # pre-review confidence label or probability.
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_basis: str = ""
    semantic_record_ids: tuple[str, ...] = ()
    supporting_evidence_ids: tuple[EvidenceId, ...] = ()
    contradicting_evidence_ids: tuple[EvidenceId, ...] = ()
    evidence_links: tuple[DiagnosticEvidenceLink, ...] = ()
    open_required_questions: tuple[str, ...] = ()
    open_exploratory_questions: tuple[str, ...] = ()
    unverified_claims: tuple[str, ...] = ()
    next_steps: tuple[str, ...] = ()
    review_status: Literal["NOT_STARTED", "PASS", "REJECT", "UNAVAILABLE", "DISABLED"] = "NOT_STARTED"
    review_reason: str = ""


class DiagnosticReportReview(BaseModel):
    """Independent assessment of a report, including INCONCLUSIVE reports."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: Literal["PASS", "REJECT"]
    evidence_quality: Literal["SUFFICIENT", "PARTIAL", "INSUFFICIENT"]
    reasoning_quality: Literal["COHERENT", "PARTIAL", "UNSUPPORTED"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    confidence_reason: str = Field(min_length=1)
    report_complete: bool
    unsupported_claims: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    reason: str = Field(min_length=1)


class IncidentMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    steps: int
    tool_calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float | None = None
    llm_calls: int = 0
    # llm_calls/planner_calls count admitted role requests, including both Phase 3
    # requests. steps and gateway logical_planner_turns identify logical turns;
    # physical attempts and measured usage come only from provider_accounting.
    llm_attempts: int = 0
    provider_calls: int = 0
    provider_accounting: dict[str, Any] = Field(default_factory=dict)
    budget_rejected_attempts: int = 0
    schema_repair_calls: int = 0
    provider_failures: int = 0
    provider_contract_failures: int = 0
    semantic_adjudication_failures: int = 0
    token_accounting_error_count: int = 0
    unattributed_prompt_tokens: int = 0
    token_telemetry: tuple[dict[str, Any], ...] = ()
    planner_calls: int = 0
    review_calls: int = 0
    wall_clock_seconds: float = 0.0
    review_rounds: int
    first_pass_accepted: bool
    evidence_sufficient_step: int | None = None
    final_candidate_step: int | None = None
    tool_calls_after_evidence_sufficient: int = 0
    termination_reason: str = ""
    reflection_calls: int = 0
    reflection_delta_accepts: int = 0
    reflection_no_delta_count: int = 0
    duplicate_calls: int = 0
    circuit_open_count: int = 0
    blocked_tool_call_count: int = 0
    schema_repair_count: int = 0
    contract_repair_count: int = 0
    review_recovery_cycles: int = 0
    obligation_created_count: int = 0
    obligation_blocked_count: int = 0
    blocking_contradiction_count: int = 0
    stable_rounds: int = 0
    execution_mode: Literal["baseline", "dag"] = "baseline"
    context_projection_mode: Literal["existing", "task_aware"] = "task_aware"
    dag_task_count: int = 0
    dag_completed_task_count: int = 0
    dag_blocked_task_count: int = 0
    dag_failed_task_count: int = 0
    dag_local_replan_count: int = 0
    dag_task_completion_rate: float = 0.0


TerminalOutcomeClass = Literal["SEMANTIC", "EXECUTION_FAILURE"]
TerminalOutcomeStatus = Literal[
    "CONVERGED",
    "INCONCLUSIVE_BUDGET",
    "INCONCLUSIVE_CAPABILITY",
    "INCONCLUSIVE_NO_PROGRESS",
    "INCONCLUSIVE_CONTRACT",
    "EXECUTION_FAILURE",
]


class TerminalOutcome(BaseModel):
    """Canonical terminal result, independent of the serialized run status.

    ``IncidentRunResult.status`` remains the execution summary field.
    New callers must use this field to distinguish a legal semantic stop from
    a provider/runtime execution failure.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    outcome_class: TerminalOutcomeClass
    status: TerminalOutcomeStatus
    reason: str = ""
    failure_category: str = ""


class IncidentRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: str = "incident-run-v1"
    case_id: str
    status: Literal["PASS", "INCONCLUSIVE", "FAILED"]
    candidate: RootCauseCandidate | None = None
    review: ReviewDecision | None = None
    report: DiagnosticReport | None = None
    report_review: DiagnosticReportReview | None = None
    semantic_records: tuple[SemanticDecisionRecord, ...] = ()
    evidence: tuple[IncidentEvidence, ...]
    hypotheses: tuple[IncidentHypothesis, ...]
    actions: tuple[dict[str, Any], ...]
    metrics: IncidentMetrics
    trace_path: str
    error_type: str = ""
    error_message: str = ""
    failure_category: str = ""
    # Additive field: historical run.json artifacts without this field remain
    # readable through the default ``None`` value.
    terminal_outcome: TerminalOutcome | None = None
