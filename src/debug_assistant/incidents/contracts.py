from __future__ import annotations

import re
from typing import Annotated, Any, Literal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


# Evidence IDs are a runtime-owned namespace.  The pattern is deliberately
# narrower than "known Evidence": the Harness still checks that an ev-* ID is
# present in the current EvidenceMemory.  This type only prevents model output
# from smuggling observation IDs or arbitrary strings into semantic contracts.
EvidenceId = Annotated[str, StringConstraints(pattern=r"^ev-")]


_FAULT_CODE_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")
from .fault_taxonomy import is_canonical_fault_code


def _migrate_fault_fields(value: Any) -> Any:
    """Populate the new fault projections without changing old payloads.

    ``fault`` was historically a single natural-language field.  It remains
    accepted and serialized as a compatibility projection; new producers may
    provide the structured code and the explanatory text independently.
    """
    if not isinstance(value, dict):
        return value
    result = dict(value)
    legacy = str(result.get("fault") or "").strip()
    code = str(result.get("fault_code") or "").strip()
    explanation = str(result.get("fault_explanation") or "").strip()
    if legacy and not code and _FAULT_CODE_PATTERN.fullmatch(legacy):
        result["fault_code"] = legacy
        code = legacy
    if legacy and not explanation:
        result["fault_explanation"] = legacy
        explanation = legacy
    if (code or explanation) and not legacy:
        # Keep the old public attribute meaningful for callers that still use
        # ``candidate.fault``.  The structured fields remain authoritative for
        # new evaluation.
        result["fault"] = code or explanation
    return result


ObligationStatus = Literal[
    "OPEN", "SATISFIED", "BLOCKED_BY_CAPABILITY",
    "NOT_APPLICABLE_BY_CAPABILITY",
]
ContradictionSeverity = Literal["WEAK", "BLOCKING"]
ContradictionStatus = Literal["OPEN", "RESOLVED", "EXPLAINED"]
SourceMechanismStatus = Literal[
    "unknown", "gap", "sufficient", "not_applicable", "blocked",
]


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
    critical: bool = True
    status: ObligationStatus = "OPEN"
    supporting_evidence_ids: tuple[EvidenceId, ...] = ()
    blocked_capabilities: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_status(cls, value: Any) -> Any:
        """Read historical traces without reintroducing the old state."""
        if not isinstance(value, dict):
            return value
        result = dict(value)
        if result.get("status") == "WAIVED_WITH_EVIDENCE":
            result["status"] = "SATISFIED"
        return result

    @property
    def blocks_finalization(self) -> bool:
        return self.critical and self.status in {"OPEN", "BLOCKED_BY_CAPABILITY"}


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


class ReflectionObligationReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    obligation_id: str = Field(min_length=1)
    status: ObligationStatus
    reason: str = Field(min_length=1)
    supporting_evidence_ids: tuple[EvidenceId, ...] = ()


class ReflectionContradictionReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_id: EvidenceId
    status: ContradictionStatus
    severity: ContradictionSeverity = "BLOCKING"
    claim: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ReflectionHypothesisDelta(BaseModel):
    """Provider proposal for a semantic hypothesis change.

    The proposal contains no runtime-owned version or identifier.  The
    Harness compares its canonical fields with the current hypothesis before
    accepting it and assigns the next version in the trace.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    claim: str = ""
    component: str = ""
    fault: str = ""
    fault_code: str = ""
    fault_explanation: str = ""
    mechanism: str = ""
    source_mechanism_status: SourceMechanismStatus | None = None
    supporting_evidence_ids: tuple[EvidenceId, ...] = ()
    contradicting_evidence_ids: tuple[EvidenceId, ...] = ()


class ReflectionEvidenceGapProposal(BaseModel):
    """A gap proposal without a model-controlled gap ID."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    claim: str = Field(min_length=1)
    critical: bool = True


class ReflectionObligationProposal(BaseModel):
    """A new obligation proposal; ID and lifecycle status belong to Runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    claim: str = Field(min_length=1)
    critical: bool = True
    supporting_evidence_ids: tuple[EvidenceId, ...] = ()


class ReflectionContradictionUpdate(BaseModel):
    """A bounded update to a contradiction keyed by a canonical Evidence ID."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_id: EvidenceId
    claim: str = Field(min_length=1)
    severity: ContradictionSeverity = "BLOCKING"
    status: ContradictionStatus = "OPEN"
    reason: str = Field(min_length=1)


class ReflectionFeedback(BaseModel):
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
    # Structured Delta is additive to the legacy feedback fields.  Runtime
    # accepts only a real canonical change and owns IDs/status/versioning.
    hypothesis_delta: ReflectionHypothesisDelta = Field(default_factory=ReflectionHypothesisDelta)
    proposed_evidence_gaps: tuple[ReflectionEvidenceGapProposal, ...] = ()
    proposed_obligations: tuple[ReflectionObligationProposal, ...] = ()
    contradiction_updates: tuple[ReflectionContradictionUpdate, ...] = ()
    next_action_constraint: str = ""
    hypothesis_stable: bool = False
    highest_information_gain_direction: str = ""
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
        return _migrate_fault_fields(value)

    def required_gap_projection(self) -> tuple[str, ...]:
        """Derive the legacy gap view from obligations when they are present."""
        if self.verification_obligations:
            return tuple(
                obligation.claim
                for obligation in self.verification_obligations
                if obligation.blocks_finalization
            )
        return self.required_gaps

    def contradicting_evidence_projection(self) -> tuple[str, ...]:
        """Derive the flat contradiction view from structured contradictions.

        Structured contradictions own severity and lifecycle.  The flat IDs are
        accepted only for older Planner responses and are never allowed to
        override a structured contradiction set supplied in the same state.
        """
        if self.contradictions:
            return tuple(
                item.evidence_id
                for item in self.contradictions
                if item.blocks_finalization
            )
        return self.contradicting_evidence_ids

    def finalization_diagnostics(self, known_evidence_ids: set[str]) -> dict[str, Any]:
        """Return the deterministic reasons this Hypothesis cannot finalize.

        ``VerificationObligation`` and structured contradictions are canonical;
        the flat fields are compatibility projections.  Keeping diagnostics
        beside ``can_finalize`` prevents tracing from implementing a second,
        subtly different completion predicate.
        """
        obligations = self.verification_obligations
        failures: dict[str, Any] = {}
        if not self.evidence_sufficient:
            failures["evidence_sufficient"] = False
        missing_fields = [
            field for field in ("component", "fault", "mechanism")
            if not getattr(self, field).strip()
        ]
        if missing_fields:
            failures["missing_fields"] = missing_fields
        if len(self.supporting_evidence_ids) < 2:
            failures["supporting_evidence_count"] = {
                "actual": len(self.supporting_evidence_ids), "minimum": 2,
            }
        invalid_supporting = [
            evidence_id for evidence_id in self.supporting_evidence_ids
            if not evidence_id.startswith("ev-") or evidence_id not in known_evidence_ids
        ]
        if invalid_supporting:
            failures["unknown_supporting_evidence_ids"] = sorted(set(invalid_supporting))

        invalid_obligation_evidence = []
        open_obligations = []
        for obligation in obligations:
            invalid_ids = [
                evidence_id for evidence_id in obligation.supporting_evidence_ids
                if not evidence_id.startswith("ev-") or evidence_id not in known_evidence_ids
            ]
            if invalid_ids or (
                obligation.status == "SATISFIED"
                and not obligation.supporting_evidence_ids
            ):
                invalid_obligation_evidence.append({
                    "obligation_id": obligation.id,
                    "evidence_ids": sorted(set(invalid_ids)),
                    "terminal_without_evidence": (
                        obligation.status == "SATISFIED"
                        and not obligation.supporting_evidence_ids
                    ),
                })
            if obligation.blocks_finalization:
                open_obligations.append({
                    "obligation_id": obligation.id,
                    "claim": obligation.claim,
                    "status": obligation.status,
                })
        if invalid_obligation_evidence:
            failures["invalid_obligation_evidence"] = invalid_obligation_evidence
        if open_obligations:
            failures["open_obligations"] = open_obligations

        effective_contradiction_ids = self.contradicting_evidence_projection()
        all_contradiction_ids = (
            tuple(item.evidence_id for item in self.contradictions)
            if self.contradictions else effective_contradiction_ids
        )
        invalid_contradiction_ids = [
            evidence_id for evidence_id in all_contradiction_ids
            if not evidence_id.startswith("ev-") or evidence_id not in known_evidence_ids
        ]
        if invalid_contradiction_ids:
            failures["unknown_contradiction_evidence_ids"] = sorted(set(invalid_contradiction_ids))
        if self.contradictions and any(
            contradiction.blocks_finalization for contradiction in self.contradictions
        ):
            failures["blocking_contradictions"] = [
                contradiction.evidence_id for contradiction in self.contradictions
                if contradiction.blocks_finalization
            ]
        elif effective_contradiction_ids:
            # The flat field is retained for old Planner/provider outputs. Such
            # IDs have no severity metadata, so preserve the historical
            # fail-closed interpretation as BLOCKING + OPEN.
            failures["legacy_contradicting_evidence_ids"] = sorted(
                set(effective_contradiction_ids)
            )
        gaps = self.required_gap_projection()
        if gaps:
            failures["required_gaps"] = list(gaps)
        if self.fault_code and not is_canonical_fault_code(self.fault_code):
            failures["invalid_fault_code"] = self.fault_code
        return failures

    def can_finalize(self, known_evidence_ids: set[str]) -> bool:
        return not self.finalization_diagnostics(known_evidence_ids)


class RootCauseCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    component: str = Field(min_length=1)
    # ``fault`` is retained for old run.json readers and old Providers.  New
    # Candidate payloads should use the structured pair below.
    fault: str = Field(min_length=1)
    fault_code: str = ""
    fault_explanation: str = ""
    mechanism: str = Field(min_length=1)
    evidence_ids: tuple[EvidenceId, ...] = Field(min_length=2)
    confidence: float = Field(ge=0.0, le=1.0)
    claim_evidence_mapping: tuple[ClaimEvidenceMapping, ...] = ()
    causal_chain_summary: tuple[CausalChainLink, ...] = ()
    source_claims: tuple[SourceClaim, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def migrate_fault_contract(cls, value: Any) -> Any:
        return _migrate_fault_fields(value)

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


class ReviewRecoverability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    recoverable: bool = False
    unrecoverable: bool = False


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def migrate_review_contract(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        result = dict(value)
        if "decision" not in result:
            for alias in ("outcome", "status"):
                if alias in result:
                    result["decision"] = result.pop(alias)
                    break
        decision = str(result.get("decision") or "").strip().upper()
        result["decision"] = {
            "ACCEPT": "PASS", "ACCEPTED": "PASS", "PASS": "PASS",
            "REJECTED": "REJECT", "FAIL": "REJECT", "FAILED": "REJECT",
        }.get(decision, result.get("decision"))
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
                result.get("missing_evidence")
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


class IncidentMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    steps: int
    tool_calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float | None = None
    llm_calls: int = 0
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
    execution_mode: Literal["legacy", "dag"] = "legacy"
    context_projection_mode: Literal["existing", "task_aware"] = "task_aware"
    dag_task_count: int = 0
    dag_satisfied_task_count: int = 0
    dag_blocked_task_count: int = 0
    dag_contradicted_task_count: int = 0
    dag_local_replan_count: int = 0
    dag_task_closure_rate: float = 0.0


class IncidentRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str
    status: Literal["PASS", "INCONCLUSIVE", "FAILED"]
    candidate: RootCauseCandidate | None = None
    review: ReviewDecision | None = None
    evidence: tuple[IncidentEvidence, ...]
    hypotheses: tuple[IncidentHypothesis, ...]
    actions: tuple[dict[str, Any], ...]
    metrics: IncidentMetrics
    trace_path: str
    error_type: str = ""
    error_message: str = ""
    failure_category: str = ""
