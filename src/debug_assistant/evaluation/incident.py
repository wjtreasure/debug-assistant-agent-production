from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from debug_assistant.evaluation.semantic import (
    DeterministicSemanticGrader,
    SemanticGrade,
    SemanticGrader,
    SemanticRubric,
    UnavailableSemanticGrader,
)
from debug_assistant.incidents.contracts import IncidentRunResult
from debug_assistant.skills.catalog import INCIDENT_SKILLS


_FAULT_CODE_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")


class CloudOpsGroundTruth(BaseModel):
    """Evaluator-only gold contract.

    ``fault`` remains a compatibility projection for old fixtures.  New gold
    should provide ``fault_code`` and a rubric for the explanatory text.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    component: str
    fault_code: str = ""
    fault: str = ""
    fault_explanation: str = ""
    fault_explanation_rubric: SemanticRubric = Field(default_factory=SemanticRubric)
    component_aliases: tuple[str, ...] = ()
    # Legacy aliases are used only by the compatibility adapter when an old
    # Candidate has no structured fault_code.
    fault_aliases: tuple[str, ...] = ()
    mechanism: str | None = None
    mechanism_rubric: SemanticRubric = Field(default_factory=SemanticRubric)
    mechanism_aliases: tuple[str, ...] = ()
    gold_source: str = ""
    gold_version: str = ""
    source_snapshot: str = ""

    @model_validator(mode="before")
    @classmethod
    def migrate_fault_contract(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        result = dict(value)
        legacy = str(result.get("fault") or "").strip()
        code = str(result.get("fault_code") or "").strip()
        explanation = str(result.get("fault_explanation") or "").strip()
        if not code and legacy and _FAULT_CODE_PATTERN.fullmatch(legacy):
            result["fault_code"] = legacy
        if not explanation and legacy:
            result["fault_explanation"] = legacy
        if not legacy and (code or explanation):
            result["fault"] = code or explanation
        return result

    @property
    def canonical_fault_code(self) -> str:
        return _normalize(self.fault_code or self.fault)


class EvidencePattern(BaseModel):
    """One evaluator-only evidence pattern.

    A pattern is an AND of its fields.  ``contains`` is an all-of keyword list;
    ``any_of`` belongs to the containing group.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str | None = None
    evidence_type: str | None = None
    contains: tuple[str, ...] = ()
    attributes: dict[str, Any] = Field(default_factory=dict)


class EvidenceGroup(BaseModel):
    """Required group: all groups must pass, one pattern in each may pass."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    any_of: tuple[EvidencePattern, ...] = Field(min_length=1)


class EvidenceRequirement(BaseModel):
    """Old one-pattern evidence requirement, retained for old fixtures/tests."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str
    contains: tuple[str, ...]


class CloudOpsEvaluatorData(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ground_truth: CloudOpsGroundTruth
    required_evidence_groups: tuple[EvidenceGroup, ...] = ()
    # Compatibility projection.  It is never passed to runtime or planner.
    key_evidence: tuple[EvidenceRequirement, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def migrate_evidence_requirements(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        result = dict(value)
        groups = result.get("required_evidence_groups")
        requirements = result.get("key_evidence")
        if not groups and requirements:
            result["required_evidence_groups"] = [
                {
                    "name": f"legacy-{index + 1}",
                    "any_of": [item],
                }
                for index, item in enumerate(requirements)
            ]
        elif not requirements and groups:
            # This projection is for old report readers only.  A group with
            # multiple alternatives cannot be represented losslessly here.
            projected = []
            for item in groups:
                if isinstance(item, EvidenceGroup):
                    pattern = item.any_of[0] if item.any_of else None
                    source = pattern.source if pattern else ""
                    contains = pattern.contains if pattern else ()
                else:
                    patterns = item.get("any_of") if isinstance(item, dict) else ()
                    pattern = patterns[0] if patterns else {}
                    source = pattern.get("source", "") if isinstance(pattern, dict) else ""
                    contains = pattern.get("contains", ()) if isinstance(pattern, dict) else ()
                if pattern is not None:
                    projected.append({"source": source or "", "contains": contains})
            result["key_evidence"] = projected
        return result

    @property
    def evidence_groups(self) -> tuple[EvidenceGroup, ...]:
        return self.required_evidence_groups


class CloudOpsEvaluatorLoader:
    """Evaluator-only loader; never imported by IncidentHarness."""

    def load(self, evaluator_dir: str | Path) -> CloudOpsEvaluatorData:
        root = Path(evaluator_dir).resolve()
        truth = json.loads((root / "ground_truth.json").read_text(encoding="utf-8"))
        groups_path = root / "required_evidence_groups.json"
        groups = json.loads(groups_path.read_text(encoding="utf-8")) if groups_path.is_file() else []
        key_path = root / "key_evidence.json"
        key_evidence = json.loads(key_path.read_text(encoding="utf-8")) if key_path.is_file() else []
        return CloudOpsEvaluatorData(
            ground_truth=truth,
            required_evidence_groups=groups,
            key_evidence=key_evidence,
        )


class IncidentValidityQuality(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_status: str
    candidate_present: bool
    candidate_accepted: bool
    review_result: str
    timeout: bool
    contract_failure: bool
    inconclusive: bool


class IncidentResultQuality(BaseModel):
    """Outcome metrics, kept separate from investigation process metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root_cause_correct: bool
    component_correct: bool
    fault_correct: bool
    fault_code_correct: bool
    fault_explanation_semantic_score: int | None
    fault_explanation_semantic_status: str
    mechanism_evaluated: bool
    mechanism_correct: bool | None = None
    mechanism_score: int | None = None
    evidence_supported: bool
    evidence_validity: float
    required_evidence_coverage: float
    unsupported_claim_rate: float
    key_evidence_coverage: float


class IncidentComponentQuality(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    correct: bool


class IncidentFaultQuality(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    code_correct: bool
    explanation_semantic_score: int | None
    explanation_semantic_status: str
    explanation_grade: SemanticGrade
    legacy_fault_schema: bool


class IncidentMechanismQuality(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    evaluated: bool
    score: int | None
    correct: bool | None
    semantic_grade: SemanticGrade | None = None


class IncidentEvidenceQuality(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    cited_count: int
    valid_count: int
    evidence_validity: float
    evidence_supported: bool
    required_evidence_coverage: float
    groups_satisfied: int
    groups_total: int
    unsupported_claim_rate: float
    material_claims: int
    supported_claims: int


class IncidentProcessQuality(BaseModel):
    """Deterministic properties of the Agent trajectory, not its answer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    skill_path: tuple[str, ...]
    useful_tool_calls: int
    irrelevant_tool_calls: int
    repeated_tool_calls: int
    duplicate_action_requests: int = 0
    duplicate_tool_calls_executed: int = 0
    evidence_coverage: float
    no_progress_termination: bool
    review_outcome: str
    candidate_present: bool
    new_evidence_per_step: float = 0.0
    obligation_close_rate: float = 1.0
    contradiction_resolution_rate: float = 1.0
    reflection_triggers: int = 0
    reflection_recoveries: int = 0
    review_reject_recovery: bool = False


class IncidentTrajectoryQuality(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    skill_path: tuple[str, ...]
    useful_tool_calls: int
    irrelevant_tool_calls: int
    repeated_tool_calls: int
    no_progress_termination: bool
    new_evidence_per_step: float
    obligation_close_rate: float
    contradiction_resolution_rate: float
    reflection_triggers: int
    reflection_recoveries: int
    review_reject_recovery: bool


class IncidentCostReliability(BaseModel):
    """Measured run cost; zero means the run did not expose that measurement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    planner_calls: int
    review_calls: int
    llm_calls: int
    tool_calls: int
    steps: int
    tokens: int
    cost: float | None = None
    wall_clock_seconds: float
    terminal_status: str
    failure_category: str


class IncidentReliabilityQuality(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    timeout: bool
    schema_repair_count: int
    contract_failure: bool
    inconclusive: bool
    terminal_status: str
    review_result: str


class IncidentEfficiencyQuality(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    steps: int
    tool_calls: int
    llm_calls: int
    tokens: int
    cost: float | None = None
    wall_clock_seconds: float
    repeated_tool_calls: int
    irrelevant_tool_calls: int


class IncidentOverallQuality(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    strict_task_success: bool
    component_correct: bool
    fault_code_correct: bool
    fault_explanation_semantic_score: int | None
    mechanism_score: int | None
    evidence_validity: float
    required_evidence_coverage: float
    unsupported_claim_rate: float


class IncidentGoldMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str
    source: str
    version: str
    required_evidence_groups: int


class IncidentEvalResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_status: str
    candidate_present: bool
    candidate_accepted: bool
    component_correct: bool
    fault_correct: bool
    fault_code_correct: bool
    fault_explanation_semantic: int | None
    fault_explanation_semantic_score: int | None
    fault_explanation_semantic_status: str
    mechanism_evaluated: bool
    mechanism_correct: bool | None = None
    mechanism_score: int | None = None
    mechanism_status: str
    root_cause_correct: bool
    strict_task_success: bool
    root_cause_fault_correct: bool
    evidence_supported: bool
    evidence_validity: float
    required_evidence_coverage: float
    unsupported_claim_rate: float
    key_evidence_coverage: float
    repeated_tool_calls: int
    duplicate_action_requests: int
    duplicate_tool_calls_executed: int
    irrelevant_tool_calls: int
    review_result: str
    timeout: bool
    schema_repair_count: int
    contract_failure: bool
    inconclusive: bool
    steps: int
    tool_calls: int
    tokens: int
    validity: IncidentValidityQuality
    component: IncidentComponentQuality
    fault: IncidentFaultQuality
    mechanism: IncidentMechanismQuality
    evidence: IncidentEvidenceQuality
    trajectory: IncidentTrajectoryQuality
    reliability: IncidentReliabilityQuality
    efficiency: IncidentEfficiencyQuality
    overall: IncidentOverallQuality
    gold: IncidentGoldMetadata
    # Existing nested projections are retained so old consumers do not need a
    # flag day migration.
    result_quality: IncidentResultQuality
    process_quality: IncidentProcessQuality
    cost_reliability: IncidentCostReliability


class IncidentEvaluator:
    """Evaluate an immutable run after completion.

    The evaluator is the only owner of gold, rubrics, and evidence groups.  It
    is intentionally not imported by the incident runtime or planner.
    """

    def __init__(self, *, semantic_grader: SemanticGrader | None = None):
        self.semantic_grader = semantic_grader or DeterministicSemanticGrader()
        self.unavailable_grader = UnavailableSemanticGrader()

    def evaluate(self, run: IncidentRunResult, expected: CloudOpsEvaluatorData) -> IncidentEvalResult:
        truth = expected.ground_truth
        groups = expected.evidence_groups
        observed = tuple(run.evidence)
        group_matches = tuple(
            _evidence_group_satisfied(group, observed) for group in groups
        )
        groups_satisfied = sum(group_matches)
        required_coverage = groups_satisfied / len(groups) if groups else 1.0

        fingerprints = [
            str(action.get("fingerprint"))
            for action in run.actions
            if action.get("tool") != "finalize_diagnosis" and action.get("fingerprint")
        ]
        duplicate_tool_calls_executed = len(fingerprints) - len(set(fingerprints))
        duplicate_action_requests = _duplicate_action_requests(run, duplicate_tool_calls_executed)
        suggested = set().union(*(set(skill.suggested_tools) for skill in INCIDENT_SKILLS.values()))
        irrelevant = sum(
            1 for action in run.actions
            if action.get("tool") not in suggested and action.get("tool") != "finalize_diagnosis"
        )

        accepted_components = {
            canonicalize_component(truth.component),
            *(canonicalize_component(value) for value in truth.component_aliases),
        }
        candidate_present = run.candidate is not None
        candidate_accepted = run.status == "PASS" and candidate_present
        candidate = run.candidate if candidate_accepted else None
        candidate_component = canonicalize_component(candidate.component) if candidate else ""
        component_correct = bool(candidate and candidate_component in accepted_components)

        candidate_fault_code, candidate_explanation, legacy_fault_schema = _candidate_fault_projection(candidate)
        truth_fault_code = truth.canonical_fault_code
        if candidate:
            if candidate.fault_code.strip():
                # Structured codes are exact taxonomy identifiers.  Aliases and
                # free-text phrase matching never participate in this branch.
                fault_code_correct = (
                    canonicalize_fault_code(candidate_fault_code) == truth_fault_code
                )
            else:
                # Old run.json / Provider compatibility adapter only.
                fault_code_correct = (
                    canonicalize_fault(
                        candidate_fault_code,
                        truth_fault_code,
                        truth.fault_aliases,
                    ) == truth_fault_code
                )
        else:
            fault_code_correct = False

        explanation_rubric = truth.fault_explanation_rubric
        explanation_grade = self._grade(
            candidate_explanation if candidate else "",
            truth.fault_explanation or truth.fault,
            explanation_rubric,
            field="fault_explanation",
        )
        explanation_score = explanation_grade.score

        mechanism_grade: SemanticGrade | None = None
        mechanism_score: int | None = None
        mechanism_evaluated = False
        mechanism_correct: bool | None = None
        if truth.mechanism_rubric.available:
            mechanism_grade = self._grade(
                candidate.mechanism if candidate else "",
                truth.mechanism or "",
                truth.mechanism_rubric,
                field="mechanism",
            )
            mechanism_score = mechanism_grade.score
            mechanism_evaluated = mechanism_grade.status == "AVAILABLE"
            mechanism_correct = mechanism_score == 2 if mechanism_evaluated else None
        elif _mechanism_is_deterministically_evaluable(truth):
            mechanism_evaluated = True
            mechanism_correct = bool(
                candidate
                and canonicalize_mechanism(
                    candidate.mechanism, truth.mechanism or "", truth.mechanism_aliases,
                ) == canonicalize_free_text(truth.mechanism or "")
            )
            mechanism_score = 2 if mechanism_correct else 0

        cited_ids = set(candidate.evidence_ids) if candidate else set()
        known_ids = {item.evidence_id for item in observed}
        valid_cited_ids = {
            evidence_id for evidence_id in cited_ids
            if evidence_id.startswith("ev-") and evidence_id in known_ids
        }
        evidence_validity = len(valid_cited_ids) / len(cited_ids) if cited_ids else 0.0
        evidence_supported = bool(
            candidate
            and len(cited_ids) >= 2
            and len(valid_cited_ids) == len(cited_ids)
        )
        material_claims, supported_claims = _claim_support_counts(candidate, valid_cited_ids)
        unsupported_claim_rate = (
            (material_claims - supported_claims) / material_claims
            if material_claims else 0.0
        )

        skill_path = tuple(dict.fromkeys(
            str(action.get("skill")) for action in run.actions if action.get("skill")
        ))
        investigation_actions = [
            action for action in run.actions if action.get("tool") != "finalize_diagnosis"
        ]
        if run.review is not None:
            review_outcome = run.review.decision
        elif run.metrics.review_rounds > 0:
            review_outcome = (
                "TIMEOUT"
                if run.error_type == "LLMDeadlineExceeded"
                or "timeout" in run.metrics.termination_reason.casefold()
                or "deadline" in run.metrics.termination_reason.casefold()
                else "ATTEMPTED_NO_DECISION"
            )
        else:
            review_outcome = "NOT_RUN"
        timeout = _run_timed_out(run)
        contract_failure = _run_contract_failed(run)
        inconclusive = run.status == "INCONCLUSIVE"
        trace_events = _load_trace_events(run)
        trajectory_metrics = _trajectory_metrics(run, trace_events)

        # This is the new outcome gate.  ``root_cause_correct`` below remains
        # the historical compatibility metric and is deliberately less strict.
        strict_task_success = bool(
            candidate_accepted
            and component_correct
            and fault_code_correct
            and explanation_grade.status == "AVAILABLE"
            and explanation_score == 2
            and mechanism_evaluated
            and mechanism_score == 2
            and evidence_supported
            and required_coverage == 1.0
            and unsupported_claim_rate == 0.0
        )
        root_cause_correct = bool(
            candidate
            and component_correct
            and fault_code_correct
            and evidence_supported
            and (mechanism_correct is not False)
        )

        return IncidentEvalResult(
            run_status=run.status,
            candidate_present=candidate_present,
            candidate_accepted=candidate_accepted,
            component_correct=component_correct,
            fault_correct=fault_code_correct,
            fault_code_correct=fault_code_correct,
            fault_explanation_semantic=explanation_score,
            fault_explanation_semantic_score=explanation_score,
            fault_explanation_semantic_status=explanation_grade.status,
            mechanism_evaluated=mechanism_evaluated,
            mechanism_correct=mechanism_correct,
            mechanism_score=mechanism_score,
            mechanism_status=("AVAILABLE" if mechanism_evaluated else "UNAVAILABLE"),
            root_cause_correct=root_cause_correct,
            strict_task_success=strict_task_success,
            root_cause_fault_correct=fault_code_correct,
            evidence_supported=evidence_supported,
            evidence_validity=evidence_validity,
            required_evidence_coverage=required_coverage,
            unsupported_claim_rate=unsupported_claim_rate,
            key_evidence_coverage=required_coverage,
            repeated_tool_calls=duplicate_tool_calls_executed,
            duplicate_action_requests=duplicate_action_requests,
            duplicate_tool_calls_executed=duplicate_tool_calls_executed,
            irrelevant_tool_calls=irrelevant,
            review_result=review_outcome,
            timeout=timeout,
            schema_repair_count=run.metrics.schema_repair_count,
            contract_failure=contract_failure,
            inconclusive=inconclusive,
            steps=run.metrics.steps,
            tool_calls=run.metrics.tool_calls,
            tokens=run.metrics.total_tokens,
            validity=IncidentValidityQuality(
                run_status=run.status,
                candidate_present=candidate_present,
                candidate_accepted=candidate_accepted,
                review_result=review_outcome,
                timeout=timeout,
                contract_failure=contract_failure,
                inconclusive=inconclusive,
            ),
            component=IncidentComponentQuality(correct=component_correct),
            fault=IncidentFaultQuality(
                code_correct=fault_code_correct,
                explanation_semantic_score=explanation_score,
                explanation_semantic_status=explanation_grade.status,
                explanation_grade=explanation_grade,
                legacy_fault_schema=legacy_fault_schema,
            ),
            mechanism=IncidentMechanismQuality(
                evaluated=mechanism_evaluated,
                score=mechanism_score,
                correct=mechanism_correct,
                semantic_grade=mechanism_grade,
            ),
            evidence=IncidentEvidenceQuality(
                cited_count=len(cited_ids),
                valid_count=len(valid_cited_ids),
                evidence_validity=evidence_validity,
                evidence_supported=evidence_supported,
                required_evidence_coverage=required_coverage,
                groups_satisfied=groups_satisfied,
                groups_total=len(groups),
                unsupported_claim_rate=unsupported_claim_rate,
                material_claims=material_claims,
                supported_claims=supported_claims,
            ),
            trajectory=IncidentTrajectoryQuality(
                skill_path=skill_path,
                useful_tool_calls=max(0, len(investigation_actions) - irrelevant),
                irrelevant_tool_calls=irrelevant,
                repeated_tool_calls=duplicate_tool_calls_executed,
                no_progress_termination="no_progress" in run.metrics.termination_reason,
                **trajectory_metrics,
            ),
            reliability=IncidentReliabilityQuality(
                timeout=timeout,
                schema_repair_count=run.metrics.schema_repair_count,
                contract_failure=contract_failure,
                inconclusive=inconclusive,
                terminal_status=run.status,
                review_result=review_outcome,
            ),
            efficiency=IncidentEfficiencyQuality(
                steps=run.metrics.steps,
                tool_calls=run.metrics.tool_calls,
                llm_calls=run.metrics.llm_calls,
                tokens=run.metrics.total_tokens,
                cost=run.metrics.cost,
                wall_clock_seconds=run.metrics.wall_clock_seconds,
                repeated_tool_calls=duplicate_tool_calls_executed,
                irrelevant_tool_calls=irrelevant,
            ),
            overall=IncidentOverallQuality(
                strict_task_success=strict_task_success,
                component_correct=component_correct,
                fault_code_correct=fault_code_correct,
                fault_explanation_semantic_score=explanation_score,
                mechanism_score=mechanism_score,
                evidence_validity=evidence_validity,
                required_evidence_coverage=required_coverage,
                unsupported_claim_rate=unsupported_claim_rate,
            ),
            gold=IncidentGoldMetadata(
                case_id=truth.case_id,
                source=truth.gold_source or "unspecified",
                version=truth.gold_version or "unspecified",
                required_evidence_groups=len(groups),
            ),
            result_quality=IncidentResultQuality(
                root_cause_correct=root_cause_correct,
                component_correct=component_correct,
                fault_correct=fault_code_correct,
                fault_code_correct=fault_code_correct,
                fault_explanation_semantic_score=explanation_score,
                fault_explanation_semantic_status=explanation_grade.status,
                mechanism_evaluated=mechanism_evaluated,
                mechanism_correct=mechanism_correct,
                mechanism_score=mechanism_score,
                evidence_supported=evidence_supported,
                evidence_validity=evidence_validity,
                required_evidence_coverage=required_coverage,
                unsupported_claim_rate=unsupported_claim_rate,
                key_evidence_coverage=required_coverage,
            ),
            process_quality=IncidentProcessQuality(
                skill_path=skill_path,
                useful_tool_calls=max(0, len(investigation_actions) - irrelevant),
                irrelevant_tool_calls=irrelevant,
                repeated_tool_calls=duplicate_tool_calls_executed,
                duplicate_action_requests=duplicate_action_requests,
                duplicate_tool_calls_executed=duplicate_tool_calls_executed,
                evidence_coverage=required_coverage,
                no_progress_termination="no_progress" in run.metrics.termination_reason,
                review_outcome=review_outcome,
                candidate_present=candidate_present,
                **trajectory_metrics,
            ),
            cost_reliability=IncidentCostReliability(
                planner_calls=run.metrics.planner_calls,
                review_calls=run.metrics.review_calls,
                llm_calls=run.metrics.llm_calls,
                tool_calls=run.metrics.tool_calls,
                steps=run.metrics.steps,
                tokens=run.metrics.total_tokens,
                cost=run.metrics.cost,
                wall_clock_seconds=run.metrics.wall_clock_seconds,
                terminal_status=run.status,
                failure_category=run.failure_category,
            ),
        )

    def _grade(self, candidate: str, reference: str, rubric: SemanticRubric, *, field: str) -> SemanticGrade:
        if not rubric.available:
            return self.unavailable_grader.grade(candidate, reference, rubric, field=field)
        return self.semantic_grader.grade(candidate, reference, rubric, field=field)


def _normalize(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())).strip("_")


def canonicalize_free_text(value: str) -> str:
    """Normalize surface form only; do not infer domain semantics."""
    return _normalize(value)


def normalize_evidence_text(value: str) -> str:
    """Normalize formatting noise without changing evidence semantics."""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def canonicalize_component(value: str) -> str:
    """Remove generic Kubernetes resource decoration from one component name."""
    without_explanation = re.sub(r"\([^)]*\)", " ", str(value).lower())
    normalized = _normalize(without_explanation)
    tokens = normalized.split("_") if normalized else []
    decorations = {"service", "deployment", "pod", "container", "workload"}
    while len(tokens) > 1 and tokens[-1] in decorations:
        tokens.pop()
    return "_".join(tokens)


def canonicalize_fault_code(value: str) -> str:
    """Canonicalize a structured code without aliasing or phrase matching."""
    normalized = _normalize(value)
    return normalized if _FAULT_CODE_PATTERN.fullmatch(normalized or "") else ""


def canonicalize_fault(value: str, canonical: str, aliases: tuple[str, ...]) -> str:
    """Compatibility adapter for old natural-language fault fields only."""
    return FaultCanonicalizer().canonicalize(value, canonical=canonical, aliases=aliases)


def _normalize_fault_phrase(value: str) -> str:
    tokens = [token for token in _normalize(value).split("_") if token not in {"set"}]
    return "_".join(tokens)


UNKNOWN_FAULT = "UNKNOWN"


class FaultCanonicalizer:
    """Deterministic exact/approved-alias canonicalizer for legacy payloads."""

    def canonicalize(self, value: str, *, canonical: str, aliases: tuple[str, ...] = ()) -> str:
        normalized = _normalize_fault_phrase(value)
        canonical_normalized = _normalize(canonical)
        if not normalized or not canonical_normalized:
            return UNKNOWN_FAULT
        phrases = {
            canonical_normalized,
            *(_normalize_fault_phrase(alias) for alias in aliases),
        }
        if any(_approved_phrase_match(normalized, phrase) for phrase in phrases if phrase):
            return canonical_normalized
        return UNKNOWN_FAULT


def _approved_phrase_match(candidate: str, phrase: str) -> bool:
    """Match a complete approved phrase, optionally embedded in prose."""
    return candidate == phrase or f"_{phrase}_" in f"_{candidate}_"


def canonicalize_mechanism(value: str, canonical: str, aliases: tuple[str, ...] = ()) -> str:
    """Match only the canonical mechanism label or explicitly approved aliases."""
    normalized = canonicalize_free_text(value)
    target = canonicalize_free_text(canonical)
    accepted = {target, *(canonicalize_free_text(alias) for alias in aliases)}
    return target if normalized and normalized in accepted else UNKNOWN_FAULT


def _mechanism_is_deterministically_evaluable(truth: CloudOpsGroundTruth) -> bool:
    value = str(truth.mechanism or "").strip()
    if not value:
        return False
    structured = bool(re.fullmatch(r"[a-z][a-z0-9_]*", value))
    return structured or bool(truth.mechanism_aliases)


def _candidate_fault_projection(candidate) -> tuple[str, str, bool]:
    if candidate is None:
        return "", "", False
    if candidate.fault_code.strip():
        return (
            candidate.fault_code.strip(),
            candidate.fault_explanation.strip() or candidate.fault.strip(),
            False,
        )
    # Old candidates may have had their ``fault`` changed with model_copy;
    # always use that field as the compatibility source in this branch.
    return candidate.fault.strip(), candidate.fault.strip(), True


def _evidence_group_satisfied(group: EvidenceGroup, evidence: tuple[Any, ...]) -> bool:
    return any(
        _evidence_pattern_matches(pattern, item)
        for pattern in group.any_of
        for item in evidence
    )


def _evidence_pattern_matches(pattern: EvidencePattern, item: Any) -> bool:
    if pattern.source and str(getattr(item, "source", "")) != pattern.source:
        return False
    if pattern.evidence_type:
        values = {
            str(getattr(item, "source", "")),
            str((getattr(item, "provenance", {}) or {}).get("evidence_type", "")),
            str((getattr(item, "provenance", {}) or {}).get("context_kind", "")),
            *(str(value) for value in getattr(item, "tags", ()) or ()),
        }
        if pattern.evidence_type.casefold() not in {value.casefold() for value in values}:
            return False
    text = normalize_evidence_text(" ".join(
        str(getattr(item, field, "") or "")
        for field in ("target", "summary", "excerpt")
    ))
    if any(normalize_evidence_text(token) not in text for token in pattern.contains):
        return False
    for key, expected in pattern.attributes.items():
        actual = _evidence_attribute(item, key)
        if not _attribute_equal(actual, expected):
            return False
    return True


def _evidence_attribute(item: Any, key: str) -> Any:
    if hasattr(item, key):
        return getattr(item, key)
    current: Any = getattr(item, "provenance", {}) or {}
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _attribute_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (list, tuple, set)):
        return any(_attribute_equal(item, expected) for item in actual)
    if isinstance(expected, str):
        return normalize_evidence_text(str(actual or "")) == normalize_evidence_text(expected)
    return actual == expected


def _claim_support_counts(candidate, valid_cited_ids: set[str]) -> tuple[int, int]:
    if candidate is None:
        return 0, 0
    rows: list[tuple[str, tuple[str, ...]]] = []
    rows.extend((item.claim, tuple(item.evidence_ids)) for item in candidate.claim_evidence_mapping)
    rows.extend(
        (f"{item.cause} -> {item.effect}", tuple(item.evidence_ids))
        for item in candidate.causal_chain_summary
    )
    if not rows:
        rows = [(candidate.fault_explanation or candidate.fault, tuple(candidate.evidence_ids))]
    supported = sum(1 for _claim, ids in rows if set(ids).intersection(valid_cited_ids))
    return len(rows), supported


def _trajectory_metrics(run: IncidentRunResult, trace_events: tuple[dict, ...]) -> dict[str, Any]:
    hypothesis = run.hypotheses[-1] if run.hypotheses else None
    obligations = tuple(hypothesis.verification_obligations) if hypothesis else ()
    contradictions = tuple(hypothesis.contradictions) if hypothesis else ()
    closed_obligations = sum(
        item.status in {"SATISFIED", "WAIVED_WITH_EVIDENCE"} for item in obligations
    )
    resolved_contradictions = sum(
        item.status in {"RESOLVED", "EXPLAINED"} for item in contradictions
    )
    reflection_triggers = sum(item.get("type") == "REFLECTION_TRIGGER" for item in trace_events)
    reflection_recoveries = sum(item.get("type") == "REFLECTION_RESULT" for item in trace_events)
    review_reject_triggered = any(
        item.get("type") == "REFLECTION_TRIGGER"
        and str((item.get("payload") or {}).get("reason") or "") == "review_reject"
        for item in trace_events
    )
    return {
        "new_evidence_per_step": len(run.evidence) / max(1, run.metrics.steps),
        "obligation_close_rate": (
            closed_obligations / len(obligations) if obligations else 1.0
        ),
        "contradiction_resolution_rate": (
            resolved_contradictions / len(contradictions) if contradictions else 1.0
        ),
        "reflection_triggers": reflection_triggers,
        "reflection_recoveries": reflection_recoveries,
        "review_reject_recovery": bool(
            review_reject_triggered
            and run.review is not None
            and run.review.decision == "PASS"
        ),
    }


def _load_trace_events(run: IncidentRunResult) -> tuple[dict, ...]:
    try:
        path = Path(run.trace_path)
        if not path.is_file():
            return ()
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
        return tuple(rows)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        return ()


def _duplicate_action_requests(run: IncidentRunResult, executed: int) -> int:
    """Count duplicate requests, including requests stopped by duplicate guards."""
    count = max(0, int(executed))
    for event in _load_trace_events(run):
        event_type = str(event.get("type") or "")
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}
        reason = str(payload.get("reason") or "").casefold()
        if event_type == "ACTION_REJECTED" and ("duplicate" in reason or "repeated" in reason):
            count += 1
        elif event_type == "PROGRESS_GUARD_REJECTED" and ("duplicate" in reason or "repeat" in reason):
            count += 1
    return count


def _run_timed_out(run: IncidentRunResult) -> bool:
    text = " ".join((run.error_type, run.metrics.termination_reason, run.failure_category)).casefold()
    return any(token in text for token in ("timeout", "deadline", "timed_out"))


def _run_contract_failed(run: IncidentRunResult) -> bool:
    text = " ".join((run.error_type, run.metrics.termination_reason, run.failure_category)).casefold()
    return "contract" in text or "schema" in text
