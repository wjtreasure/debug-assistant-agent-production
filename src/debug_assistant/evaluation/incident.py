from __future__ import annotations

import json
from pathlib import Path
import re

from pydantic import BaseModel, ConfigDict

from debug_assistant.incidents.contracts import IncidentRunResult
from debug_assistant.skills.catalog import INCIDENT_SKILLS


class CloudOpsGroundTruth(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str
    component: str
    fault: str
    component_aliases: tuple[str, ...] = ()
    fault_aliases: tuple[str, ...] = ()
    mechanism: str | None = None
    mechanism_aliases: tuple[str, ...] = ()


class EvidenceRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source: str
    contains: tuple[str, ...]


class CloudOpsEvaluatorData(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ground_truth: CloudOpsGroundTruth
    key_evidence: tuple[EvidenceRequirement, ...]


class CloudOpsEvaluatorLoader:
    """Evaluator-only loader. This module is not imported by IncidentHarness."""

    def load(self, evaluator_dir: str | Path) -> CloudOpsEvaluatorData:
        root = Path(evaluator_dir).resolve()
        truth = json.loads((root / "ground_truth.json").read_text(encoding="utf-8"))
        evidence = json.loads((root / "key_evidence.json").read_text(encoding="utf-8"))
        return CloudOpsEvaluatorData(ground_truth=truth, key_evidence=evidence)


class IncidentResultQuality(BaseModel):
    """Outcome metrics, kept separate from investigation process metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root_cause_correct: bool
    component_correct: bool
    fault_correct: bool
    mechanism_evaluated: bool
    mechanism_correct: bool | None = None
    evidence_supported: bool
    key_evidence_coverage: float


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


class IncidentCostReliability(BaseModel):
    """Measured run cost; zero means the run did not expose that measurement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    planner_calls: int
    review_calls: int
    llm_calls: int
    tool_calls: int
    steps: int
    tokens: int
    wall_clock_seconds: float
    terminal_status: str
    failure_category: str


class IncidentEvalResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_status: str
    candidate_present: bool
    candidate_accepted: bool
    component_correct: bool
    fault_correct: bool
    mechanism_evaluated: bool
    mechanism_correct: bool | None = None
    root_cause_correct: bool
    root_cause_fault_correct: bool
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
    result_quality: IncidentResultQuality
    process_quality: IncidentProcessQuality
    cost_reliability: IncidentCostReliability


class IncidentEvaluator:
    """Minimal evaluator; receives evaluator-only data after the Agent run ends."""

    def evaluate(self, run: IncidentRunResult, expected: CloudOpsEvaluatorData) -> IncidentEvalResult:
        truth = expected.ground_truth
        observed = [
            (item.source, normalize_evidence_text(item.summary)) for item in run.evidence
        ]
        covered = 0
        for requirement in expected.key_evidence:
            if any(source == requirement.source and all(
                    normalize_evidence_text(token) in text for token in requirement.contains
                   )
                   for source, text in observed):
                covered += 1
        fingerprints = [
            str(action.get("fingerprint"))
            for action in run.actions
            if action.get("tool") != "finalize_diagnosis" and action.get("fingerprint")
        ]
        duplicate_tool_calls_executed = len(fingerprints) - len(set(fingerprints))
        duplicate_action_requests = _duplicate_action_requests(run, duplicate_tool_calls_executed)
        suggested = set().union(*(set(skill.suggested_tools) for skill in INCIDENT_SKILLS.values()))
        irrelevant = sum(1 for action in run.actions
                         if action.get("tool") not in suggested and action.get("tool") != "finalize_diagnosis")
        accepted_components = {
            canonicalize_component(truth.component),
            *(canonicalize_component(value) for value in truth.component_aliases),
        }
        candidate_present = run.candidate is not None
        candidate_accepted = run.status == "PASS" and candidate_present
        accepted_candidate = run.candidate if candidate_accepted else None
        candidate_component = canonicalize_component(accepted_candidate.component) if accepted_candidate else ""
        candidate_fault = canonicalize_fault(
            accepted_candidate.fault if accepted_candidate else "", truth.fault, truth.fault_aliases,
        )
        candidate = accepted_candidate
        cited_ids = set(candidate.evidence_ids) if candidate else set()
        known_ids = {item.evidence_id for item in run.evidence}
        evidence_supported = bool(
            candidate
            and len(cited_ids) >= 2
            and all(evidence_id.startswith("ev-") for evidence_id in cited_ids)
            and cited_ids.issubset(known_ids)
        )
        mechanism_evaluated = _mechanism_is_deterministically_evaluable(truth)
        mechanism_correct = None
        if mechanism_evaluated:
            mechanism_correct = bool(
                candidate
                and canonicalize_mechanism(
                    candidate.mechanism, truth.mechanism or "", truth.mechanism_aliases,
                ) == canonicalize_free_text(truth.mechanism or "")
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
            # A Review stage can be entered without producing a decision (for
            # example, when the first pass or schema repair times out).  This
            # is observably different from never entering Review.
            review_outcome = (
                "TIMEOUT"
                if run.error_type == "LLMDeadlineExceeded"
                or "timeout" in run.metrics.termination_reason.casefold()
                or "deadline" in run.metrics.termination_reason.casefold()
                else "ATTEMPTED_NO_DECISION"
            )
        else:
            review_outcome = "NOT_RUN"
        component_correct = bool(candidate and candidate_component in accepted_components)
        fault_correct = bool(candidate and candidate_fault == _normalize(truth.fault))
        root_cause_correct = bool(
            candidate and candidate_component in accepted_components
            and fault_correct
            and evidence_supported
            and (mechanism_correct is not False)
        )
        timeout = _run_timed_out(run)
        contract_failure = _run_contract_failed(run)
        return IncidentEvalResult(
            run_status=run.status,
            candidate_present=candidate_present,
            candidate_accepted=candidate_accepted,
            component_correct=component_correct,
            fault_correct=fault_correct,
            mechanism_evaluated=mechanism_evaluated,
            mechanism_correct=mechanism_correct,
            root_cause_correct=root_cause_correct,
            # Compatibility alias retained for existing reports.
            root_cause_fault_correct=fault_correct,
            key_evidence_coverage=covered / len(expected.key_evidence) if expected.key_evidence else 1.0,
            # Compatibility field now has the explicit executed-call meaning.
            repeated_tool_calls=duplicate_tool_calls_executed,
            duplicate_action_requests=duplicate_action_requests,
            duplicate_tool_calls_executed=duplicate_tool_calls_executed,
            irrelevant_tool_calls=irrelevant,
            review_result=review_outcome,
            timeout=timeout,
            schema_repair_count=run.metrics.schema_repair_count,
            contract_failure=contract_failure,
            inconclusive=run.status == "INCONCLUSIVE",
            steps=run.metrics.steps, tool_calls=run.metrics.tool_calls,
            tokens=run.metrics.total_tokens,
            result_quality=IncidentResultQuality(
                root_cause_correct=root_cause_correct,
                component_correct=component_correct,
                fault_correct=fault_correct,
                mechanism_evaluated=mechanism_evaluated,
                mechanism_correct=mechanism_correct,
                evidence_supported=evidence_supported,
                key_evidence_coverage=covered / len(expected.key_evidence) if expected.key_evidence else 1.0,
            ),
            process_quality=IncidentProcessQuality(
                skill_path=skill_path,
                useful_tool_calls=max(0, len(investigation_actions) - irrelevant),
                irrelevant_tool_calls=irrelevant,
                repeated_tool_calls=duplicate_tool_calls_executed,
                duplicate_action_requests=duplicate_action_requests,
                duplicate_tool_calls_executed=duplicate_tool_calls_executed,
                evidence_coverage=covered / len(expected.key_evidence) if expected.key_evidence else 1.0,
                no_progress_termination="no_progress" in run.metrics.termination_reason,
                review_outcome=review_outcome,
                candidate_present=candidate_present,
            ),
            cost_reliability=IncidentCostReliability(
                planner_calls=run.metrics.planner_calls,
                review_calls=run.metrics.review_calls,
                llm_calls=run.metrics.llm_calls,
                tool_calls=run.metrics.tool_calls,
                steps=run.metrics.steps,
                tokens=run.metrics.total_tokens,
                wall_clock_seconds=run.metrics.wall_clock_seconds,
                terminal_status=run.status,
                failure_category=run.failure_category,
            ),
        )


def _normalize(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())).strip("_")


def canonicalize_free_text(value: str) -> str:
    """Normalize only surface form; do not infer domain semantics."""
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


def canonicalize_fault(value: str, canonical: str, aliases: tuple[str, ...]) -> str:
    """Map only explicit taxonomy phrases to a canonical label.

    The evaluator deliberately has no semantic similarity fallback.  Matching a
    phrase inside a longer candidate is allowed only because the phrase itself is
    an approved taxonomy alias; unrelated text returns ``UNKNOWN``.
    """
    return FaultCanonicalizer().canonicalize(value, canonical=canonical, aliases=aliases)


def _normalize_fault_phrase(value: str) -> str:
    tokens = [token for token in _normalize(value).split("_") if token not in {"set"}]
    return "_".join(tokens)


UNKNOWN_FAULT = "UNKNOWN"


class FaultCanonicalizer:
    """Deterministic exact/approved-alias fault canonicalizer."""

    def canonicalize(self, value: str, *, canonical: str, aliases: tuple[str, ...] = ()) -> str:
        normalized = _normalize_fault_phrase(value)
        canonical_normalized = _normalize(canonical)
        if not normalized or not canonical_normalized:
            return UNKNOWN_FAULT
        approved = set(aliases)
        phrases = {
            canonical_normalized,
            *(_normalize_fault_phrase(alias) for alias in approved),
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
    # A structured canonical mechanism is a stable identifier. Open natural
    # language is evaluable only when the fixture supplies explicit aliases.
    structured = bool(re.fullmatch(r"[a-z][a-z0-9_]*", value))
    return structured or bool(truth.mechanism_aliases)


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
        if event_type == "ACTION_REJECTED" and (
            "duplicate" in reason or "repeated" in reason
        ):
            count += 1
        elif event_type == "PROGRESS_GUARD_REJECTED" and (
            "duplicate" in reason or "repeat" in reason
        ):
            count += 1
    return count


def _run_timed_out(run: IncidentRunResult) -> bool:
    text = " ".join((run.error_type, run.metrics.termination_reason, run.failure_category)).casefold()
    return any(token in text for token in ("timeout", "deadline", "timed_out"))


def _run_contract_failed(run: IncidentRunResult) -> bool:
    text = " ".join((run.error_type, run.metrics.termination_reason, run.failure_category)).casefold()
    return "contract" in text or "schema" in text
