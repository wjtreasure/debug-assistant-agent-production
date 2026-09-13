from __future__ import annotations

import json
import time
from typing import Any

from pydantic import ValidationError

from debug_assistant.contracts import compact_validation_error
from debug_assistant.incidents.contracts import IncidentEvidence, ReviewDecision, RootCauseCandidate
from debug_assistant.llm.base import LLMDeadlineExceeded, complete_json_compat


_SYSTEM = """You are the Final Review Agent. Review only the supplied Diagnosis Agent candidate,
claim-evidence mapping, causal chain summary, cited Evidence, source-mechanism coverage, open
critical obligations, and open blocking contradictions.
Do not re-diagnose, call tools, propose a replacement root cause, or modify the hypothesis.
Return exactly: decision (PASS or REJECT), unsupported_claims, missing_evidence,
contradictions, causal_chain_valid, causal_gaps, suggested_investigation, and reason.
PASS only when the cited Evidence supports both the named component and causal mechanism and
the structured causal chain is coherent. Claim specificity and certainty must not exceed what
the cited Evidence directly establishes: do not turn a symptom into an implementation cause,
and do not add an unobserved numeric relationship, burst, load spike, or dependency effect.
For example, a direct CFS-throttling observation does not by itself prove that aggregate usage
exceeded the configured CPU limit. Abductive diagnosis is allowed; mathematical proof is not
required. If application source is declared and available but source-mechanism coverage is
unknown or gap, or a critical source obligation is open, reject a candidate that has not cited
bounded source Evidence resolving that causal mechanism; report the missing fact in
missing_evidence or causal_gaps. Do not demand source Evidence when coverage is not_applicable
or the mechanism is already source-backed. When a finding category is empty, return an empty array []. Do not put
"None", "None found", or explanatory prose into a finding array; put explanation in reason instead."""

_REPAIR_SYSTEM = """Repair only the JSON shape of a Final Review Agent response.
Do not re-diagnose, add evidence, change the substantive decision, or propose a replacement root cause.
Return exactly these fields and types: decision string (PASS or REJECT); unsupported_claims,
missing_evidence, contradictions, and causal_gaps as arrays of strings; causal_chain_valid as a boolean;
suggested_investigation and reason as strings.
When a finding category has no finding, use an empty array []; put any explanation in reason.
This is the only repair attempt."""


class ReviewSchemaError(ValueError):
    """The bounded review-format repair could not satisfy ReviewDecision."""


_REVIEW_FINDING_FIELDS = (
    "unsupported_claims",
    "missing_evidence",
    "contradictions",
    "causal_gaps",
)


def _normalize_review_payload(data, *, actions=None, drops=None, warnings=None):
    """Apply only shape-preserving, deterministic Review normalizations.

    Compatible Providers frequently return one finding as a scalar string or
    a list where the contract asks for a tuple of strings.  Those conversions
    do not change the finding's meaning and should not spend a second LLM call.
    Anything less certain is left for Pydantic and the bounded repair call.
    """
    if not isinstance(data, dict):
        return data
    normalized = dict(data)
    known_fields = set(ReviewDecision.model_fields)
    extra_fields = sorted(set(normalized) - known_fields)
    for field in extra_fields:
        normalized.pop(field, None)
    if extra_fields:
        if actions is not None:
            actions.append("top_level:ignored_extra_fields:" + ",".join(extra_fields))
        if drops is not None:
            drops.extend(f"top_level.{field}:unknown_field" for field in extra_fields)
        if warnings is not None:
            warnings.append("top_level:ignored_unknown_fields:" + ",".join(extra_fields))
    for field in _REVIEW_FINDING_FIELDS:
        value = normalized.get(field)
        if isinstance(value, str):
            normalized[field] = [value]
            if actions is not None:
                actions.append(f"{field}:scalar_to_singleton_array")
        elif isinstance(value, tuple):
            normalized[field] = list(value)
            if actions is not None:
                actions.append(f"{field}:tuple_to_array")
    for field in ("suggested_investigation", "reason"):
        value = normalized.get(field)
        if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            normalized[field] = "\n".join(value)
            if actions is not None:
                actions.append(f"{field}:string_array_to_string")
    return normalized


def _meaningful_findings(values, *, category: str) -> tuple[str, ...]:
    """Normalize common model no-finding sentinels in structured arrays.

    ReviewDecision deliberately represents findings as arrays. Some providers still
    emit prose such as ``"None. The cited evidence supports ..."`` in those arrays.
    Treat only an unambiguous leading no-finding statement as empty; preserve every
    other string so the deterministic PASS consistency check remains fail-closed.
    """
    no_finding_prefixes = {
        "unsupported_claims": ("no unsupported claim", "no unsupported claims"),
        "missing_evidence": ("no missing evidence", "no additional evidence"),
        "contradictions": ("no contradiction", "no contradictions"),
    }[category]
    meaningful: list[str] = []
    for value in values:
        normalized = " ".join(str(value).split()).strip()
        lowered = normalized.casefold()
        no_finding = lowered in {"none", "none.", "none:", "none;"}
        no_finding = no_finding or lowered.startswith(("none. ", "none: ", "none; ", "none - ", "none — "))
        if lowered.startswith("none found"):
            suffix = lowered[len("none found"):]
            no_finding = suffix == "" or suffix[0] in " .,:;!?-—"
        if not no_finding:
            no_finding = any(
                lowered == prefix
                or (lowered.startswith(prefix) and lowered[len(prefix)] in " .,:;!?-—")
                for prefix in no_finding_prefixes
            )
        if no_finding:
            continue
        if normalized:
            meaningful.append(str(value))
    return tuple(meaningful)


def enforce_review_consistency(decision: ReviewDecision) -> ReviewDecision:
    """Fail closed when a PASS response also reports unsupported material.

    The Review model may explain that a gap is optional in prose, but the
    structured contract already has an unambiguous representation: PASS has no
    unsupported claims, missing evidence, or contradictions. The Harness, not
    the model, owns this terminal-state invariant.
    """
    cleaned = decision.model_copy(update={
        "unsupported_claims": _meaningful_findings(
            decision.unsupported_claims, category="unsupported_claims",
        ),
        "missing_evidence": _meaningful_findings(
            decision.missing_evidence, category="missing_evidence",
        ),
        "contradictions": _meaningful_findings(
            decision.contradictions, category="contradictions",
        ),
    })
    if cleaned.decision != "PASS":
        return cleaned
    findings = (
        cleaned.unsupported_claims
        + cleaned.missing_evidence
        + cleaned.contradictions
    )
    if not findings and cleaned.causal_chain_valid and not cleaned.causal_gaps:
        return cleaned
    return cleaned.model_copy(update={
        "decision": "REJECT",
        "reason": (
            "Deterministic Review consistency check rejected an internally "
            "inconsistent or causally incomplete PASS response. " + cleaned.reason
        ),
    })


class FinalReviewAgent:
    def __init__(self, llm, model: str = ""):
        self.llm = llm
        self.model = model
        self.last_usage = {}
        self.last_schema_repaired = False
        self.last_repair_attempted = False
        self.last_deterministic_normalized = False
        self.last_schema_error = ""
        self.last_call_count = 0
        self.last_attempts: list[dict] = []
        self.last_failure_type = ""
        self.last_normalization_actions: list[str] = []
        self.last_metadata_drops: list[str] = []
        self.last_metadata_warnings: list[str] = []
        self.last_prompt_breakdown: dict[str, Any] = {}
        self.last_prompt_breakdowns: list[dict[str, Any]] = []

    def review(self, candidate: RootCauseCandidate, evidence: tuple[IncidentEvidence, ...],
               *, blocking_contradictions: tuple[dict, ...] = (),
               source_mechanism_context: dict | None = None,
               open_critical_obligations: tuple[dict, ...] = (),
               logical_timeout_seconds: float | None = None,
               on_attempt_started=None, prompt_budget=None) -> ReviewDecision:
        cited = set(candidate.evidence_ids)
        payload = {
            "candidate": candidate.model_dump(),
            "claim_evidence_mapping": [item.model_dump() for item in candidate.claim_evidence_mapping],
            "causal_chain_summary": [item.model_dump() for item in candidate.causal_chain_summary],
            "cited_evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "source": item.source,
                    "target": item.target,
                    "summary": item.summary,
                }
                for item in evidence if item.evidence_id in cited
            ],
            "open_blocking_contradictions": list(blocking_contradictions),
            "source_mechanism_context": source_mechanism_context or {},
            "open_critical_obligations": list(open_critical_obligations),
        }
        self.last_schema_repaired = False
        self.last_repair_attempted = False
        self.last_deterministic_normalized = False
        self.last_schema_error = ""
        self.last_usage = {}
        self.last_call_count = 0
        self.last_attempts = []
        self.last_failure_type = ""
        self.last_normalization_actions = []
        self.last_metadata_drops = []
        self.last_metadata_warnings = []
        self.last_prompt_breakdowns = []

        logical_deadline = None
        if logical_timeout_seconds is not None:
            logical_deadline = time.monotonic() + max(0.0, float(logical_timeout_seconds))

        def remaining_timeout() -> float | None:
            if logical_deadline is None:
                return None
            return max(0.0, logical_deadline - time.monotonic())

        def call_stage(stage: str, system: str, user: str):
            if prompt_budget is not None:
                budget_decision = prompt_budget.check_prompt(
                    "review", system, user,
                    breakdown={"evidence": user},
                )
                self.last_prompt_breakdown = {
                    "estimated_prompt_tokens": budget_decision.estimated_prompt_tokens,
                    "input_hard_capacity": budget_decision.input_hard_capacity,
                    "budget_state": budget_decision.state.value,
                    "token_breakdown": dict(budget_decision.breakdown),
                }
                self.last_prompt_breakdowns.append(dict(self.last_prompt_breakdown))
            timeout = remaining_timeout()
            if timeout is not None and timeout <= 0:
                raise LLMDeadlineExceeded(
                    f"Final Review logical timeout exhausted before {stage}"
                )
            self.last_call_count += 1
            attempt = {
                "stage": stage,
                "call_index": self.last_call_count,
                "schema_repair": stage == "schema_repair",
                "effective_timeout": timeout,
                "elapsed_seconds": 0.0,
                "status": "started",
                "failure_type": "",
            }
            self.last_attempts.append(attempt)
            if on_attempt_started is not None:
                on_attempt_started(dict(attempt))
            started = time.monotonic()
            try:
                result = complete_json_compat(
                    self.llm, system, user,
                    model=self.model or None,
                    logical_timeout_seconds=timeout,
                    on_attempt_started=on_attempt_started,
                )
            except BaseException as exc:
                attempt.update({
                    "elapsed_seconds": time.monotonic() - started,
                    "status": "failed",
                    "failure_type": type(exc).__name__,
                })
                self.last_failure_type = type(exc).__name__
                raise
            attempt.update({
                "elapsed_seconds": time.monotonic() - started,
                "status": "completed",
            })
            return result

        raw = call_stage(
            "first_pass", _SYSTEM, json.dumps(payload, ensure_ascii=False),
        )
        self._add_usage()
        normalized = _normalize_review_payload(
            raw,
            actions=self.last_normalization_actions,
            drops=self.last_metadata_drops,
            warnings=self.last_metadata_warnings,
        )
        self.last_deterministic_normalized = normalized != raw
        try:
            return ReviewDecision.model_validate(normalized)
        except ValidationError as first_error:
            self.last_attempts[-1].update({
                "status": "schema_invalid",
                "failure_type": "schema_validation",
            })
            self.last_schema_error = "validation error: " + json.dumps(
                compact_validation_error(first_error), ensure_ascii=False,
            )
            repair_payload = {
                "invalid_response": normalized,
                "validation_errors": first_error.errors(include_url=False),
            }
            self.last_repair_attempted = True
            raw_repaired = call_stage(
                "schema_repair", _REPAIR_SYSTEM,
                json.dumps(repair_payload, ensure_ascii=False),
            )
            self._add_usage()
            repaired = _normalize_review_payload(
                raw_repaired,
                actions=self.last_normalization_actions,
                drops=self.last_metadata_drops,
                warnings=self.last_metadata_warnings,
            )
            try:
                decision = ReviewDecision.model_validate(repaired)
            except ValidationError as second_error:
                self.last_attempts[-1].update({
                    "status": "schema_invalid",
                    "failure_type": "schema_validation",
                })
                self.last_schema_error += "\nRepair validation error: " + json.dumps(
                    compact_validation_error(second_error), ensure_ascii=False,
                )
                raise ReviewSchemaError("Final Review schema repair failed") from second_error
            self.last_deterministic_normalized = (
                self.last_deterministic_normalized or repaired != raw_repaired
            )
            self.last_schema_repaired = True
            return decision

    def _add_usage(self) -> None:
        usage = dict(getattr(self.llm, "last_usage", {}) or {})
        prompt_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        completion_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        self.last_usage["prompt_tokens"] = int(self.last_usage.get("prompt_tokens", 0) or 0) + prompt_tokens
        self.last_usage["completion_tokens"] = int(self.last_usage.get("completion_tokens", 0) or 0) + completion_tokens
