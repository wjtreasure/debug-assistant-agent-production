from __future__ import annotations

import json
import time
from typing import Any

from debug_assistant.context.models import ContextView

from pydantic import ValidationError

from debug_assistant.contracts import compact_validation_error
from debug_assistant.harness.dynamic_budget import PromptBudgetExceeded
from debug_assistant.incidents.contracts import (
    IncidentEvidence,
    ReviewDecision,
    RootCauseCandidate,
    candidate_claim_inventory,
)
from debug_assistant.llm.base import LLMDeadlineExceeded, LLMOutputError, extract_json, estimate_tokens_char4, strict_response_format


_SYSTEM = """You are the Final Review Agent. Review only the supplied Diagnosis Agent candidate
when present, or the supplied INCONCLUSIVE diagnostic report when no candidate is present.
Use the cited Evidence, semantic relationships, source-mechanism coverage, open critical
obligations, open blocking contradictions, and explicit uncertainty in the report.
Do not re-diagnose, call tools, propose a replacement root cause, or modify the hypothesis.
Return exactly: decision (PASS or REJECT), confidence_score (0.0 to 1.0), confidence_reason,
unsupported_claims, missing_evidence,
blocking_contradictions, recoverability {recoverable, unrecoverable},
targeted_followup, causal_chain_valid, causal_gaps, gap_proposals, and reason.
For an INCONCLUSIVE report, PASS means the report is an honest, evidence-traceable
assessment; it never promotes the run to a confirmed RCA. Score the most likely explanation
only to the extent supported by the supplied evidence, and state what remains unconfirmed.
gap_proposals is an array of structured missing-fact proposals. Each item must identify an exact
claim from candidate_claim_inventory with claim_kind and unsupported_claim, supply an explicit
completion_contract.target_fact and additional_facts with source_profile_ref from source_profiles
(null means unrestricted provenance), and define one atomic
target_component, verification_question, evidence_type, and
name one or more suggested_tools from evidence_acquisition_tools. Existing canonical_questions
are immutable; do not redeclare their identity or CompletionContract. Do not create IDs, lifecycle
status, or criticality. General advice and gaps without an executable evidence path stay only in
the explanatory finding fields and must not become gap_proposals.
PASS only when the cited Evidence supports both the named component and causal mechanism and
the structured causal chain is coherent. Claim specificity and certainty must not exceed what
the cited Evidence directly establishes. Abductive diagnosis is allowed; mathematical proof is
not required. Check each claim only against its cited Evidence and supplied canonical blockers.
The supplied diagnostic_report is the frozen delivery view for this same review; check that its
summary, evidence links, uncertainty and open questions do not overstate the candidate.
Do not add a stronger prerequisite unless the candidate claim or an open critical Obligation
explicitly requires it. When a finding category is empty, return an empty array []. Do not put
"None", "None found", or explanatory prose into a finding array; put explanation in reason instead."""

def review_schema(candidate, profiles, tools):
    """Strict terminal protocol, sharing the canonical Question compiler."""
    from copy import deepcopy
    from debug_assistant.agent.decision_surface import (
        LegalActionSurface, strict_schema, validate_command, without_nulls, enum, array,
    )
    from debug_assistant.agent.question_protocol import install_question_contract, compile_question
    schema = strict_schema(ReviewDecision.model_json_schema())
    definitions = schema['$defs']
    for alias in ('contradictions', 'suggested_investigation'):
        schema['properties'].pop(alias)
    # A terminal critic must explicitly state findings and causal validity.
    # Null must never acquire the legacy causal_chain_valid=True default.
    for name, field in list(schema['properties'].items()):
        if 'anyOf' in field:
            schema['properties'][name] = next(branch for branch in field['anyOf']
                if branch.get('type') != 'null')
    schema['required'] = list(schema['properties'])
    for name, field in definitions['ReviewRecoverability']['properties'].items():
        definitions['ReviewRecoverability']['properties'][name] = {'type': 'boolean'}
    gap = install_question_contract(definitions, definitions['ReviewGapProposal'], profiles)
    claims = [claim for rows in (candidate_claim_inventory(candidate).values()
                                 if candidate is not None else ()) for claim in rows]
    gap['properties']['unsupported_claim'] = enum(claims)
    gap['properties']['suggested_tools'] = array(enum(tools), minItems=1) if tools else array({'type': 'string'}, maxItems=0)
    definitions['ReviewGapProposal'] = gap
    if not tools:
        field = schema['properties']['gap_proposals']
        for branch in field.get('anyOf', (field,)):
            if branch.get('type') == 'array':
                branch['maxItems'] = 0
    if candidate is None:
        field = schema['properties']['gap_proposals']
        for branch in field.get('anyOf', (field,)):
            if branch.get('type') == 'array':
                branch['maxItems'] = 0

    class TerminalReview:
        name = 'final_review'
        source_profiles = profiles

        @classmethod
        def model_json_schema(cls, **kwargs):
            return deepcopy(schema)

        @classmethod
        def provider_json_schema(cls):
            return LegalActionSurface(cls.name, schema, {}, {}).provider_json_schema()

        @classmethod
        def packing_json_schema(cls):
            return LegalActionSurface(cls.name, schema, {}, {}).packing_json_schema()

        @classmethod
        def model_validate(cls, raw, **kwargs):
            validate_command(schema, raw)
            compiled = deepcopy(raw)
            compiled['gap_proposals'] = [compile_question(row, profiles)
                for row in (raw.get('gap_proposals') or ())]
            for row in compiled['gap_proposals']:
                row['claim'] = row['verification_question']
                row['evidence_requirement'] = '; '.join(fact['description']
                    for fact in row['completion_contract']['required_facts'])
            return ReviewDecision.model_validate(without_nulls(compiled), **kwargs)

    return TerminalReview


class ReviewSchemaError(ValueError):
    """The bounded review-format repair could not satisfy ReviewDecision."""


_REVIEW_FINDING_FIELDS = (
    "unsupported_claims",
    "missing_evidence",
    "contradictions",
    "blocking_contradictions",
    "causal_gaps",
)


def _normalize_review_bool(value: Any, *, field: str, actions=None):
    """Normalize unambiguous Provider boolean spellings without changing meaning."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes"}:
            if actions is not None:
                actions.append(f"{field}:string_to_true")
            return True
        if normalized in {"false", "no"}:
            if actions is not None:
                actions.append(f"{field}:string_to_false")
            return False
    return value


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
    decision = str(normalized.get("decision") or "").strip().upper()
    if "blocking_contradictions" not in normalized and "contradictions" in normalized:
        normalized["blocking_contradictions"] = normalized["contradictions"]
    if "contradictions" not in normalized and "blocking_contradictions" in normalized:
        normalized["contradictions"] = normalized["blocking_contradictions"]
    if "targeted_followup" not in normalized and "suggested_investigation" in normalized:
        normalized["targeted_followup"] = normalized["suggested_investigation"]
    if "suggested_investigation" not in normalized and "targeted_followup" in normalized:
        normalized["suggested_investigation"] = normalized["targeted_followup"]
    if "recoverability" not in normalized:
        followup = normalized.get("targeted_followup") or normalized.get("suggested_investigation")
        normalized["recoverability"] = {
            "recoverable": decision != "PASS" and bool(
                normalized.get("gap_proposals")
                or normalized.get("missing_evidence")
                or normalized.get("blocking_contradictions")
                or followup
            ),
            "unrecoverable": False,
        }
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
        if value is None:
            normalized[field] = []
            if actions is not None:
                actions.append(f"{field}:null_to_empty_array")
        elif isinstance(value, str):
            normalized[field] = [value]
            if actions is not None:
                actions.append(f"{field}:scalar_to_singleton_array")
        elif isinstance(value, tuple):
            normalized[field] = list(value)
            if actions is not None:
                actions.append(f"{field}:tuple_to_array")
    for field in ("suggested_investigation", "targeted_followup", "reason"):
        value = normalized.get(field)
        if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            normalized[field] = "\n".join(value)
            if actions is not None:
                actions.append(f"{field}:string_array_to_string")
    for field in ("causal_chain_valid",):
        if field in normalized:
            normalized[field] = _normalize_review_bool(
                normalized[field], field=field, actions=actions,
            )
    recoverability = normalized.get("recoverability")
    if isinstance(recoverability, dict):
        normalized["recoverability"] = dict(recoverability)
        for field in ("recoverable", "unrecoverable"):
            if field in recoverability:
                normalized["recoverability"][field] = _normalize_review_bool(
                    recoverability[field],
                    field=f"recoverability.{field}",
                    actions=actions,
                )
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
        "blocking_contradictions": ("no blocking contradiction", "no blocking contradictions"),
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
        "blocking_contradictions": _meaningful_findings(
            decision.blocking_contradictions, category="blocking_contradictions",
        ),
    })
    if cleaned.decision != "PASS":
        return cleaned
    findings = (
        cleaned.unsupported_claims
        + cleaned.missing_evidence
        + cleaned.contradictions
        + cleaned.blocking_contradictions
    )
    if (
        not findings
        and not cleaned.gap_proposals
        and cleaned.causal_chain_valid
        and not cleaned.causal_gaps
    ):
        return cleaned
    return cleaned.model_copy(update={
        "decision": "REJECT",
        "reason": (
            "Deterministic Review consistency check rejected an internally "
            "inconsistent or causally incomplete PASS response. " + cleaned.reason
        ),
    })


class FinalReviewAgent:
    def __init__(self, llm, model: str = "", *, provider_gateway=None):
        from debug_assistant.incidents.provider_gateway import ProviderOperationalGateway
        self.llm = llm
        self.provider_gateway = provider_gateway or ProviderOperationalGateway(llm).bind(llm, "review")
        self._usage_cursor = 0
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
        self.last_symbol_normalizations: list[dict] = []
        self.last_metadata_drops: list[str] = []
        self.last_metadata_warnings: list[str] = []
        self.last_prompt_breakdown: dict[str, Any] = {}
        self.last_prompt_breakdowns: list[dict[str, Any]] = []

    @staticmethod
    def render_request(candidate, evidence, *, blocking_contradictions=(),
                       source_mechanism_context=None, open_critical_obligations=(),
                       evidence_acquisition_tools=(), source_profiles=None, context_view=None,
                       canonical_questions=(), diagnostic_report=None):
        cited = set(candidate.evidence_ids) if candidate is not None else {
            item.evidence_id for item in evidence
        }
        payload = {
            "candidate": candidate.model_dump() if candidate is not None else None,
            "candidate_claim_inventory": (
                candidate_claim_inventory(candidate) if candidate is not None else {}
            ),
            "evidence_acquisition_tools": list(evidence_acquisition_tools),
            "source_profiles": source_profiles or {},
            "cited_evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "source": item.source,
                    "target": item.target,
                    "summary": item.summary,
                    "observation_id": item.observation_id,
                    "provenance": dict(item.provenance),
                }
                for item in evidence if item.evidence_id in cited
            ],
            "open_blocking_contradictions": list(blocking_contradictions),
            "source_mechanism_context": source_mechanism_context or {},
            "open_critical_obligations": list(open_critical_obligations),
            "canonical_questions": list(canonical_questions),
            "diagnostic_report": (
                diagnostic_report.model_dump(mode="json")
                if hasattr(diagnostic_report, "model_dump") else diagnostic_report
            ),
        }
        if context_view is not None:
            if context_view.role != 'review':
                raise ValueError('Review requires its own ContextView')
            payload.update(context_view.reference_metadata)
            payload['diagnostic_context'] = context_view.text
        return _SYSTEM, json.dumps(payload, ensure_ascii=False)

    @classmethod
    def render_envelope(cls, candidate, evidence, context_view, **kwargs):
        system, user = cls.render_request(candidate, evidence, context_view=context_view, **kwargs)
        schema = review_schema(candidate, kwargs.get('source_profiles') or {}, kwargs.get('evidence_acquisition_tools', ()))
        return system, user + '\n\n' + json.dumps(strict_response_format(schema, for_packing=True), ensure_ascii=False)

    def review(self, candidate: RootCauseCandidate | None, evidence: tuple[IncidentEvidence, ...],
               *, blocking_contradictions: tuple[dict, ...] = (),
               source_mechanism_context: dict | None = None,
               open_critical_obligations: tuple[dict, ...] = (),
               logical_timeout_seconds: float | None = None,
               on_attempt_started=None, prompt_budget=None,
               max_output_tokens: int | None = None,
               repair_max_output_tokens: int | None = None,
               allow_terminal_reserve: bool = False,
               evidence_acquisition_tools: tuple[str, ...] = (),
               source_profiles: dict | None = None,
               context_view: ContextView | None = None,
               canonical_questions: tuple[dict, ...] = (),
               diagnostic_report=None) -> ReviewDecision:
        system, user = self.render_request(candidate, evidence,
            blocking_contradictions=blocking_contradictions, source_mechanism_context=source_mechanism_context,
            open_critical_obligations=open_critical_obligations, evidence_acquisition_tools=evidence_acquisition_tools,
            source_profiles=source_profiles, context_view=context_view, canonical_questions=canonical_questions,
            diagnostic_report=diagnostic_report)
        self.last_schema_repaired = False
        self.last_repair_attempted = False
        self.last_deterministic_normalized = False
        self.last_schema_error = ""
        self.last_usage = {}
        self.last_call_count = 0
        self._usage_cursor = len(self.provider_gateway.records)
        self.last_attempts = []
        self.last_failure_type = ""
        self.last_normalization_actions = []
        self.last_symbol_normalizations = []
        symbol_context = {"normalization_events": self.last_symbol_normalizations}
        self.last_metadata_drops = []
        self.last_metadata_warnings = []
        self.last_prompt_breakdowns = []

        from debug_assistant.agent.decision_surface import DecisionSurfaceError
        schema = review_schema(candidate, source_profiles or {}, evidence_acquisition_tools)
        deadline = None if logical_timeout_seconds is None else time.monotonic() + logical_timeout_seconds
        error = None
        for index in range(2):
            stage = 'first_pass' if index == 0 else 'schema_repair'
            prompt = user if error is None else user + '\nCONTRACT_REJECTION:\n' + self.last_schema_error
            timeout = None if deadline is None else deadline - time.monotonic()
            if timeout is not None and timeout <= 0:
                raise LLMDeadlineExceeded('Final Review logical timeout exhausted')
            output_tokens = repair_max_output_tokens if index else max_output_tokens
            self.last_prompt_breakdown = {'prompt_components': {
                'system_tokens': estimate_tokens_char4(_SYSTEM), 'tool_schema_tokens': 0,
                'control_state_tokens': 0, 'diagnostic_context_tokens': estimate_tokens_char4(prompt)}}
            if prompt_budget is not None:
                decision = prompt_budget.check_prompt('review', _SYSTEM, prompt,
                    breakdown={'evidence': prompt}, response_format=strict_response_format(schema),
                    completion_reserve_tokens=output_tokens, allow_terminal_reserve=allow_terminal_reserve)
                self.last_prompt_breakdown.update(estimated_prompt_tokens=decision.estimated_prompt_tokens,
                    input_hard_capacity=decision.input_hard_capacity, budget_state=decision.state.value,
                    token_breakdown=dict(decision.breakdown))
            self.last_prompt_breakdowns.append(dict(self.last_prompt_breakdown))
            self.last_call_count += 1
            attempt = {'stage': stage, 'call_index': self.last_call_count,
                'schema_repair': bool(index), 'effective_timeout': timeout,
                'elapsed_seconds': 0.0, 'status': 'started', 'failure_type': ''}
            self.last_attempts.append(attempt)
            if on_attempt_started is not None:
                on_attempt_started(dict(attempt))
            started = time.monotonic()
            try:
                response = self.provider_gateway.complete_structured(_SYSTEM, prompt, schema=schema,
                    request_kind='review_schema_repair' if index else 'review',
                    contract_retry_reason='review_schema_invalid' if index else '',
                    model=self.model or None, logical_timeout_seconds=timeout,
                    on_attempt_started=on_attempt_started, max_output_tokens=output_tokens,
                    enable_thinking=False)
                raw = getattr(response, 'structured', None)
                if raw is None:
                    raw = extract_json(getattr(response, 'content', response))
                result = schema.model_validate(raw, context=symbol_context)
                attempt['status'] = 'completed'
                self.last_schema_repaired = bool(index)
                return result
            except (ValidationError, DecisionSurfaceError, LLMOutputError) as exc:
                error = exc
                self.last_schema_error = str(exc)
                self.last_repair_attempted = index == 0 or self.last_repair_attempted
                attempt.update(status='schema_invalid', failure_type='schema_validation')
            except BaseException as exc:
                self.last_failure_type = type(exc).__name__
                attempt.update(status='failed', failure_type=type(exc).__name__)
                raise
            finally:
                attempt['elapsed_seconds'] = time.monotonic() - started
                self._add_usage()
        self.last_failure_type = 'review_contract_exhausted'
        raise ReviewSchemaError('Final Review remained invalid after one correction: '
            + self.last_schema_error) from error

    def _add_usage(self) -> None:
        from debug_assistant.incidents.provider_gateway import aggregate_provider_usage
        records = self.provider_gateway.records
        usage = aggregate_provider_usage(records[self._usage_cursor:])
        for key, value in usage.items():
            self.last_usage[key] = self.last_usage.get(key, 0) + value
        self._usage_cursor = len(records)
