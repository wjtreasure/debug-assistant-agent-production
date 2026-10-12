from __future__ import annotations

import copy
import json
import time
from debug_assistant.context.models import ContextView, PromptSnapshot, redact_snapshot_value
from pydantic import ValidationError
from debug_assistant.incidents.semantic_transition import (
    constrain_disposition_schema, disposition_targets, validate_disposition_targets,
    validate_required_dispositions,
)
from debug_assistant.llm.base import (
    LLMDeadlineExceeded, LLMOutputError, extract_json, estimate_tokens_char4, strict_response_format,
)
from debug_assistant.incidents.contracts import (
    ReflectionFeedback,
)


class ReflectionContractExhausted(ValueError):
    """Incident Reflection schema remained invalid after one bounded repair."""


def reflection_schema(snapshot):
    """The same Question contract/profile compiler used by normal decisions."""
    from debug_assistant.agent.decision_surface import (
        strict_schema, LegalActionSurface, enum, validate_command, without_nulls,
        validate_reference_selection,
    )
    from debug_assistant.agent.question_protocol import install_question_contract, compile_question
    from debug_assistant.incidents.semantic_transition import contradiction_reference_ids
    projection = snapshot.get('rca_state', {})
    obligations = projection.get('obligations', snapshot.get('verification_obligations', ()))
    targets = disposition_targets(obligations)
    required = tuple(row['id'] for row in obligations
                     if row.get('critical', True) and row['id'] in targets)
    profiles = snapshot.get('source_profiles', {})
    schema = strict_schema(ReflectionFeedback.model_json_schema())
    constrain_disposition_schema(schema, targets, required)
    definitions = schema['$defs']
    definitions['ReflectionObligationProposal'] = install_question_contract(
        definitions, definitions['ReflectionObligationProposal'], profiles)
    # Suggested executables are literal canonical Registry names, never free aliases.
    tools = snapshot.get('available_tools', ())
    field = definitions['ReflectionObligationProposal']['properties']['suggested_tools']
    for branch in field.get('anyOf', (field,)):
        if branch.get('type') == 'array':
            branch['items'] = enum(tools) if tools else {'type': 'string'}
            branch.update(maxItems=len(tools), uniqueItems=True)
    contradictions = contradiction_reference_ids(snapshot)
    for name in ('ReflectionContradictionUpdate', 'ReflectionContradictionReview'):
        definitions[name]['properties']['evidence_id'] = enum(contradictions) if contradictions else {'type': 'string'}
    for name in ('contradiction_updates', 'contradiction_reviews', 'contradicting_evidence_ids'):
        for branch in schema['properties'][name].get('anyOf', (schema['properties'][name],)):
            if branch.get('type') == 'array':
                branch['maxItems'] = len(contradictions)
                if name == 'contradicting_evidence_ids' and contradictions:
                    branch.update(items=enum(contradictions), uniqueItems=True)

    class TurnFeedback:
        name = 'incident_reflection'
        source_profiles = profiles

        @classmethod
        def model_json_schema(cls, **kwargs):
            return copy.deepcopy(schema)

        @classmethod
        def provider_json_schema(cls):
            return LegalActionSurface(cls.name, schema, {}, {}).provider_json_schema()

        @classmethod
        def packing_json_schema(cls):
            return LegalActionSurface(cls.name, schema, {}, {}).packing_json_schema()

        @classmethod
        def model_validate(cls, raw, **kwargs):
            validate_command(schema, raw)
            compiled = copy.deepcopy(raw)
            # source_profile_ref:null is a declared semantic choice. Compile
            # Questions before erasing nullable optional transport fields.
            compiled['proposed_obligations'] = [compile_question(question, profiles)
                for question in (raw.get('proposed_obligations') or ())]
            result = without_nulls(compiled)
            for name in ('contradiction_updates', 'contradiction_reviews'):
                validate_reference_selection(result.get(name, []), 'evidence_id', [name])
            for question in result['proposed_obligations']:
                question['claim'] = question['verification_question']
                question['evidence_requirement'] = '; '.join(
                    fact['description'] for fact in question['completion_contract']['required_facts'])
            feedback = ReflectionFeedback.model_validate(result, **kwargs)
            validate_disposition_targets(feedback.hypothesis_delta.transition, targets)
            if feedback.hypothesis_delta.transition is not None:
                validate_required_dispositions(feedback.hypothesis_delta.transition, required)
            return feedback

    return TurnFeedback


class IncidentReflectionAgent:
    """Triggered, tool-less semantic reflection for the Incident runtime.

    The agent receives a compact structured snapshot and returns feedback only.
    DiagnosisHarness validates Evidence IDs and decides how that feedback changes
    control flow; this class never executes a Tool or mutates a Hypothesis.
    """

    _SYSTEM = """You are the Incident Reflection Agent. Review the compact structured
    diagnosis state supplied by the Harness. Do not call tools, access ground truth,
    invent evidence. Return only structured feedback:
    identify supported/unsupported claims, remaining gaps, obligation and contradiction
    reviews, and emit a structured hypothesis_delta plus proposed_evidence_gaps,
    proposed_obligations, contradiction_updates, and next_action_constraint when
    appropriate. Hypothesis changes require transition INITIAL/REFINE/REPLACE; explicit
    RETAIN/SUPERSEDE dispositions select only OPEN / BLOCKED_BY_CAPABILITY targets from the schema enum.
    Completed Obligations need no disposition; use [] when no legal target remains.
    New follow-up questions do not need a disposition on an already-completed question. Runtime
    allocates Hypothesis identity/revision and new Obligation IDs after validation. All cited Evidence IDs must
    be existing ev-* IDs from the input. The selected_semantic_projection owns domain
    strategy. Use feasible_skill_cards for an optional skill_reselection proposal;
    return its exact canonical skill_id or an empty string to retain the current Skill.
    A new Skill's full semantic graph becomes visible in the next Planner turn.
    Typed proposed_obligations use completion_contract.target_fact and additional_facts.
    Select source_profile_ref from source_profiles; null declares no provenance restriction.
    Existing questions are immutable canonical objects; reference them in lifecycle dispositions
    rather than recreating their identity. claim-only gaps remain audit findings.
    Canonical coverage and lifecycle remain Runtime-owned. Do not return raw chain-of-thought
    or the complete action history."""

    def __init__(self, llm, model: str = "", *, provider_gateway=None):
        from debug_assistant.incidents.provider_gateway import ProviderOperationalGateway
        self.llm = llm
        self.provider_gateway = provider_gateway or ProviderOperationalGateway(llm).bind(llm, "reflection")
        self._usage_cursor = 0
        self.model = model
        self.last_usage: dict = {}
        self.last_call_count = 0
        self.last_schema_repaired = False
        self.last_schema_error = ""
        self.last_failure_type = ""
        self.last_normalization_actions: list[str] = []
        self.last_symbol_normalizations: list[dict] = []
        self.last_metadata_drops: list[str] = []
        self.last_metadata_warnings: list[str] = []
        self.last_prompt_breakdown: dict = {}
        self.last_prompt_breakdowns: list[dict] = []
        self.last_prompt_snapshots: list[PromptSnapshot] = []

    @classmethod
    def render_request(cls, snapshot: dict, context_view: ContextView | None = None):
        payload = dict(snapshot)
        if context_view is not None:
            if context_view.role != "reflection":
                raise ValueError("Reflection requires its own ContextView")
            payload.update(context_view.reference_metadata)
            payload["diagnostic_context"] = context_view.text
        elif payload.get("visible_observations"):
            raise TypeError("Reflection Observation bodies require ContextManager ContextView")
        user = ("STRUCTURED_REFLECTION_INPUT:\n"
                + json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
        return cls._SYSTEM, user

    @classmethod
    def render_envelope(cls, snapshot, context_view):
        system, user = cls.render_request(snapshot, context_view)
        return system, user + "\n\n" + json.dumps(strict_response_format(reflection_schema(snapshot), for_packing=True), ensure_ascii=False)

    def reflect(self, snapshot: dict, *, context_view: ContextView | None = None, logical_timeout_seconds: float | None = None,
                on_attempt_started=None, prompt_budget=None,
                max_output_tokens: int | None = None) -> ReflectionFeedback:
        from debug_assistant.agent.decision_surface import DecisionSurfaceError
        from debug_assistant.incidents.semantic_transition import SemanticTransitionError
        self.last_usage = {}
        self._usage_cursor = len(self.provider_gateway.records)
        self.last_call_count = 0
        self.last_schema_repaired = False
        self.last_schema_error = self.last_failure_type = ''
        self.last_normalization_actions = self.last_symbol_normalizations = []
        self.last_metadata_drops = self.last_metadata_warnings = []
        self.last_prompt_breakdowns = []
        self.last_prompt_snapshots = []
        schema = reflection_schema(snapshot)
        system, user = self.render_request(snapshot, context_view)
        deadline = None if logical_timeout_seconds is None else time.monotonic() + logical_timeout_seconds
        error = None
        # Same one-shot contract correction allowance; preserve the original state/view.
        for attempt in range(2):
            prompt = user if error is None else user + '\nCONTRACT_REJECTION:\n' + self.last_schema_error
            timeout = None if deadline is None else deadline - time.monotonic()
            if timeout is not None and timeout <= 0:
                raise LLMDeadlineExceeded('Incident Reflection logical timeout exhausted')
            self.last_prompt_breakdown = {
                'system_chars': len(system), 'context_chars': len(prompt), 'raw_history_included': False,
                'prompt_components': {'system_tokens': estimate_tokens_char4(system),
                    'diagnostic_context_tokens': estimate_tokens_char4(prompt), 'tool_schema_tokens': 0,
                    'control_state_tokens': 0}}
            if prompt_budget is not None:
                decision = prompt_budget.check_prompt('reflection', system, prompt,
                    breakdown={'observation': prompt}, response_format=strict_response_format(schema))
                self.last_prompt_breakdown.update(estimated_prompt_tokens=decision.estimated_prompt_tokens,
                    input_hard_capacity=decision.input_hard_capacity, budget_state=decision.state.value,
                    token_breakdown=dict(decision.breakdown))
            self.last_prompt_breakdowns.append(dict(self.last_prompt_breakdown))
            self.last_prompt_snapshots.append(PromptSnapshot.from_render(
                redact_snapshot_value(system + '\n\n' + prompt),
                token_count=estimate_tokens_char4(system + '\n\n' + prompt),
                system_prompt=redact_snapshot_value(system), user_prompt=redact_snapshot_value(prompt),
                request_kind='reflection', context_view=context_view,
                acquired_source_ranges=context_view.acquired_source_ranges if context_view else ()))
            self.last_call_count += 1
            try:
                response = self.provider_gateway.complete_structured(system, prompt,
                    model=self.model or None, schema=schema, logical_timeout_seconds=timeout,
                    on_attempt_started=on_attempt_started, max_output_tokens=max_output_tokens,
                    enable_thinking=False)
                raw = getattr(response, 'structured', None)
                if raw is None:
                    raw = extract_json(getattr(response, 'content', response))
                return schema.model_validate(raw)
            except (ValidationError, DecisionSurfaceError, SemanticTransitionError, LLMOutputError) as exc:
                error = exc
                self.last_schema_error = str(exc)
                self.last_schema_repaired = True
            finally:
                self._add_usage()
        self.last_failure_type = 'reflection_contract_exhausted'
        raise ReflectionContractExhausted('incident reflection remained invalid after one correction: '
            + self.last_schema_error) from error

    # A descriptive alias makes the role convenient for direct deterministic tests.
    review = reflect

    def _add_usage(self) -> None:
        from debug_assistant.incidents.provider_gateway import aggregate_provider_usage
        records = self.provider_gateway.records
        usage = aggregate_provider_usage(records[self._usage_cursor:])
        for key, value in usage.items():
            self.last_usage[key] = self.last_usage.get(key, 0) + value
        self._usage_cursor = len(records)
