from __future__ import annotations
import copy
import json
import time
import inspect
from dataclasses import dataclass
from typing import Any, Iterable, get_args
from pydantic import ValidationError
from debug_assistant.models import ActionProposal, ActionKind, AgentState
from debug_assistant.contracts import (AgentActionContract, PlannerIntent, QuestionType, compact_validation_error,
                                       render_contract, render_contract_compact)
from debug_assistant.llm.base import complete_json_compat, extract_json
from debug_assistant.llm.base import LLMOutputError, LLMResponse, LLMToolCall, ProviderCapabilities
from debug_assistant.skills.catalog import INCIDENT_SKILLS, SKILLS, render_skill_catalog
from debug_assistant.skills.loader import SkillLibrary
from debug_assistant.tools.registry import PARALLEL_ALLOWED_TOOLS
from debug_assistant.tools.repository import REPOSITORY_SOURCE_MAX_LINES
from debug_assistant.incidents.contracts import (
    Contradiction, SourceMechanismStatus, VerificationObligation,
)
from debug_assistant.agent.output_normalization import (
    INCIDENT_REQUIRED_CONTROL_FIELDS,
    INCIDENT_SKILL_CONTROL_FIELDS,
    IncidentLLMOutputNormalizer,
)


INCIDENT_ENVELOPE_TOOL = "diagnosis_action"


class PlannerContractError(ValueError):
    """Bounded, sanitized planner contract failure."""
    def __init__(self, message, *, validation_errors=None, output=None, repair_rejection_reason=None):
        actions = output.get('actions') if isinstance(output,dict) else None
        self.metadata={
            'validation_errors': validation_errors or [],
            'kind': output.get('kind') if isinstance(output,dict) and isinstance(output.get('kind'),str) else None,
            'tool': output.get('tool') if isinstance(output,dict) and isinstance(output.get('tool'),str) else None,
            'arguments_type': type(output.get('arguments')).__name__ if isinstance(output,dict) else type(output).__name__,
            'actions_type': type(output.get('actions')).__name__ if isinstance(output,dict) else None,
            'actions_count': len(actions) if isinstance(actions,list) else None,
            'child_types': [type(x).__name__ for x in actions] if isinstance(actions,list) else None,
            'output_shape': 'object' if isinstance(output,dict) else type(output).__name__,
        }
        if repair_rejection_reason:
            self.metadata['repair_rejection_reason'] = repair_rejection_reason
        if isinstance(output, dict):
            # Contract artifacts are intentionally limited to provider-visible
            # JSON and validation state. Hidden reasoning is never copied here.
            for key in (
                "raw_output", "parsed_output", "normalized_output",
                "validation_error", "repair_prompt", "repair_output",
                "repair_result",
            ):
                if key in output:
                    self.metadata[key] = output[key]
        super().__init__(message)


class NativePlannerContractError(PlannerContractError):
    """A malformed native provider response, distinct from legacy JSON parsing."""

    def __init__(self, message, *, error_type="provider_contract_mismatch", validation_errors=None,
                 output=None, index=None, tool=None):
        super().__init__(message, validation_errors=validation_errors, output=output)
        self.metadata["error_type"] = error_type
        if index is not None:
            self.metadata["index"] = index
        if tool is not None:
            self.metadata["tool"] = tool


class PlannerContractExhausted(PlannerContractError):
    """The bounded native Planner contract recovery path was exhausted."""

    def __init__(self, cause: PlannerContractError):
        super().__init__(
            "planner structured contract remained invalid after bounded retry",
            validation_errors=list(getattr(cause, "metadata", {}).get("validation_errors", ())),
        )
        self.metadata.update(getattr(cause, "metadata", {}))
        self.metadata["error_type"] = "planner_contract_exhausted"


@dataclass(frozen=True, slots=True)
class NativeSkillSelection:
    call_id: str
    skill: str
    reason: str
    current_hypothesis: str
    evidence_gap: str
    candidate_component: str = ""
    candidate_fault: str = ""
    candidate_mechanism: str = ""
    supporting_evidence_ids: tuple[str, ...] = ()
    contradicting_evidence_ids: tuple[str, ...] = ()
    required_evidence_gaps: tuple[str, ...] = ()
    evidence_sufficiency: str = "insufficient"
    remaining_evidence_need: str = ""
    source_mechanism_status: SourceMechanismStatus = "unknown"
    mechanism_category: str = ""
    verification_obligations: tuple[VerificationObligation, ...] = ()
    contradictions: tuple[Contradiction, ...] = ()
    candidate_fault_code: str = ""
    candidate_fault_explanation: str = ""
    # Optional provider metadata is useful when valid, but it must not make a
    # valid tool request unusable merely because an older/model-specific
    # provider serialized that additive block imperfectly.  The warning is
    # surfaced by the Harness trace; the flat compatibility projections remain
    # the only fallback state used for this selection.
    reasoning_metadata_warnings: tuple[str, ...] = ()
    reasoning_metadata_normalizations: tuple[str, ...] = ()
    reasoning_metadata_drops: tuple[str, ...] = ()
    compatibility_normalizations: tuple[str, ...] = ()
    obligation_id: str = ""
    expected_information_gain: str = ""


@dataclass(frozen=True, slots=True)
class NativePlannerResult:
    """Semantic planner metadata plus provider-native tool requests.

    Execution shape is compiled later by ``ToolOrchestrator``. Incident runs also
    carry a validated skill decision per tool call; repository runs retain their
    previous schema and leave ``skill_selections`` empty.
    """

    response: LLMResponse
    tool_calls: tuple[LLMToolCall, ...]
    reason: str = ""
    information_need: str = ""
    expected_evidence: str = ""
    retain_context_ids: tuple[str, ...] = ()
    obligation_ids: tuple[str, ...] = ()
    intent: PlannerIntent | None = None
    assistant_text: str | None = None
    skill_selections: tuple[NativeSkillSelection, ...] = ()
    tool_call_audits: tuple[dict[str, Any], ...] = ()


class NativeToolPlanner:
    def __init__(self, llm, tools, model: str = "", *, planner_state_envelope: bool = False):
        self.llm, self.tools, self.model = llm, tools, model
        # Opt-in optimized protocol. The direct-call protocol remains available
        # for old providers and compatibility tests.
        self.planner_state_envelope = bool(planner_state_envelope)
        self.last_prompt_breakdown = {}
        self.last_prompt_breakdowns: list[dict[str, Any]] = []

    @staticmethod
    def _contract_output(response: LLMResponse, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build bounded, auditable diagnostics for a Planner contract error."""
        output = dict(context or {})
        raw_output = getattr(response, "raw_output", None)
        if isinstance(raw_output, dict):
            output["raw_output"] = copy.deepcopy(raw_output)
        elif raw_output:
            output["raw_output"] = str(raw_output)[:12000]
        structured = getattr(response, "structured", None)
        if structured is not None:
            try:
                json.dumps(structured, ensure_ascii=False)
                output["parsed_output"] = copy.deepcopy(structured)
            except (TypeError, ValueError):
                output["parsed_output"] = str(structured)[:12000]
        return output

    def propose(self, state: AgentState, context: str, *, logical_timeout_seconds=None,
                on_attempt_started=None, prompt_budget=None,
                visible_tool_names: Iterable[str] | None = None,
                max_output_tokens: int | None = None) -> NativePlannerResult:
        if not getattr(getattr(self.llm, "capabilities", ProviderCapabilities()), "tool_calling", False):
            raise PlannerContractError("provider does not support native tool calling",
                                       validation_errors=["tool_calling=false"])
        incident_mode = state.task.metadata.get("task_kind") == "incident"
        system = (
            "You are a read-only repository investigation planner. Request only the "
            "repository tools needed for the current goal. Tool arguments are validated "
            "by the Harness; do not invent tool names. Return tool calls, not an execution plan. "
            "Search results are discovery only: whenever grep, code_search, or symbol_search "
            "returns a repository path with a line or symbol, the next investigation turn "
            "must include a bounded read_file for the matching source, covering the hit and "
            "a substantial nearby context window (at least 100 lines when no complete symbol "
            "range is available; never use a narrow abbreviated slice). A broad read in another "
            "file is not a substitute. If OPEN_CRITICAL_EVIDENCE_OBLIGATIONS names an exact "
            "file, symbol, or line range, honor that scope first and use explicit supporting "
            "actions only for callers, tests, or history."
        )
        if incident_mode:
            system = (
                "You are the Diagnosis Agent for a read-only production incident. On every action, "
                "choose the next Skill from the supplied catalog based on the INCIDENT, CURRENT_HYPOTHESIS, "
                "EVIDENCE_GAP, and PREVIOUS_ACTIONS. Never infer or route on a benchmark fault label. "
                "Every function call must include the Skill decision and the structured hypothesis-completion fields. "
                "Explicitly compare CONTINUE INVESTIGATION with FINALIZE CURRENT DIAGNOSIS. Root-cause completion "
                "requires a component, causal mechanism, supporting evidence, no critical required gap, and no "
                "critical contradiction. Impact scope is optional and must never block finalization. If evidence is "
                "sufficient, strongly prefer finalize_diagnosis. Continuing from a sufficient hypothesis requires a "
                "remaining_evidence_need that names evidence capable of changing the root-cause judgment; otherwise finalize. "
                "Final Review receives only the ev-* Evidence IDs cited by finalize_diagnosis; uncited Evidence is invisible "
                "to Review. Cite every collected observation needed to support each component and mechanism claim. "
                "REJECTED_ACTIONS are policy-rejected requests, not new observations; never repeat their fingerprints. "
                "A snapshot result with status UNAVAILABLE and semantic_negative=false means only that the benchmark did not "
                "capture that observation; it is not evidence that a resource is missing or a service is unreachable. "
                "Keep Direct Observation, evidence-backed inference, and unverified hypothesis distinct. Prefer the narrowest "
                "candidate fully supported by the cited Evidence. Do not claim that a measured value reaches or exceeds a "
                "configured threshold unless the cited Evidence directly establishes that numeric relation; an aggregate CPU "
                "value below a limit does not disprove a direct CFS-throttling observation. Do not state traffic bursts, load "
                "spikes, dependency amplification, or other causal triggers as facts without direct Evidence. "
                "For Kubernetes get_resources, name is an exact captured resource name, not a logical service or application "
                "name; when the exact name is unknown, omit name or list first, then use the returned exact name. "
                "When TOOL_BUDGET_REMAINING is zero, do not request another observation tool; use "
                "finalize_diagnosis if the existing evidence supports a component and mechanism. "
                "For read_file, always provide start_line plus line_count; line_count is the number "
                "of requested lines, inclusive of start_line, and must be between 1 and "
                + str(REPOSITORY_SOURCE_MAX_LINES) + ". For a concrete source file involved in the "
                "hypothesis, prefer one broad bounded read that covers the relevant file or complete "
                "implementation region, rather than mechanically splitting a file into 200-line pages. "
                "Never provide end_line for read_file. "
                "Use finalize_diagnosis only when the cited evidence supports both component and causal mechanism. "
                "At finalize_diagnosis, component/fault/mechanism/evidence_ids are compatibility projections; "
                "the Runtime freezes the already validated current Hypothesis as the Candidate core. "
                "When known, keep candidate_fault_code as a lowercase snake_case taxonomy identifier and "
                "candidate_fault_explanation as the evidence-grounded natural-language explanation; never "
                "guess a taxonomy code solely to satisfy a field. "
                "Use claim_evidence_mapping and causal_chain_summary to explain that frozen diagnosis, and "
                "keep their Evidence IDs within the Hypothesis supporting Evidence. "
                "If a bound application source workspace is available and runtime evidence leaves an implementation gap, "
                "use code_investigation; file indexes and search hits locate candidates, while only bounded read_file output "
                "can support a source-backed behavior claim. If the proposed mechanism is an application implementation "
                "behavior that runtime evidence has not established, do not finalize from telemetry or configuration alone "
                "when the source workspace is available; use the code Skill to verify it. The supporting_evidence_ids "
                "field on the same finalize_diagnosis call must include every evidence_ids value, even when both fields "
                "refer to the same current hypothesis. When SOURCE_MECHANISM_COVERAGE is declared and the workspace "
                "is available, explicitly set source_mechanism_status to one of unknown, gap, sufficient, or "
                "not_applicable. Use gap when the current causal mechanism still needs source verification, "
                "sufficient only after a cited read_file CODE Evidence supports it, and not_applicable only when "
                "the current mechanism is demonstrably independent of application implementation. A gap or unknown "
                "source status must be represented by a critical VerificationObligation and investigated with "
                "code_investigation before finalization.\n\n"
                + render_skill_catalog(skills=INCIDENT_SKILLS)
            )
        try:
            schemas = self.tools.function_schemas(visible_tools=visible_tool_names)
        except TypeError:
            # Generic repository registries predate stage-aware incident
            # exposure. Keep them compatible and apply no accidental filter.
            schemas = self.tools.function_schemas()
        if visible_tool_names is not None and not hasattr(self.tools, "function_schemas"):
            visible = {str(name) for name in visible_tool_names}
            schemas = [
                schema for schema in schemas
                if schema.get("function", {}).get("name") in visible
            ]
        actual_schemas = schemas
        contract_catalog = ""
        if incident_mode and self.planner_state_envelope:
            contract_catalog = _render_incident_tool_contract_catalog(actual_schemas)
            schemas = [_build_incident_envelope_schema(actual_schemas)]
        elif incident_mode:
            schemas = [
                _compact_incident_provider_schema(_with_incident_skill_controls(schema))
                for schema in schemas
            ]
        if incident_mode and self.planner_state_envelope:
            user = (
                f"{context}\n\nPLANNER_TOOL_CONTRACTS (live executable argument names; "
                f"Runtime remains authoritative):\n{contract_catalog}\n\n"
                f"CURRENT_GOAL: {state.task.issue}\n"
            )
        else:
            user = f"{context}\n\nCURRENT_GOAL: {state.task.issue}\n"
        self.last_prompt_breakdown = {"system_chars": len(system), "context_chars": len(user),
                                      "tool_schema_count": len(schemas)}
        self.last_prompt_breakdowns = []
        if prompt_budget is not None:
            decision = prompt_budget.check_prompt(
                "planner", system, user, tools=schemas,
                breakdown={"incident": context},
            )
            self.last_prompt_breakdown.update({
                "estimated_prompt_tokens": decision.estimated_prompt_tokens,
                "input_hard_capacity": decision.input_hard_capacity,
                "budget_state": decision.state.value,
                "token_breakdown": dict(decision.breakdown),
            })
            self.last_prompt_breakdowns.append(dict(self.last_prompt_breakdown))
        method = getattr(self.llm, "complete_with_tools", None)
        if method is None:
            raise PlannerContractError("provider has no complete_with_tools API")
        kwargs = {"tools": schemas, "model": self.model or None}
        try:
            parameters = inspect.signature(method).parameters
            has_varkw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
        except (TypeError, ValueError):
            parameters, has_varkw = {}, False
        if "logical_timeout_seconds" in parameters or has_varkw:
            kwargs["logical_timeout_seconds"] = logical_timeout_seconds
        if "on_attempt_started" in parameters or has_varkw:
            kwargs["on_attempt_started"] = on_attempt_started
        if ("max_output_tokens" in parameters or has_varkw) and max_output_tokens is not None:
            kwargs["max_output_tokens"] = int(max_output_tokens)
        try:
            response = method(system, user, **kwargs)
        except LLMOutputError as exc:
            raise NativePlannerContractError(
                str(exc), error_type=getattr(exc, "error_type", "provider_contract_mismatch"),
                validation_errors=[getattr(exc, "error_type", "provider_contract_mismatch")],
                output={
                    "raw_output": copy.deepcopy(getattr(self.llm, "last_raw_output", {})),
                    "validation_error": str(exc),
                },
                index=getattr(exc, "index", None), tool=getattr(exc, "tool", None),
            ) from exc
        if not isinstance(response, LLMResponse):
            raise NativePlannerContractError(
                "provider returned a non-typed native response",
                error_type="provider_contract_mismatch",
                output={"validation_error": "response_type=" + type(response).__name__},
            )

        # Some OpenAI-compatible providers advertise tool calling but emit the
        # requested action as ordinary JSON content.  Keep this compatibility
        # boundary narrow: only an unambiguous tool_name/tool plus an object of
        # arguments is promoted, and the normal native-tool validation below
        # remains authoritative.  Natural-language no-tool turns stay intact.
        if incident_mode and not response.tool_calls and isinstance(response.content, str):
            try:
                content_action = extract_json(response.content)
            except (TypeError, ValueError, LLMOutputError):
                content_action = None
            if isinstance(content_action, dict):
                content_tool = content_action.get("tool_name") or content_action.get("tool")
                content_arguments = content_action.get("arguments")
                if isinstance(content_tool, str) and content_tool.strip():
                    promoted_arguments = dict(content_arguments) if isinstance(content_arguments, dict) else {}
                    for key, value in content_action.items():
                        if key not in {"tool_name", "tool", "arguments", "kind"}:
                            promoted_arguments.setdefault(key, value)
                    response = LLMResponse(
                        content=response.content,
                        structured=content_action,
                        tool_calls=(LLMToolCall("content-fallback-1", content_tool.strip(), promoted_arguments),),
                        usage=response.usage,
                        raw_output=response.raw_output,
                    )

        current_normalized_output: dict[str, Any] = {}

        def contract_error(message: str, *, error_type: str,
                           validation_errors=None, output=None, index=None,
                           tool=None):
            context = dict(output or {})
            if current_normalized_output:
                context.setdefault("normalized_output", copy.deepcopy(current_normalized_output))
            context.setdefault("validation_error", message)
            return NativePlannerContractError(
                message, error_type=error_type, validation_errors=validation_errors,
                output=self._contract_output(response, context), index=index, tool=tool,
            )
        names = {spec.name for spec in self.tools.specs()}
        allowed_tool_fields = {
            spec.name: tuple(spec.args_model.model_fields)
            for spec in self.tools.specs()
        }
        output_normalizer = IncidentLLMOutputNormalizer(allowed_tool_fields)
        known_evidence_ids = {
            str(getattr(item, "evidence_id", ""))
            for item in getattr(state, "evidence", ())
            if getattr(item, "evidence_id", None)
        }
        sanitized_calls = []
        selections = []
        tool_call_audits: list[dict[str, Any]] = []
        for call in response.tool_calls:
            if not isinstance(call, LLMToolCall) or not isinstance(call.arguments, dict):
                raise contract_error(
                    "provider returned a malformed native tool call",
                    error_type="malformed_tool_call",
                    validation_errors=["malformed_tool_call"],
                )
            provider_tool_name = call.name
            call_name = call.name
            call_arguments = dict(call.arguments)
            current_normalized_output = {}
            protocol_mode = "direct"
            if incident_mode and self.planner_state_envelope and call.name == INCIDENT_ENVELOPE_TOOL:
                protocol_mode = "envelope"
                envelope_result = output_normalizer.normalize_envelope(call.arguments)
                envelope = envelope_result.arguments
                call_name = envelope.pop("tool_name", "")
                executable_arguments = envelope.pop("arguments", None)
                if not isinstance(call_name, str) or not call_name:
                    raise contract_error(
                        "diagnosis_action is missing tool_name",
                        error_type="missing_executable_tool",
                        validation_errors=["tool_name"], output={"tool": call.name},
                    )
                if not isinstance(executable_arguments, dict):
                    raise contract_error(
                        "diagnosis_action.arguments must be an object",
                        error_type="malformed_executable_arguments",
                        validation_errors=["arguments"], output={"tool": call.name},
                    )
                # Planner controls live at the envelope level and never become
                # part of the executable Tool's Pydantic argument model.
                call_arguments = {**executable_arguments, **envelope}
                current_normalized_output = dict(call_arguments)
            elif incident_mode and self.planner_state_envelope and call.name in names:
                protocol_mode = "legacy_direct_compat"
            if call_name not in names:
                raise contract_error(
                    f"unknown tool: {call_name}", error_type="unknown_tool",
                    validation_errors=["unknown_tool"], output={"tool": call.name},
                )
            if incident_mode:
                normalized = output_normalizer.normalize_tool_arguments(
                    call_name, call_arguments,
                )
                args = normalized.arguments
                current_normalized_output = dict(args)
                raw_arguments = dict(call_arguments)
                compatibility_normalizations = normalized.actions
                control = {key: args.pop(key, None) for key in INCIDENT_SKILL_CONTROL_FIELDS}
                skill = control["skill"]
                reason = control["skill_reason"]
                hypothesis = control["current_hypothesis"]
                evidence_gap = control["evidence_gap"]
                component = control["candidate_component"]
                fault = control["candidate_fault"]
                mechanism = control["candidate_mechanism"]
                supporting = control["supporting_evidence_ids"]
                contradicting = control["contradicting_evidence_ids"]
                required_gaps = control["required_evidence_gaps"]
                sufficiency = control["evidence_sufficiency"]
                remaining_need = control["remaining_evidence_need"]
                obligation_id = control.get("obligation_id") or ""
                expected_information_gain = control.get("expected_information_gain") or ""
                source_mechanism_status = control.get("source_mechanism_status") or "unknown"
                mechanism_category = control["mechanism_category"] or ""
                fault_code = control.get("candidate_fault_code") or ""
                fault_explanation = control.get("candidate_fault_explanation") or ""
                raw_obligations = control["verification_obligations"]
                raw_contradictions = control["contradictions"]
                if skill not in INCIDENT_SKILLS:
                    raise contract_error(
                        f"unknown skill: {skill}", error_type="unknown_skill",
                        validation_errors=["unknown_skill"], output={"tool": call.name},
                    )
                if not all(isinstance(value, str) and value.strip() for value in (reason, hypothesis, evidence_gap)):
                    raise contract_error(
                        "incident tool call is missing skill decision context",
                        error_type="missing_skill_context",
                        validation_errors=["skill_reason", "current_hypothesis", "evidence_gap"],
                        output={"tool": call.name},
                    )
                if not all(isinstance(value, str) for value in (component, fault, mechanism, remaining_need)):
                    raise contract_error(
                        "incident hypothesis completion fields must be strings",
                        error_type="malformed_hypothesis_completion",
                        validation_errors=["candidate_component", "candidate_fault", "candidate_mechanism", "remaining_evidence_need"],
                        output={"tool": call.name},
                    )
                if not all(isinstance(value, str) for value in (fault_code, fault_explanation)):
                    raise contract_error(
                        "structured fault fields must be strings",
                        error_type="malformed_hypothesis_completion",
                        validation_errors=["candidate_fault_code", "candidate_fault_explanation"],
                        output={"tool": call.name},
                    )
                if not all(isinstance(value, list) and all(isinstance(item, str) for item in value)
                           for value in (supporting, contradicting, required_gaps)):
                    raise contract_error(
                        "incident evidence linkage fields must be string arrays",
                        error_type="malformed_hypothesis_completion",
                        validation_errors=["supporting_evidence_ids", "contradicting_evidence_ids", "required_evidence_gaps"],
                        output={"tool": call.name},
                    )
                if not isinstance(mechanism_category, str):
                    raise contract_error(
                        "mechanism_category must be a string",
                        error_type="malformed_hypothesis_completion",
                        validation_errors=["mechanism_category"], output={"tool": call.name},
                    )
                if source_mechanism_status not in {
                    "unknown", "gap", "sufficient", "not_applicable", "blocked",
                }:
                    raise contract_error(
                        "invalid source_mechanism_status",
                        error_type="malformed_hypothesis_completion",
                        validation_errors=["source_mechanism_status"], output={"tool": call.name},
                    )
                obligation_result = _parse_optional_reasoning_items(
                    raw_obligations, field_name="verification_obligations",
                    model_type=VerificationObligation,
                )
                contradiction_result = _parse_optional_reasoning_items(
                    raw_contradictions, field_name="contradictions",
                    model_type=Contradiction,
                )
                obligations = obligation_result.items
                structured_contradictions = contradiction_result.items
                reasoning_metadata_warnings = (
                    *obligation_result.validation_warnings,
                    *contradiction_result.validation_warnings,
                )
                reasoning_metadata_normalizations = (
                    *obligation_result.normalization_actions,
                    *contradiction_result.normalization_actions,
                )
                reasoning_metadata_drops = (
                    *obligation_result.dropped_items,
                    *contradiction_result.dropped_items,
                )
                structured_ids = [
                    item.evidence_id for item in structured_contradictions
                ] + [
                    evidence_id
                    for item in obligations
                    for evidence_id in item.supporting_evidence_ids
                ]
                invalid_structured_ids = [
                    evidence_id for evidence_id in structured_ids
                    if not evidence_id.startswith("ev-")
                    or evidence_id not in known_evidence_ids
                ]
                if invalid_structured_ids:
                    raise contract_error(
                        "structured incident metadata may cite only ev-* Evidence IDs that are present",
                        error_type="invalid_evidence_id_contract",
                        validation_errors=["verification_obligations", "contradictions"],
                        output={"tool": call.name},
                    )
                invalid_evidence_ids = [
                    item for item in supporting + contradicting
                    if not item.startswith("ev-") or item not in known_evidence_ids
                ]
                if invalid_evidence_ids:
                    raise contract_error(
                        "incident hypotheses may cite only ev-* Evidence IDs that are present",
                        error_type="invalid_evidence_id_contract",
                        validation_errors=["supporting_evidence_ids", "contradicting_evidence_ids"],
                        output={"tool": call.name},
                    )
                if sufficiency not in {"insufficient", "sufficient"}:
                    raise contract_error(
                        "invalid evidence_sufficiency",
                        error_type="malformed_hypothesis_completion",
                        validation_errors=["evidence_sufficiency"], output={"tool": call.name},
                    )
                sanitized_calls.append(LLMToolCall(call.id, call_name, args))
                tool_call_audits.append({
                    "call_id": call.id,
                    "tool": call_name,
                    "provider_tool": provider_tool_name,
                    "protocol_mode": protocol_mode,
                    "raw_arguments": dict(call.arguments),
                    "executable_arguments": raw_arguments,
                    "normalized_arguments": dict(args),
                    "normalization_actions": list(normalized.actions),
                    "dropped_fields": list(normalized.dropped_fields),
                    "envelope_normalization_actions": list(
                        envelope_result.actions if protocol_mode == "envelope" else ()
                    ),
                    "envelope_dropped_fields": list(
                        envelope_result.dropped_fields if protocol_mode == "envelope" else ()
                    ),
                    "allowed_argument_fields": sorted(allowed_tool_fields.get(call_name, ())),
                    "status": "normalized_pending_runtime_validation",
                })
                selections.append(NativeSkillSelection(
                    call.id, skill, reason.strip(), hypothesis.strip(), evidence_gap.strip(),
                    component.strip(), fault.strip(), mechanism.strip(),
                    tuple(item.strip() for item in supporting if item.strip()),
                    tuple(item.strip() for item in contradicting if item.strip()),
                    tuple(item.strip() for item in required_gaps if item.strip()),
                    sufficiency, remaining_need.strip(), source_mechanism_status,
                    mechanism_category.strip(),
                    obligations, structured_contradictions,
                    fault_code.strip(), fault_explanation.strip(),
                    reasoning_metadata_warnings,
                    reasoning_metadata_normalizations,
                    reasoning_metadata_drops,
                    compatibility_normalizations,
                    obligation_id.strip(), expected_information_gain.strip(),
                ))
            else:
                sanitized_calls.append(call)
                tool_call_audits.append({
                    "call_id": call.id,
                    "tool": call.name,
                    "raw_arguments": dict(call.arguments),
                    "normalized_arguments": dict(call.arguments),
                    "normalization_actions": [],
                    "dropped_fields": [],
                    "allowed_argument_fields": sorted(allowed_tool_fields.get(call.name, ())),
                    "status": "pending_runtime_validation",
                })
        metadata = response.structured if isinstance(response.structured, dict) else {}
        assistant_text = response.content.strip() if isinstance(response.content, str) else ""
        raw_intent = {
            "information_need": metadata.get("information_need") if isinstance(metadata.get("information_need"), str) else None,
            "target": metadata.get("target") if isinstance(metadata.get("target"), str) else None,
            "question_type": metadata.get("question_type") if isinstance(metadata.get("question_type"), str) else None,
            "evidence_goal": metadata.get("evidence_goal") if isinstance(metadata.get("evidence_goal"), str) else None,
            "reason": metadata.get("reason") if isinstance(metadata.get("reason"), str) else None,
        }
        if not raw_intent["information_need"] and assistant_text:
            raw_intent["information_need"] = assistant_text
        if not raw_intent["reason"] and assistant_text:
            raw_intent["reason"] = assistant_text
        try:
            intent = PlannerIntent.model_validate(raw_intent)
        except ValidationError:
            # Metadata is advisory. Preserve tool execution and use deterministic
            # runtime fallback linkage when a provider emits an invalid intent.
            intent = PlannerIntent(
                information_need=raw_intent["information_need"] if isinstance(raw_intent["information_need"], str) else None,
                reason=raw_intent["reason"] if isinstance(raw_intent["reason"], str) else None,
            )
        return NativePlannerResult(
            response=response, tool_calls=tuple(sanitized_calls),
            reason=intent.reason or "",
            information_need=intent.information_need or "",
            expected_evidence=str(metadata.get("expected_evidence") or ""),
            retain_context_ids=tuple(str(x) for x in (metadata.get("retain_context_ids") or [])),
            obligation_ids=tuple(str(x) for x in (metadata.get("obligation_ids") or [])),
            intent=intent,
            assistant_text=assistant_text or None,
            skill_selections=tuple(selections),
            tool_call_audits=tuple(tool_call_audits),
        )


@dataclass(frozen=True, slots=True)
class OptionalReasoningParseResult:
    """Sanitized result for additive Planner reasoning metadata."""

    items: tuple[Any, ...] = ()
    dropped_items: tuple[str, ...] = ()
    normalization_actions: tuple[str, ...] = ()
    validation_warnings: tuple[str, ...] = ()


def _optional_item_schema(model_type) -> dict[str, Any]:
    """Publish the nested schema from the exact Pydantic runtime model.

    The provider-facing schema stays strict and therefore cannot advertise an
    arbitrary object that the runtime contract would reject.  The parser still
    accepts a manually received additive annotation as a compatibility input,
    records a warning, and removes it before validation.
    """
    return copy.deepcopy(model_type.model_json_schema())


def _optional_evidence_reference_paths(item: dict[str, Any], *, field_name: str, index: int):
    """Find explicit non-ev citations before an invalid optional row is dropped."""
    references = []
    if field_name == "verification_obligations":
        raw_ids = item.get("supporting_evidence_ids")
        if isinstance(raw_ids, (list, tuple)):
            references.extend(
                (f"{field_name}[{index}].supporting_evidence_ids[{offset}]", value)
                for offset, value in enumerate(raw_ids)
            )
        elif raw_ids is not None:
            references.append((f"{field_name}[{index}].supporting_evidence_ids", raw_ids))
    elif field_name == "contradictions":
        references.append((f"{field_name}[{index}].evidence_id", item.get("evidence_id")))
    return tuple(
        (path, value) for path, value in references
        if isinstance(value, str) and not value.startswith("ev-")
    )


def _parse_optional_reasoning_items(raw, *, field_name: str, model_type) -> OptionalReasoningParseResult:
    """Apply safe shape tolerance while retaining strict semantic checks.

    This parser intentionally does not coerce enum values, booleans, IDs, or
    claims.  A malformed optional row is isolated from valid siblings.  An
    explicit non-ev citation is different: dropping it would hide a core
    Evidence contract violation, so it raises a bounded Planner contract error.
    """
    if raw is None:
        return OptionalReasoningParseResult()

    normalization_actions = []
    validation_warnings = []
    dropped_items = []
    if isinstance(raw, dict):
        required_fields = {
            name for name, field in model_type.model_fields.items() if field.is_required()
        }
        if not required_fields.issubset(raw):
            return OptionalReasoningParseResult(
                dropped_items=(f"{field_name}:block:ambiguous_object_shape",),
                validation_warnings=(f"{field_name}:discarded_ambiguous_object",),
            )
        items = [raw]
        normalization_actions.append(f"{field_name}:object_to_singleton_array")
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
        if isinstance(raw, tuple):
            normalization_actions.append(f"{field_name}:tuple_to_array")
    else:
        return OptionalReasoningParseResult(
            dropped_items=(f"{field_name}:block:non_array",),
            validation_warnings=(
                f"{field_name}:discarded_non_array:{type(raw).__name__}",
            ),
        )

    allowed_fields = set(model_type.model_fields)
    parsed = []
    for index, item in enumerate(items):
        item_path = f"{field_name}[{index}]"
        if not isinstance(item, dict):
            dropped_items.append(f"{item_path}:non_object")
            validation_warnings.append(
                f"{item_path}:discarded_non_object:{type(item).__name__}"
            )
            continue

        invalid_references = _optional_evidence_reference_paths(
            item, field_name=field_name, index=index,
        )
        if invalid_references:
            paths = [path for path, _ in invalid_references]
            raise NativePlannerContractError(
                "structured incident metadata may cite only ev-* Evidence IDs",
                error_type="invalid_evidence_id_contract",
                validation_errors=paths,
            )

        unknown_fields = sorted(set(item) - allowed_fields)
        if unknown_fields:
            normalization_actions.append(
                f"{item_path}:ignored_extra_fields:{','.join(unknown_fields)}"
            )
            validation_warnings.append(
                f"{item_path}:ignored_unknown_fields:{','.join(unknown_fields)}"
            )
        sanitized = {
            key: value for key, value in item.items() if key in allowed_fields
        }
        try:
            parsed_item = model_type.model_validate(sanitized)
        except ValidationError as exc:
            compact_errors = []
            for error in exc.errors(include_url=False):
                location = ".".join(str(part) for part in error.get("loc", ())) or "item"
                compact_errors.append(f"{location}:{error.get('type', 'validation_error')}")
            detail = ",".join(compact_errors[:4]) or "validation_error"
            dropped_items.append(f"{item_path}:schema_invalid")
            # Keep a block-level category for existing trace consumers while
            # also retaining the precise item path for new diagnostics.
            validation_warnings.append(f"{field_name}:discarded_invalid:{item_path}:{detail}")
            validation_warnings.append(f"{item_path}:discarded_invalid:{detail}")
            continue

        defaulted = sorted(
            name for name, field in model_type.model_fields.items()
            if name not in sanitized and not field.is_required()
        )
        if defaulted:
            normalization_actions.append(
                f"{item_path}:applied_defaults:{','.join(defaulted)}"
            )
        parsed.append(parsed_item)

    return OptionalReasoningParseResult(
        items=tuple(parsed),
        dropped_items=tuple(dropped_items),
        normalization_actions=tuple(normalization_actions),
        validation_warnings=tuple(validation_warnings),
    )


def _render_incident_tool_contract_catalog(schemas: Iterable[dict[str, Any]]) -> str:
    """Render a compact live catalog for the single Planner envelope.

    The catalog gives the model enough argument names to form ``arguments``;
    the actual Pydantic ToolSpec remains the authority at execution time.
    """
    rows = []
    for schema in schemas:
        function = schema.get("function", {})
        name = str(function.get("name") or "")
        parameters = function.get("parameters") or {}
        properties = parameters.get("properties") or {}
        required = parameters.get("required") or []
        if not name:
            continue
        fields = []
        for field, spec in properties.items():
            kind = spec.get("type", "value") if isinstance(spec, dict) else "value"
            suffix = "!" if field in required else ""
            fields.append(f"{field}:{kind}{suffix}")
        rows.append(f"- {name}: " + ", ".join(fields))
    return "\n".join(rows) or "(no executable tools exposed)"


def _build_incident_envelope_schema(schemas: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Build one native function carrying Planner state once per call.

    Directly enriching every executable Tool duplicated the same 12 required
    and 6 optional control fields 16 times. The envelope keeps native Tool
    Calling while making the Runtime unwrap and validate the selected Tool.
    """
    actual = list(schemas)
    names = [
        str(schema.get("function", {}).get("name"))
        for schema in actual
        if schema.get("function", {}).get("name")
    ]
    control_source = _with_incident_skill_controls({
        "type": "function",
        "function": {"name": INCIDENT_ENVELOPE_TOOL, "parameters": {"type": "object", "properties": {}}},
    })
    control_properties = control_source["function"]["parameters"]["properties"]
    properties = {
        "tool_name": {"type": "string", "enum": names},
        "arguments": {
            "type": "object",
            "description": "Executable arguments for tool_name; Runtime validates the live ToolSpec.",
            "additionalProperties": True,
        },
        **copy.deepcopy(control_properties),
    }
    required = ["tool_name", "arguments"] + [
        name for name in INCIDENT_REQUIRED_CONTROL_FIELDS
        if name not in {"tool_name", "arguments"}
    ]
    return _compact_incident_provider_schema({
        "type": "function",
        "function": {
            "name": INCIDENT_ENVELOPE_TOOL,
            "description": "Select exactly one live read-only incident Tool and report Planner state.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    })


def _with_incident_skill_controls(schema: dict[str, Any]) -> dict[str, Any]:
    """Add planner-only controls without changing the executable tool contract."""
    enriched = copy.deepcopy(schema)
    function = enriched.setdefault("function", {})
    parameters = function.setdefault("parameters", {"type": "object"})
    properties = parameters.setdefault("properties", {})
    properties.update({
        "skill": {"type": "string", "enum": list(INCIDENT_SKILLS), "description": "Selected investigation Skill."},
        "skill_reason": {"type": "string", "minLength": 1, "description": "Why this Skill closes the current evidence gap."},
        "current_hypothesis": {"type": "string", "minLength": 1, "description": "Current falsifiable hypothesis before the action."},
        "evidence_gap": {"type": "string", "minLength": 1, "description": "Specific missing evidence this action should obtain."},
        "candidate_component": {"type": "string", "description": "Current root-cause component, or empty when unknown."},
        "candidate_fault": {"type": "string", "description": "Compatibility natural-language fault projection, or empty when unknown."},
        "candidate_fault_code": {"type": "string", "description": "Structured lowercase snake_case fault taxonomy code, when known; do not invent one."},
        "candidate_fault_explanation": {"type": "string", "description": "Evidence-grounded natural-language explanation of the fault, separate from the taxonomy code."},
        "candidate_mechanism": {"type": "string", "description": "Current causal mechanism, or empty when unknown."},
        "supporting_evidence_ids": {"type": "array", "items": {"type": "string", "pattern": "^ev-"}, "description": "Existing ev-* Evidence IDs supporting the current root cause. Include every fact needed to substantiate the component and causal mechanism because Final Review cannot see uncited evidence."},
        "contradicting_evidence_ids": {"type": "array", "items": {"type": "string", "pattern": "^ev-"}, "description": "Existing ev-* Evidence IDs that critically contradict the current root cause."},
        "required_evidence_gaps": {"type": "array", "items": {"type": "string"}, "description": "Unresolved critical gaps required for root-cause diagnosis; exclude optional impact analysis."},
        "evidence_sufficiency": {"type": "string", "enum": ["insufficient", "sufficient"], "description": "Whether component and mechanism are already supported for Final Review."},
        "remaining_evidence_need": {"type": "string", "description": "If continuing despite sufficient evidence, the critical evidence that could change the root-cause judgment; otherwise empty."},
        "source_mechanism_status": {
            "type": "string",
            "enum": ["unknown", "gap", "sufficient", "not_applicable", "blocked"],
            "description": "Planner/Reflection semantic status of source-backed application mechanism coverage.",
        },
        # Compatibility alias accepted from providers that name this field
        # after the candidate rather than the source-coverage contract.
        "candidate_mechanism_status": {
            "type": "string",
            "enum": ["unknown", "gap", "sufficient", "not_applicable", "blocked"],
            "description": "Deprecated compatibility alias for source_mechanism_status; Runtime canonicalizes it.",
        },
        # These fields are optional planner metadata.  Runtime remains backward
        # compatible with older providers that emit only required_evidence_gaps
        # and the flat contradicting_evidence_ids projection.
        "mechanism_category": {"type": "string", "description": "Stable semantic category for the mechanism, when known."},
        "verification_obligations": {
            "type": "array",
            "items": _optional_item_schema(VerificationObligation),
            "description": "Optional structured verification obligations.",
        },
        "contradictions": {
            "type": "array",
            "items": _optional_item_schema(Contradiction),
            "description": "Optional structured contradiction metadata.",
        },
        "obligation_id": {"type": "string", "description": "Open verification obligation targeted by this action, when known."},
        "expected_information_gain": {"type": "string", "enum": ["low", "medium", "high"], "description": "Expected information gain of this action."},
    })
    required = list(parameters.get("required") or [])
    # The original incident controls are required for native compatibility and
    # deterministic hypothesis linkage.  The structured obligation and
    # contradiction fields are additive metadata: older providers may omit
    # them and the Harness will use the legacy projections.
    # ``mechanism_category`` is an optional descriptive projection just like
    # the two nested metadata blocks.  The canonical causal fields are the
    # candidate component/fault/mechanism strings above.
    required_controls = INCIDENT_REQUIRED_CONTROL_FIELDS
    parameters["required"] = required + [name for name in required_controls if name not in required]
    return enriched


def _compact_incident_provider_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Keep executable schema semantics while removing repeated prose.

    Incident controls are attached to every visible function. Their detailed
    descriptions were repeated for every Planner call and dominated the Qwen
    prompt budget. Names, types, enums, patterns and required fields stay
    intact; Pydantic remains the Runtime validation authority.
    """
    function_description = schema.get("function", {}).get("description", "")

    def compact(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key in {"title", "default", "examples", "deprecated", "$comment", "description"}:
                    continue
                result[key] = compact(item)
            return result
        if isinstance(value, list):
            return [compact(item) for item in value]
        return value

    result = compact(copy.deepcopy(schema))
    if function_description:
        result.setdefault("function", {})["description"] = function_description
    properties = (
        result.get("function", {}).get("parameters", {}).get("properties", {})
    )
    original_properties = (
        schema.get("function", {}).get("parameters", {}).get("properties", {})
    )
    evidence_description = original_properties.get("evidence_ids", {}).get("description")
    if evidence_description and "evidence_ids" in properties:
        properties["evidence_ids"]["description"] = evidence_description
    return result


class PlannerFacade:
    """Select native tool calling or the bounded V1.4 structured fallback."""
    def __init__(self, llm, tools, model: str = "", *, native_enabled: bool = True,
                 max_parallel_actions: int = 4):
        self.llm, self.tools, self.model = llm, tools, model
        self.legacy = Planner(llm, tools, model, max_parallel_actions=max_parallel_actions)
        self.native = NativeToolPlanner(llm, tools, model)
        capabilities = getattr(llm, "capabilities", ProviderCapabilities())
        self.mode = "native_tool_calling" if native_enabled and capabilities.tool_calling else "legacy_structured_json"

    @property
    def last_prompt_breakdown(self):
        actor = self.native if self.mode == "native_tool_calling" else self.legacy
        return actor.last_prompt_breakdown

    @property
    def last_action_normalization(self):
        return getattr(self.legacy, "last_action_normalization", None)

    def propose(self, state, context, *, logical_timeout_seconds=None, on_attempt_started=None):
        if self.mode == "native_tool_calling":
            return self.native.propose(state, context, logical_timeout_seconds=logical_timeout_seconds,
                                       on_attempt_started=on_attempt_started)
        return self.legacy.propose(state, context, logical_timeout_seconds=logical_timeout_seconds)


def normalize_planner_action(data, *, max_parallel_actions: int = 4):
    """Normalize only the unambiguous one-child parallel structural degeneration."""
    if not isinstance(data, dict) or data.get('kind') != 'parallel':
        return data, None
    actions = data.get('actions')
    if not isinstance(actions, list) or len(actions) != 1 or not isinstance(actions[0], dict):
        return data, None
    child = actions[0]
    normalized = dict(data)
    normalized.update({
        'kind': 'tool',
        'tool': child.get('tool'),
        'arguments': child.get('arguments'),
        'actions': [],
    })
    return normalized, {
        'reason': 'parallel_single_child',
        'from_kind': 'parallel',
        'to_kind': 'tool',
        'child_count': 1,
    }


_REPAIRABLE_FIELDS = frozenset({'kind', 'arguments', 'actions', 'information_need_structured'})
_IMMUTABLE_FIELDS = frozenset({
    'skill', 'reason', 'confidence', 'tool', 'expected_evidence', 'information_need',
    'retain_context_ids',
})
_SEMANTIC_ARGUMENT_KEYS = frozenset({'path', 'query', 'target', 'file', 'symbol', 'commit', 'evidence_id', 'obligation_id'})


def _runtime_catalog(tools) -> dict[str, Any]:
    """Build the Planner-facing contract from the live registries/policy."""
    specs = getattr(tools, 'specs', lambda: [])()
    tool_names = tuple(sorted(str(spec.name) for spec in specs))
    skill_names = tuple(str(name) for name in SKILLS)
    parallel_names = tuple(name for name in tool_names if name in PARALLEL_ALLOWED_TOOLS)
    kinds = tuple(str(kind.value) for kind in ActionKind)
    question_types = tuple(str(x) for x in get_args(QuestionType))
    return {
        'skills': skill_names,
        'tools': tool_names,
        'parallel_tools': parallel_names,
        'kinds': kinds,
        'question_types': question_types,
    }


def _render_runtime_catalog(catalog: dict[str, Any]) -> str:
    lines = [
        'RUNTIME_CATALOG (single source of truth; values outside these lists are invalid):',
        f"VALID_SKILLS: [{', '.join(catalog['skills'])}]",
        f"VALID_TOOLS: [{', '.join(catalog['tools'])}]",
        f"PARALLEL_ALLOWED_TOOLS: [{', '.join(catalog['parallel_tools'])}]",
        f"VALID_KINDS: [{', '.join(catalog['kinds'])}]",
        f"VALID_QUESTION_TYPES: [{', '.join(catalog['question_types'])}]",
    ]
    return '\n'.join(lines)


def _protected_values(value: Any) -> dict[str, Any]:
    """Return semantic argument values that a format repair must not alter."""
    found = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _SEMANTIC_ARGUMENT_KEYS:
                found[key] = copy.deepcopy(item)
            found.update({f'{key}.{nested}': nested_value for nested, nested_value in _protected_values(item).items()})
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.update({f'{index}.{nested}': nested_value for nested, nested_value in _protected_values(item).items()})
    return found


def _validate_repair_patch(primary: Any, repaired: Any) -> tuple[dict[str, Any], str | None]:
    """Merge a repair patch while retaining primary semantic intent.

    Full-object responses remain accepted for compatibility with old providers/tests,
    but only repairable fields are taken from them. Immutable differences are rejected.
    """
    if not isinstance(primary, dict) or not isinstance(repaired, dict):
        return {}, 'invalid_structural_patch'
    if repaired.get('repair_failed') is not None:
        return {}, 'invalid_structural_patch'
    for field in _IMMUTABLE_FIELDS:
        if field in repaired and repaired.get(field) != primary.get(field):
            # Legacy full-object repair responses often restate a rationale. Keep
            # the primary rationale; a format repair must not be allowed to alter
            # it, but this harmless restatement need not reject the whole repair.
            if field == 'reason':
                continue
            # A one-child parallel response may expose the same semantic tool in a
            # child while setting the group-level tool to null. Normalization below
            # converts that unambiguous representation back to one tool action.
            same_child_tool=(field == 'tool' and primary.get('kind') == 'tool' and
                             repaired.get('kind') == 'parallel' and repaired.get('tool') is None and
                             isinstance(repaired.get('actions'),list) and len(repaired['actions']) == 1 and
                             isinstance(repaired['actions'][0],dict) and repaired['actions'][0].get('tool') == primary.get('tool'))
            if not same_child_tool:
                return {}, 'immutable_field_modified'
    unknown = set(repaired) - (_REPAIRABLE_FIELDS | _IMMUTABLE_FIELDS | {'repair_failed'})
    if unknown:
        return {}, 'invalid_structural_patch'
    patch = {key: copy.deepcopy(value) for key, value in repaired.items() if key in _REPAIRABLE_FIELDS}

    if 'arguments' in patch:
        original = primary.get('arguments')
        replacement = patch['arguments']
        if not isinstance(replacement, dict):
            return {}, 'invalid_structural_patch'
        if isinstance(original, dict):
            # Preserve every existing argument, including values not in the named
            # protected set. Repair may add schema defaults, but cannot rewrite data.
            for key, value in original.items():
                if key not in replacement or replacement[key] != value:
                    return {}, 'semantic_change_attempted'
        elif original is not None and not isinstance(original, (list, tuple)):
            return {}, 'semantic_change_attempted'

    if 'information_need_structured' in patch:
        original = primary.get('information_need_structured')
        replacement = patch['information_need_structured']
        if original is None:
            if replacement is not None:
                return {}, 'semantic_change_attempted'
        elif not isinstance(original, dict) or not isinstance(replacement, dict):
            return {}, 'invalid_structural_patch'
        else:
            for key in ('target', 'evidence_goal'):
                if replacement.get(key) != original.get(key):
                    return {}, 'semantic_change_attempted'

    if 'actions' in patch:
        original = primary.get('actions')
        replacement = patch['actions']
        if not isinstance(replacement, list) or (original is not None and not isinstance(original, list)):
            return {}, 'semantic_change_attempted'
        if original is None:
            # Backward-compatible full-object repair: the primary omitted the
            # structural container, so one child can be normalized if it preserves
            # the existing top-level tool intent.
            if len(replacement) != 1 or not isinstance(replacement[0],dict) or replacement[0].get('tool') != primary.get('tool'):
                return {}, 'semantic_change_attempted'
        elif len(replacement) != len(original):
            return {}, 'semantic_change_attempted'
        for before, after in zip(original or [], replacement):
            if not isinstance(before, dict) or not isinstance(after, dict):
                return {}, 'invalid_structural_patch'
            if before.get('tool') != after.get('tool') or before.get('action_id') != after.get('action_id'):
                return {}, 'semantic_change_attempted'
            if _protected_values(before.get('arguments')) != _protected_values(after.get('arguments')):
                return {}, 'semantic_change_attempted'

    merged = copy.deepcopy(primary)
    merged.update(patch)
    return merged, None

SYSTEM="""You are the planner inside a read-only software debugging agent. Diagnose the issue; never propose edits, patches, write commands, package installation, network side effects, or repository mutation. Every conclusion must be grounded in repository evidence. Choose one next action, not a workflow plan. You may choose kind="parallel" only for 2-4 independent read-only tool calls that serve the same information need; child arguments must not depend on sibling results. Prefer falsification over confirmation. Do not repeat equivalent calls. High confidence does not grant permission. Tool argument names and constraints are strict: use only fields shown in the tool catalog. For read_file, use start_line plus line_count, where line_count is an integer from 1 through 800. line_count is the number of lines to read, so do not calculate or provide end_line. When a concrete source file is implicated, prefer one broad bounded read covering the relevant file or complete implementation region instead of mechanically splitting it into 200-line pages. Context IDs are optional hints: only reference IDs that appear in CONTEXT_CATALOG.

When another tool call is necessary, describe the unresolved question in both information_need and information_need_structured when possible. Keep structured fields semantically stable across paraphrases. Generic examples:
- Exact-symbol issue: target="Parser.visit_unknown", question_type="location", evidence_goal="locate unknown-node dispatch implementation".
- Behavioral issue: target="schema compatibility decision", question_type="location", evidence_goal="find implementation deciding whether schemas are compatible".
Do not copy example targets when they are unrelated to the current issue. Repository paths should preferably be canonical repo-relative paths such as astroid/modutils.py. read_file can recover a uniquely identifiable read-only suffix/basename, but ambiguous paths require replanning. grep glob semantics are explicit: *.py matches basenames recursively; patterns containing / are anchored to the repository root and are not fuzzy-resolved."""

class Planner:
    def __init__(self,llm,tools,model='',compact_prompt=False,skill_library=None,max_parallel_actions=4): self.llm=llm; self.tools=tools; self.model=model; self.compact_prompt=compact_prompt; self.last_prompt_breakdown={}; self.last_action_normalization=None; self.last_repair_rejection_reason=None; self.skill_library=skill_library or SkillLibrary(); self.max_parallel_actions=max(2,int(max_parallel_actions))
    def propose(self,state:AgentState,context:str,logical_timeout_seconds:float|None=None,
                prompt_budget=None, max_output_tokens: int | None = None) -> ActionProposal:
        contract=(render_contract_compact(AgentActionContract,"AGENT_ACTION_SCHEMA") if self.compact_prompt else render_contract(AgentActionContract,"AGENT_ACTION_SCHEMA"))
        skills=render_skill_catalog(compact=self.compact_prompt)
        catalog=_runtime_catalog(self.tools)
        catalog_text=_render_runtime_catalog(catalog)
        active_name=(state.actions[-1].skill if state.actions else None)
        active_skill=self.skill_library.render_active(active_name)
        tools_text=self.tools.render(compact=self.compact_prompt)
        instruction=("Use only the current RUNTIME_CATALOG: skill MUST be a VALID_SKILLS value; "
                     "tool MUST be a VALID_TOOLS value; every parallel child tool MUST be in "
                     "PARALLEL_ALLOWED_TOOLS. Never invent general, planner, search, noop, "
                     "placeholder, or any other value outside the catalog. "
                     "retain_context_ids is optional. If present, use only IDs from CONTEXT_CATALOG. "
                     "information_need must state the precise unresolved fact that justifies another tool call. "
                     "When practical, also fill information_need_structured with a stable target, "
                     "question_type, and evidence_goal; use null rather than inventing a field. "
                     "Choose retrieval mode by the information need: exact identifiers favor lexical/symbol/grep; "
                     "behavioral concepts favor semantic; uncertain or weak lexical vocabulary favors hybrid. "
                     "Retrieval candidates are not evidence until source is read.")
        user=f"{context}\n\nSKILLS (progressive catalog):\n{skills}\n"
        if active_skill: user+=f"\nACTIVE_SKILL_GUIDANCE ({active_name}):\n{active_skill}\n"
        examples=("MINIMAL_VALID_SHAPES:\n"
                   "tool: {kind: tool, skill: <VALID_SKILLS>, tool: <VALID_TOOLS>, arguments: {...}, actions: []}\n"
                   "parallel: {kind: parallel, skill: <VALID_SKILLS>, tool: null, arguments: {}, actions: [two existing independent children]}\n"
                   "INVALID: parallel with one child; skill=general; tool=noop; parallel child=git_log when absent from PARALLEL_ALLOWED_TOOLS.")
        user+=f"\n{catalog_text}\n\n{examples}\n\nTOOLS (strict schemas; suggested skill/tool affinity is guidance, not permission):\n{tools_text}\n\n{contract}\n{instruction}"
        self.last_prompt_breakdown={'system_chars':len(SYSTEM),'context_chars':len(context),'skill_catalog_chars':len(skills),'tool_catalog_chars':len(tools_text),'runtime_catalog_chars':len(catalog_text),'active_skill_chars':len(active_skill),'contract_chars':len(contract),'instruction_chars':len(instruction),'valid_skill_count':len(catalog['skills']),'valid_tool_count':len(catalog['tools']),'parallel_tool_count':len(catalog['parallel_tools']),'question_type_count':len(catalog['question_types'])}
        if prompt_budget is not None:
            decision = prompt_budget.check_prompt(
                "planner", SYSTEM, user,
                breakdown={"incident": context, "tools": tools_text},
            )
            self.last_prompt_breakdown.update({
                "estimated_prompt_tokens": decision.estimated_prompt_tokens,
                "input_hard_capacity": decision.input_hard_capacity,
                "budget_state": decision.state.value,
                "token_breakdown": dict(decision.breakdown),
            })
        call_started=time.monotonic()
        data=complete_json_compat(self.llm,SYSTEM,user,model=self.model or None,
                                  logical_timeout_seconds=logical_timeout_seconds,
                                  max_output_tokens=max_output_tokens)
        self.last_action_normalization=None
        self.last_repair_rejection_reason=None
        data, normalization = normalize_planner_action(data, max_parallel_actions=self.max_parallel_actions)
        self.last_action_normalization=normalization
        try:
            c=AgentActionContract.model_validate(data)
        except ValidationError as exc:
            details=compact_validation_error(exc)
            repair_schema=(f'''PLANNER_FORMAT_REPAIR
Return ONLY a structural repair patch object, not a new plan. You are NOT replanning.
Allowed patch fields: kind, arguments, actions, information_need_structured.
Do not return or change skill, tool, reason, confidence, file paths, queries, targets,
evidence, information_need, or retain_context_ids. Do not invent skills or tools.
Never add noop, placeholder, fake, or new child actions. Preserve existing child tool and
semantic arguments exactly. Only question_type may be canonicalized to VALID_QUESTION_TYPES.
parallel requires 2..{self.max_parallel_actions} existing independent child actions;
parallel with one valid child is normalized to one tool before repair. If it cannot be
repaired without changing intent, return {{"repair_failed": "structural intent cannot be preserved"}}.
            {catalog_text}''')
            remaining=None if logical_timeout_seconds is None else max(0.0,float(logical_timeout_seconds)-(time.monotonic()-call_started))
            if remaining is not None and remaining <= 0:
                raise PlannerContractError(f'planner contract validation failed: {details}',validation_errors=details,output=data) from exc
            repair_user=json.dumps({'validation_errors':details,'kind':self._safe_scalar(data,'kind'),'tool':self._safe_scalar(data,'tool'),'skill':self._safe_scalar(data,'skill'),'arguments_type':type(data.get('arguments')).__name__ if isinstance(data,dict) else type(data).__name__,'actions_type':type(data.get('actions')).__name__ if isinstance(data,dict) else None,'actions_count':len(data.get('actions')) if isinstance(data,dict) and isinstance(data.get('actions'),list) else None},ensure_ascii=False)
            try:
                if prompt_budget is not None:
                    repair_decision = prompt_budget.check_prompt(
                        "planner", repair_schema, repair_user,
                        breakdown={"incident": repair_user},
                    )
                    repair_breakdown = {
                        "stage": "planner_schema_repair",
                        "estimated_prompt_tokens": repair_decision.estimated_prompt_tokens,
                        "input_hard_capacity": repair_decision.input_hard_capacity,
                        "budget_state": repair_decision.state.value,
                        "token_breakdown": dict(repair_decision.breakdown),
                    }
                    self.last_prompt_breakdowns.append(repair_breakdown)
                repaired=complete_json_compat(self.llm,repair_schema,repair_user,model=self.model or None,
                                              logical_timeout_seconds=remaining,
                                              max_output_tokens=max_output_tokens)
                merged, rejection = _validate_repair_patch(data, repaired)
                if rejection:
                    self.last_repair_rejection_reason=rejection
                    raise PlannerContractError('planner repair rejected: '+rejection, validation_errors=details, output=repaired, repair_rejection_reason=rejection)
                repaired, normalization = normalize_planner_action(merged, max_parallel_actions=self.max_parallel_actions)
                self.last_action_normalization=normalization
                c=AgentActionContract.model_validate(repaired)
                data=repaired
            except Exception as repair_exc:
                details2=compact_validation_error(repair_exc) if isinstance(repair_exc,ValidationError) else str(repair_exc)[:500]
                rejection=getattr(repair_exc,'metadata',{}).get('repair_rejection_reason') if isinstance(repair_exc,PlannerContractError) else self.last_repair_rejection_reason
                raise PlannerContractError(f'planner contract validation failed after bounded repair: {details2}',validation_errors=details2,output=(repaired if 'repaired' in locals() else data),repair_rejection_reason=rejection) from repair_exc
        return ActionProposal(
            kind=ActionKind(c.kind), skill=c.skill, reason=c.reason, confidence=c.confidence,
            tool=c.tool, arguments=c.arguments, expected_evidence=c.expected_evidence,
            information_need=c.information_need, information_need_structured=(c.information_need_structured.model_dump() if c.information_need_structured else None), retain_context_ids=c.retain_context_ids,
            actions=[x.model_dump() for x in c.actions],
        )

    @staticmethod
    def _safe_scalar(data, key):
        value=data.get(key) if isinstance(data,dict) else None
        return value if isinstance(value,(str,int,float,bool)) or value is None else type(value).__name__


# Explicit migration name for callers that want to make the fallback boundary visible.
StructuredJSONPlannerFallback = Planner
