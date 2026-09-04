from __future__ import annotations
import copy
import json
import time
import inspect
from dataclasses import dataclass
from typing import Any, get_args
from pydantic import ValidationError
from debug_assistant.models import ActionProposal, ActionKind, AgentState
from debug_assistant.contracts import (AgentActionContract, PlannerIntent, QuestionType, compact_validation_error,
                                       render_contract, render_contract_compact)
from debug_assistant.llm.base import complete_json_compat
from debug_assistant.llm.base import LLMOutputError, LLMResponse, LLMToolCall, ProviderCapabilities
from debug_assistant.skills.catalog import SKILLS, render_skill_catalog
from debug_assistant.skills.loader import SkillLibrary
from debug_assistant.tools.registry import PARALLEL_ALLOWED_TOOLS


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


@dataclass(frozen=True, slots=True)
class NativePlannerResult:
    """Semantic planner metadata plus provider-native tool requests.

    It intentionally has no ``kind``, ``skill`` or ``parallel`` field. Execution
    shape is compiled later by ``ToolOrchestrator``.
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


class NativeToolPlanner:
    def __init__(self, llm, tools, model: str = ""):
        self.llm, self.tools, self.model = llm, tools, model
        self.last_prompt_breakdown = {}

    def propose(self, state: AgentState, context: str, *, logical_timeout_seconds=None,
                on_attempt_started=None) -> NativePlannerResult:
        if not getattr(getattr(self.llm, "capabilities", ProviderCapabilities()), "tool_calling", False):
            raise PlannerContractError("provider does not support native tool calling",
                                       validation_errors=["tool_calling=false"])
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
        user = f"{context}\n\nCURRENT_GOAL: {state.task.issue}\n"
        schemas = self.tools.function_schemas()
        self.last_prompt_breakdown = {"system_chars": len(system), "context_chars": len(user),
                                      "tool_schema_count": len(schemas)}
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
        try:
            response = method(system, user, **kwargs)
        except LLMOutputError as exc:
            raise NativePlannerContractError(
                str(exc), error_type=getattr(exc, "error_type", "provider_contract_mismatch"),
                validation_errors=[getattr(exc, "error_type", "provider_contract_mismatch")],
                index=getattr(exc, "index", None), tool=getattr(exc, "tool", None),
            ) from exc
        if not isinstance(response, LLMResponse):
            raise NativePlannerContractError(
                "provider returned a non-typed native response",
                error_type="provider_contract_mismatch",
            )
        names = {spec.name for spec in self.tools.specs()}
        for call in response.tool_calls:
            if not isinstance(call, LLMToolCall) or not isinstance(call.arguments, dict):
                raise NativePlannerContractError(
                    "provider returned a malformed native tool call",
                    error_type="malformed_tool_call",
                    validation_errors=["malformed_tool_call"],
                )
            if call.name not in names:
                raise NativePlannerContractError(
                    f"unknown tool: {call.name}", error_type="unknown_tool",
                    validation_errors=["unknown_tool"], output={"tool": call.name},
                )
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
            response=response, tool_calls=response.tool_calls,
            reason=intent.reason or "",
            information_need=intent.information_need or "",
            expected_evidence=str(metadata.get("expected_evidence") or ""),
            retain_context_ids=tuple(str(x) for x in (metadata.get("retain_context_ids") or [])),
            obligation_ids=tuple(str(x) for x in (metadata.get("obligation_ids") or [])),
            intent=intent,
            assistant_text=assistant_text or None,
        )


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

SYSTEM="""You are the planner inside a read-only software debugging agent. Diagnose the issue; never propose edits, patches, write commands, package installation, network side effects, or repository mutation. Every conclusion must be grounded in repository evidence. Choose one next action, not a workflow plan. You may choose kind="parallel" only for 2-4 independent read-only tool calls that serve the same information need; child arguments must not depend on sibling results. Prefer falsification over confirmation. Do not repeat equivalent calls. High confidence does not grant permission. Tool argument names and constraints are strict: use only fields shown in the tool catalog. For read_file, start_line/end_line are inclusive and a request may contain at most 200 lines, so end_line - start_line + 1 <= 200. Context IDs are optional hints: only reference IDs that appear in CONTEXT_CATALOG.

When another tool call is necessary, describe the unresolved question in both information_need and information_need_structured when possible. Keep structured fields semantically stable across paraphrases. Generic examples:
- Exact-symbol issue: target="Parser.visit_unknown", question_type="location", evidence_goal="locate unknown-node dispatch implementation".
- Behavioral issue: target="schema compatibility decision", question_type="location", evidence_goal="find implementation deciding whether schemas are compatible".
Do not copy example targets when they are unrelated to the current issue. Repository paths should preferably be canonical repo-relative paths such as astroid/modutils.py. read_file can recover a uniquely identifiable read-only suffix/basename, but ambiguous paths require replanning. grep glob semantics are explicit: *.py matches basenames recursively; patterns containing / are anchored to the repository root and are not fuzzy-resolved."""

class Planner:
    def __init__(self,llm,tools,model='',compact_prompt=False,skill_library=None,max_parallel_actions=4): self.llm=llm; self.tools=tools; self.model=model; self.compact_prompt=compact_prompt; self.last_prompt_breakdown={}; self.last_action_normalization=None; self.last_repair_rejection_reason=None; self.skill_library=skill_library or SkillLibrary(); self.max_parallel_actions=max(2,int(max_parallel_actions))
    def propose(self,state:AgentState,context:str,logical_timeout_seconds:float|None=None) -> ActionProposal:
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
        call_started=time.monotonic()
        data=complete_json_compat(self.llm,SYSTEM,user,model=self.model or None,logical_timeout_seconds=logical_timeout_seconds)
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
                repaired=complete_json_compat(self.llm,repair_schema,repair_user,model=self.model or None,logical_timeout_seconds=remaining)
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
