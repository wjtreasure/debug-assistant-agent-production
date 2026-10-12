from __future__ import annotations
import copy
import hashlib
import json
import inspect
import re
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Annotated, Any, Iterable, Literal
from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError, model_validator
from debug_assistant.agent.decision_surface import (
    DecisionSurfaceError, interpretation_surface,
)
from debug_assistant.agent.decision_turn import NextActionIntent, decision_turn_surface, compile_next_action
from debug_assistant.context.requests import ContextRequestBatch
from debug_assistant.memory.evidence_projector import EvidenceFactProposal, EvidenceProposal
from debug_assistant.agent.requirement_targeting import requirement_table, resolve_fact, RequirementTargetError, targeting_feedback
from debug_assistant.incidents.semantic_transition import (
    disposition_targets, validate_disposition_targets, SemanticTransitionError,
)
from debug_assistant.models import AgentState
from debug_assistant.context.models import ContextView, PromptSnapshot, redact_snapshot_value as _redact_snapshot_value
from debug_assistant.contracts import PlannerIntent
from debug_assistant.llm.base import (
    LLMError, LLMOutputError, LLMResponse, LLMToolCall, ProviderCapabilities,
    extract_json,
    estimate_tokens_char4,
    strict_response_format,
)
from debug_assistant.skills.catalog import (
    CANONICAL_INCIDENT_SKILLS,
)
from debug_assistant.skills.projection import SkillViews, load_skill_views, action_choices
from debug_assistant.incidents.contracts import (
    CompletionContract, Contradiction, SourceClaim, SourceMechanismStatus,
)
from debug_assistant.incidents.fault_taxonomy import ProposedFaultCode, render_public_fault_taxonomy
from debug_assistant.symbols import CanonicalOutputModel, CanonicalSymbolError
from debug_assistant.incidents.verification_plan import (
    PlanAdmissionDecision, VerificationPlanProposal, VerificationPlanError,
    EnrichedVerificationPlan, enrich_verification_plan,
    registry_schema_sha256, validate_verification_plan,
)
from debug_assistant.incidents.action_runtime import ActionRuntime


SOURCE_READ_MAX_LINES = 800

_INCIDENT_SEMANTIC_SYSTEM = """Interpret the visible observations and canonical Evidence.
Return exactly one submit_interpretation_decision command matching the strict response schema. Propose semantics and one next_action_intent in the same reasoning turn.
KEEP_CURRENT accumulates Evidence without changing causal meaning or revision.
REFINE changes causal meaning; REPLACE changes hypothesis identity. New revisions need fresh qualification.
Select exact Runtime references. Requirements express completion, not truth: semantic roles do not prove causality.
Choose the Evidence relation to the hypothesis and to the question independently.
Evidence coverage is an explicit action: COVER_EXISTING_REQUIREMENT selects the nonempty exact requirement_refs this source supports; NO_RELEVANT_COVERAGE links background information to obligation_ref and covers no fact. SUPPORTS relation or a prose claim never closes a requirement. Select only refs justified by the source and its CompletionContract.
Use acknowledgements for successful operational-only or irrelevant results; they create no Evidence or coverage.
Failed Tool results are already recorded operationally and have no ACK/Evidence action.
INITIAL creates a hypothesis and explicit questions/contracts without Evidence; retained observations are interpreted after commit.
Questions must define observable facts necessary to verify the causal explanation, not merely repeat the symptom. Select each fact's source_profile_ref from next_action_catalog.source_profiles; that complete Registry kind/source/producer group is its provenance constraint. Explicit null means no provenance restriction. source_read denotes application source code, while incident_snapshot denotes captured operational data. A profile never implies that the requested content exists, has been acquired, or is true.
Contradiction citations select already admitted contradictory relationships in the current scope. To interpret new or reused content as contradictory, use Evidence with the explicit relation first; a structured contradiction claim cannot create an Evidence relationship.
Runtime owns IDs, scopes, provenance, qualification, atomic commit and Tool execution.
CONTEXT_REQUEST selects retained local content without Tool I/O or semantic mutation.
Select question_mode EXISTING to use canonical requirements: question must be null and next_action_intent.target selects requirement_ref. Select NEW only to create one complete decision.question with its target_fact and additional_facts; its next_action_intent has no target field and binds this exact proposal after admission. This is the sole Question owner. No PROPOSED_QUESTION target or independent empty proposal exists.
Existing verification identities and CompletionContracts are immutable: use canonical requirement/obligation refs, never re-declare them in a new question. Use explicit lifecycle dispositions when changing hypothesis.
Read canonical_semantic_input.obligations before choosing question_mode. An existing OPEN Question remains the same object across Tool results and local context recovery; acquiring another source for it uses EXISTING. Changing the operation or wording of a CompletionContract does not authorize recreating that identity. Retry feedback describes the rejected decision; correct the decision itself, rather than returning the same NEW proposal.
Executable arguments are literal typed values from visible context; supply every field, using null only where allowed. Acquisition history and Registry parameter combinations remain Runtime admission constraints. Never select an already ACQUIRED operation with the same canonical arguments: use its retained observation/Evidence references instead. A new query requires a distinct meaningful acquisition; changing its question does not make the data new.
Runtime commits semantics first, checks Finalization, then admits the intent against committed canonical state. The intent never executes a Tool directly."""



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
    """A malformed native provider response, distinct from JSON envelopes."""

    def __init__(self, message, *, error_type="provider_contract_mismatch", validation_errors=None,
                 output=None, index=None, tool=None):
        super().__init__(message, validation_errors=validation_errors, output=output)
        self.metadata["error_type"] = error_type
        if index is not None:
            self.metadata["index"] = index
        if tool is not None:
            self.metadata["tool"] = tool
        self.metadata["error_layer"] = _planner_error_layer(error_type, tool=tool)


class PlannerContractExhausted(PlannerContractError):
    """The bounded native Planner contract recovery path was exhausted."""

    def __init__(self, cause: PlannerContractError):
        super().__init__(
            "planner structured contract remained invalid after bounded retry",
            validation_errors=list(getattr(cause, "metadata", {}).get("validation_errors", ())),
        )
        self.metadata.update(getattr(cause, "metadata", {}))
        self.metadata["error_type"] = "planner_contract_exhausted"


def _planner_error_layer(error_type: str, *, tool: str | None = None) -> str:
    """Keep transport, parser, Tool, semantic, and finalization failures distinct."""
    value = str(error_type or "").casefold()
    if any(token in value for token in ("transport", "timeout", "provider")):
        return "transport"
    if any(token in value for token in ("json", "parse", "malformed_tool_call", "response_type")):
        return "parser"
    if tool == "finalize_diagnosis" or "final" in value:
        return "finalization"
    if any(token in value for token in ("tool", "unknown_tool", "executable")):
        return "tool"
    return "semantic"


class PlannerObligationProposal(BaseModel):
    """Planner-owned semantic proposal without Runtime lifecycle fields.

    Obligation identity, lifecycle status, Evidence links and capability
    blocking are assigned by the Runtime.  Keeping this boundary as a
    dedicated model prevents provider JSON from being mistaken for canonical
    RCA state.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim: str = Field(min_length=1)
    evidence_requirement: str = ""
    target_component: str = ""
    verification_question: str = ""
    evidence_type: str = ""
    completion_contract: CompletionContract
    verification_role: Literal["NECESSARY", "EXPLORATORY"] = "NECESSARY"


from debug_assistant.incidents.contracts import HypothesisTransition


class PlannerHypothesisProposal(BaseModel):
    """Small provider proposal; Runtime remains the semantic authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transition: HypothesisTransition
    claim: str = ""
    component: str = ""
    fault: str = ""
    mechanism: str = ""
    supporting_evidence_ids: tuple[str, ...] = ()
    contradicting_evidence_ids: tuple[str, ...] = ()
    mechanism_category: str = ""
    fault_code: ProposedFaultCode = ""
    fault_explanation: str = ""
    source_claims: tuple[SourceClaim, ...] = ()

    @model_validator(mode="after")
    def validate_lifecycle_fields(self):
        fields = set(type(self).model_fields) - {"transition"}
        if self.transition.kind == "KEEP_CURRENT":
            if any(getattr(self, key) for key in fields):
                raise ValueError("KEEP_CURRENT forbids hypothesis fields; propose Evidence relationships separately")
        elif not self.claim.strip():
            raise ValueError("hypothesis_claim_required")
        return self


class OperationalAcknowledgement(BaseModel):
    """A visible result was considered without claiming Evidence or coverage."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    observation_ref: str = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=1000)


class InterpretationDelta(CanonicalOutputModel):
    """Semantic proposal only; Runtime atomically commits before Planning."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    skill_id: str = Field(min_length=1)
    evidence_need: str = Field(min_length=1)
    hypothesis: PlannerHypothesisProposal
    operational_acknowledgements: tuple[OperationalAcknowledgement, ...] = ()
    new_obligation_proposals: tuple[PlannerObligationProposal, ...] = ()
    evidence_proposals: tuple[EvidenceFactProposal, ...] = ()
    contradictions: tuple[Contradiction, ...] = ()


class PlanningDecision(CanonicalOutputModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    plan: VerificationPlanProposal


class SemanticInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["SEMANTIC_DELTA"]
    delta: InterpretationDelta


class ContextInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["CONTEXT_REQUEST"]
    batch: ContextRequestBatch


class InterpretationDecision(RootModel[Annotated[SemanticInterpretation | ContextInterpretation,
                                               Field(discriminator="kind")]]):
    model_config = ConfigDict(frozen=True)



def pending_context_targets(view, projection):
    """Pending acquired bodies must be visible before their semantic turn."""
    if not projection.get('active_hypothesis'):
        return ()  # INITIAL/contracts still commit before bootstrap interpretation.
    return tuple(key for key in projection.get('pending_observation_ids', ())
                 if key not in view.visible_observation_ids)


def acknowledgement_targets(view):
    """Match Runtime ACK admission: successful, canonically visible observations."""
    return tuple(row['observation_id'] for row in view.reference_metadata.get('visible_observations', ())
                 if row.get('ok') is True and row['observation_id'] in view.visible_observation_ids)


class PlannerTransportProfile(str, Enum):
    """Declared Planner transport selected once for the session."""

    SEMANTIC_PLAN = "semantic_plan"


def _semantic_prompt_projection(projection):
    """Detached metadata view; canonical contracts and ContextManager bodies stay owned."""
    result = copy.deepcopy(projection)
    # Definitions occur once in obligations; this table is only an exact ref index.
    result['canonical_requirements'] = [{key: row[key] for key in
        ('requirement_ref', 'obligation_id', 'slot')} for row in projection.get('canonical_requirements', ())]
    result['obligations'] = [{key: value for key, value in row.items()
        if key not in {'required_fact_slots', 'verification_key'}}
        for row in result.get('obligations', ())]
    # Each acquisition still exposes exact tool arguments, result/state and intents.
    # The operational snapshot identity is already carried once at top level.
    result['acquisitions'] = [{key: value for key, value in row.items()
        if key not in {'snapshot_identity'}}
        for row in result.get('acquisitions', ())]
    return result


@dataclass(frozen=True, slots=True)
class InterpretationTurnInput:
    turn_id: str
    state_version: str
    state_snapshot_hash: str
    task_kind: str
    goal: str
    semantic_projection_json: str
    active_skill_id: str
    feasible_skill_ids: tuple[str, ...]
    skill_semantic_guidance_json: str
    context_view: ContextView
    local_context_feedback_json: str = ""
    action_input: PlanningTurnInput | None = None

    @property
    def semantic_projection(self):
        projection = json.loads(self.semantic_projection_json)
        projection["canonical_requirements"] = requirement_table(projection)
        projection.update(self.context_view.reference_metadata)
        return projection

    @property
    def context(self):
        return self.context_view.text

    @property
    def known_evidence_ids(self):
        return self.context_view.visible_evidence_refs

    @property
    def acquired_source_ranges(self):
        return self.context_view.acquired_source_ranges

    @property
    def visible_source_ranges(self):
        return self.context_view.visible_source_ranges


@dataclass(frozen=True, slots=True)
class PlanningTurnInput:
    """Immutable action view rebuilt from actual committed canonical state."""

    turn_id: str
    state_version: str
    state_snapshot_hash: str
    task_kind: str
    goal: str
    context: str
    semantic_projection_json: str
    active_skill_id: str
    feasible_skill_ids: tuple[str, ...]
    skill_views: tuple[SkillViews, ...]
    skill_tool_names_by_id: tuple[tuple[str, tuple[str, ...]], ...]
    visible_tool_names: tuple[str, ...]
    executable_tool_schemas_json: str
    budget_projection_json: str
    action_ledger_json: str
    acquired_source_ranges: tuple[tuple[str, int, int], ...]
    context_visible_source_ranges: tuple[tuple[str, int, int], ...]
    visible_source_ranges: tuple[tuple[str, int, int], ...]
    known_evidence_ids: tuple[str, ...]
    action_choices_json: str

    @property
    def executable_tool_schemas(self) -> list[dict[str, Any]]:
        return json.loads(self.executable_tool_schemas_json)

    @property
    def semantic_projection(self) -> dict[str, Any]:
        return json.loads(self.semantic_projection_json)

    @property
    def action_choices(self) -> tuple[dict[str, Any], ...]:
        return tuple(json.loads(self.action_choices_json))




def _snapshot_response(response: LLMResponse | None) -> dict[str, Any] | None:
    if response is None:
        return None
    return _redact_snapshot_value({
        "content": response.content,
        "structured": response.structured,
        "tool_calls": [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in response.tool_calls
        ],
        "usage": dict(response.usage or {}),
        "raw_output": dict(response.raw_output or {}),
    })


def _provider_response_for_snapshot(llm: Any) -> LLMResponse | None:
    """Build a typed snapshot envelope even when native parsing failed."""
    raw_output = getattr(llm, "last_raw_output", {}) or {}
    raw_content = getattr(llm, "last_raw_content", None)
    usage = getattr(llm, "last_usage", {}) or {}
    if not raw_output and not raw_content and not usage:
        return None
    return LLMResponse(
        content=raw_content if isinstance(raw_content, str) else None,
        usage=dict(usage),
        raw_output=raw_output if isinstance(raw_output, dict) else {},
    )


def _parse_interpretation_decision(response: LLMResponse, *, symbol_domains=None,
                                  normalization_events=None) -> InterpretationDecision | None:
    """Parse the declared semantic request; return None only when absent."""
    structured = response.structured
    payload = (
        structured
        if isinstance(structured, dict)
        else None
    )
    content = response.content.strip() if isinstance(response.content, str) else ""
    if payload is None and content:
        try:
            payload = extract_json(content)
        except Exception as exc:
            raise NativePlannerContractError(
                "Planner semantic delta is not valid JSON",
                error_type="semantic_delta_parse_error",
                validation_errors=["semantic_delta"],
                output={"semantic_delta_error": str(exc)[:500]},
            ) from exc
    if payload is None:
        return None
    try:
        return InterpretationDecision.model_validate(payload, context={
            "symbol_domains": symbol_domains or {},
            "normalization_events": normalization_events,
        })
    except ValidationError as exc:
        symbol_error = next((item.get("ctx", {}).get("error") for item in exc.errors()
                             if isinstance(item.get("ctx", {}).get("error"), CanonicalSymbolError)), None)
        error = NativePlannerContractError(
            "Planner semantic delta failed validation",
            error_type="semantic_delta_validation_error",
            validation_errors=[str(error.get("loc", ())) for error in exc.errors()],
            output={"semantic_delta": payload},
        )
        error.metadata["validation_details"] = [
            {"field": list(item["loc"]), "code": item["type"], "message": item["msg"]}
            for item in exc.errors(include_input=False, include_url=False)
        ]
        if symbol_error is not None:
            error.metadata["symbol_reason_code"] = symbol_error.reason_code
            error.metadata["symbol_domain"] = symbol_error.domain
        raise error from exc




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
    verification_obligations: tuple[PlannerObligationProposal, ...] = ()
    contradictions: tuple[Contradiction, ...] = ()
    candidate_fault_code: str = ""
    candidate_fault_explanation: str = ""
    # Audit projections retain the trace shape; semantic objects arrive only
    # from the strict InterpretationDelta parser.
    reasoning_metadata_warnings: tuple[str, ...] = ()
    reasoning_metadata_normalizations: tuple[str, ...] = ()
    reasoning_metadata_drops: tuple[str, ...] = ()
    planner_control_rejections: tuple[tuple[str, str], ...] = ()
    normalization_actions: tuple[str, ...] = ()
    obligation_id: str = ""
    expected_information_gain: str = ""
    source_claims: tuple[SourceClaim, ...] = ()
    evidence_proposals: tuple[EvidenceProposal, ...] = ()
    operational_acknowledgements: tuple[OperationalAcknowledgement, ...] = ()
    hypothesis_transition: HypothesisTransition | None = None
    hypothesis_id: str = ""
    revision: int = 0


@dataclass(frozen=True, slots=True)
class NativePlannerResult:
    """Incident semantic selection/plan, or nonincident native Tool requests."""

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
    semantic_only: bool = False
    continuation: PlanAdmissionDecision | None = None
    plan: EnrichedVerificationPlan | None = None
    context_requests: ContextRequestBatch | None = None
    next_action_intent: NextActionIntent | None = None


class NativeToolPlanner:
    def __init__(self, llm, tools, model: str = "", *,
                 public_fault_taxonomy: bool = False,
                 planner_contract_version: str = "planner_turn_v1",
                 transport_profile: PlannerTransportProfile | str = (
                     PlannerTransportProfile.SEMANTIC_PLAN
                 )):
        self.llm, self.tools, self.model = llm, tools, model
        self.public_fault_taxonomy = bool(public_fault_taxonomy)
        self.planner_contract_version = str(planner_contract_version)
        try:
            self._transport_profile = PlannerTransportProfile(transport_profile)
        except ValueError as exc:
            raise ValueError(f"unsupported Planner transport profile: {transport_profile}") from exc
        self.allowed_incident_skills = dict(CANONICAL_INCIDENT_SKILLS)
        self.last_prompt_breakdown = {}
        self.last_prompt_breakdowns: list[dict[str, Any]] = []
        self.last_request_snapshot: dict[str, Any] | None = None
        self.last_request_snapshots: list[dict[str, Any]] = []
        self.last_turn_input: PlanningTurnInput | None = None

    @property
    def transport_profile(self) -> PlannerTransportProfile:
        return self._transport_profile

    def _build_turn_input(
        self,
        *,
        state: AgentState,
        context: str,
        active_skill_id: str,
        feasible_skill_ids: tuple[str, ...],
        skill_tool_names_by_id: dict[str, tuple[str, ...]],
        visible_tool_names: tuple[str, ...],
        executable_tool_schemas: list[dict[str, Any]],
        context_snapshot: dict[str, Any],
        semantic_projection: dict[str, Any],
        budget_projection: dict[str, Any],
        action_ledger: tuple[dict[str, Any], ...],
    ) -> PlanningTurnInput:
        state_version = (
            "committed-state-v1:"
            f"events={semantic_projection.get('state_version', 0)};"
            f"step={int(state.step)};"
            f"tool_calls={int(state.tool_calls)};"
            f"observations={len(state.observations)};"
            f"evidence={len(state.evidence)}"
        )
        acquired_ranges = tuple(
            tuple(item) for item in context_snapshot.get("acquired_source_ranges", ())
        )
        visible_ranges = tuple(
            tuple(item) for item in context_snapshot.get("visible_source_ranges", ())
        )
        context_visible_ranges = visible_ranges
        schemas_json = json.dumps(
            executable_tool_schemas,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        budget_json = json.dumps(
            budget_projection,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        ledger_json = json.dumps(
            action_ledger,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        skill_views = load_skill_views(feasible_skill_ids,
            getattr(self.tools, "decision_tool_aliases", getattr(self.tools, "official_aliases", {})))
        visible_operations = {row["function"]["name"] for row in executable_tool_schemas}
        choices = action_choices(skill_views, visible_operations,
            companions={spec.name: getattr(spec, 'companion_tools', ()) for spec in self.tools.specs()})
        choices = tuple(row for row in choices if (
            (not active_skill_id or row["skill_id"] == active_skill_id)
            and row["operation"] in skill_tool_names_by_id.get(row["skill_id"], visible_operations)
        ))
        # Workflow node recipes are prior guidance, not distinct acquisitions.
        # One Registry operation has one exact command choice per selected Skill;
        # callers can still issue distinct arguments/dependent queries for it.
        operations = {}
        for row in choices:
            operations.setdefault((row["skill_id"], row["operation"]),
                {key: value for key, value in row.items() if key != "args_template"})
        choices = tuple(operations.values())
        planning_projection = dict(semantic_projection)
        planning_projection.pop("existing_evidence", None)
        # Canonical contracts are already present in obligations. Preserve the
        # ready subset by exact IDs instead of serializing the contracts again.
        if planning_projection.get("active_hypothesis"):
            planning_projection["active_hypothesis"] = {
                key: value for key, value in planning_projection["active_hypothesis"].items()
                if key != "verification_obligations"
            }
        if "ready_obligations" in planning_projection:
            planning_projection["ready_obligation_ids"] = [
                row["id"] for row in planning_projection.pop("ready_obligations")
            ]
        planning_projection["visible_observations"] = [
            {key: value for key, value in row.items() if key != "content"}
            for row in semantic_projection.get("visible_observations", ())
        ]
        semantic_json = json.dumps({**planning_projection,
            "evidence_gaps": [{**gap, "uncovered_fact_slots": [slot
                for slot in gap["required_fact_slots"] if not gap["coverage"].get(slot)]}
                for gap in semantic_projection.get("evidence_gaps", ())],
            "gap_choice_surface": ActionRuntime.gap_choices(choices, self.tools, semantic_projection)},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        material = {
            "schema_version": "planner-turn-input-v2",
            "state_version": state_version,
            "task_kind": str(state.task.metadata.get("task_kind", "")),
            "goal": str(state.task.issue),
            "context": str(context),
            "semantic_projection_json": semantic_json,
            "active_skill_id": str(active_skill_id),
            "feasible_skill_ids": feasible_skill_ids,
            "skill_views": skill_views,
            "action_choices_json": json.dumps(choices, sort_keys=True, separators=(",", ":")),
            "skill_tool_names_by_id": tuple(sorted(skill_tool_names_by_id.items())),
            "visible_tool_names": visible_tool_names,
            "executable_tool_schemas_json": schemas_json,
            "budget_projection_json": budget_json,
            "action_ledger_json": ledger_json,
            "acquired_source_ranges": acquired_ranges,
            "context_visible_source_ranges": context_visible_ranges,
            "visible_source_ranges": visible_ranges,
            "known_evidence_ids": tuple(sorted(
                str(getattr(item, "evidence_id", ""))
                for item in state.evidence
                if getattr(item, "evidence_id", None)
            )),
        }
        digest = hashlib.sha256(
            json.dumps(material, ensure_ascii=False, sort_keys=True,
                       default=asdict).encode("utf-8")
        ).hexdigest()
        turn_fields = dict(material)
        turn_fields.pop("schema_version")
        return PlanningTurnInput(
            turn_id=f"planner-turn-{digest[:16]}",
            state_snapshot_hash=digest,
            **turn_fields,
        )

    def create_turn_input(
        self, state: AgentState, context: str, *,
        visible_tool_names: Iterable[str] | None = None,
        active_skill_id: str = "",
        feasible_skill_ids: Iterable[str] | None = None,
        skill_tool_names_by_id: dict[str, Iterable[str]] | None = None,
        context_snapshot: dict[str, Any] | None = None,
        semantic_projection: dict[str, Any] | None = None,
        budget_projection: dict[str, Any] | None = None,
        action_ledger: Iterable[dict[str, Any]] | None = None,
    ) -> PlanningTurnInput:
        """Build the one immutable input consumed by a Planner session."""
        feasible = tuple(dict.fromkeys(
            str(item) for item in (
                feasible_skill_ids
                if feasible_skill_ids is not None
                else self.allowed_incident_skills
            )
            if str(item) in self.allowed_incident_skills
        ))
        if active_skill_id and active_skill_id not in feasible:
            raise NativePlannerContractError(
                "the active Skill requires current canonical feasibility",
                error_type="active_skill_unavailable", validation_errors=["active_skill_id"],
            )
        tool_names = {
            str(key): tuple(dict.fromkeys(str(item) for item in value))
            for key, value in dict(skill_tool_names_by_id or {}).items()
            if str(key) in feasible
        }
        schemas = self.tools.function_schemas(visible_tools=visible_tool_names)
        if state.task.metadata.get("task_kind") == "incident":
            visible = {schema["function"]["name"] for schema in schemas}
            schemas = [
                {**spec.function_schema(), "schema_sha256": registry_schema_sha256(spec)}
                for spec in self.tools.specs()
                if spec.name in visible and spec.capability != "diagnosis_control"
            ]
        return self._build_turn_input(
            state=state,
            context=context,
            active_skill_id=active_skill_id,
            feasible_skill_ids=feasible,
            skill_tool_names_by_id=tool_names,
            visible_tool_names=tuple(str(item) for item in (visible_tool_names or ())),
            executable_tool_schemas=schemas,
            context_snapshot=copy.deepcopy(context_snapshot or {}),
            semantic_projection=copy.deepcopy(semantic_projection or {}),
            budget_projection=copy.deepcopy(budget_projection or {}),
            action_ledger=tuple(copy.deepcopy(tuple(action_ledger or ()))),
        )

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

    @staticmethod
    def _semantic_only_selection(delta: InterpretationDelta, projection: dict) -> NativeSkillSelection:
        hypothesis = delta.hypothesis
        if hypothesis.transition.kind == "KEEP_CURRENT":
            current = projection.get("active_hypothesis")
            if not current:
                raise NativePlannerContractError("KEEP_CURRENT requires current Hypothesis", error_type="current_hypothesis_missing")
            hypothesis = hypothesis.model_copy(update={key: current.get(key, getattr(hypothesis, key))
                for key in type(hypothesis).model_fields if key != "transition"})
            hypothesis = hypothesis.model_copy(update={"source_claims": tuple(
                SourceClaim.model_validate(row) for row in hypothesis.source_claims)})
        return NativeSkillSelection(
            call_id="semantic-only",
            hypothesis_transition=hypothesis.transition,
            operational_acknowledgements=delta.operational_acknowledgements,
            skill=delta.skill_id,
            reason=delta.evidence_need.strip(),
            current_hypothesis=hypothesis.claim.strip(),
            evidence_gap=delta.evidence_need.strip(),
            candidate_component=hypothesis.component.strip(),
            candidate_fault=hypothesis.fault.strip(),
            candidate_mechanism=hypothesis.mechanism.strip(),
            supporting_evidence_ids=tuple(hypothesis.supporting_evidence_ids),
            contradicting_evidence_ids=tuple(hypothesis.contradicting_evidence_ids),
            required_evidence_gaps=(),
            evidence_sufficiency="insufficient",
            remaining_evidence_need="",
            source_mechanism_status="unknown",
            mechanism_category=hypothesis.mechanism_category.strip(),
            verification_obligations=tuple(delta.new_obligation_proposals),
            contradictions=delta.contradictions,
            candidate_fault_code=hypothesis.fault_code.strip(),
            candidate_fault_explanation=hypothesis.fault_explanation.strip(),
            source_claims=tuple(hypothesis.source_claims),
            evidence_proposals=tuple(resolve_fact(item,
                projection=projection,
                transition=hypothesis.transition, index=index,
            ) for index, item in enumerate(delta.evidence_proposals)),
            obligation_id="",
        )


    def _request_snapshot(
        self, *, system: str, user: str, schemas: list[dict[str, Any]],
        max_output_tokens: int | None, turn_input: PlanningTurnInput,
        response: LLMResponse | None = None, status: str = "request_built",
        logical_timeout_seconds: float | None = None,
        enable_thinking: bool | None = None,
        request_kind: str = "planner",
        contract_retry: bool = False,
        context_resolution_round: int = 0,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        capabilities = getattr(self.llm, "capabilities", None)
        provider = str(
            getattr(capabilities, "provider", "")
            or getattr(self.llm, "provider", "")
            or type(self.llm).__name__
        )
        model = str(
            self.model
            or getattr(capabilities, "model", "")
            or getattr(self.llm, "default_model", "")
        )
        transport_mode = "json_schema_strict" if response_format else self.transport_profile.value
        payload = {
            "schema_version": "planner-request-snapshot-v4",
            "choice_contract_version": "node-local-choice-v1" if turn_input.task_kind == "incident" else "",
            "status": status,
            "provider": provider,
            "model": model,
            "temperature": getattr(self.llm, "temperature", None),
            "max_tokens": max_output_tokens,
            "logical_timeout_seconds": logical_timeout_seconds,
            "enable_thinking": enable_thinking,
            "transport_profile": self.transport_profile.value,
            "transport_mode": transport_mode,
            "requested_transport_mode": transport_mode,
            "effective_transport_mode": transport_mode,
            "request_kind": request_kind,
            "contract_retry": bool(contract_retry),
            "context_resolution_round": context_resolution_round,
            "attempt_kind": "contract_retry" if contract_retry else "context_resolution" if context_resolution_round else "first",
            "turn_id": turn_input.turn_id,
            "planner_contract_version": self.planner_contract_version,
            "state_version": turn_input.state_version,
            "state_snapshot_sha256": turn_input.state_snapshot_hash,
            "system_prompt": system,
            "user_prompt": user,
            "tool_schemas": schemas,
            "response_format": response_format,
            "tool_schema_sha256": hashlib.sha256(json.dumps(schemas, sort_keys=True).encode()).hexdigest(),
            "executable_registry_sha256": hashlib.sha256(turn_input.executable_tool_schemas_json.encode()).hexdigest()
                if isinstance(turn_input, PlanningTurnInput) else None,
            "response": _snapshot_response(response),
            "redacted": True,
        }
        for marker, name in (("SELECTED_SEMANTIC_PROJECTION:\n", "semantic_projection"),
                             ("ACTION_CHOICES:\n", "action_choices"),
                             ("FEASIBLE_SKILL_CARDS:\n", "skill_cards")):
            if marker in system:
                payload[name] = json.JSONDecoder().raw_decode(system.split(marker, 1)[1])[0]
        safe = _redact_snapshot_value(payload)
        prompt_text = f"{system}\n\n{user}"
        safe_prompt_text = _redact_snapshot_value(prompt_text)
        safe_tool_schemas = _redact_snapshot_value(schemas)
        prompt_snapshot = PromptSnapshot.from_render(
            safe_prompt_text,
            token_count=(
                int((response.usage or {}).get("prompt_tokens", 0))
                if response is not None and (response.usage or {}).get("prompt_tokens") is not None
                else estimate_tokens_char4(prompt_text + (json.dumps(response_format) if response_format else ''))
            ),
            acquired_source_ranges=tuple(
                turn_input.acquired_source_ranges
            ),
            visible_source_ranges=tuple(
                turn_input.visible_source_ranges if request_kind == "semantic_interpretation"
                else ()
            ),
            context_view=turn_input.context_view if isinstance(turn_input, InterpretationTurnInput) else None,
            provider=str(safe.get("provider") or ""),
            model=str(safe.get("model") or ""),
            system_prompt=str(safe.get("system_prompt") or ""),
            user_prompt=str(safe.get("user_prompt") or ""),
            tool_schemas=tuple(safe_tool_schemas),
            transport_profile=self.transport_profile.value,
            request_kind=request_kind,
            state_snapshot_hash=turn_input.state_snapshot_hash,
            state_version=turn_input.state_version,
            temperature=safe.get("temperature"),
            max_output_tokens=max_output_tokens,
            redacted_request_metadata={
                "logical_timeout_seconds": logical_timeout_seconds,
                "enable_thinking": enable_thinking,
                "planner_contract_version": self.planner_contract_version,
                "turn_id": turn_input.turn_id,
                "contract_retry": bool(contract_retry),
                "context_resolution_round": context_resolution_round,
                "response_format": response_format,
                "attempt_kind": "contract_retry" if contract_retry else "context_resolution" if context_resolution_round else "first",
                "skill_context": {key: safe[key] for key in
                                  ("semantic_projection", "skill_cards", "action_choices")
                                  if key in safe},
            },
        ).as_dict()
        safe["prompt_snapshot"] = _redact_snapshot_value(prompt_snapshot)
        request_material = {
            key: safe[key]
            for key in (
                "provider", "model", "temperature", "max_tokens",
                "logical_timeout_seconds", "enable_thinking", "transport_mode",
                "request_kind", "planner_contract_version", "system_prompt",
                "state_version", "state_snapshot_sha256", "user_prompt",
                "tool_schemas", "response_format", "prompt_snapshot",
            )
        }
        safe["request_sha256"] = hashlib.sha256(
            json.dumps(request_material, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return safe

    def create_interpretation_input(self, state, context, **kwargs):
        if not isinstance(context, ContextView) or context.role != "interpretation":
            raise TypeError("Interpretation requires a ContextManager interpretation ContextView")
        view = context
        kwargs["context_snapshot"] = {"acquired_source_ranges": view.acquired_source_ranges,
                                      "visible_source_ranges": view.visible_source_ranges}
        # One detached decision input; execution admission still uses the post-commit view.
        full = self.create_turn_input(state, "", **kwargs)
        projection = copy.deepcopy(full.semantic_projection)
        for key in ("visible_observations", "existing_evidence", "local_context_references", "gap_choice_surface"):
            projection.pop(key, None)
        semantic_json = json.dumps(projection, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if full.active_skill_id:
            guidance = {"selected": json.loads(next(view.semantic_json for view in full.skill_views
                                                    if view.skill_id == full.active_skill_id))}
        else:
            guidance = {"cards": [{key: value for key, value in json.loads(view.card_json).items()
                                  if key in {"skill_id", "description", "entry"}} for view in full.skill_views]}
        # Runtime supplies semantic context independently of the executable surface.
        material = {"state_version": full.state_version, "goal": full.goal, "context": view.text,
                    "semantic_projection_json": semantic_json, "guidance": guidance, "context_view": asdict(view),
                    "action_snapshot_hash": full.state_snapshot_hash}
        digest = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
        return InterpretationTurnInput(full.turn_id, full.state_version, digest, full.task_kind,
            full.goal, semantic_json, full.active_skill_id, full.feasible_skill_ids,
            json.dumps(guidance, ensure_ascii=False), view, action_input=full)

    @staticmethod
    def with_context_view(turn_input, view, *, local_feedback=None):
        if view.role != 'interpretation':
            raise TypeError('Interpretation requires its role ContextView')
        material = {'state_version': turn_input.state_version, 'goal': turn_input.goal,
                    'context': view.text, 'semantic_projection_json': turn_input.semantic_projection_json,
                    'guidance': json.loads(turn_input.skill_semantic_guidance_json), 'context_view': asdict(view),
                    'action_snapshot_hash': turn_input.action_input.state_snapshot_hash}
        feedback = json.dumps(local_feedback, ensure_ascii=False) if local_feedback else ''
        if feedback:
            material['local_context_feedback'] = feedback
        digest = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
        return replace(turn_input, context_view=view, state_snapshot_hash=digest, local_context_feedback_json=feedback)

    def _structured_request(self, turn_input, *, schema, system, user, request_kind,
                            provider_gateway, prompt_budget, logical_timeout_seconds,
                            on_attempt_started, max_output_tokens, contract_retry, contract_feedback,
                            terminal_projection_allowed=False, **kwargs):
        self.last_turn_input = turn_input
        self.last_request_snapshot = None
        self.last_request_snapshots = []
        self.last_prompt_breakdowns = []
        if provider_gateway is None:
            from debug_assistant.incidents.provider_gateway import ProviderOperationalGateway
            provider_gateway = ProviderOperationalGateway(self.llm)
        if not callable(getattr(self.llm, "complete_structured", None)):
            raise NativePlannerContractError("strict semantic schema transport required", error_type="transport_profile_unsupported")
        if contract_retry:
            user += "\nBOUNDED_CONTRACT_RETRY: Return a valid response for this unchanged request snapshot."
        if contract_feedback:
            user += "\nCONTRACT_FEEDBACK: " + json.dumps(contract_feedback, ensure_ascii=False)
        response_format = strict_response_format(schema)
        cap = min(int(max_output_tokens or 2400), 1200 if terminal_projection_allowed else 2400)
        timeout = max(1.0, float(logical_timeout_seconds) * 0.4) if logical_timeout_seconds is not None else None
        self.last_prompt_breakdown = {"system_chars": len(system), "context_chars": len(user),
            "state_snapshot_sha256": turn_input.state_snapshot_hash,
            "prompt_components": {"system_tokens": estimate_tokens_char4(system),
                                  "diagnostic_context_tokens": estimate_tokens_char4(user),
                                  "response_schema_tokens": estimate_tokens_char4(json.dumps(response_format))}}
        if prompt_budget is not None:
            budget_kwargs = {"completion_reserve_tokens": cap, "response_format": response_format}
            signature = inspect.signature(prompt_budget.check_prompt)
            if "allow_terminal_projection" in signature.parameters or any(
                item.kind is inspect.Parameter.VAR_KEYWORD for item in signature.parameters.values()):
                budget_kwargs["allow_terminal_projection"] = bool(terminal_projection_allowed)
            decision = prompt_budget.check_prompt("planner", system, user, **budget_kwargs)
            self.last_prompt_breakdown.update(estimated_prompt_tokens=decision.estimated_prompt_tokens,
                input_hard_capacity=decision.input_hard_capacity, budget_state=decision.state.value,
                token_breakdown=dict(decision.breakdown))
        self.last_prompt_breakdowns.append(dict(self.last_prompt_breakdown))
        snapshot_kwargs = dict(system=system, user=user, schemas=[], response_format=response_format, max_output_tokens=cap,
            turn_input=turn_input, logical_timeout_seconds=timeout, enable_thinking=False,
            request_kind=request_kind, contract_retry=contract_retry,
            context_resolution_round=kwargs.get("context_resolution_round", 0))
        self.last_request_snapshot = self._request_snapshot(**snapshot_kwargs, status="provider_call_started")
        self.last_request_snapshots.append(self.last_request_snapshot)
        try:
            response = provider_gateway.complete_structured(system, user, schema=schema, model=self.model or None,
                logical_timeout_seconds=timeout, hard_timeout_seconds=timeout, on_attempt_started=on_attempt_started,
                max_output_tokens=cap, enable_thinking=False, request_kind=request_kind,
                turn_id=turn_input.turn_id, state_snapshot_hash=turn_input.state_snapshot_hash,
                transport_profile=self.transport_profile.value,
                contract_retry_reason=str((contract_feedback or {}).get("reason_code", "")))
        except LLMError as exc:
            self.last_request_snapshot = self._request_snapshot(**snapshot_kwargs, status="provider_output_error",
                response=_provider_response_for_snapshot(self.llm))
            self.last_request_snapshots[-1] = self.last_request_snapshot
            if isinstance(exc, LLMOutputError):
                raise NativePlannerContractError(str(exc), error_type=request_kind + "_parser_error") from exc
            raise
        if not isinstance(response, LLMResponse):
            raise NativePlannerContractError("strict semantic response required", error_type="response_type_invalid")
        self.last_request_snapshot = self._request_snapshot(**snapshot_kwargs, status="provider_response_received", response=response)
        self.last_request_snapshots[-1] = self.last_request_snapshot
        if response.tool_calls:
            raise NativePlannerContractError("semantic response cannot execute Tools", error_type="semantic_command_required")
        arguments = response.structured
        if arguments is None:
            arguments = extract_json(response.content)
        try:
            canonical = schema.compile(arguments)
        except DecisionSurfaceError as exc:
            error = NativePlannerContractError(str(exc), error_type="semantic_command_schema_invalid")
            error.metadata['validation_details'] = exc.details
            raise error from exc
        self.last_request_snapshot['command_arguments'] = _redact_snapshot_value(arguments)
        self.last_request_snapshot['compiled_proposal'] = _redact_snapshot_value(canonical)
        # Keep the original strict response in the physical snapshot. Internal
        # typed proposals enter the existing validator/transaction, never Tool dispatch.
        return LLMResponse(content=None, structured=canonical, usage=response.usage, raw_output=response.raw_output)

    @staticmethod
    def interpretation_schema(view, projection, *, active_skill='', feasible_skills=(), provenance_categories=None,
                              context_resolution_remaining=None):
        return interpretation_surface(view, projection, active_skill=active_skill,
            feasible_skills=feasible_skills, provenance_categories=provenance_categories,
            context_resolution_remaining=context_resolution_remaining)

    def interpretation_command(self, turn_input, *, context_resolution_remaining=None):
        categories = {name: set() for name in ('context_kinds', 'information_sources', 'source_tools')}
        from debug_assistant.memory.evidence_projector import EvidenceProjector
        official = getattr(self.tools, 'official_aliases', {})
        decision = getattr(self.tools, 'decision_tool_aliases', official)
        shadowed = {operation for name, operation in official.items() if decision.get(name) != operation}
        for spec in self.tools.specs():
            tool = self.tools.get(spec.name)
            if tool is None or spec.side_effect != 'none' or spec.capability == 'knowledge_read':
                continue
            declared = getattr(tool, 'provenance_categories', lambda: {})()
            if declared.get('information_sources') and all(value in EvidenceProjector._NON_INCIDENT_SOURCES
                                                           for value in declared['information_sources']):
                continue
            if spec.name not in shadowed:
                categories['source_tools'].add(spec.name)
            for name, values in declared.items():
                categories[name].update(values)
        for observation in turn_input.context_view.reference_metadata.get('visible_observations', ()):
            if observation.get('ok') is not True:
                continue
            metadata = observation.get('metadata', {})
            if metadata.get('information_source') in EvidenceProjector._NON_INCIDENT_SOURCES:
                continue
            if observation.get('tool'):
                categories['source_tools'].add(observation['tool'])
            for category, key in (('context_kinds', 'context_kind'), ('information_sources', 'information_source')):
                if metadata.get(key):
                    categories[category].add(metadata[key])
        semantic = self.interpretation_schema(turn_input.context_view, turn_input.semantic_projection,
            active_skill=turn_input.active_skill_id, feasible_skills=turn_input.feasible_skill_ids,
            provenance_categories=categories, context_resolution_remaining=context_resolution_remaining)
        return decision_turn_surface(semantic, turn_input.action_input, self.tools)

    def render_interpretation_envelope(self, turn_input, **kwargs):
        """Packing-only complete envelope, including the actual strict response schema."""
        system, user = self.render_interpretation_request(turn_input, **kwargs)
        command = self.interpretation_command(turn_input,
            context_resolution_remaining=0 if kwargs.get("consolidation_allowed") else kwargs.get('context_resolution_remaining'))
        if kwargs.get("consolidation_allowed"):
            command = self._consolidation_surface(command)
        return system, user + "\n\n" + json.dumps(strict_response_format(command, for_packing=True), ensure_ascii=False)

    def render_interpretation_request(self, turn_input, *, terminal_projection_allowed=False,
                                      context_resolution_remaining=None, budget_status=None,
                                      consolidation_allowed=False):
        """Single role envelope, shared by ContextManager packing and transport."""
        projection = turn_input.semantic_projection
        feasible, active = turn_input.feasible_skill_ids, turn_input.active_skill_id
        system = _INCIDENT_SEMANTIC_SYSTEM
        if context_resolution_remaining is not None:
            system += (f"\nLOCAL_CONTEXT_RECOVERY: {context_resolution_remaining} resolution rounds remain. "
                "Batch all needed retained references together (at most 8 distinct references per batch). "
                "The requests array selects reference/selector pairs; select each reference at most once. "
                "At zero rounds, decide from visible content; unseen content cannot establish Evidence.")
        pending = pending_context_targets(turn_input.context_view, projection)
        if pending:
            system += "\nRequired pending local context targets: " + json.dumps(pending)
        guidance = json.loads(turn_input.skill_semantic_guidance_json)
        marker = "SELECTED_SEMANTIC_PROJECTION" if active else "FEASIBLE_SKILL_CARDS"
        system += "\n\n" + marker + ":\n" + json.dumps(guidance.get("selected", guidance.get("cards")), ensure_ascii=False)
        if self.public_fault_taxonomy:
            system += "\nPUBLIC_FAULT_CODE_CONTRACT: hypothesis.fault_code uses a public canonical ID.\n" + render_public_fault_taxonomy()
        if terminal_projection_allowed:
            system += "\nTERMINAL_FIELD_RULE: return candidate fields only; reconcile supplied Review feedback."
        payload = {"current_goal": turn_input.goal, "diagnostic_context": turn_input.context, "state_version": turn_input.state_version,
            "state_snapshot_hash": turn_input.state_snapshot_hash, "canonical_semantic_input": _semantic_prompt_projection(projection)}
        projection_budget = (json.loads(turn_input.action_input.budget_projection_json)
                             if turn_input.action_input is not None else {})
        budget = dict(budget_status or {key: projection_budget[key] for key in (
            "phase", "remaining_api_calls", "remaining_phase_calls", "remaining_total_tokens",
            "remaining_seconds", "remaining_steps", "reserved_review_calls", "reserved_consolidation_calls")
            if key in projection_budget})
        if budget:
            payload["budget_status"] = budget
            system += ("\nBUDGET_RULE: remaining counts include every real API request, including local context, "
                       "corrections and Review. Near the limit, consolidate existing evidence and explicitly "
                       "dispose of unnecessary exploration. Unverified necessary facts remain blockers; "
                       "insufficient evidence must end INCONCLUSIVE, never invent support.")
        if consolidation_allowed:
            system += ("\nCONSOLIDATION: No new Tool or ContextRequest. Use visible evidence for lawful "
                       "qualification and Question lifecycle decisions. Preserve necessary unresolved facts "
                       "and contradictions. FinalizationPolicy remains authoritative.")
        if turn_input.action_input is not None:
            choices = turn_input.action_input.action_choices
            descriptions = {row['function']['name']: row['function'].get('description', '')
                            for row in turn_input.action_input.executable_tool_schemas}
            operations = tuple(dict.fromkeys(row['operation'] for row in choices))
            from debug_assistant.tools.provenance import registry_provenance_profiles
            profiles = registry_provenance_profiles(self.tools, operations)
            catalog = {'tools': [{'operation': name, 'description': descriptions.get(name, '')} for name in operations],
                       'skill_operations': {skill: [row['operation'] for row in choices if row['skill_id'] == skill]
                                            for skill in dict.fromkeys(row['skill_id'] for row in choices)}}
            payload['next_action_catalog'] = catalog
            from debug_assistant.tools.provenance import observation_provenance_profiles, provenance_profile_catalog
            catalog['source_profiles'] = provenance_profile_catalog(profiles + observation_provenance_profiles(
                projection.get('visible_observations', ())))
        if turn_input.local_context_feedback_json:
            payload['local_context_feedback'] = json.loads(turn_input.local_context_feedback_json)
        return system, json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _consolidation_surface(command):
        schema = copy.deepcopy(command.schema)
        branches = schema["properties"]["decision"]["anyOf"]
        schema["properties"]["decision"]["anyOf"] = [branch for branch in branches
            if branch["properties"]["kind"]["enum"] == ["SEMANTIC_DELTA"]]
        for branch in schema["properties"]["decision"]["anyOf"]:
            branch["properties"]["next_action_intent"] = {"type": "null"}
        return replace(command, schema=schema)

    def _interpret(self, turn_input, **kwargs):
        projection = turn_input.semantic_projection
        feasible, active = turn_input.feasible_skill_ids, turn_input.active_skill_id
        if not feasible or active and active not in feasible:
            raise NativePlannerContractError("feasible Skill required", error_type="infeasible_skill_selection")
        closing = kwargs.get('consolidation_allowed', False)
        remaining = 0 if closing else kwargs.get('context_resolution_remaining')
        budget = kwargs.get('prompt_budget')
        live_budget = kwargs.get('budget_status_provider')
        live_status = (live_budget() if callable(live_budget) else
                       budget.model_status() if hasattr(budget, 'model_status') else None)
        if live_status and turn_input.action_input is not None:
            projected = json.loads(turn_input.action_input.budget_projection_json)
            live_status.update({key: projected[key] for key in ('remaining_steps',) if key in projected})
        system, user = self.render_interpretation_request(turn_input,
            terminal_projection_allowed=kwargs.get('terminal_projection_allowed', False),
            context_resolution_remaining=remaining, budget_status=live_status, consolidation_allowed=closing)
        command = self.interpretation_command(turn_input, context_resolution_remaining=remaining)
        if closing:
            command = self._consolidation_surface(command)
        if command.bindings.get('unavailable'):
            from debug_assistant.context.requests import ContextResolutionError
            raise ContextResolutionError('context_resolution_unavailable')
        response = self._structured_request(turn_input, schema=command, system=system, user=user,
                                            request_kind="semantic_interpretation", **kwargs)
        canonical = dict(response.structured)
        raw_intent = canonical.pop('next_action_intent', None)
        intent = NextActionIntent(**raw_intent) if raw_intent else None
        response = replace(response, structured=canonical)
        normalization_events = []
        self.last_request_snapshot["normalization_events"] = normalization_events
        try:
            decision = _parse_interpretation_decision(response, normalization_events=normalization_events, symbol_domains={
                "skill_id": ("canonical_skill_id", tuple(dict(CANONICAL_INCIDENT_SKILLS)))})
        except PlannerContractError as exc:
            parsed = response.structured
            if parsed is None and isinstance(response.content, str):
                try:
                    parsed = extract_json(response.content)
                except (ValueError, LLMError):
                    parsed = None
            delta_payload = parsed.get('delta') if isinstance(parsed, dict) else None
            proposals = delta_payload.get('evidence_proposals') if isinstance(delta_payload, dict) else None
            if isinstance(proposals, (list, tuple)):
                for detail in exc.metadata.get('validation_details', ()):
                    field = detail.get('field', ())
                    if 'evidence_proposals' in field:
                        index = field[field.index('evidence_proposals') + 1]
                        if isinstance(index, int) and index < len(proposals) and isinstance(proposals[index], dict):
                            item = proposals[index]
                            exc.metadata['requirement_feedback'] = targeting_feedback(projection, index,
                                item.get('obligation_id'),
                                item.get('requirement_refs') if isinstance(item.get('requirement_refs'), (list, tuple)) else (),
                                item.get('hypothesis_target'))
                            break
            raise
        if decision is None:
            raise NativePlannerContractError("missing interpretation", error_type="semantic_delta_missing")
        pending = pending_context_targets(turn_input.context_view, projection)
        if pending and (not isinstance(decision.root, ContextInterpretation) or
                        not set(pending).issubset({item.observation_id for item in decision.root.batch.requests})):
            error = NativePlannerContractError('Pending acquired bodies require local context before semantic commit',
                                              error_type='pending_observation_context_required')
            error.metadata['evidence_feedback'] = {'required_context_targets': list(pending),
                                                  'legal_actions': ['CONTEXT_REQUEST']}
            raise error
        if isinstance(decision.root, ContextInterpretation):
            self.last_request_snapshot["interpretation_decision"] = "CONTEXT_REQUEST"
            return NativePlannerResult(response=response, tool_calls=(), context_requests=decision.root.batch)
        delta = decision.root.delta
        legal_acknowledgements = acknowledgement_targets(turn_input.context_view)
        if any(item.observation_ref not in legal_acknowledgements for item in delta.operational_acknowledgements):
            error = NativePlannerContractError('ACK requires a successful visible Observation',
                                              error_type='operational_acknowledgement_invalid')
            error.metadata['evidence_feedback'] = {'legal_acknowledgement_targets': list(legal_acknowledgements)}
            raise error
        try:
            validate_disposition_targets(delta.hypothesis.transition,
                                         disposition_targets(projection.get('obligations', ())))
        except SemanticTransitionError as exc:
            error = NativePlannerContractError(str(exc), error_type=exc.code)
            error.metadata['disposition_feedback'] = exc.details
            raise error from exc
        terminal_projection_allowed = kwargs.get("terminal_projection_allowed", False)
        if delta.skill_id not in ((active,) if active else feasible):
            raise NativePlannerContractError("selected Skill is not currently feasible",
                                              error_type="infeasible_skill_selection")
        if terminal_projection_allowed and (
            delta.new_obligation_proposals or delta.evidence_proposals or delta.contradictions
        ):
            raise NativePlannerContractError("terminal projection accepts candidate fields only",
                                              error_type="terminal_semantic_mutation")
        if not projection.get("evidence_hypothesis_id") and not projection.get("active_hypothesis") and delta.evidence_proposals:
            raise NativePlannerContractError("Bootstrap requires INITIAL plus explicit contracts with no Evidence; next Interpretation uses committed canonical references", error_type="bootstrap_evidence_requires_committed_targets")
        references = (
            *delta.hypothesis.supporting_evidence_ids, *delta.hypothesis.contradicting_evidence_ids,
            *(item.evidence_id for item in delta.contradictions),
            *(key for item in delta.hypothesis.source_claims for key in item.evidence_ids),
        )
        if any(key not in turn_input.known_evidence_ids or not key.startswith("ev-") for key in references):
            raise NativePlannerContractError("Evidence references must be visible canonical IDs",
                                              error_type="invalid_evidence_id_contract")
        obligations = {item["id"] for item in projection.get("obligations", ())}
        observation_ids = {item["observation_id"] for item in projection.get("visible_observations", ())
                           if "observation_id" in item}
        reusable = {row["evidence_ref"]: row for row in projection.get("existing_evidence", ())}
        for index, proposal in enumerate(delta.evidence_proposals):
            rejected_refs = [key for key in proposal.observation_refs if key not in observation_ids]
            unknown_evidence = proposal.evidence_ref is not None and (
                proposal.evidence_ref not in reusable or proposal.evidence_ref not in turn_input.known_evidence_ids
            )
            if (not proposal.obligation_id or proposal.obligation_id not in obligations
                    or not projection.get("incident_id")
                    or rejected_refs or unknown_evidence
                    or not projection.get("evidence_hypothesis_id")):
                error = NativePlannerContractError(
                    "Evidence proposals require canonical scope and a visible Observation or reusable Evidence",
                    error_type="evidence_proposal_reference_invalid",
                    validation_errors=["evidence_proposals"],
                )
                error.metadata['requirement_feedback'] = targeting_feedback(projection, index,
                    proposal.obligation_id, proposal.requirement_refs, proposal.hypothesis_target)
                error.metadata["evidence_feedback"] = {
                    "rejected_refs": rejected_refs + ([proposal.evidence_ref] if unknown_evidence else []),
                    "reason_code": ("unknown_evidence_ref" if unknown_evidence else "observation_not_visible"
                                    if rejected_refs else "evidence_scope_invalid"),
                    "visible_observation_refs": sorted(observation_ids),
                    "reusable_evidence_refs": sorted(reusable),
                    "admitted_evidence_for_rejected_observations": {
                        ref: [key for key, row in reusable.items() if ref in row.get("observation_refs", ())]
                        for ref in rejected_refs
                    },
                }
                raise error
        try:
            selection = self._semantic_only_selection(delta, projection)
        except RequirementTargetError as exc:
            error = NativePlannerContractError(str(exc), error_type=exc.code)
            error.metadata['requirement_feedback'] = exc.feedback
            raise error from exc
        self.last_request_snapshot["semantic_contract_accepted"] = True
        return NativePlannerResult(response=response, tool_calls=(), reason=selection.reason,
            information_need=selection.evidence_gap, expected_evidence=selection.evidence_gap,
            skill_selections=(selection,), semantic_only=True, next_action_intent=intent)

    def compile_next_action(self, turn_input, intent, response):
        """Pure Runtime Phase B admission; no Provider call or semantic mutation."""
        try:
            proposal = compile_next_action(intent, turn_input, self.tools)
        except VerificationPlanError as exc:
            raise NativePlannerContractError(str(exc), error_type=exc.reason_code) from exc
        return self.validate_committed_plan(turn_input, proposal, response)

    def validate_committed_plan(self, turn_input, proposal, response):
        projection = turn_input.semantic_projection
        active, choices = turn_input.active_skill_id, turn_input.action_choices
        if not active:
            raise NativePlannerContractError("committed selected Skill required", error_type="active_skill_unavailable")
        open_rows = [row for row in projection.get("obligations", ()) if row["status"] == "OPEN"]
        gaps = projection.get("gap_choice_surface", ())
        decision = PlanningDecision(plan=proposal)
        hashes = {row["function"]["name"]: row["schema_sha256"] for row in turn_input.executable_tool_schemas}
        try:
            enriched = enrich_verification_plan(decision.plan, hashes, choices, skill_id=active)
            by_id = {row["id"]: row for row in open_rows}
            for raw, task in zip(decision.plan.tasks, enriched.tasks, strict=True):
                if task.target_obligation_id not in by_id:
                    raise VerificationPlanError("plan_target_unknown", task_id=task.task_id)
                available = [row for row in gaps if row["obligation_id"] == task.target_obligation_id
                    and (task.target_fact_slot is None or row["target_fact_slot"] == task.target_fact_slot)]
                if not any(raw.choice_ref in row["available_choice_refs"] for row in available):
                    raise VerificationPlanError("plan_gap_choice_unavailable", task_id=task.task_id)
            validate_verification_plan(enriched, self.tools, skill_id=active,
                allowed_operations=tuple(hashes), state_fields=projection)
        except VerificationPlanError as exc:
            error = NativePlannerContractError(str(exc), error_type=exc.reason_code, validation_errors=["plan", exc.task_id])
            error.metadata["validation_details"] = [{"task_id": exc.task_id, "reason_code": exc.reason_code, "detail": exc.detail}]
            error.metadata["target_feedback"] = {"existing_targets": [{"id": row["id"], "claim": row.get("claim", "")} for row in open_rows]}
            rejected = next((raw for raw in decision.plan.tasks if raw.task_id == exc.task_id), None)
            node_ref = rejected.choice_ref.rsplit(".A", 1)[0] if rejected else ""
            error.metadata["choice_feedback"] = {"allowed_choices": [row for row in choices if row["node_ref"] == node_ref],
                "rejected_choice_ref": rejected.choice_ref if rejected else "", "selected_skill_id": active}
            raise error from exc
        if self.last_request_snapshot is None:
            self.last_request_snapshot = {}
        self.last_request_snapshot["choice_resolutions"] = [{"choice_ref": raw.choice_ref,
            "node_id": task.node_id, "operation": task.operation, "target_obligation_id": task.target_obligation_id}
            for raw, task in zip(decision.plan.tasks, enriched.tasks, strict=True)]
        return NativePlannerResult(response=response, tool_calls=(), plan=enriched)

    def propose(self, state: AgentState | PlanningTurnInput | InterpretationTurnInput, context: str = "", *, logical_timeout_seconds=None,
                on_attempt_started=None, prompt_budget=None,
                visible_tool_names: Iterable[str] | None = None,
                active_skill_id: str = "",
                feasible_skill_ids: Iterable[str] | None = None,
                skill_tool_names_by_id: dict[str, Iterable[str]] | None = None,
                max_output_tokens: int | None = None,
                enable_thinking: bool | None = None,
                terminal_projection_allowed: bool | None = None,
                context_snapshot: dict[str, Any] | None = None,
                semantic_projection: dict[str, Any] | None = None,
                budget_projection: dict[str, Any] | None = None,
                action_ledger: Iterable[dict[str, Any]] | None = None,
                provider_gateway=None,
                contract_retry: bool = False,
                contract_feedback: dict[str, Any] | None = None,
                context_resolution_round: int = 0,
                context_resolution_remaining: int | None = None,
                consolidation_allowed: bool = False,
                budget_status_provider=None,
                ) -> NativePlannerResult:
        request_kwargs = dict(provider_gateway=provider_gateway, prompt_budget=prompt_budget,
            logical_timeout_seconds=logical_timeout_seconds, on_attempt_started=on_attempt_started,
            max_output_tokens=max_output_tokens, terminal_projection_allowed=terminal_projection_allowed,
            contract_retry=contract_retry, contract_feedback=contract_feedback,
            context_resolution_round=context_resolution_round,
            context_resolution_remaining=context_resolution_remaining, consolidation_allowed=consolidation_allowed,
            budget_status_provider=budget_status_provider)
        if isinstance(state, InterpretationTurnInput):
            return self._interpret(state, **request_kwargs)
        supplied_turn_input = state if isinstance(state, PlanningTurnInput) else None
        incident_mode = supplied_turn_input.task_kind == "incident" if supplied_turn_input else state.task.metadata.get("task_kind") == "incident"
        if incident_mode:
            if supplied_turn_input is None:
                raise TypeError("incident requests require an explicit InterpretationTurnInput or committed PlanningTurnInput")
            raise TypeError("Planning is deterministic post-commit admission; use compile_next_action")
        if not getattr(getattr(self.llm, "capabilities", ProviderCapabilities()), "tool_calling", False):
            raise PlannerContractError("provider does not support native tool calling",
                                       validation_errors=["tool_calling=false"])
        feasible_skills = (
            supplied_turn_input.feasible_skill_ids
            if supplied_turn_input is not None
            else tuple(dict.fromkeys(
                str(item) for item in (
                    feasible_skill_ids
                    if feasible_skill_ids is not None
                    else self.allowed_incident_skills
                )
                if str(item) in self.allowed_incident_skills
            ))
        )
        tool_names_by_id = (
            dict(supplied_turn_input.skill_tool_names_by_id)
            if supplied_turn_input is not None else {
                str(key): tuple(dict.fromkeys(str(item) for item in value))
                for key, value in dict(skill_tool_names_by_id or {}).items()
                if str(key) in feasible_skills
            }
        )
        system = (
            "You are a read-only CloudOps incident planner. Request only the "
            "CloudOps tools needed for the current goal. Tool arguments are validated "
            "by the Runtime; do not invent tool names. Return tool calls, not an execution plan. "
            "Source-search results are discovery only: whenever grep, code_search, or symbol_search "
            "returns a repository path with a line or symbol, the next investigation turn "
            "must include a bounded read_file for the matching source, covering the hit and "
            "a substantial nearby context window (at least 100 lines when no complete symbol "
            "range is available; never use a narrow abbreviated slice). A broad read in another "
            "file is not a substitute. If OPEN_CRITICAL_EVIDENCE_OBLIGATIONS names an exact "
            "file, symbol, or line range, honor that scope first and use explicit supporting "
            "actions only for callers, tests, or history."
        )
        schemas = (
            supplied_turn_input.executable_tool_schemas
            if supplied_turn_input is not None
            else self.tools.function_schemas(visible_tools=visible_tool_names)
        )
        turn_input = supplied_turn_input or self.create_turn_input(
            state,
            context,
            active_skill_id=active_skill_id,
            feasible_skill_ids=feasible_skills,
            skill_tool_names_by_id=tool_names_by_id,
            visible_tool_names=visible_tool_names,
            context_snapshot=context_snapshot,
            semantic_projection=semantic_projection,
            budget_projection=budget_projection,
            action_ledger=action_ledger,
        )
        active_skill_id = turn_input.active_skill_id
        self.last_turn_input = turn_input
        context = turn_input.context
        state_version = turn_input.state_version
        state_snapshot_sha256 = turn_input.state_snapshot_hash
        state_snapshot_marker = (
            "\n\nPLANNER_STATE_VERSION: " + state_version
            + "\nPLANNER_STATE_SNAPSHOT_SHA256: " + state_snapshot_sha256
        )
        user = f"{context}\n\nCURRENT_GOAL: {turn_input.goal}\n" + state_snapshot_marker
        self.last_request_snapshots = []
        if provider_gateway is None:
            from debug_assistant.incidents.provider_gateway import ProviderOperationalGateway
            provider_gateway = ProviderOperationalGateway(self.llm)
        retry_instruction = ""
        marker = "\n\nDIAGNOSTIC_CONTEXT:\n"
        control_text, diagnostic_text = (
            context.split(marker, 1) if marker in context else ("", context)
        )
        self.last_prompt_breakdown = {
            "system_chars": len(system), "context_chars": len(user),
            "state_snapshot_sha256": state_snapshot_sha256,
            "tool_schema_count": len(schemas),
            "prompt_components": {
                "system_tokens": estimate_tokens_char4(system),
                "tool_schema_tokens": estimate_tokens_char4(
                    json.dumps(schemas, ensure_ascii=False, sort_keys=True)
                ),
                "control_state_tokens": estimate_tokens_char4(control_text),
                "diagnostic_context_tokens": estimate_tokens_char4(
                    diagnostic_text + "\n\nCURRENT_GOAL: " + turn_input.goal
                ),
            },
        }
        self.last_prompt_breakdowns = []
        if prompt_budget is not None:
            budget_kwargs = {"tools": schemas, "breakdown": {"incident": user}}
            try:
                budget_signature = inspect.signature(prompt_budget.check_prompt)
                accepts_budget_kwargs = any(
                    item.kind is inspect.Parameter.VAR_KEYWORD
                    for item in budget_signature.parameters.values()
                )
            except (TypeError, ValueError):
                budget_signature = None
                accepts_budget_kwargs = False
            if max_output_tokens is not None and (
                accepts_budget_kwargs or (
                    budget_signature is not None
                    and "completion_reserve_tokens" in budget_signature.parameters
                )
            ):
                budget_kwargs["completion_reserve_tokens"] = int(max_output_tokens)
            decision = prompt_budget.check_prompt("planner", system, user, **budget_kwargs)
            self.last_prompt_breakdown.update({
                "estimated_prompt_tokens": decision.estimated_prompt_tokens,
                "input_hard_capacity": decision.input_hard_capacity,
                "budget_state": decision.state.value,
                "token_breakdown": dict(decision.breakdown),
            })
            self.last_prompt_breakdowns.append(dict(self.last_prompt_breakdown))
        method = provider_gateway.complete_with_tools
        kwargs = {"tools": schemas, "model": self.model or None}
        action_timeout = (
            max(1.0, float(logical_timeout_seconds) * 0.6)
            if logical_timeout_seconds is not None else None
        )
        action_enable_thinking = enable_thinking
        user += retry_instruction
        kwargs["logical_timeout_seconds"] = action_timeout
        kwargs["on_attempt_started"] = on_attempt_started
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = int(max_output_tokens)
        kwargs["enable_thinking"] = action_enable_thinking
        action_snapshot = self._request_snapshot(
            system=system, user=user, schemas=schemas,
            max_output_tokens=max_output_tokens,
            turn_input=turn_input, status="provider_call_started",
            logical_timeout_seconds=action_timeout,
            enable_thinking=action_enable_thinking,
            request_kind="executable_action",
        )
        self.last_request_snapshots.append(action_snapshot)
        self.last_request_snapshot = action_snapshot
        try:
            response = method(
                system,
                user,
                request_kind="executable_action",
                turn_id=turn_input.turn_id,
                state_snapshot_hash=turn_input.state_snapshot_hash,
                hard_timeout_seconds=action_timeout,
                **kwargs,
            )
        except LLMOutputError as exc:
            action_snapshot = self._request_snapshot(
                system=system, user=user, schemas=schemas,
                max_output_tokens=max_output_tokens,
                turn_input=turn_input, status="provider_output_error",
                logical_timeout_seconds=action_timeout,
                enable_thinking=action_enable_thinking,
                request_kind="executable_action",
            )
            self.last_request_snapshots[-1] = action_snapshot
            self.last_request_snapshot = action_snapshot
            raise NativePlannerContractError(
                str(exc), error_type=getattr(exc, "error_type", "provider_contract_mismatch"),
                validation_errors=[getattr(exc, "error_type", "provider_contract_mismatch")],
                output={
                    "raw_output": copy.deepcopy(getattr(self.llm, "last_raw_output", {})),
                    "validation_error": str(exc),
                },
                index=getattr(exc, "index", None), tool=getattr(exc, "tool", None),
            ) from exc
        except LLMError as exc:
            # Some OpenAI-compatible adapters wrap malformed native Tool JSON
            # in a generic LLMError after the provider response has arrived.
            # Preserve that response in the snapshot and classify only the
            # observable native-payload case as a bounded parser contract
            # failure; transport/auth failures remain provider errors.
            provider_response = _provider_response_for_snapshot(self.llm)
            self.last_request_snapshots[-1] = self._request_snapshot(
                system=system, user=user, schemas=schemas,
                max_output_tokens=max_output_tokens,
                turn_input=turn_input, status="provider_output_error",
                logical_timeout_seconds=action_timeout,
                enable_thinking=action_enable_thinking,
                response=provider_response,
                request_kind="executable_action",
            )
            self.last_request_snapshot = self.last_request_snapshots[-1]
            raw_tool_calls = (
                provider_response.raw_output.get("tool_calls", [])
                if provider_response is not None else []
            )
            cause = getattr(exc, "__cause__", None)
            is_native_payload_error = bool(raw_tool_calls) and (
                isinstance(cause, (ValueError, json.JSONDecodeError))
                or "delimiter" in str(exc).casefold()
                or "json" in str(exc).casefold()
                or "arguments" in str(exc).casefold()
            )
            if is_native_payload_error:
                raise NativePlannerContractError(
                    str(exc), error_type="tool_arguments_invalid_json",
                    validation_errors=["tool_calls.arguments"],
                    output={
                        "raw_output": provider_response.raw_output
                        if provider_response is not None else {},
                        "validation_error": str(exc),
                    },
                ) from exc
            raise
        if not isinstance(response, LLMResponse):
            action_snapshot = self._request_snapshot(
                system=system, user=user, schemas=schemas,
                max_output_tokens=max_output_tokens,
                turn_input=turn_input, status="provider_response_type_error",
                logical_timeout_seconds=action_timeout,
                enable_thinking=action_enable_thinking,
                request_kind="executable_action",
            )
            self.last_request_snapshots[-1] = action_snapshot
            self.last_request_snapshot = action_snapshot
            raise NativePlannerContractError(
                "provider returned a non-typed native response",
                error_type="provider_contract_mismatch",
                output={"validation_error": "response_type=" + type(response).__name__},
            )
        action_snapshot = self._request_snapshot(
            system=system, user=user, schemas=schemas,
            max_output_tokens=max_output_tokens, turn_input=turn_input,
            response=response, status="provider_response_received",
            logical_timeout_seconds=action_timeout,
            enable_thinking=action_enable_thinking,
            request_kind="executable_action",
        )
        self.last_request_snapshots[-1] = action_snapshot
        self.last_request_snapshot = action_snapshot

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
        names = {
            str(schema.get("function", {}).get("name"))
            for schema in schemas
            if schema.get("function", {}).get("name")
        }
        allowed_tool_fields = {
            spec.name: tuple(spec.args_model.model_fields)
            for spec in self.tools.specs()
        }
        sanitized_calls = []
        tool_call_audits: list[dict[str, Any]] = []
        for call in response.tool_calls:
            if not isinstance(call, LLMToolCall) or not isinstance(call.arguments, dict):
                raise contract_error(
                    "provider returned a malformed native tool call",
                    error_type="malformed_tool_call",
                    validation_errors=["malformed_tool_call"],
                )
            call_name = call.name
            current_normalized_output = {}
            if call_name not in names:
                raise contract_error(
                    f"unknown tool: {call_name}", error_type="unknown_tool",
                    validation_errors=["unknown_tool"], output={"tool": call.name},
                )
            sanitized_calls.append(call)
            tool_call_audits.append({
                "call_id": call.id, "tool": call.name,
                "raw_arguments": dict(call.arguments), "normalized_arguments": dict(call.arguments),
                "normalization_actions": [], "dropped_fields": [],
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
            tool_call_audits=tuple(tool_call_audits),
        )
