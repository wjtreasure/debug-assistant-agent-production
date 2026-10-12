"""Immutable execution proposals and deterministic admission/compilation.

Runtime owns the post-commit state, permission and execution decisions. These
pure contracts do not interpret Observations or mutate semantic state.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Annotated, Any, Literal, Mapping

from pydantic import (
    BaseModel, ConfigDict, Field, JsonValue, StrictInt, StrictStr,
    TypeAdapter, ValidationError, model_validator,
)

from debug_assistant.llm.base import LLMToolCall
from .action_runtime import ActionRuntime, AcquisitionKey


class VerificationPlanError(ValueError):
    def __init__(self, reason_code: str, *, task_id: str = "", detail: str = ""):
        self.reason_code = reason_code
        self.task_id = task_id
        self.detail = detail
        super().__init__(reason_code + (": " + detail if detail else ""))


class ParameterBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["literal", "state_field", "observation_field"]
    value: JsonValue = None
    field_path: tuple[StrictStr | StrictInt, ...] = ()
    upstream_task_id: str | None = None

    @model_validator(mode="after")
    def exact_reference(self):
        if self.kind == "literal":
            if "value" not in self.model_fields_set:
                raise ValueError("literal binding requires an explicit value")
            if self.field_path or self.upstream_task_id is not None:
                raise ValueError("literal binding cannot contain a reference")
        else:
            if not self.field_path or self.value is not None:
                raise ValueError("field binding requires an exact path and no literal value")
            if any(isinstance(key, bool) or key == "" or (isinstance(key, int) and key < 0)
                   for key in self.field_path):
                raise ValueError("field path requires nonempty keys or nonnegative indices")
            if (self.kind == "observation_field") != bool(self.upstream_task_id):
                raise ValueError("only Observation bindings require upstream_task_id")
        return self


class SemanticIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    target_obligation_id: str = Field(min_length=1)
    target_fact_slot: str = Field(min_length=1)


class PlannedAction(BaseModel):
    """Canonical executable task; never a Planner response schema."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1, max_length=128)
    target_obligation_id: str = Field(min_length=1)
    target_fact_slot: str | None = Field(default=None, min_length=1)
    skill_id: str = Field(min_length=1)
    node_id: str | None = Field(default=None, min_length=1)
    operation: str = Field(min_length=1)
    semantic_intents: tuple[SemanticIntent, ...] = ()
    dependencies: tuple[str, ...] = ()
    args: dict[str, JsonValue] | None = None
    bindings: dict[str, ParameterBinding] | None = None

    @model_validator(mode="after")
    def complete_execution_contract(self):
        if (self.args is None) == (self.bindings is None):
            raise ValueError("task requires exactly one of args or bindings")
        if len(set(self.dependencies)) != len(self.dependencies) or self.task_id in self.dependencies:
            raise ValueError("dependencies must be unique and exclude the task itself")
        return self


class PlannedChoice(BaseModel):
    """LLM chooses one frozen legal node/action pair and an exact question."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1, max_length=128)
    choice_ref: str = Field(min_length=1)
    target_obligation_id: str = Field(min_length=1)
    target_fact_slot: str | None = Field(default=None, min_length=1)
    semantic_intents: tuple[SemanticIntent, ...] = ()
    dependencies: tuple[str, ...] = ()
    args: dict[str, JsonValue] | None = None
    bindings: dict[str, ParameterBinding] | None = None

    complete_execution_contract = model_validator(mode="after")(
        PlannedAction.complete_execution_contract
    )


class VerificationPlanProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tasks: tuple[PlannedChoice, ...] = Field(min_length=0, max_length=4)

    @model_validator(mode="after")
    def bounded_dag(self):
        by_id = {task.task_id: task for task in self.tasks}
        if len(by_id) != len(self.tasks):
            raise ValueError("task IDs must be unique")
        for task in self.tasks:
            if any(key not in by_id for key in task.dependencies):
                raise ValueError("dependency must reference a task in this plan")
            for binding in (task.bindings or {}).values():
                if binding.kind == "observation_field" and binding.upstream_task_id not in task.dependencies:
                    raise ValueError("Observation binding must reference an explicit dependency")
        pending = set(by_id)
        while pending:
            ready = {key for key in pending if not pending.intersection(by_id[key].dependencies)}
            if not ready:
                raise ValueError("plan dependency cycle")
            pending -= ready
        return self


class EnrichedAction(PlannedAction):
    """Operational schema identity attached from the immutable turn surface."""

    schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EnrichedVerificationPlan(VerificationPlanProposal):
    tasks: tuple[EnrichedAction, ...] = Field(min_length=0, max_length=4)


def enrich_verification_plan(
    proposal: VerificationPlanProposal, schema_hashes: Mapping[str, str],
    choices: tuple[Mapping[str, Any], ...], *, skill_id: str,
) -> EnrichedVerificationPlan:
    by_ref = {row["choice_ref"]: row for row in choices}
    if len(by_ref) != len(choices):
        raise VerificationPlanError("plan_choice_ambiguous")
    tasks = []
    for task in proposal.tasks:
        row = by_ref.get(task.choice_ref)
        if row is None or row["skill_id"] != skill_id:
            raise VerificationPlanError("plan_choice_invalid", task_id=task.task_id,
                                        detail=task.choice_ref)
        operation = row["operation"]
        if operation not in schema_hashes:
            raise VerificationPlanError("plan_operation_not_visible", task_id=task.task_id)
        tasks.append(EnrichedAction(**task.model_dump(exclude={"choice_ref"}),
            skill_id=skill_id, node_id=row["node_id"], operation=operation,
            schema_sha256=schema_hashes[operation]))
    return EnrichedVerificationPlan(tasks=tuple(tasks))


class AdmittedVerificationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tasks_json: str
    state_snapshot_hash: str
    committed_state_version: int = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def freeze_tasks(cls, value):
        if isinstance(value, Mapping) and "tasks" in value:
            value = dict(value)
            tasks = value.pop("tasks")
            if "tasks_json" in value:
                raise ValueError("supply tasks or tasks_json, never both")
            value["tasks_json"] = EnrichedVerificationPlan(tasks=tasks).model_dump_json()
        return value

    @property
    def tasks(self) -> tuple[EnrichedAction, ...]:
        """Fresh decoded tasks prevent nested dict mutation of admitted actions."""
        return EnrichedVerificationPlan.model_validate_json(self.tasks_json).tasks

    @property
    def plan_sha256(self) -> str:
        return hashlib.sha256(self.tasks_json.encode("utf-8")).hexdigest()

    @model_validator(mode="after")
    def canonical_targets(self):
        if any(not task.target_obligation_id
               for task in self.tasks):
            raise ValueError("admitted tasks require canonical Obligation IDs")
        return self


@dataclass(frozen=True, slots=True)
class PlanAdmissionDecision:
    action: Literal["execute", "replan", "reject", "finalize"]
    reason: str
    admitted_plan: AdmittedVerificationPlan | None = None
    committed_state_version: int = 0
    detail: str = ""

def registry_schema_sha256(spec: Any) -> str:
    material = json.dumps(spec.args_model.model_json_schema(), ensure_ascii=False,
                          sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _registry_spec(task: EnrichedAction, registry: Any):
    specs = [spec for spec in registry.specs() if spec.name == task.operation]
    if len(specs) != 1:
        raise VerificationPlanError("plan_operation_unknown", task_id=task.task_id)
    spec = specs[0]
    if task.schema_sha256 != registry_schema_sha256(spec):
        raise VerificationPlanError("plan_schema_stale", task_id=task.task_id)
    return spec


def _exact_field(root: Any, path: tuple[str | int, ...], task_id: str) -> Any:
    value = root
    for key in path:
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        if isinstance(value, Mapping) and isinstance(key, str) and key in value:
            value = value[key]
        elif isinstance(value, (list, tuple)) and type(key) is int and 0 <= key < len(value):
            value = value[key]
        else:
            raise VerificationPlanError("plan_binding_unresolved", task_id=task_id)
    return value


def compile_planned_action(
    task: EnrichedAction, registry: Any, *, state_fields: Mapping[str, Any],
    upstream_observations: Mapping[str, Any],
) -> LLMToolCall:
    """Resolve exact fields and validate the complete canonical executable args."""
    spec = _registry_spec(task, registry)
    args = dict(task.args) if task.args is not None else {}
    for name, binding in (task.bindings or {}).items():
        if binding.kind == "literal":
            args[name] = binding.value
        elif binding.kind == "state_field":
            args[name] = _exact_field(state_fields, binding.field_path, task.task_id)
        else:
            if binding.upstream_task_id not in task.dependencies:
                raise VerificationPlanError("plan_binding_dependency_required", task_id=task.task_id)
            if binding.upstream_task_id not in upstream_observations:
                raise VerificationPlanError("plan_binding_unresolved", task_id=task.task_id)
            args[name] = _exact_field(upstream_observations[binding.upstream_task_id],
                                      binding.field_path, task.task_id)
    required = {key for key, field in spec.args_model.model_fields.items() if field.is_required()}
    if set(args) - set(spec.args_model.model_fields) or not required.issubset(args):
        raise VerificationPlanError("plan_args_invalid", task_id=task.task_id,
                                    detail="unknown executable argument")
    try:
        validated = spec.args_model.model_validate_json(json.dumps(args, allow_nan=False), strict=True)
    except (ValidationError, ValueError, TypeError) as exc:
        raise VerificationPlanError("plan_args_invalid", task_id=task.task_id,
                                    detail="canonical typed argument validation failed") from exc
    canonical_args = validated.model_dump(mode="json")
    if any(canonical_args.get(key) != value for key, value in args.items()):
        raise VerificationPlanError("plan_args_changed", task_id=task.task_id,
                                    detail="canonical validation changed a frozen argument")
    canonical_args, error = registry.validate_arguments(task.operation, canonical_args)
    if error:
        raise VerificationPlanError("plan_args_invalid", task_id=task.task_id,
                                    detail=json.dumps(error))
    return LLMToolCall(task.task_id, task.operation, canonical_args)


def validate_verification_plan(
    plan: EnrichedVerificationPlan, registry: Any, *, skill_id: str,
    allowed_operations: tuple[str, ...],
    state_fields: Mapping[str, Any],
) -> None:
    """Validate action structure/schema against the committed planning view."""
    for task in plan.tasks:
        if task.skill_id != skill_id or task.operation not in allowed_operations:
            raise VerificationPlanError("plan_skill_operation_invalid", task_id=task.task_id)
        spec = _registry_spec(task, registry)
        supplied = set(task.args if task.args is not None else task.bindings or {})
        required = {key for key, field in spec.args_model.model_fields.items() if field.is_required()}
        if supplied - set(spec.args_model.model_fields) or not required.issubset(supplied):
            raise VerificationPlanError("plan_args_required", task_id=task.task_id,
                detail=json.dumps({"missing_keys": sorted(required - supplied),
                                   "unsupported_keys": sorted(supplied - set(spec.args_model.model_fields)),
                                   "allowed_keys": sorted(spec.args_model.model_fields)}))
        for name, binding in (task.bindings or {}).items():
            if binding.kind == "observation_field":
                continue
            value = (binding.value if binding.kind == "literal"
                     else _exact_field(state_fields, binding.field_path, task.task_id))
            field = spec.args_model.model_fields[name]
            try:
                TypeAdapter(Annotated[field.annotation, field]).validate_json(
                    json.dumps(value, allow_nan=False), strict=True,
                )
            except (ValidationError, ValueError, TypeError) as exc:
                raise VerificationPlanError("plan_args_invalid", task_id=task.task_id,
                                            detail="binding violates canonical parameter type") from exc
        if not any(binding.kind == "observation_field" for binding in (task.bindings or {}).values()):
            compile_planned_action(task, registry, state_fields=state_fields, upstream_observations={})


def validate_plan_acquisitions(plan: EnrichedVerificationPlan, registry: Any, *,
                               projection: Mapping[str, Any]) -> None:
    """Reject successful data duplicates; upstream bindings wait for the execution barrier."""
    obligations = {row["id"]: row for row in projection.get("obligations", ())}
    acquired = projection.get("acquisitions", ())
    seen = set()
    for task in plan.tasks:
        target = task.target_obligation_id
        slots = obligations.get(target, {}).get("required_fact_slots", ())
        slot = task.target_fact_slot or (slots[0] if len(slots) == 1 else None)
        if slot not in slots:
            raise VerificationPlanError("plan_fact_slot_required", task_id=task.task_id,
                                        detail=json.dumps({"allowed_fact_slots": slots}))
        if any(binding.kind == "observation_field" for binding in (task.bindings or {}).values()):
            continue
        call = compile_planned_action(task, registry, state_fields=projection, upstream_observations={})
        key = AcquisitionKey.from_validated_arguments(call.name, call.arguments,
            projection.get("acquisition_snapshot_identity", "run:frozen"))
        prior = ActionRuntime.matching_acquisitions(key, acquired)
        if prior or key.value in seen:
            reason = "plan_acquisition_already_acquired" if prior else "plan_duplicate_acquisition"
            raise VerificationPlanError(reason, task_id=task.task_id,
                detail=json.dumps({"acquisition_key": key.value, "arguments": call.arguments,
                                   "observation_ids": [row.get("observation_id") for row in prior]}))
        seen.add(key.value)


def admit_verification_plan(
    plan: EnrichedVerificationPlan, *, state: Any, state_snapshot_hash: str,
) -> AdmittedVerificationPlan:
    """Bind exact canonical OPEN IDs and contract slots."""
    by_id = {item.id: item for item in state.obligations}
    tasks = []
    for task in plan.tasks:
        target = task.target_obligation_id
        if target is None or target not in by_id:
            raise VerificationPlanError("plan_target_unknown", task_id=task.task_id)
        if by_id[target].status != "OPEN":
            raise VerificationPlanError("plan_target_terminal", task_id=task.task_id)
        slots = by_id[target].required_fact_slots
        for intent in task.semantic_intents:
            item = by_id.get(intent.target_obligation_id)
            if item is None or item.status != "OPEN" or intent.target_fact_slot not in item.required_fact_slots:
                raise VerificationPlanError("plan_semantic_intent_invalid", task_id=task.task_id)
        slot = task.target_fact_slot
        if slot is None and len(slots) == 1:
            slot = slots[0]
        if slot not in slots:
            raise VerificationPlanError("plan_fact_slot_required", task_id=task.task_id,
                                        detail=json.dumps({"allowed_fact_slots": slots}))
        tasks.append(EnrichedAction.model_validate({**task.model_dump(),
                     "target_obligation_id": target,
                     "target_fact_slot": slot}))
    return AdmittedVerificationPlan(tasks=tuple(tasks), state_snapshot_hash=state_snapshot_hash,
                                    committed_state_version=state.event_count)
