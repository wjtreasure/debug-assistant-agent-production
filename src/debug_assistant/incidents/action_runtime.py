from __future__ import annotations

import json
import hashlib
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

from debug_assistant.harness.retry import RetryPolicy
from debug_assistant.harness.tool_executor import execute_with_retry
from debug_assistant.models import ActionKind, ActionProposal, ToolObservation


@dataclass(frozen=True, slots=True)
class AcquisitionKey:
    operation: str
    canonical_arguments_json: str
    snapshot_identity: str

    @classmethod
    def from_validated_arguments(cls, operation: str, arguments: dict[str, Any],
                                 snapshot_identity: str) -> AcquisitionKey:
        """Serialize existing Registry validation output; never normalize again."""
        if not snapshot_identity:
            raise ValueError("runtime acquisition snapshot identity required")
        return cls(operation, json.dumps(arguments, sort_keys=True, ensure_ascii=False,
                                         separators=(",", ":"), allow_nan=False), snapshot_identity)

    @property
    def value(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class InvestigationUse:
    obligation_id: str | None = None
    target_fact_slot: str | None = None
    hypothesis_scope: tuple[str, ...] = ()
    task_id: str | None = None
    node_id: str | None = None


@dataclass(frozen=True, slots=True)
class DuplicateAcquisition:
    acquisition_key: str
    existing_observation_id: str
    status: str = "ACQUIRED"
    reason_code: str = "acquisition_already_acquired"


@dataclass(frozen=True, slots=True)
class ActionExecutionContext:
    action: ActionProposal
    permitted_tool_names: tuple[str, ...]
    visible_tool_names: tuple[str, ...]
    capability_tool_names: tuple[str, ...]
    absolute_deadline: float | None = None
    requested_for: InvestigationUse = InvestigationUse()


@dataclass(frozen=True, slots=True)
class ActionAdmission:
    allowed: bool
    reason_code: str
    arguments: dict[str, Any]
    acquisition_key: str
    error: dict[str, Any] | None = None
    duplicate: DuplicateAcquisition | None = None


class ActionRuntime:
    """Sole acquisition owner: existing operational action ledger, no second registry."""

    def __init__(self, registry, *, action_ledger: list[dict[str, Any]], circuit,
                 retry_policy: RetryPolicy, snapshot_identity: str = "run:frozen"):
        self.registry = registry
        self.action_ledger = action_ledger
        self.circuit = circuit
        self.retry_policy = retry_policy
        self.snapshot_identity = snapshot_identity
        self.reacquire_reason = ""

    def acquired_keys(self) -> frozenset[str]:
        """Successful immutable acquisitions; attempts/reuses do not advance this set."""
        return frozenset(row['acquisition_key'] for row in self.action_ledger
            if row.get('acquisition_key') and row.get('acquisition_state') == 'ACQUIRED'
            and row.get('observation_id'))

    def authorize_refresh(self, snapshot_identity: str, *, reason: str) -> None:
        """Runtime policy seam; no model-facing refresh field."""
        if reason not in {"snapshot_changed", "freshness_expired", "explicit_runtime_refresh"}:
            raise ValueError("runtime freshness reason required")
        if not snapshot_identity or snapshot_identity == self.snapshot_identity:
            raise ValueError("refresh requires a new runtime freshness identity")
        self.snapshot_identity, self.reacquire_reason = snapshot_identity, reason

    @staticmethod
    def matching_acquisitions(key: AcquisitionKey, acquisitions) -> list[dict[str, Any]]:
        value = key.value
        return [row for row in acquisitions if row.get("acquisition_key") == value
                and row.get("acquisition_state") == "ACQUIRED" and row.get("observation_id")]

    @staticmethod
    def acquisition_projection(actions) -> list[dict[str, Any]]:
        """Detached ledger facts; InvestigationUse never determines data identity."""
        return [{key: deepcopy(row[key]) for key in (
            "tool", "arguments", "acquisition_key", "snapshot_identity", "requested_for",
            "observation_id", "outcome", "error_type", "acquisition_state", "reacquire_reason", "semantic_intents",
        ) if key in row} for row in actions if row.get("acquisition_key")]

    @staticmethod
    def gap_choices(choices, registry, projection) -> list[dict[str, Any]]:
        from debug_assistant.tools.provenance import registry_provenance_profiles, profile_matches_constraints
        specs = {spec.name: spec for spec in registry.specs()}
        profiles = registry_provenance_profiles(registry)
        operations = {choice['operation'] for choice in choices}
        acquisitions = projection.get("acquisitions", ())
        snapshot = projection.get("acquisition_snapshot_identity", "run:frozen")
        rows = []
        coverage = {row['obligation_id']: row.get('coverage', {}) for row in projection.get('evidence_gaps', ())}
        for obligation in projection.get("obligations", ()):
            if obligation.get("status") != "OPEN":
                continue
            facts = {fact['slot']: fact for fact in obligation['completion_contract']['required_facts']}
            for slot in obligation["required_fact_slots"]:
                if coverage.get(obligation["id"], {}).get(slot):
                    continue
                available, exhausted = [], []
                constraints = facts[slot].get('evidence_constraints', {})
                producers = {tool for profile in profiles if profile_matches_constraints(profile, constraints)
                             for tool in profile['source_tools']} & operations
                eligible = producers | {companion for tool in producers
                                        for companion in specs[tool].companion_tools}
                for choice in choices:
                    operation = choice["operation"]
                    if any(constraints.values()) and operation not in eligible:
                        continue
                    if operation in obligation.get("blocked_capabilities", ()):
                        continue
                    duplicate = False
                    if not specs[operation].args_model.model_fields:
                        arguments, error = registry.validate_arguments(operation, {})
                        if not error:
                            key = AcquisitionKey.from_validated_arguments(operation, arguments, snapshot)
                            duplicate = bool(ActionRuntime.matching_acquisitions(key, acquisitions))
                    (exhausted if duplicate else available).append(choice["choice_ref"])
                rows.append({"obligation_id": obligation["id"], "target_fact_slot": slot,
                    "available_choice_refs": available, "exhausted_choice_refs": exhausted,
                    "exhausted_reason": "acquisition_already_acquired" if exhausted else None})
        return rows

    def validate_arguments(self, tool_name: str, arguments: dict[str, Any]):
        return self.registry.validate_arguments(tool_name, arguments)

    def tool_spec(self, tool_name: str):
        return next((spec for spec in self.registry.specs() if spec.name == tool_name), None)

    def validate_surface(self, tool_name: str, *, permitted_tools, visible_tools, capability_tools) -> str:
        if tool_name not in permitted_tools:
            return "ACTION_PERMISSION_REQUIRED"
        if tool_name not in visible_tools:
            return "ACTION_VISIBILITY_REQUIRED"
        if tool_name not in capability_tools:
            return "ACTION_CAPABILITY_REQUIRED"
        return ""

    def resource_is_directory(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        if tool_name != "read_file":
            return False
        source_root = getattr(getattr(self.registry, "source_binding", None), "source_root", None)
        raw_path = str(arguments.get("path") or "").strip()
        if source_root is None or not raw_path:
            return False
        try:
            normalized = raw_path.replace("\\", "/")
            if normalized == "code":
                normalized = "."
            elif normalized.startswith("code/"):
                normalized = normalized[5:] or "."
            root = Path(source_root).resolve(strict=True)
            candidate = Path(normalized)
            if not candidate.is_absolute():
                candidate = root / candidate
            candidate = candidate.resolve(strict=False)
            return (candidate == root or root in candidate.parents) and candidate.is_dir()
        except (OSError, RuntimeError, ValueError):
            return False

    def admit(self, context: ActionExecutionContext) -> ActionAdmission:
        action = context.action
        tool_name = str(action.tool or "")
        reason = self.validate_surface(
            tool_name, permitted_tools=context.permitted_tool_names,
            visible_tools=context.visible_tool_names,
            capability_tools=context.capability_tool_names,
        )
        if action.kind is not ActionKind.TOOL:
            reason = "ACTION_TOOL_REQUIRED"
        if reason:
            return ActionAdmission(False, reason, {}, "")
        spec = self.tool_spec(tool_name)
        if spec is None:
            return ActionAdmission(False, "ACTION_CANONICAL_TOOL_REQUIRED", {}, "")
        if spec.side_effect != "none":
            return ActionAdmission(False, "ACTION_READ_ONLY_PERMISSION_REQUIRED", {}, "")
        arguments, error = self.validate_arguments(tool_name, action.arguments)
        if error:
            return ActionAdmission(False, "ACTION_SCHEMA_REQUIRED", {}, "", error)
        key = AcquisitionKey.from_validated_arguments(tool_name, arguments, self.snapshot_identity)
        acquisition_key = key.value
        if self.resource_is_directory(tool_name, arguments):
            return ActionAdmission(False, "ACTION_FILE_RESOURCE_REQUIRED", arguments, acquisition_key)
        if self.registry.get(tool_name) is None:
            return ActionAdmission(False, "ACTION_EXECUTABLE_TOOL_REQUIRED", arguments, acquisition_key)
        prior = self.matching_acquisitions(key, self.action_ledger)
        if prior:
            duplicate = DuplicateAcquisition(key.value, prior[-1]["observation_id"])
            return ActionAdmission(False, "ACTION_ACQUISITION_ALREADY_ACQUIRED", arguments,
                                   acquisition_key, duplicate=duplicate)
        return ActionAdmission(True, "ACTION_EXECUTION_ALLOWED", arguments, acquisition_key)

    def execute(self, context: ActionExecutionContext, *,
                before_execution: Callable[[], None] | None = None,
                record: Callable[[str, dict[str, Any]], None] | None = None,
                action_record: dict[str, Any] | None = None) -> ToolObservation | DuplicateAcquisition:
        record = record or (lambda *_: None)
        admission = self.admit(context)
        tool_name = str(context.action.tool or "")
        if admission.duplicate is not None:
            record("ACQUISITION_DUPLICATE", {**asdict(admission.duplicate),
                "requested_for": asdict(context.requested_for)})
            return admission.duplicate
        if not admission.allowed:
            return ToolObservation(tool_name, False, admission.reason_code, {
                "failure_category": "action_validation", "reason_code": admission.reason_code,
                "validation_error": admission.error,
            }, admission.reason_code)
        if not self.circuit.before_call(tool_name):
            record("TOOL_CALL_BLOCKED_BY_CIRCUIT", {
                "tool": tool_name, "health": self.circuit.summary().get(tool_name, {}),
            })
            return self.circuit.blocked_observation(tool_name)
        if before_execution is not None:
            before_execution()
        row = action_record if action_record is not None else {}
        row.update({"tool": tool_name, "arguments": deepcopy(admission.arguments),
            "acquisition_key": admission.acquisition_key, "snapshot_identity": self.snapshot_identity,
            "requested_for": asdict(context.requested_for), "acquisition_state": "ATTEMPTED",
            "reacquire_reason": self.reacquire_reason or None})
        if not any(item is row for item in self.action_ledger):
            self.action_ledger.append(row)
        if self.reacquire_reason:
            record("REACQUIRE_ALLOWED", {"acquisition_key": admission.acquisition_key,
                "snapshot_identity": self.snapshot_identity, "reason_code": self.reacquire_reason})
        record("ACQUISITION_ATTEMPTED", {"acquisition_key": admission.acquisition_key})
        observation = execute_with_retry(
            self.registry.get(tool_name), admission.arguments,
            policy=self.retry_policy, absolute_deadline=context.absolute_deadline,
            on_retry=lambda attempt, failed: record("TOOL_RETRY", {
                "tool": tool_name, "attempt": attempt, "error_type": failed.error_type,
            }),
        )
        row.update({"acquisition_state": "ACQUIRED" if observation.ok else "ATTEMPTED",
                    "observation_id": observation.observation_id, "outcome": "ok" if observation.ok else "error",
                    "error_type": observation.error_type})
        record("ACQUISITION_RESULT", {"acquisition_key": admission.acquisition_key,
            "acquisition_state": row["acquisition_state"], "observation_id": observation.observation_id})
        normalization = (observation.metadata or {}).get("normalization_action")
        if isinstance(normalization, dict):
            record("PATH_NORMALIZED", {
                "tool": tool_name, "normalization_action": dict(normalization),
                "observation_id": observation.observation_id,
            })
        if self.circuit.observe(tool_name, observation) == "opened":
            record("TOOL_CIRCUIT_OPEN", {
                "tool": tool_name, "health": self.circuit.summary().get(tool_name, {}),
            })
        return observation
