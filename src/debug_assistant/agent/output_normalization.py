"""Provider-output normalization for incident tool calls.

OpenAI-compatible providers are not equally strict about native tool schemas.
This module is the single compatibility boundary between provider JSON and the
typed Runtime.  It may repair transport-level drift, but it must not invent
facts, Evidence IDs, permissions, or required tool arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


INCIDENT_REQUIRED_CONTROL_FIELDS = (
    "skill", "skill_reason", "current_hypothesis", "evidence_gap",
    "candidate_component", "candidate_fault", "candidate_mechanism",
    "supporting_evidence_ids", "contradicting_evidence_ids",
    "required_evidence_gaps", "evidence_sufficiency", "remaining_evidence_need",
)

INCIDENT_SKILL_CONTROL_FIELDS = INCIDENT_REQUIRED_CONTROL_FIELDS + (
    "candidate_fault_code", "candidate_fault_explanation",
    "source_mechanism_status", "mechanism_category",
    "verification_obligations", "contradictions", "obligation_id",
    "expected_information_gain",
)

_OPTIONAL_STRING_CONTROLS = (
    "candidate_component", "candidate_fault", "candidate_mechanism",
    "candidate_fault_code", "candidate_fault_explanation",
    "mechanism_category", "remaining_evidence_need", "obligation_id",
    "expected_information_gain",
)
_UNKNOWN_STATE_STRING_CONTROLS = (
    "candidate_component", "candidate_fault", "candidate_mechanism",
    "remaining_evidence_need",
)
_OPTIONAL_LIST_CONTROLS = (
    "supporting_evidence_ids", "contradicting_evidence_ids",
    "required_evidence_gaps", "verification_obligations", "contradictions",
)
_OPTIONAL_FINALIZATION_METADATA = (
    "claim_evidence_mapping", "causal_chain_summary",
)
_OPTIONAL_FINALIZATION_METADATA_FIELDS = {
    "claim_evidence_mapping": frozenset({"claim", "evidence_ids"}),
    "causal_chain_summary": frozenset({"cause", "effect", "evidence_ids"}),
}

# Keep this registry deliberately explicit.  A new alias must be justified by
# a captured provider trace and added here, rather than hidden in a tool branch.
INCIDENT_COMPATIBILITY_ALIASES: Mapping[str, str] = {
    "candidate_mechanism_status": "source_mechanism_status",
}


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """Normalized arguments plus an audit trail of compatibility actions."""

    arguments: dict[str, Any]
    actions: tuple[str, ...] = ()
    dropped_fields: tuple[str, ...] = ()


class IncidentLLMOutputNormalizer:
    """Normalize provider tool JSON before executable Pydantic validation.

    ``allowed_tool_fields`` comes from the live ``ToolSpec.args_model`` and is
    therefore the Runtime's parameter whitelist.  Incident control fields are
    allowed in addition to executable fields because the native Planner carries
    both in one provider call.  Unknown provider fields are discarded and
    recorded; missing required canonical fields remain validation errors.
    """

    def __init__(
        self,
        allowed_tool_fields: Mapping[str, Iterable[str]],
        *,
        control_fields: Iterable[str] = INCIDENT_SKILL_CONTROL_FIELDS,
    ) -> None:
        self._allowed_tool_fields = {
            str(tool_name): frozenset(str(field) for field in fields)
            for tool_name, fields in allowed_tool_fields.items()
        }
        self._control_fields = frozenset(control_fields)

    def normalize_tool_arguments(
        self,
        tool_name: str,
        raw_arguments: Mapping[str, Any],
    ) -> NormalizationResult:
        args = dict(raw_arguments)
        actions: list[str] = []
        dropped_fields: list[str] = []

        self._normalize_aliases(args, actions, dropped_fields)
        self._normalize_finalize_projection(tool_name, args, actions)
        self._drop_malformed_optional_finalization_metadata(
            tool_name, args, actions, dropped_fields,
        )
        self._normalize_nullable_controls(args, actions)
        self._drop_unknown_fields(tool_name, args, actions, dropped_fields)

        return NormalizationResult(
            arguments=args,
            actions=tuple(actions),
            dropped_fields=tuple(dropped_fields),
        )

    @staticmethod
    def normalize_envelope(raw_arguments: Mapping[str, Any]) -> NormalizationResult:
        """Normalize only transport aliases around ``diagnosis_action``.

        A few OpenAI-compatible adapters use ``tool``/``tool_args`` for the
        nested executable selection.  These aliases carry no diagnosis; the
        Runtime still validates the selected name and arguments against the
        live registry.  Missing values remain missing and therefore fail
        closed in the Planner contract.
        """
        args = dict(raw_arguments)
        actions: list[str] = []
        dropped: list[str] = []
        for alias, canonical in (("tool", "tool_name"), ("tool_args", "arguments")):
            if alias not in args:
                continue
            value = args.pop(alias)
            if canonical not in args or args[canonical] is None:
                args[canonical] = value
                actions.append(f"{alias}->{canonical}")
            else:
                dropped.append(alias)
                actions.append(f"{alias}:dropped_canonical_present")
        return NormalizationResult(args, tuple(actions), tuple(dropped))

    def _normalize_aliases(
        self,
        args: dict[str, Any],
        actions: list[str],
        dropped_fields: list[str],
    ) -> None:
        for alias, canonical in INCIDENT_COMPATIBILITY_ALIASES.items():
            if alias not in args:
                continue
            alias_value = args.pop(alias)
            if canonical not in args or args[canonical] is None:
                args[canonical] = alias_value
                actions.append(f"{alias}->{canonical}")
            else:
                dropped_fields.append(alias)
                actions.append(f"{alias}:dropped_canonical_present")

    @staticmethod
    def _normalize_finalize_projection(
        tool_name: str,
        args: dict[str, Any],
        actions: list[str],
    ) -> None:
        # ``evidence_ids`` is an executable-tool projection.  Copying the
        # already-present support list is transport compatibility, not a new
        # diagnosis; the downstream min-length and known-ID checks still apply.
        support_ids = args.get("supporting_evidence_ids")
        if (
            tool_name == "finalize_diagnosis"
            and isinstance(support_ids, list)
            and ("evidence_ids" not in args or args["evidence_ids"] is None)
        ):
            args["evidence_ids"] = list(support_ids)
            actions.append("supporting_evidence_ids->evidence_ids")

    @staticmethod
    def _drop_malformed_optional_finalization_metadata(
        tool_name: str,
        args: dict[str, Any],
        actions: list[str],
        dropped_fields: list[str],
    ) -> None:
        """Drop malformed optional explanation blocks before Tool validation.

        Qwen-compatible providers have emitted prose strings and structured
        objects with the wrong keys for these fields even though the fields
        are optional and the live ToolSpec requires exact structured objects.
        Dropping a malformed optional block preserves the validated Candidate
        core; it never repairs or invents Evidence, component, fault, or
        mechanism fields.
        """
        if tool_name != "finalize_diagnosis":
            return
        for field in _OPTIONAL_FINALIZATION_METADATA:
            if field not in args:
                continue
            value = args[field]
            if value is None:
                args.pop(field, None)
                actions.append(f"{field}:null_optional_metadata_dropped")
                dropped_fields.append(field)
                continue
            expected = _OPTIONAL_FINALIZATION_METADATA_FIELDS[field]
            if not isinstance(value, list) or not all(
                isinstance(item, dict)
                and set(item) == expected
                and isinstance(item.get("evidence_ids"), (list, tuple))
                and item["evidence_ids"]
                and all(
                    isinstance(evidence_id, str) and evidence_id.startswith("ev-")
                    for evidence_id in item["evidence_ids"]
                )
                and all(isinstance(item.get(key), str) and item[key].strip() for key in expected - {"evidence_ids"})
                for item in value
            ):
                args.pop(field, None)
                actions.append(f"{field}:malformed_optional_metadata_dropped")
                dropped_fields.append(field)

    @staticmethod
    def _normalize_nullable_controls(
        args: dict[str, Any],
        actions: list[str],
    ) -> None:
        # Unknown-state hypothesis projections are semantically empty, not
        # missing diagnosis facts. Some native providers omit them on the
        # first telemetry call even though the live schema describes them as
        # empty-string-compatible. Required context such as skill_reason,
        # current_hypothesis, and evidence_gap remains strict and fail-closed.
        for field in _UNKNOWN_STATE_STRING_CONTROLS:
            if field not in args:
                args[field] = ""
                actions.append(f"{field}:missing->empty_string")
        for field in _OPTIONAL_STRING_CONTROLS:
            if field in args and args[field] is None:
                args[field] = ""
                actions.append(f"{field}:null->empty_string")
        for field in _OPTIONAL_LIST_CONTROLS:
            if field in args and args[field] is None:
                args[field] = []
                actions.append(f"{field}:null->empty_list")
            elif field in args and isinstance(args[field], tuple):
                args[field] = list(args[field])
                actions.append(f"{field}:tuple_to_array")
            elif field in args and isinstance(args[field], str):
                # A scalar string is a common provider shape drift. Treat it
                # as one item, never as a comma-separated list and never as a
                # fabricated Evidence ID. Missing fields stay strict.
                args[field] = [args[field]] if args[field] else []
                actions.append(f"{field}:scalar_to_array")
        if args.get("source_mechanism_status") is None and "source_mechanism_status" in args:
            args["source_mechanism_status"] = "unknown"
            actions.append("source_mechanism_status:null->unknown")
        if args.get("evidence_sufficiency") is None and "evidence_sufficiency" in args:
            args["evidence_sufficiency"] = "insufficient"
            actions.append("evidence_sufficiency:null->insufficient")

    def _drop_unknown_fields(
        self,
        tool_name: str,
        args: dict[str, Any],
        actions: list[str],
        dropped_fields: list[str],
    ) -> None:
        allowed = set(self._allowed_tool_fields.get(tool_name, ())) | set(self._control_fields)
        for field in sorted(set(args) - allowed):
            args.pop(field, None)
            dropped_fields.append(field)
            actions.append(f"{field}:dropped_unknown")
