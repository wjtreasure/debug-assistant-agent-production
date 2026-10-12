"""Stateless lifecycle validation and short-lived semantic publication staging."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .contracts import HypothesisTransition, IncidentHypothesis
from .rca_state import RCAState


class SemanticTransitionError(ValueError):
    def __init__(self, code: str, detail: str = ""):
        self.code = code
        self.details = detail if isinstance(detail, dict) else {}
        super().__init__(f"{code}: {detail}")


def grounded_contradiction_ids(proposed_ids, facts, scopes=()) -> tuple[str, ...]:
    """Citations select admitted relationships in the exact requested scope."""
    def value(fact, key):
        return fact.get(key, '') if isinstance(fact, dict) else getattr(fact, key, '')
    grounded = {value(fact, 'evidence_id') for fact in facts
                if (not scopes or value(fact, 'hypothesis_fingerprint') in scopes)
                and 'CONTRADICTS' in {value(fact, 'relation'), value(fact, 'hypothesis_relation')}}
    return tuple(dict.fromkeys(ref for ref in proposed_ids if ref in grounded))


def contradiction_reference_ids(projection) -> tuple[str, ...]:
    evidence = projection.get('existing_evidence', ())
    facts = [dict(row, evidence_id=item['evidence_ref'], hypothesis_fingerprint=row.get('hypothesis_scope', ''))
             for item in evidence for row in item.get('relations', ())]
    return grounded_contradiction_ids((item['evidence_ref'] for item in evidence), facts,
                                      projection.get('hypothesis_scope', ()))


def disposition_targets(obligations) -> tuple[str, ...]:
    """Read-only legal surface of the canonical lifecycle; no semantic repair."""
    return tuple(item["id"] for item in obligations
                 if item.get("status") in {"OPEN", "BLOCKED_BY_CAPABILITY"})


def constrain_disposition_schema(schema: dict, targets: tuple[str, ...], required=()) -> None:
    schema["$defs"]["ObligationDisposition"]["properties"]["obligation_id"] = {
        "type": "string", **({"enum": list(targets)} if targets else {}),
    }
    field = schema["$defs"]["HypothesisTransition"]["properties"]["obligation_dispositions"]
    for branch in field.get('anyOf', (field,)):
        if branch.get('type') == 'array':
            branch.update(minItems=len(required), maxItems=len(targets))
            if required:
                schema['$defs']['HypothesisTransition']['properties']['obligation_dispositions'] = branch


def validate_disposition_targets(transition, targets) -> None:
    if transition is not None and {item.obligation_id for item in transition.obligation_dispositions} - set(targets):
        raise SemanticTransitionError("obligation_disposition_invalid", {
            "legal_disposition_targets": list(targets), "legal_actions": ["RETAIN", "SUPERSEDE"],
        })


def validate_required_dispositions(transition, required) -> None:
    missing = set(required) - {item.obligation_id for item in transition.obligation_dispositions}
    if missing:
        raise SemanticTransitionError("critical_obligation_disposition_required", str(sorted(missing)))


def resolve_hypothesis(state: RCAState, transition: HypothesisTransition | None,
                       proposed: IncidentHypothesis) -> IncidentHypothesis:
    current = state.active_hypothesis_projection()
    if not proposed.claim.strip():
        raise SemanticTransitionError("hypothesis_claim_required")
    if current is not None and (not current.hypothesis_id or current.revision < 1):
        raise SemanticTransitionError("canonical_hypothesis_identity_required")
    if transition is None:
        raise SemanticTransitionError("hypothesis_transition_required")
    if (transition.kind == "INITIAL") != (current is None):
        raise SemanticTransitionError("hypothesis_transition_invalid", transition.kind)
    dispositions = transition.obligation_dispositions
    if len({item.obligation_id for item in dispositions}) != len(dispositions):
        raise SemanticTransitionError("obligation_disposition_conflict")
    validate_disposition_targets(transition, disposition_targets(item.model_dump() for item in state.obligations))
    if current is not None:
        validate_required_dispositions(transition, state.open_critical_obligation_ids)
    if transition.kind == "KEEP_CURRENT":
        causal_fields = ("claim", "component", "fault", "fault_code", "fault_explanation",
                         "mechanism", "mechanism_category", "source_claims")
        if any(getattr(proposed, key) != getattr(current, key) for key in causal_fields):
            raise SemanticTransitionError("keep_current_causal_change")
        return current
    if transition.kind == "REFINE":
        identity, revision = current.hypothesis_id, current.revision + 1
    else:
        # Allocation derives only from canonical history, never fault/claim text.
        identity, revision = f"hyp-{state.incident_id}-{len(state.hypotheses) + 1}", 1
    return proposed.model_copy(update={"hypothesis_id": identity, "revision": revision})


def validate_disposition_conflicts(transition: HypothesisTransition, before: RCAState,
                                   staged: RCAState) -> None:
    for disposition in transition.obligation_dispositions:
        if disposition.action == "SUPERSEDE":
            target = staged.obligation(disposition.obligation_id)
            if target.status not in {"OPEN", "BLOCKED_BY_CAPABILITY"}:
                raise SemanticTransitionError("semantic_transition_conflict",
                                              f"{target.id}: SUPERSEDE + {target.status}")


@dataclass
class SemanticTraceBuffer:
    """Detached proposal diagnostics; never writes the committed trace."""
    records: list[tuple[str, Any]] = field(default_factory=list)

    def record(self, event_type: str, payload: Any) -> None:
        self.records.append((event_type, deepcopy(payload)))

    def validate(self) -> None:
        rejected = {"EVIDENCE_PROPOSAL_REJECTED", "EVIDENCE_SET_STATE_REJECTED",
                    "OBLIGATION_PROPOSAL_REJECTED", "CONTRADICTION_PROPOSAL_REJECTED",
                    "CONTRADICTION_PROJECTION_REUSED", "REFLECTION_DELTA_REJECTED"}
        for kind, payload in self.records:
            if kind in rejected:
                code = payload.get("reason_code") or next(iter(payload.get("admission_reasons", ())), "semantic_proposal_invalid")
                raise SemanticTransitionError(code, payload)

    def publish(self, trace: Any) -> None:
        for kind, payload in self.records:
            trace.record(kind, payload)
