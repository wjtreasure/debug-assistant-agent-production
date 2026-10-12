"""Snapshot-local references derived from canonical CompletionContracts only."""
from __future__ import annotations

import hashlib
import json

from debug_assistant.memory.evidence_projector import EvidenceProposal


def requirement_table(projection: dict) -> tuple[dict, ...]:
    # Identity follows canonical semantic owners, never a role-view/operational
    # projection hash. Renderer and admission may carry different metadata.
    hypothesis = projection.get('active_hypothesis') or {}
    scope = [projection.get('incident_id'), hypothesis.get('hypothesis_id'), hypothesis.get('revision')]
    rows = []
    for obligation in projection.get('obligations', ()):
        for fact in obligation['completion_contract']['required_facts']:
            identity = json.dumps([scope, obligation['id'], obligation['completion_contract'], fact['slot']],
                                  sort_keys=True, separators=(',', ':'))
            rows.append({'requirement_ref': 'req-' + hashlib.sha256(identity.encode()).hexdigest(),
                         'obligation_id': obligation['id'], **fact})
    return tuple(rows)


class RequirementTargetError(ValueError):
    def __init__(self, code, feedback):
        self.code, self.feedback = code, feedback
        super().__init__(code)


def targeting_feedback(projection, index, target, rejected, scope):
    allowed = [row for row in requirement_table(projection) if row['obligation_id'] == target]
    lookup = {row['requirement_ref']: row for row in allowed}
    feedback = {'proposal_index': index, 'target_obligation_id': target,
                'rejected_refs': [ref for ref in rejected if ref not in lookup],
                'allowed_requirement_refs': [row['requirement_ref'] for row in allowed],
                'legal_obligation_targets': list(dict.fromkeys(row['obligation_id'] for row in requirement_table(projection))),
                'requirements': [{key: row[key] for key in ('requirement_ref', 'description', 'role')} for row in allowed],
                'current_scope': projection.get('evidence_hypothesis_id'),
                'current_revision': (projection.get('active_hypothesis') or {}).get('revision'),
                'scope_intention': scope,
                'legal_actions': ['COVER_EXISTING_REQUIREMENT (required exact obligation_id and requirement_refs)',
                                 'NO_RELEVANT_COVERAGE (required exact obligation_id; empty requirement_refs)']}
    return feedback


def resolve_fact(item, *, projection, transition, index):
    """Resolve exact references; no aliases, qualification or semantic inference."""
    rows = requirement_table(projection)
    allowed = [row for row in rows if row['obligation_id'] == item.obligation_id]
    lookup = {row['requirement_ref']: row for row in allowed}
    feedback = targeting_feedback(projection, index, item.obligation_id, item.requirement_refs, item.hypothesis_target)
    def reject(code):
        raise RequirementTargetError(code, feedback)
    if not allowed:
        reject('requirement_target_invalid')
    kind = transition.kind
    if item.hypothesis_target == 'CURRENT' and (kind == 'INITIAL' or not projection.get('evidence_hypothesis_id')):
        reject('evidence_scope_invalid')
    if item.requirement_refs and (feedback['rejected_refs'] or len(set(item.requirement_refs)) != len(item.requirement_refs)):
        reject('requirement_ref_invalid')
    if item.coverage_action == 'COVER_EXISTING_REQUIREMENT':
        if not item.requirement_refs:
            reject('coverage_action_invalid')
    elif item.requirement_refs:
        reject('coverage_action_invalid')
    fields = item.model_dump(exclude={'requirement_refs', 'coverage_action'})
    return EvidenceProposal(**fields, covered_fact_slots=tuple(lookup[ref]['slot'] for ref in item.requirement_refs),
                            incident_id=projection.get('incident_id', ''), hypothesis_id='')
