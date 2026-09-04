from __future__ import annotations
from debug_assistant.harness.obligations import ObligationStatus
from debug_assistant.memory.hypothesis import normalize_target

class SemanticInvariantError(ValueError):
    pass


def _gap_target(row) -> str:
    if hasattr(row,'model_dump'):
        row=row.model_dump()
    if isinstance(row,dict):
        return normalize_target(str(row.get('target') or ''))
    return normalize_target(str(row or ''))


def validate_semantic_candidate(obligations, hypothesis_state, *, submitted_reflection_ids: set[str] | None=None):
    """Deterministic cross-state checks run immediately before semantic commit.

    These assertions intentionally validate only machine-provable relationships.  They
    never try to decide whether two natural-language causal claims are semantically
    contradictory.
    """
    enforce_submission = submitted_reflection_ids is not None
    submitted=set(submitted_reflection_ids or set())
    required_items=[]
    resolved_targets=set()
    reviewed_support=set()
    has_critical_semantic_obligations=False

    for obj in obligations.items.values():
        semantic=obj.goal_type in {'behavior','causality','caller','contradiction'}
        has_critical_semantic_obligations = has_critical_semantic_obligations or (semantic and obj.critical)
        if obj.status is ObligationStatus.SATISFIED and obj.active_required:
            raise SemanticInvariantError(f'{obj.obligation_id}: resolved obligation cannot remain active_required')
        if obj.status is ObligationStatus.SUPERSEDED:
            if not obj.superseded_by:
                raise SemanticInvariantError(f'{obj.obligation_id}: superseded obligation missing superseded_by')
            child=obligations.items.get(obj.superseded_by)
            if child is None or child.refined_from != obj.obligation_id:
                raise SemanticInvariantError(f'{obj.obligation_id}: refinement linkage is inconsistent')
        if obj.last_presented_reflection_id and enforce_submission and obj.last_presented_reflection_id not in submitted:
            raise SemanticInvariantError(f'{obj.obligation_id}: PRESENTED without submitted LLM request')
        if obj.status is ObligationStatus.SATISFIED and semantic:
            if obj.last_review_decision != 'resolved' or not obj.last_reviewed_reflection_id:
                raise SemanticInvariantError(f'{obj.obligation_id}: semantic resolved without committed review')
            if obj.last_presented_reflection_id != obj.last_reviewed_reflection_id:
                raise SemanticInvariantError(f'{obj.obligation_id}: semantic review was not over same-reflection presented evidence')
            reviewed_support.update(obj.evidence_ids)
        if obj.status in {ObligationStatus.SATISFIED,ObligationStatus.SUPERSEDED}:
            resolved_targets.add(normalize_target(obj.target))
        if obj.active_required and obj.critical and obj.status not in {ObligationStatus.SATISFIED,ObligationStatus.SUPERSEDED}:
            required_items.append(obj)

    hs=hypothesis_state
    required=getattr(hs,'required_missing_evidence',[]) if hs is not None else []
    gap_targets={_gap_target(x) for x in required}
    stale=sorted(x for x in resolved_targets if x and x in gap_targets)
    if stale:
        raise SemanticInvariantError(f'resolved/superseded obligation still present in hypothesis required gaps: {stale[:3]}')

    if hs is not None and getattr(hs,'status','') in {'supported','confirmed'}:
        support=set(getattr(hs,'supporting_evidence_ids',[]) or [])
        if not support:
            raise SemanticInvariantError('supported hypothesis requires supporting evidence')
        if required:
            raise SemanticInvariantError('supported hypothesis cannot retain required gaps')
        if not getattr(hs,'evidence_sufficient',False):
            raise SemanticInvariantError('supported hypothesis requires evidence_sufficient=true')
        if has_critical_semantic_obligations and not (support & reviewed_support):
            raise SemanticInvariantError('supported hypothesis requires at least one semantically reviewed supporting evidence item')

    # Tracker is the semantic source of truth; hypothesis cannot claim no gaps while tracker has active gaps.
    if required_items and not required:
        raise SemanticInvariantError('hypothesis missing active required obligations')
    return True
