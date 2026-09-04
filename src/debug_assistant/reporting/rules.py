from __future__ import annotations

def _source_covers(ev,claim):
    if ev is None or ev.source!='read_file': return False
    if claim.get('file') and ev.file!=claim.get('file'): return False
    a=claim.get('line_start'); b=claim.get('line_end')
    if a is None or b is None:return True
    return ev.source_start_line is not None and ev.source_end_line is not None and ev.source_start_line<=a and ev.source_end_line>=b

def derive_claim_status(claim, *, evidence_by_id, obligations=None, hypothesis=None):
    ctype=claim.get('claim_type'); ids=list(claim.get('evidence_ids') or [])
    evs=[evidence_by_id.get(x) for x in ids if x in evidence_by_id]
    if ctype=='source_fact':
        if ids and len(evs)==len(ids) and any(_source_covers(ev,claim) for ev in evs): return 'observed'
        if evs:return 'acquired_unreviewed'
        return 'hypothesis'
    if ctype=='causal_inference':
        oids=list(claim.get('obligation_ids') or [])
        if obligations is not None and oids:
            objs=[obligations.items.get(x) for x in oids]
            if all(o is not None and o.last_review_decision=='resolved' and o.last_reviewed_reflection_id for o in objs):return 'supported_inference'
        return 'hypothesis'
    if ctype=='diagnosis':
        status=getattr(hypothesis,'status',None) if hypothesis is not None else None
        return 'supported' if status in {'supported','confirmed'} else 'hypothesis'
    return 'hypothesis'

def apply_reporting_rules(report, *, evidence, repository_index=None, obligations=None, hypothesis=None):
    corrections=[]; by_id={e.evidence_id:e for e in evidence}
    corrected_claims=[]
    for claim in list(getattr(report,'claims',[]) or []):
        row=dict(claim); old=row.get('status'); new=derive_claim_status(row,evidence_by_id=by_id,obligations=obligations,hypothesis=hypothesis)
        row['status']=new; corrected_claims.append(row)
        if old!=new:corrections.append({'kind':'claim_status','text':row.get('text','')[:120],'from_status':old,'to_status':new})
    report.claims=corrected_claims
    grounded=[]
    for cp in list(report.recommended_change_points or []):
        row=dict(cp); file=row.get('file') or ''; symbol=row.get('symbol') or ''
        if repository_index is not None and symbol:
            exact=repository_index.resolve_symbol(symbol,file or None)
            if len(exact)==1:
                actual=exact[0]['path']
                if file!=actual:
                    corrections.append({'kind':'change_point_file','symbol':symbol,'from_file':file,'to_file':actual}); row['file']=actual
                grounded.append(row)
            else:
                global_exact=repository_index.resolve_symbol(symbol,None)
                if len(global_exact)==1:
                    corrections.append({'kind':'change_point_file','symbol':symbol,'from_file':file,'to_file':global_exact[0]['path']}); row['file']=global_exact[0]['path']; grounded.append(row)
                else:
                    corrections.append({'kind':'change_point_dropped','file':file,'symbol':symbol,'reason':'unresolvable_or_ambiguous_symbol'})
        else:grounded.append(row)
    report.recommended_change_points=grounded
    return report,corrections
