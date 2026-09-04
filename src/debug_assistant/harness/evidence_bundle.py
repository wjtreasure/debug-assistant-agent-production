from __future__ import annotations
from dataclasses import dataclass
from debug_assistant.context.indexes import extract_numbered_range

_GOAL_RANK={'contradiction':0,'causality':1,'behavior':2,'caller':3,'location':4,'test':5,'history':6,'evidence':7}

@dataclass(slots=True)
class BundleItem:
    obligation_id: str
    evidence_id: str
    observation_id: str
    file: str
    start_line: int
    end_line: int
    content: str
    evidence_fingerprint: str

@dataclass(slots=True)
class EvidenceBundle:
    bundle_id: str
    root_id: str | None
    items: list[BundleItem]
    text: str
    chars: int


def select_ready_obligations(tracker,max_items=3):
    rows=tracker.presentation_candidates()
    if not rows:return []
    rows=sorted(rows,key=lambda o:(0 if o.active_required else 1,0 if o.critical else 1,0 if o.last_reviewed_evidence_fingerprint!=tracker.evidence_fingerprint(o) else 1,_GOAL_RANK.get(o.goal_type,9),0 if (o.symbol_ranges or o.line_hint) else 1,o.obligation_id))
    root=rows[0].information_need_root_id
    same=[x for x in rows if x.information_need_root_id==root]
    return same[:max(1,int(max_items))]


def build_evidence_bundle(tracker,state_evidence,observation_store,*,bundle_id,max_items=3,max_chars=16000):
    selected=select_ready_obligations(tracker,max_items=max_items)
    if not selected:return None
    evidence_by_id={e.evidence_id:e for e in state_evidence}
    items=[]; seen=set(); used=0; sections=[]
    for obj in selected:
        chosen=None
        for eid in obj.evidence_ids:
            ev=evidence_by_id.get(eid)
            if not ev or not ev.raw_observation_id:continue
            scope=tracker.presentation_scope(obj,ev)
            raw=observation_store.get(ev.raw_observation_id)
            if not scope or raw is None:continue
            path,a,b=scope; content,_,_=extract_numbered_range(raw.content,a,b)
            if not content:continue
            chosen=(ev,raw,path,a,b,content);break
        if not chosen:continue
        ev,raw,path,a,b,content=chosen
        key=(raw.observation_id,path,a,b)
        fp=tracker.evidence_fingerprint(obj)
        if key in seen:
            # Preserve a presentation plan for every atomic obligation even when the
            # source projection is shared.  The text is emitted once, but each
            # obligation still gets its own PRESENTED/REVIEWED execution facts.
            sections.append(f'OBLIGATION {obj.obligation_id}: reuse source {path}:{a}-{b}')
            items.append(BundleItem(obj.obligation_id,ev.evidence_id,raw.observation_id,path,a,b,'',fp))
            continue
        seen.add(key)
        header=f'=== OBLIGATION {obj.obligation_id} [{obj.goal_type}] {obj.target} ===\nSOURCE {path}:{a}-{b}\n'
        remaining=max(0,int(max_chars)-used-len(header))
        if remaining<=0:break
        clipped=content[:remaining]
        block=header+clipped
        sections.append(block);used+=len(block)
        items.append(BundleItem(obj.obligation_id,ev.evidence_id,raw.observation_id,path,a,b,clipped,fp))
        if used>=max_chars:break
    if not items:return None
    return EvidenceBundle(bundle_id,selected[0].information_need_root_id,items,'\n\n'.join(sections),sum(len(x) for x in sections))
