from __future__ import annotations
from pathlib import Path
import json

def summarize_trace(path):
    events=[json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
    counts={}
    for e in events: counts[e['type']]=counts.get(e['type'],0)+1
    end=next((e['payload'] for e in reversed(events) if e['type']=='RUN_END'),{})
    state=end.get('summary',{})
    return {"events":len(events),"tool_calls":counts.get('TOOL_OBSERVATION',0),"route_rejections":counts.get('ROUTE_REJECTED',0),"loop_blocks":counts.get('LOOP_BLOCKED',0),"reflections":counts.get('REFLECTION',0),"evidence_added":counts.get('EVIDENCE_ADDED',0),"status":state.get('status')}
