from __future__ import annotations
from pathlib import Path
import json

def summarize_trace(path):
    events=[json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
    counts={}
    for e in events: counts[e['type']]=counts.get(e['type'],0)+1
    end=next((e['payload'] for e in reversed(events) if e['type'] in ('RUN_END','RUN_FAILED')),{})
    state=end.get('summary',{})
    usage=next((e['payload'] for e in reversed(events) if e['type']=='LLM_USAGE'),{'totals':{}}).get('totals',{})
    ctx=[e['payload'] for e in events if e['type']=='CONTEXT_BUILT']
    prompts=[e['payload'] for e in events if e['type']=='LLM_CALL_USAGE']
    return {
        'events':len(events),'status':state.get('status'),'report_source':state.get('report_source'),
        'tool_calls':counts.get('TOOL_OBSERVATION',0),'observation_reuse_count':counts.get('OBSERVATION_REUSED',0),
        'route_rejections':counts.get('ACTION_REJECTED',0),'loop_blocks':counts.get('LOOP_BLOCKED',0),
        'reflections':counts.get('REFLECTION',0),'evidence_added':counts.get('EVIDENCE_ADDED',0),
        'no_progress_count':counts.get('NO_PROGRESS',0),'hypothesis_updates':counts.get('HYPOTHESIS_UPDATED',0),
        'termination_advisories':counts.get('TERMINATION_ADVISORY',0),'fallback_reports':counts.get('FALLBACK_REPORT_BUILT',0),
        'llm_calls':usage.get('calls',len(prompts)),'prompt_tokens':usage.get('prompt_tokens',0),'completion_tokens':usage.get('completion_tokens',0),
        'total_tokens':usage.get('tokens',0),
        'avg_prompt_tokens':(sum(int(x.get('prompt_tokens',0) or 0) for x in prompts)/len(prompts) if prompts else 0),
        'max_prompt_tokens':max([int(x.get('prompt_tokens',0) or 0) for x in prompts] or [0]),
        'context_build_count':len(ctx),'avg_context_chars':(sum(int(x.get('used_chars',0)) for x in ctx)/len(ctx) if ctx else 0),
        'max_context_chars':max([int(x.get('used_chars',0)) for x in ctx] or [0]),
        'avg_working_set_size':(sum(int(x.get('working_set_size',0)) for x in ctx)/len(ctx) if ctx else 0),
        'invalid_context_requests':sum(len(x.get('invalid_requested_ids') or []) for x in ctx),
    }
