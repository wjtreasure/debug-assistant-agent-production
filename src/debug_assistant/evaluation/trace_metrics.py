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
    prompt_breakdowns=[e['payload'] for e in events if e['type']=='PROMPT_BREAKDOWN']
    first_stable_total=state.get('tokens_at_first_stable_diagnosis')
    total_tokens=int(usage.get('tokens',0) or 0)
    post_stable=(max(0,total_tokens-int(first_stable_total)) if first_stable_total is not None else None)
    return {
        'events':len(events),'status':state.get('status'),'report_source':state.get('report_source'),
        'tool_calls':counts.get('TOOL_OBSERVATION',0),'observation_reuse_count':counts.get('OBSERVATION_REUSED',0),
        'rehydration_count':counts.get('OBSERVATION_REHYDRATED',0),'redundant_request_count':counts.get('REDUNDANT_CONTEXT_REQUEST',0),
        'route_rejections':counts.get('ACTION_REJECTED',0),'loop_blocks':counts.get('LOOP_BLOCKED',0),
        'reflections':counts.get('REFLECTION',0),'evidence_added':counts.get('EVIDENCE_ADDED',0),
        'no_progress_count':counts.get('NO_PROGRESS',0),'progress_count':counts.get('PROGRESS',0),'hypothesis_updates':counts.get('HYPOTHESIS_UPDATED',0),
        'termination_advisories':counts.get('TERMINATION_ADVISORY',0),'fallback_reports':counts.get('FALLBACK_REPORT_BUILT',0),
        'forced_finalization':bool(state.get('forced_finalization') or counts.get('FORCE_FINALIZATION',0)),
        'budget_critical_entered':bool(state.get('budget_critical_entered') or counts.get('BUDGET_CRITICAL',0)),
        'budget_exhausted_events':counts.get('BUDGET_EXHAUSTED',0),
        'first_supported_hypothesis_step':state.get('first_supported_hypothesis_step'),
        'first_stable_diagnosis_step':state.get('first_stable_diagnosis_step'),
        'prompt_tokens_at_first_stable_diagnosis':state.get('prompt_tokens_at_first_stable_diagnosis'),
        'completion_tokens_at_first_stable_diagnosis':state.get('completion_tokens_at_first_stable_diagnosis'),
        'tokens_at_first_stable_diagnosis':first_stable_total,
        'post_stable_diagnosis_tokens':post_stable,
        'post_stable_diagnosis_ratio':(post_stable/total_tokens if post_stable is not None and total_tokens else None),
        'llm_calls':usage.get('calls',len(prompts)),'prompt_tokens':usage.get('prompt_tokens',0),'completion_tokens':usage.get('completion_tokens',0),
        'total_tokens':total_tokens,
        'avg_prompt_tokens':(sum(int(x.get('prompt_tokens',0) or 0) for x in prompts)/len(prompts) if prompts else 0),
        'max_prompt_tokens':max([int(x.get('prompt_tokens',0) or 0) for x in prompts] or [0]),
        'context_build_count':len(ctx),'avg_context_chars':(sum(int(x.get('used_chars',0)) for x in ctx)/len(ctx) if ctx else 0),
        'max_context_chars':max([int(x.get('used_chars',0)) for x in ctx] or [0]),
        'avg_working_set_size':(sum(int(x.get('working_set_size',0)) for x in ctx)/len(ctx) if ctx else 0),
        'invalid_context_requests':sum(len(x.get('invalid_requested_ids') or []) for x in ctx),
        'prompt_breakdown_events':len(prompt_breakdowns),
        'prompt_breakdown_chars':{
            key:sum(int(x.get(key,0) or 0) for x in prompt_breakdowns)
            for key in sorted({k for x in prompt_breakdowns for k in x if k.endswith('_chars')})
        },
    }
