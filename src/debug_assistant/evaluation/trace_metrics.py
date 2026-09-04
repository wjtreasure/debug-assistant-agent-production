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
    tool_events=[e for e in events if e['type']=='TOOL_OBSERVATION']
    reflection_failures=[e for e in events if e['type']=='REFLECTION_FAILED']
    reflection_timeout_failures=sum(
        1 for e in reflection_failures
        if 'timeout' in str((e.get('payload') or {}).get('error_type','')).lower()
        or 'deadline' in str((e.get('payload') or {}).get('error_type','')).lower()
    )
    deadline_timeout_events=sum(
        1 for e in events
        if e['type']=='LLM_DEADLINE_EXCEEDED'
        and str((e.get('payload') or {}).get('stage','')).lower()=='reflection'
    )
    # A logical deadline normally emits both LLM_DEADLINE_EXCEEDED and
    # REFLECTION_FAILED. Count the failed reflection once, using the provider
    # event only when the stage failure event is absent.
    reflection_timeout_count=max(reflection_timeout_failures, deadline_timeout_events)
    planner_steps=counts.get('ACTION_PROPOSED',0)+counts.get('NATIVE_PLANNER_RESULT',0)
    tool_failures=sum(1 for e in tool_events if not bool((e.get('payload') or {}).get('ok',True)))
    tool_failures += counts.get('TOOL_EXECUTION_FAILED',0)
    status=str(state.get('status') or '')
    return {
        'events':len(events),'status':state.get('status'),'report_source':state.get('report_source'),
        'planner_steps':planner_steps,
        'tool_calls':len(tool_events),'tool_failures':tool_failures,'tool_retries':counts.get('TOOL_RETRY',0),
        'evidence_count':counts.get('EVIDENCE_ADDED',0),
        'hypothesis_count':state.get('hypotheses',counts.get('HYPOTHESIS_UPDATED',0)),
        'reflection_count':state.get('reflection_count',counts.get('REFLECTION',0)+len(reflection_failures)),
        'reflection_timeout_count':reflection_timeout_count,
        'reporter_count':counts.get('REPORTER_CONTEXT_BUILT',0),
        'budget_exhaustion':bool(status=='budget_exhausted' or counts.get('BUDGET_EXHAUSTED',0)),
        'partial_success':status=='partial_success','failure':status=='failed' or bool(counts.get('RUN_FAILED',0)),
        'observation_reuse_count':counts.get('OBSERVATION_REUSED',0),
        'rehydration_count':counts.get('OBSERVATION_REHYDRATED',0),'redundant_request_count':counts.get('REDUNDANT_CONTEXT_REQUEST',0),
        'route_rejections':counts.get('ACTION_REJECTED',0)+counts.get('OBLIGATION_ACTION_REJECTED',0),
        'obligation_scope_rejections':counts.get('OBLIGATION_ACTION_REJECTED',0),
        'loop_blocks':counts.get('LOOP_BLOCKED',0),
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
                'llm_calls':usage.get('calls',len(prompts)),
        'logical_llm_calls':usage.get('logical_llm_calls',usage.get('calls',len(prompts))),
        'provider_attempts':usage.get('provider_attempts',usage.get('calls',len(prompts))),
        'failed_provider_attempts':usage.get('failed_provider_attempts',0),
        'llm_retry_count':usage.get('retry_count',0),
        'llm_deadline_exceeded_calls':usage.get('deadline_exceeded_calls',0),
        'prompt_tokens':usage.get('prompt_tokens',0),'completion_tokens':usage.get('completion_tokens',0),
        'total_tokens':total_tokens,
        'avg_prompt_tokens':(sum(int(x.get('prompt_tokens',0) or 0) for x in prompts)/len(prompts) if prompts else 0),
        'max_prompt_tokens':max([int(x.get('prompt_tokens',0) or 0) for x in prompts] or [0]),
        'context_build_count':len(ctx),'avg_context_chars':(sum(int(x.get('used_chars',0)) for x in ctx)/len(ctx) if ctx else 0),
        'max_context_chars':max([int(x.get('used_chars',0)) for x in ctx] or [0]),
        'avg_working_set_size':(sum(int(x.get('working_set_size',0)) for x in ctx)/len(ctx) if ctx else 0),
        'avg_active_item_count':(sum(int(x.get('active_item_count',0)) for x in ctx)/len(ctx) if ctx else 0),
        'max_active_item_count':max([int(x.get('active_item_count',0)) for x in ctx] or [0]),
        'avg_cold_item_count':(sum(int(x.get('cold_item_count',0)) for x in ctx)/len(ctx) if ctx else 0),
        'eviction_count':sum(int(x.get('eviction_count',0)) for x in ctx),
        'projection_count':sum(int(x.get('projection_count',0)) for x in ctx),
        'avg_known_context_chars':(sum(int(x.get('known_context_chars',0)) for x in ctx)/len(ctx) if ctx else 0),
        'rehydration_rate':(counts.get('OBSERVATION_REHYDRATED',0)/max(1,counts.get('TOOL_OBSERVATION',0)+counts.get('OBSERVATION_REHYDRATED',0)+counts.get('REDUNDANT_CONTEXT_REQUEST',0))),
        'invalid_context_requests':sum(len(x.get('invalid_requested_ids') or []) for x in ctx),
        'prompt_breakdown_events':len(prompt_breakdowns),
        'prompt_breakdown_chars':{
            key:sum(int(x.get(key,0) or 0) for x in prompt_breakdowns)
            for key in sorted({k for x in prompt_breakdowns for k in x if k.endswith('_chars')})
        },
    }
