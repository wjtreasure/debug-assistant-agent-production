from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json, re, inspect, copy, uuid, time, hashlib
from debug_assistant.models import AgentState, ActionKind, TaskSpec, RuntimeStage, RuntimeFailure, ToolObservation
from debug_assistant.llm.factory import build_llm
from debug_assistant.llm.base import LLMError, LLMDeadlineExceeded, LLMTransportTimeout
from debug_assistant.tools.registry import ToolRegistry
from debug_assistant.repository.index import RepositoryIndex, IndexDeadlineExceeded
from debug_assistant.repository.safe_fs import SafeRepositoryFS
from debug_assistant.repository.paths import ResolutionMode, RepositoryPathError
from debug_assistant.repository.chunks import build_chunk_manifest
from debug_assistant.repository.embeddings import EmbeddingCache, SiliconFlowEmbeddingProvider
from debug_assistant.repository.semantic_index import SemanticIndex
from debug_assistant.repository.search_engine import RepositorySearchEngine
from debug_assistant import __version__
from debug_assistant.security.redaction import redact_sensitive
from debug_assistant.memory.evidence_memory import EvidenceMemory
from debug_assistant.memory.observation_store import ObservationStore
from debug_assistant.memory.coverage import ReadCoverageIndex
from debug_assistant.memory.hypothesis import HypothesisManager, normalize_target, normalize_location
from debug_assistant.context.manager import ContextManager
from debug_assistant.agent.planner import Planner, PlannerFacade, NativePlannerResult, PlannerContractError, NativePlannerContractError
from debug_assistant.agent.reflection import Reflector, TypedReflection
from debug_assistant.contracts import ReflectionDecision
from debug_assistant.agent.reporter import Reporter, ReporterContractViolation, build_finalization_context
from debug_assistant.reporting.fallback import FallbackReportBuilder
from .guards import RouterGuard, LoopGuard
from .trace import TraceRecorder
from .tool_executor import execute_with_retry
from .retry import RetryPolicy
from .budget_gate import RemainingBudgetGate
from .parallel import execute_parallel_group
from .evidence_bundle import build_evidence_bundle
from .semantic_invariants import validate_semantic_candidate
from .semantic_reducer import SemanticReducer
from debug_assistant.reporting.rules import apply_reporting_rules
from .convergence import ConvergenceController, ConvergenceMode, ProgressKind
from .budget import BudgetController
from .information_need import InformationNeedTracker
from .obligations import EvidenceObligationTracker, ObligationStatus
from .action_policy import ActionPolicy
from .tool_orchestrator import ToolOrchestrator, RequestedToolCall, ToolPlanningError
from .provider_health import ProviderCircuitBreaker, ProviderHealthSample
from .deadline import RunDeadline
from debug_assistant.context.indexes import extract_numbered_range

RUNTIME_VERSION=__version__
RELIABILITY_REVISION='r1'


def _classify_error(exc: Exception, stage: RuntimeStage) -> tuple[str, bool | None]:
    name=type(exc).__name__.lower(); msg=str(exc).lower()
    if isinstance(exc, PlannerContractError):
        return 'planner_contract_invalid', False
    if isinstance(exc, LLMDeadlineExceeded):
        return 'llm_deadline_exceeded', True
    if isinstance(exc, LLMTransportTimeout):
        return 'llm_transport_timeout', True
    if 'timeout' in name or 'timeout' in msg:
        return ('llm_timeout' if stage in {RuntimeStage.PLANNER,RuntimeStage.REFLECTION,RuntimeStage.REPORTER} else 'timeout'), True
    if isinstance(exc, LLMError):
        if 'json' in msg or 'parse' in msg: return 'llm_parse_error', True
        if 'http' in msg or 'request' in msg or 'network' in msg: return 'llm_http_error', True
        return 'llm_error', True
    if isinstance(exc, ReporterContractViolation):
        return 'reporter_contract_violation', False
    if 'validation' in msg or 'schema' in msg:
        return ('report_validation' if stage == RuntimeStage.REPORTER else 'schema_validation'), False if stage == RuntimeStage.REPORTER else True
    if stage == RuntimeStage.MEMORY_INGESTION: return 'memory_error', False
    if stage == RuntimeStage.SERIALIZATION: return 'serialization_error', False
    return 'unexpected_error', None


def _failure_from_exception(state: AgentState, stage: RuntimeStage, exc: Exception) -> RuntimeFailure:
    error_type,retryable=_classify_error(exc,stage)
    last=asdict(state.actions[-1]) if state.actions else None
    return RuntimeFailure(stage=stage.value,error_type=error_type,exception_type=type(exc).__name__,message=str(exc),retryable=retryable,
                          step=state.step,tool_calls=state.tool_calls,evidence_count=len(state.evidence),last_action=last)


def _record_new_llm_usage(trace: TraceRecorder, llm, start_index: int, stage: RuntimeStage) -> int:
    calls=getattr(llm,'calls',[]) if llm is not None else []
    for idx in range(start_index,len(calls)):
        call=dict(calls[idx]); call['stage']=stage.value; call['call_index']=idx+1
        trace.record('LLM_CALL_USAGE',call)
    return len(calls)


def _record_new_llm_events(trace: TraceRecorder, llm, start_index: int, stage: RuntimeStage) -> int:
    events=getattr(llm,'events',[]) if llm is not None else []
    for idx in range(start_index,len(events)):
        row=dict(events[idx])
        event_type=str(row.get('type') or 'LLM_PROVIDER_EVENT')
        payload=dict(row.get('payload') or {})
        payload.setdefault('stage',stage.value)
        trace.record(event_type,payload)
    return len(events)


def _usage_snapshot(llm) -> dict:
    calls=[dict(x) for x in getattr(llm,'calls',[])] if llm is not None else []
    events=[dict(x) for x in getattr(llm,'events',[])] if llm is not None else []
    event_types=[str(x.get('type') or '') for x in events]
    return {'calls':calls,'totals':{
        'calls':len(calls),
        'logical_llm_calls':event_types.count('LLM_LOGICAL_CALL_STARTED') or len(calls),
        'provider_attempts':event_types.count('LLM_ATTEMPT_STARTED') or sum(int(x.get('provider_attempts',1) or 1) for x in calls),
        'failed_provider_attempts':event_types.count('LLM_ATTEMPT_FAILED'),
        'retry_count':event_types.count('LLM_ATTEMPT_RETRYING'),
        'deadline_exceeded_calls':event_types.count('LLM_DEADLINE_EXCEEDED'),
        'prompt_tokens':sum(int(x.get('prompt_tokens',x.get('input_tokens',0)) or 0) for x in calls),
        'completion_tokens':sum(int(x.get('completion_tokens',x.get('output_tokens',0)) or 0) for x in calls),
        'tokens':sum(int(x.get('total_tokens',0) or 0) for x in calls),
        'prompt_chars':sum(int(x.get('prompt_chars',0) or 0) for x in calls),
        'completion_chars':sum(int(x.get('completion_chars',0) or 0) for x in calls),
    }}


def _truncate_observation_content(obs, limit: int) -> None:
    if len(obs.content) <= limit: return
    original=obs.content
    if obs.tool == 'read_file':
        kept=[]; used=0
        for line in original.splitlines():
            addition=len(line)+(1 if kept else 0)
            if used+addition > limit: break
            kept.append(line); used+=addition
        obs.content='\n'.join(kept) if kept else original[:limit]
        old_end=obs.metadata.get('end_line')
        if old_end is not None: obs.metadata['tool_returned_end_line']=old_end
        nums=[]
        for line in obs.content.splitlines():
            m=re.match(r'^\s*(\d+)\s*\|',line)
            if m: nums.append(int(m.group(1)))
        if nums: obs.metadata['end_line']=nums[-1]
    else:
        obs.content=original[:limit]
    obs.content += "\n...[truncated by harness]"
    obs.metadata['truncated']=True; obs.metadata['truncated_by']='harness_char_limit'; obs.metadata['char_limit']=limit


def _call_with_optional_timeout(method, *args, logical_timeout_seconds: float | None = None, on_attempt_started=None):
    """Preserve compatibility with legacy stage doubles while passing optional V1.4.x metadata."""
    try:
        sig=inspect.signature(method)
        has_varkw=any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        accepts_timeout=('logical_timeout_seconds' in sig.parameters or has_varkw)
        accepts_callback=('on_attempt_started' in sig.parameters or has_varkw)
    except (TypeError,ValueError):
        accepts_timeout=accepts_callback=False
    kwargs={}
    if accepts_timeout: kwargs['logical_timeout_seconds']=logical_timeout_seconds
    if accepts_callback: kwargs['on_attempt_started']=on_attempt_started
    return method(*args,**kwargs)


def _validate_evidence_ids(ids: list[str], evidence) -> list[str]:
    available={e.evidence_id for e in evidence}
    return [x for x in ids if x in available]


class AgentHarness:
    def __init__(self,config):
        self.config=config
        self.config.harness.features.validate()

    def run(self,task:TaskSpec,output_dir:str|None=None):
        cfg=self.config; h=cfg.harness; flags=h.features
        deadline=RunDeadline(h.max_wall_time_seconds)
        trace=TraceRecorder(h.trace_dir,task.task_id); state=AgentState(task=task)
        trace.record('RUN_START',{'task':asdict(task),'policy':'read_only','runtime_version':RUNTIME_VERSION,'reliability_revision':RELIABILITY_REVISION,'trace_schema':'2.6'})
        current_stage=RuntimeStage.SETUP
        llm=None; index=None; index_path=None; semantic_index=None; embedding_cache=None; search_engine=None
        memory=EvidenceMemory(); observations=ObservationStore(); coverage=ReadCoverageIndex(); hypothesis=None
        ctxmgr=ContextManager(h.context,enable_catalog=flags.context_catalog,enable_model_selection=flags.model_context_selection,enable_budget_packing=flags.context_budget_packing,enable_lifecycle=flags.context_lifecycle_v2,enable_projection=flags.context_projection_v2)
        fallback=FallbackReportBuilder()
        tools=planner=reflector=typed_reducer=reporter=router=loop=None
        pending_route_recovery=False; force_reflect=False; consecutive_planner_errors=0; consecutive_planner_contract_failures=0; consecutive_reflection_failures=0; usage_cursor=0; provider_event_cursor=0
        last_planner_contract_feedback=None
        rejected_since_last_evidence=False
        requested_context_ids=[]; last_information_need=""; same_information_need=0
        semantic_revision=0
        active_presentation_plans=[]
        submitted_reflection_ids=set()
        pending_reconciliation=False
        critic_compact_retry_used=False
        compact_reflection_retry_pending=False
        reflection_failures_by_signature={}
        reflection_signature=None
        reflection_context_metrics={}
        semantic_no_progress_streak=0
        last_semantic_fingerprint=None
        last_context_result={}
        need_tracker=InformationNeedTracker(max_no_gain_attempts=2) if flags.information_need_tracking else None
        obligations=None
        budget=BudgetController(max_steps=h.max_steps,max_tool_calls=h.max_tool_calls,max_llm_calls=h.max_llm_calls,max_total_tokens=h.max_total_tokens,max_wall_time_seconds=h.max_wall_time_seconds,finalization_reserve_seconds=h.finalization_reserve_seconds,started_at=state.started_at)
        current_need=None
        controller=ConvergenceController(no_progress_limit=2) if flags.convergence_control else None
        last_redundant_key=None
        action_policy=ActionPolicy()
        retry_policy=RetryPolicy(max_attempts=h.tool_retry_attempts,base_delay_seconds=h.retry_base_delay_seconds,max_delay_seconds=h.retry_max_delay_seconds)
        budget_gate=RemainingBudgetGate(state.started_at+h.max_wall_time_seconds,finalization_reserve_seconds=h.finalization_reserve_seconds,cleanup_margin_seconds=h.llm_cleanup_margin_seconds)
        provider_health=ProviderCircuitBreaker(
            window=h.provider_health_window,failure_threshold=h.provider_failure_threshold,
            consecutive_failures=h.provider_consecutive_failures,recovery_successes=h.provider_recovery_successes,
            degraded_timeout_seconds=h.provider_degraded_timeout_seconds,
        )

        def _flush_llm(stage: RuntimeStage, before_usage: int):
            nonlocal usage_cursor, provider_event_cursor, active_presentation_plans
            events=getattr(llm,'events',[]) if llm is not None else []
            for idx in range(provider_event_cursor,len(events)):
                row=dict(events[idx]); et=str(row.get('type') or 'LLM_PROVIDER_EVENT'); payload=dict(row.get('payload') or {})
                payload.setdefault('stage',stage.value); trace.record(et,payload)
                if et=='LLM_ATTEMPT_STARTED' and stage is RuntimeStage.REFLECTION:
                    for plan in active_presentation_plans:
                        if not plan.get('prepared') or plan.get('presented'): continue
                        plan['presented']=True; plan['request_submitted']=True
                        plan['logical_call_id']=payload.get('logical_call_id'); plan['provider_attempt_index']=payload.get('attempt_index')
                        submitted_reflection_ids.add(str(plan.get('reflection_id')))
                        if obligations is not None:
                            obligations.mark_presented(plan['obligation_id'],reflection_id=plan['reflection_id'],projection_id=plan['projection_id'],evidence_fingerprint=plan['evidence_fingerprint'])
                        trace.record('OBLIGATION_EVIDENCE_PRESENTED',{k:v for k,v in plan.items() if k not in {'prepared','presented','content'}})
                if et=='LLM_LOGICAL_CALL_FINISHED':
                    sample=ProviderHealthSample(
                        logical_call_id=str(payload.get('logical_call_id') or ''),stage=stage.value,
                        provider_success=bool(payload.get('provider_success',payload.get('success',False))),
                        provider_failure=bool(payload.get('provider_failure',False)),error_type=str(payload.get('error_type') or ''),
                        elapsed_seconds=float(payload.get('logical_elapsed_ms',0) or 0)/1000.0,
                        allowed_seconds=float(payload.get('logical_timeout_seconds',0) or 0),
                    )
                    transition=provider_health.observe(sample)
                    if transition=='degraded': trace.record('PROVIDER_DEGRADED',provider_health.summary())
                    elif transition=='recovered': trace.record('PROVIDER_RECOVERED',provider_health.summary())
            provider_event_cursor=len(events)
            usage_cursor=_record_new_llm_usage(trace,llm,before_usage,stage)

        def build_ctx(stage: RuntimeStage):
            nonlocal requested_context_ids
            result=ctxmgr.build(state,memory,observations,max_context_chars=h.max_context_chars,max_steps=h.max_steps,max_tool_calls=h.max_tool_calls,requested_ids=requested_context_ids)
            trace.record('CONTEXT_BUILT',{
                'stage':stage.value,'budget_chars':result.budget_chars,'used_chars':result.used_chars,'catalog_size':result.catalog_size,
                'working_set_size':result.working_set_size,'selected':result.selected,'dropped':result.dropped,
                'invalid_requested_ids':result.invalid_requested_ids,'breakdown':result.breakdown,
                'known_context_chars':result.known_context_chars,'active_item_count':result.active_item_count,
                'cold_item_count':result.cold_item_count,'eviction_count':result.eviction_count,
                'projection_count':result.projection_count,'display_coverage':result.display_coverage,
            })
            last_context_result[stage.value]=result
            if result.invalid_requested_ids: trace.record('CONTEXT_REQUEST_INVALID',{'ids':result.invalid_requested_ids,'stage':stage.value})
            requested_context_ids=[]
            extra=[]
            if need_tracker is not None:
                advisory=need_tracker.advisory(current_need)
                if advisory: extra.append('RETRIEVAL_ADVISORY:\n'+advisory)
            if obligations is not None and obligations.open_critical():
                extra.append('OPEN_CRITICAL_EVIDENCE_OBLIGATIONS:\n'+json.dumps(obligations.summary(),ensure_ascii=False,default=str))
            if flags.cost_aware_convergence:
                snap=budget.snapshot(steps=state.step,tool_calls=state.tool_calls,llm_calls=_current_llm_calls(),tokens=_current_total_tokens())
                extra.append(f'BUDGET_STATE: phase={snap.phase}; remaining_ratio={snap.remaining_ratio:.3f}; tokens={snap.tokens_used}/{h.max_total_tokens}; llm_calls={snap.llm_calls_used}/{h.max_llm_calls}; wall_time={snap.wall_time_seconds:.1f}/{h.max_wall_time_seconds}s')
            if last_planner_contract_feedback:
                extra.append('PREVIOUS_PLANNER_CONTRACT_ERROR:\n'+json.dumps(last_planner_contract_feedback,ensure_ascii=False))
            return result.text + ('\n\n'+'\n\n'.join(extra) if extra else '')

        def _current_llm_calls() -> int:
            try:
                return int((_usage_snapshot(llm).get('totals') or {}).get('logical_llm_calls',0) or 0)
            except Exception:
                return len(getattr(llm,'calls',[]) if llm is not None else [])

        def _current_total_tokens() -> int:
            try:
                return int((_usage_snapshot(llm).get('totals') or {}).get('tokens',0) or 0)
            except Exception as exc:
                # Usage is observability/budget telemetry. Failure to read it must not
                # change the primary runtime failure stage.
                try: trace.record('BUDGET_TELEMETRY_ERROR',{'error_type':type(exc).__name__,'message':str(exc)})
                except Exception: pass
                return 0

        def _record_prompt_breakdown(stage: RuntimeStage, actor) -> None:
            payload=dict(getattr(actor,'last_prompt_breakdown',{}) or {})
            if payload:
                payload.update({'stage':stage.value,'step':state.step})
                trace.record('PROMPT_BREAKDOWN',payload)
                if stage is RuntimeStage.PLANNER and all(key in payload for key in ('valid_skill_count','valid_tool_count','parallel_tool_count','question_type_count')):
                    trace.record('PLANNER_PROMPT_CONTRACT',{
                        'skill_count':payload['valid_skill_count'],
                        'tool_count':payload['valid_tool_count'],
                        'parallel_tool_count':payload['parallel_tool_count'],
                        'question_type_count':payload['question_type_count'],
                    })

        def _budget_snapshot(boundary: str | None=None):
            snap=budget.snapshot(steps=state.step,tool_calls=state.tool_calls,llm_calls=_current_llm_calls(),tokens=_current_total_tokens())
            if boundary:
                trace.record('BUDGET_SNAPSHOT',{**asdict(snap),'boundary':boundary})
            return snap

        def _stage_guard_due(stage: RuntimeStage, guard_seconds: int) -> bool:
            guard=max(0,int(guard_seconds or 0))
            # With no source evidence there is nothing useful to report yet, so stage
            # admission must not prevent the first evidence-gathering Planner turn.
            if guard <= 0 or not state.evidence:
                return False
            snap=_budget_snapshot('pre_'+stage.value+'_admission')
            due=(snap.phase=='finalize' or snap.seconds_until_forced_finalization <= guard)
            if due:
                trace.record('LLM_STAGE_SKIPPED',{
                    'stage':stage.value,'reason':'insufficient_pre_finalize_budget',
                    'guard_seconds':guard,'seconds_until_forced_finalization':snap.seconds_until_forced_finalization,
                    'phase':snap.phase,'step':state.step,
                })
            return due

        def _llm_stage_timeout(stage: RuntimeStage) -> float:
            snap=_budget_snapshot('pre_'+stage.value+'_deadline')
            # Bootstrap exception: before the first source Evidence exists, refusing the
            # first Planner because the cleanup/finalization reserve dominates a tiny
            # test/run budget guarantees a useless empty report.  Keep the absolute run
            # deadline authoritative, but do not charge cleanup reserve to that first
            # evidence-gathering Planner turn.
            bootstrap_planner=(stage is RuntimeStage.PLANNER and not state.evidence)
            cleanup=0.0 if bootstrap_planner else max(0.0,float(h.llm_cleanup_margin_seconds or 0))
            hard_remaining=max(0.0,deadline.remaining()-cleanup)
            caps={
                RuntimeStage.PLANNER:float(h.planner_llm_timeout_seconds),
                RuntimeStage.REFLECTION:float(h.reflection_llm_timeout_seconds),
                RuntimeStage.REPORTER:float(h.reporter_llm_timeout_seconds),
            }
            configured_cap=max(0.0,caps.get(stage,float(cfg.model.timeout)))
            cap=max(0.0,provider_health.cap(configured_cap,stage.value))
            if stage is RuntimeStage.REFLECTION and active_presentation_plans:
                cap=min(cap,float(h.focused_reflection_timeout_seconds))
            if stage in {RuntimeStage.PLANNER,RuntimeStage.REFLECTION}:
                available=hard_remaining if bootstrap_planner else min(hard_remaining,max(0.0,snap.seconds_until_forced_finalization))
            else:
                available=hard_remaining
            allowed=max(0.0,min(cap,available))
            trace.record('LLM_STAGE_BUDGET',{
                'stage':stage.value,'configured_cap_seconds':configured_cap,'effective_cap_seconds':cap,'provider_degraded':provider_health.degraded,'allowed_seconds':allowed,
                'hard_run_remaining_seconds':max(0.0,float(h.max_wall_time_seconds)-snap.wall_time_seconds),
                'seconds_until_forced_finalization':snap.seconds_until_forced_finalization,
                'cleanup_margin_seconds':cleanup,'phase':snap.phase,'step':state.step,
            })
            return allowed

        def _admit_reporter_repair() -> bool:
            """Admission for the single Reporter repair, under the same stage budget."""
            if _current_llm_calls() >= h.max_llm_calls:
                return False
            return deadline.remaining() > max(0.0, float(h.llm_cleanup_margin_seconds or 0)) and _llm_stage_timeout(RuntimeStage.REPORTER) > 0

        def _flush_reporter_events() -> None:
            for event_type, payload in list(getattr(reporter, 'last_events', []) or []):
                trace.record(event_type, payload)
            if reporter is not None:
                reporter.last_events=[]

        def _find_evidence(evidence_id: str):
            return next((e for e in state.evidence if e.evidence_id==evidence_id),None)

        def _trace_obligation_events():
            if obligations is None:return
            for et,payload in obligations.pop_events(): trace.record(et,payload)

        def _check_obligation_scope(tool: str, arguments: dict | None, intent: str = "") -> bool:
            """Keep precise active requirements attached to the next tool action."""
            if obligations is None:
                return True
            decision = obligations.action_scope(tool, arguments, intent)
            if decision.get('allowed', True):
                return True
            trace.record('OBLIGATION_ACTION_REJECTED', {
                'tool': tool,
                'relation': decision.get('relation'),
                'obligation_id': decision.get('obligation_id'),
                'reason': decision.get('reason'),
                'required': decision.get('required'),
            })
            state.invalid_routes += 1
            state.errors.append('obligation scope drift rejected')
            return False

        def _reconcile_pending(*, boundary: str):
            """Phase B: idempotently map immutable acquired Evidence onto Obligation readiness."""
            nonlocal pending_reconciliation, force_reflect, semantic_no_progress_streak
            if obligations is None:
                pending_reconciliation=False; return True
            snapshot=copy.deepcopy(obligations.items)
            before_ready={oid:o.evidence_ready for oid,o in obligations.items.items()}
            try:
                closed=[]
                for ev in state.evidence:
                    raw=observations.get(ev.raw_observation_id) if ev.raw_observation_id else None
                    closed.extend(obligations.note_evidence(ev,raw.content if raw else ''))
                _trace_obligation_events()
                ready=[oid for oid,o in obligations.items.items() if o.evidence_ready and not before_ready.get(oid,False)]
                if ready:
                    trace.record('EVIDENCE_OBLIGATIONS_READY',{'obligation_ids':sorted(set(ready)),'source':'pending_reconciliation','boundary':boundary})
                    force_reflect=True; semantic_no_progress_streak=0
                if closed:
                    trace.record('EVIDENCE_OBLIGATIONS_SATISFIED',{'obligation_ids':sorted(set(closed)),'source':'deterministic_nonsemantic_reconciliation','boundary':boundary})
                pending_reconciliation=False
                return True
            except Exception as exc:
                obligations.items=snapshot; pending_reconciliation=True
                trace.record('RECONCILIATION_FAILED',{'boundary':boundary,'error_type':type(exc).__name__,'message':str(exc)})
                return False

        def _prepare_obligation_bundle(reflection_id: str):
            nonlocal active_presentation_plans
            active_presentation_plans=[]
            if obligations is None:return None
            candidates=obligations.presentation_candidates()
            if not candidates:return None
            snap=_budget_snapshot('pre_obligation_presentation')
            if snap.seconds_until_forced_finalization <= max(0,int(h.obligation_review_min_seconds or 0)):
                trace.record('OBLIGATION_PRESENTATION_SKIPPED',{'reason':'insufficient_review_budget','candidate_count':len(candidates),'seconds_until_forced_finalization':snap.seconds_until_forced_finalization,'required_seconds':h.obligation_review_min_seconds,'step':state.step})
                return {'finalize':True}
            bundle=build_evidence_bundle(obligations,state.evidence,observations,bundle_id=f'B-{reflection_id}',max_items=min(h.focused_reflection_max_obligations,h.max_auto_obligation_presentations_per_reflection),max_chars=h.evidence_bundle_max_chars)
            if bundle is None:return None
            for idx,item in enumerate(bundle.items):
                plan={'reflection_id':reflection_id,'obligation_id':item.obligation_id,'evidence_id':item.evidence_id,'observation_id':item.observation_id,'file':item.file,'start_line':item.start_line,'end_line':item.end_line,'chars':len(item.content),'evidence_fingerprint':item.evidence_fingerprint,'projection_id':f'{bundle.bundle_id}:P{idx}','prepared':True,'presented':False,'content':item.content}
                active_presentation_plans.append(plan)
                trace.record('OBLIGATION_PRESENTATION_PREPARED',{k:v for k,v in plan.items() if k not in {'prepared','presented','content'}})
                trace.record('OBLIGATION_PRESENTATION_CONTEXT_CONFIRMED',{'reflection_id':reflection_id,'obligation_id':item.obligation_id,'projection_id':plan['projection_id'],'file':item.file,'start_line':item.start_line,'end_line':item.end_line,'source':'evidence_bundle'})
            trace.record('EVIDENCE_BUNDLE_BUILT',{'bundle_id':bundle.bundle_id,'information_need_root_id':bundle.root_id,'obligation_ids':[x.obligation_id for x in bundle.items],'chars':bundle.chars,'source_projection_count':len(bundle.items)})
            return bundle

        def _reflection_signature(bundle):
            if bundle is None:
                payload={'root_id':getattr(current_need,'need_id',None),
                         'obligation_ids':[],
                         'evidence_ids':sorted(e.evidence_id for e in state.evidence)}
            else:
                payload={'root_id':bundle.root_id,
                         'items':[(x.obligation_id,x.evidence_fingerprint,x.file,x.start_line,x.end_line) for x in bundle.items]}
            return hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:20]

        def _focused_reflection_context(bundle, *, compact=False, target_chars=None):
            if compact:
                item=bundle.items[0]
                projection=item.content or f'SOURCE {item.file}:{item.start_line}-{item.end_line} (shared projection already acquired)'
                compact_bundle=(f'=== OBLIGATION {item.obligation_id} ===\n'
                                f'SOURCE {item.file}:{item.start_line}-{item.end_line}\n{projection[:3500]}')
                open_rows=json.dumps([x for x in (obligations.summary() if obligations is not None else [])
                                      if x.get('obligation_id')==item.obligation_id],ensure_ascii=False,default=str)[:1800]
                issue=task.issue[:1800]
                hyp=json.dumps(state.current_hypothesis or {},ensure_ascii=False,default=str)[:2200]
                text=("COMPACT_FOCUSED_REFLECTION_RETRY\nISSUE_SUMMARY:\n"+issue+
                      "\n\nCURRENT_HYPOTHESIS:\n"+hyp+"\n\nTARGET_OBLIGATION:\n"+open_rows+
                      "\n\nEXACT_SOURCE_PROJECTION:\n"+compact_bundle)
            else:
                issue=task.issue[:5000]
                hyp=json.dumps(state.current_hypothesis or {},ensure_ascii=False,default=str)[:7000]
                open_rows=json.dumps(obligations.summary() if obligations is not None else [],ensure_ascii=False,default=str)[:5000]
                text=("FOCUSED_REFLECTION\nISSUE:\n"+issue+"\n\nCURRENT_HYPOTHESIS:\n"+hyp+
                      "\n\nOPEN_OBLIGATIONS:\n"+open_rows+"\n\nEVIDENCE_BUNDLE:\n"+bundle.text)
            cap=max(1, int(h.focused_reflection_max_chars))
            if target_chars is not None:
                cap=min(cap,max(1,int(target_chars)))
            text=text[:cap]
            trace.record('FOCUSED_REFLECTION_CONTEXT_BUILT',{
                'reflection_id':active_presentation_plans[0]['reflection_id'] if active_presentation_plans else None,
                'obligation_ids':[x['obligation_id'] for x in active_presentation_plans[:1] if compact] or [x['obligation_id'] for x in active_presentation_plans],
                'chars':len(text),'information_need_root_id':bundle.root_id,'compact':compact,
                'obligation_count':1 if compact else len(active_presentation_plans),
            })
            return text

        def update_hypothesis(review: dict, *, reflection_id: str):
            """Validate reviews independently; commit one consistent semantic candidate atomically."""
            nonlocal rejected_since_last_evidence, semantic_revision, pending_reconciliation, semantic_no_progress_streak, last_semantic_fingerprint
            valid_support=_validate_evidence_ids(review.get('supporting_evidence_ids') or [],state.evidence)
            valid_contra=_validate_evidence_ids(review.get('contradicting_evidence_ids') or [],state.evidence)
            if len(valid_support) != len(review.get('supporting_evidence_ids') or []):
                trace.record('REFLECTION_EVIDENCE_ID_INVALID',{'kind':'support','requested':review.get('supporting_evidence_ids'),'accepted':valid_support})
            if len(valid_contra) != len(review.get('contradicting_evidence_ids') or []):
                trace.record('REFLECTION_EVIDENCE_ID_INVALID',{'kind':'contradiction','requested':review.get('contradicting_evidence_ids'),'accepted':valid_contra})
            review=dict(review); review['supporting_evidence_ids']=valid_support; review['contradicting_evidence_ids']=valid_contra
            if obligations is None:
                hs=hypothesis.update(review,state.step); state.current_hypothesis=asdict(hs)
                trace.record('HYPOTHESIS_UPDATED',state.current_hypothesis); semantic_revision+=1
                trace.record('SEMANTIC_STATE_COMMIT',{'reflection_id':reflection_id,'from_revision':semantic_revision-1,'to_revision':semantic_revision})
            else:
                items_snapshot=copy.deepcopy(obligations.items); hypothesis_snapshot=copy.deepcopy(hypothesis.state); state_hyp_snapshot=copy.deepcopy(state.current_hypothesis); from_revision=semantic_revision
                valid_review_ids=[]; invalid_review_ids=[]
                try:
                    pending_semantic_events=[]
                    invalid_review_obligation_ids=set()
                    for bad in review.pop('_invalid_obligation_reviews',[]) or []:
                        oid=str(bad.get('obligation_id') or '')
                        invalid_review_ids.append(oid or f"index:{bad.get('index')}")
                        if oid:
                            invalid_review_obligation_ids.add(oid)
                        pending_semantic_events.append(('INVALID_OBLIGATION_REVIEW',{'reflection_id':reflection_id,'obligation_id':oid or None,'reason':'schema_invalid','details':bad.get('errors',[])}))
                        pending_semantic_events.append(('OBLIGATION_REVIEW_IGNORED',{'reflection_id':reflection_id,'obligation_id':oid or None,'reason':'schema_invalid'}))
                    root_id=getattr(current_need,'need_id',None)
                    def with_root(rows):
                        out=[]
                        for raw in rows or []:
                            item=raw.model_dump() if hasattr(raw,'model_dump') else dict(raw)
                            if root_id and not item.get('information_need_root_id'): item['information_need_root_id']=root_id
                            out.append(item)
                        return out
                    raw_missing=with_root(review.get('required_missing_evidence') or [])
                    raw_optional=with_root(review.get('optional_validation') or [])
                    transitions=obligations.sync(raw_missing,raw_optional)
                    # A schema-invalid review is not an instruction to change the
                    # obligation plane. Restore those rows before any legacy sync or
                    # implicit-resolution path can observe the temporary projection.
                    if invalid_review_obligation_ids:
                        for oid in invalid_review_obligation_ids:
                            if oid in items_snapshot:
                                obligations.items[oid]=copy.deepcopy(items_snapshot[oid])
                        transitions=[x for x in transitions if x.get('obligation_id') not in invalid_review_obligation_ids]
                    for et,payload in obligations.pop_events(): pending_semantic_events.append((et,payload))
                    explicit_ids=set(); review_transitions=[]
                    for item in review.get('obligation_reviews') or []:
                        rv=item.model_dump() if hasattr(item,'model_dump') else dict(item); oid=str(rv.get('obligation_id') or '')
                        if oid in explicit_ids: raise ValueError(f'duplicate obligation review id after schema validation: {oid}')
                        explicit_ids.add(oid)
                        if oid not in obligations.items:
                            invalid_review_ids.append(oid); pending_semantic_events.append(('INVALID_OBLIGATION_REVIEW',{'reflection_id':reflection_id,'obligation_id':oid,'reason':'unknown_obligation_id'})); continue
                        ok,trs=obligations.apply_explicit_review(rv,reflection_id=reflection_id)
                        for et,payload in obligations.pop_events(): pending_semantic_events.append((et,payload))
                        if not ok:
                            invalid_review_ids.append(oid); pending_semantic_events.append(('INVALID_OBLIGATION_REVIEW',{'reflection_id':reflection_id,'obligation_id':oid,'reason':'illegal_or_unpresented_transition','decision':rv.get('decision')})); continue
                        valid_review_ids.append(oid); pending_semantic_events.append(('OBLIGATION_REVIEW_APPLIED',{'reflection_id':reflection_id,'obligation_id':oid,'decision':rv.get('decision'),'source':'explicit'})); review_transitions.extend(trs)
                    if transitions or review_transitions: pending_semantic_events.append(('EVIDENCE_OBLIGATION_TRANSITIONS',{'items':transitions+review_transitions}))
                    review['required_missing_evidence']=obligations.active_required_items(); review['optional_validation']=obligations.optional_items()
                    if review['required_missing_evidence'] and review.get('evidence_sufficient'):
                        review['evidence_sufficient']=False; pending_semantic_events.append(('REFLECTION_SUFFICIENCY_CORRECTED',{'reflection_id':reflection_id,'reason':'active_required_obligation_remains'}))
                    hs=hypothesis.update(review,state.step)
                    validate_semantic_candidate(obligations,hs,submitted_reflection_ids=submitted_reflection_ids)
                    state.current_hypothesis=asdict(hs); semantic_revision=from_revision+1
                    semantic_core=lambda value: {k:v for k,v in value.items() if k not in {'updated_step','stable_diagnosis_transitions','model_claimed_changed'}}
                    invalid_review_only=(bool(invalid_review_obligation_ids) and not valid_review_ids and not transitions and
                                         semantic_core(state.current_hypothesis)==semantic_core(state_hyp_snapshot))
                    if invalid_review_only:
                        hypothesis.state=hypothesis_snapshot
                        state.current_hypothesis=state_hyp_snapshot
                        semantic_revision=from_revision
                    semantic_fp=json.dumps({'diagnosis':state.current_hypothesis.get('diagnosis_fingerprint'),'evidence':state.current_hypothesis.get('evidence_fingerprint'),'required':state.current_hypothesis.get('required_gap_fingerprint'),'obligations':[(o.obligation_id,o.status.value,o.active_required,o.last_review_decision) for o in sorted(obligations.items.values(),key=lambda x:x.obligation_id)]},sort_keys=True,default=str)
                    created=any(x.get('reason') in {'required_created','optional_created'} for x in transitions)
                    if created or semantic_fp!=last_semantic_fingerprint:
                        semantic_no_progress_streak=0
                    else:
                        semantic_no_progress_streak+=1; trace.record('SEMANTIC_NO_PROGRESS',{'streak':semantic_no_progress_streak,'limit':h.semantic_no_progress_limit,'reflection_id':reflection_id})
                    last_semantic_fingerprint=semantic_fp
                    for event_type,payload in pending_semantic_events: trace.record(event_type,payload)
                    trace.record('EVIDENCE_OBLIGATIONS_UPDATED',{'items':obligations.summary()}); trace.record('HYPOTHESIS_UPDATED',state.current_hypothesis)
                    trace.record('SEMANTIC_STATE_COMMIT',{'reflection_id':reflection_id,'from_revision':from_revision,'to_revision':semantic_revision,'valid_review_ids':valid_review_ids,'invalid_review_ids':invalid_review_ids})
                    pending_reconciliation=bool(transitions)
                except Exception as exc:
                    obligations.items=items_snapshot; hypothesis.state=hypothesis_snapshot; state.current_hypothesis=state_hyp_snapshot; semantic_revision=from_revision; pending_reconciliation=True
                    payload={'reflection_id':reflection_id,'semantic_revision':from_revision,'valid_review_ids':valid_review_ids,'invalid_review_ids':invalid_review_ids,'discarded_valid_review_ids':list(valid_review_ids),'reason':'cross_state_invariant_failed' if type(exc).__name__=='SemanticInvariantError' else 'candidate_inconsistent','error_type':type(exc).__name__,'message':str(exc)}
                    trace.record('SEMANTIC_TRANSACTION_ROLLBACK',payload); trace.record('SEMANTIC_STATE_ROLLBACK',{'reflection_id':reflection_id,'revision':from_revision,'error_type':type(exc).__name__,'message':str(exc)})
                    raise

            hs=hypothesis.state
            if controller is not None:
                old_mode=controller.state.mode
                assessment=controller.assess_reflection(
                    state.current_hypothesis,
                    usage_totals=(_usage_snapshot(llm).get('totals') or {}),
                    allow_budget_recovery=not rejected_since_last_evidence,
                )
                state.no_progress_count += int(assessment.kind is ProgressKind.NO_PROGRESS)
                state.convergence_mode=controller.state.mode.value
                state.forced_finalization=controller.state.forced_finalization
                state.budget_critical_entered=controller.state.budget_critical_entered
                state.first_supported_hypothesis_step=controller.state.first_supported_hypothesis_step
                state.first_stable_diagnosis_step=controller.state.first_stable_diagnosis_step
                state.prompt_tokens_at_first_stable_diagnosis=controller.state.prompt_tokens_at_first_stable_diagnosis
                state.completion_tokens_at_first_stable_diagnosis=controller.state.completion_tokens_at_first_stable_diagnosis
                state.tokens_at_first_stable_diagnosis=controller.state.tokens_at_first_stable_diagnosis
                trace.record('PROGRESS' if assessment.kind is ProgressKind.PROGRESS else 'NO_PROGRESS',{
                    'step':state.step,'reasons':assessment.reasons,'no_progress_streak':controller.state.no_progress_streak,
                    'diagnosis_changed':assessment.diagnosis_changed,'required_gap_changed':assessment.required_gap_changed,
                    'contradiction_changed':assessment.contradiction_changed,'support_changed':assessment.support_changed,
                })
                if old_mode != controller.state.mode:
                    trace.record('CONVERGENCE_MODE_CHANGED',{'from':old_mode.value,'to':controller.state.mode.value,'step':state.step})
                if controller.state.mode is ConvergenceMode.BUDGET_CRITICAL:
                    trace.record('BUDGET_CRITICAL',{'step':state.step,'required_missing_evidence':hs.required_missing_evidence,'no_progress_streak':controller.state.no_progress_streak})
                if controller.state.mode is ConvergenceMode.FORCE_FINALIZATION:
                    trace.record('FORCE_FINALIZATION',{'step':state.step,'hypothesis_status':hs.status,'supporting_evidence_ids':hs.supporting_evidence_ids})
                if controller.critical_failed_after_reflection(assessment):
                    state.status='budget_exhausted'
                    state.errors.append('budget critical final required-information attempt produced no diagnostic progress')
                    trace.record('BUDGET_EXHAUSTED',{'step':state.step,'reason':'budget_critical_no_progress'})
            if flags.termination_advisory and hs.status in ('supported','confirmed') and not hs.contradicting_evidence_ids:
                if controller is not None and controller.state.mode is ConvergenceMode.CONVERGENCE_REQUIRED:
                    state.termination_advisory=("CONVERGENCE_REQUIRED: the causal diagnosis is stable, no required evidence gap remains, and no direct contradiction is recorded. "
                                                "Prefer finish. Continue only for a specific causal uncertainty; optional validation alone must not block finalization.")
                elif controller is not None and controller.state.mode is ConvergenceMode.BUDGET_CRITICAL:
                    state.termination_advisory=("BUDGET_CRITICAL: investigation is closed. Do not start another tool call; finalize from the best source-backed evidence and explicitly report unresolved gaps.")
                elif review.get('evidence_sufficient'):
                    state.termination_advisory=("Current diagnosis is sufficiently supported and no unresolved contradiction is recorded. "
                                                "Prefer finish unless a specific unresolved information_need requires another tool call.")
                else:
                    state.termination_advisory=""
                if state.termination_advisory:
                    trace.record('TERMINATION_ADVISORY',{'step':state.step,'hypothesis_status':hs.status,'stable_diagnosis_transitions':hs.stable_diagnosis_transitions,'mode':state.convergence_mode})
            else: state.termination_advisory=""
            return review

        def build_report():
            nonlocal usage_cursor,current_stage,pending_reconciliation
            if pending_reconciliation: _reconcile_pending(boundary='pre_finalization')
            current_stage=RuntimeStage.REPORTER; before=len(getattr(llm,'calls',[]))
            ready_unreviewed=[]
            if obligations is not None:
                for obj in obligations.open_critical():
                    if not obj.evidence_ready:
                        continue
                    fp=obligations.evidence_fingerprint(obj)
                    if obj.last_reviewed_evidence_fingerprint != fp:
                        steps=[]
                        for eid in obj.evidence_ids:
                            ev=_find_evidence(eid)
                            if ev:
                                for tag in ev.tags:
                                    if tag.startswith('step:'):
                                        try: steps.append(int(tag.split(':',1)[1]))
                                        except ValueError: pass
                        ready_unreviewed.append({
                            'obligation_id':obj.obligation_id,'target':obj.target,'goal_type':obj.goal_type,
                            'file':(obj.canonical_files[0] if len(obj.canonical_files)==1 else None),
                            'symbol':(obj.canonical_symbols[0] if len(obj.canonical_symbols)==1 else None),
                            'line_start':(obj.line_hint[0] if obj.line_hint else (obj.symbol_ranges[0][2] if len(obj.symbol_ranges)==1 else None)),
                            'line_end':(obj.line_hint[1] if obj.line_hint else (obj.symbol_ranges[0][3] if len(obj.symbol_ranges)==1 else None)),
                            'acquired_step':min(steps) if steps else None,
                            'evidence_ids':list(obj.evidence_ids),'last_presented_reflection_id':obj.last_presented_reflection_id,
                        })
                trace.record('FINAL_STATE_RECONCILIATION',{
                    'semantic_revision':semantic_revision,'active_required_count':len(obligations.open_critical()),
                    'evidence_ready_unreviewed':ready_unreviewed,
                    'hypothesis_required_count':len((state.current_hypothesis or {}).get('required_missing_evidence') or []),
                })
            final_context,final_ctx_meta=build_finalization_context(
                task_id=task.task_id,issue=task.issue,state_summary=state.to_summary(),
                hypothesis=(hypothesis.state if hypothesis is not None else state.current_hypothesis),
                evidence=state.evidence,observation_store=observations,
                max_evidence_per_file=h.max_evidence_per_file,
                max_snippet_lines=h.max_snippet_lines,
                max_trace_summary_items=h.max_trace_summary_items,
                max_context_chars=h.max_reporter_context_tokens * 4,
            )
            if ready_unreviewed:
                final_context += '\n\nUNREVIEWED_READY_EVIDENCE: Source evidence was acquired and is READY, but no successful semantic review committed before finalization. Presentation state is recorded separately. ' + json.dumps(ready_unreviewed,ensure_ascii=False)
            trace.record('REPORTER_CONTEXT_BUILT',{**final_ctx_meta,'evidence_ready_unreviewed_count':len(ready_unreviewed)})
            candidates=list(final_ctx_meta.get('fallback_candidate_files') or [])
            try:
                report_timeout=_llm_stage_timeout(RuntimeStage.REPORTER)
                if report_timeout <= 0:
                    trace.record('LLM_STAGE_SKIPPED',{'stage':'reporter','reason':'insufficient_hard_run_budget','step':state.step})
                    raise LLMDeadlineExceeded('Reporter logical deadline unavailable before hard run deadline')
                reservation,gate_error=budget_gate.admit('reporter',include_finalization_reserve=True)
                if gate_error:
                    trace.record('BUDGET_GATE_REJECTED',{**gate_error,'stage':'reporter','step':state.step}); raise LLMDeadlineExceeded('Reporter rejected by RemainingBudgetGate')
                try:
                    report=_call_with_optional_timeout(reporter.build,task.task_id,final_context,state.evidence,logical_timeout_seconds=report_timeout)
                finally:
                    budget_gate.release(reservation)
                _flush_reporter_events()
                _record_prompt_breakdown(current_stage,reporter); state.report_source='llm'
                if not report.likely_files and candidates:
                    report.likely_files=candidates[:3]
                    hs=(state.current_hypothesis or {})
                    report.likely_file_source=('partial_hypothesis' if hs.get('status')=='partial' else 'evidence_fallback')
                    trace.record('REPORT_CANDIDATE_FALLBACK',{'likely_files':report.likely_files,'source':report.likely_file_source})
                elif report.likely_files and report.likely_file_source=='llm' and (state.current_hypothesis or {}).get('status') in {'supported','confirmed'}:
                    report.likely_file_source='hypothesis'
                report.acquired_unreviewed=[{**x,'reason':('reflection_deadline_or_failure' if state.reflection_failure_count else 'finalized_before_semantic_review')} for x in ready_unreviewed]
                report,corrections=apply_reporting_rules(report,evidence=state.evidence,repository_index=index,obligations=obligations,hypothesis=(hypothesis.state if hypothesis is not None else None))
                for corr in corrections:
                    if corr.get('kind')=='claim_status': trace.record('REPORT_STATUS_CORRECTED',corr)
                    else: trace.record('REPORT_GROUNDING_VALIDATED',corr)
                return report
            except Exception as exc:
                _flush_reporter_events()
                failure=_failure_from_exception(state,current_stage,exc)
                violation=getattr(reporter,'last_contract_violation',None)
                if violation:
                    trace.record('REPORTER_CONTRACT_VIOLATION',{**violation,'step':state.step})
                trace.record('REPORTER_FAILED',asdict(failure))
                if flags.fallback_reporter and hypothesis is not None and hypothesis.state.status in ('supported','confirmed') and hypothesis.state.description and hypothesis.state.supporting_evidence_ids and not hypothesis.state.contradicting_evidence_ids and not hypothesis.state.required_missing_evidence:
                    report=fallback.build(task.task_id,hypothesis.state,state.evidence); state.report_source='fallback'; report.acquired_unreviewed=[{**x,'reason':'reporter_failed'} for x in ready_unreviewed]
                    report,_=apply_reporting_rules(report,evidence=state.evidence,repository_index=index,obligations=obligations,hypothesis=hypothesis.state)
                    if not getattr(reporter,'last_fallback_reason',None):
                        trace.record('REPORTER_FALLBACK_TRIGGERED',{'reason':'primary_failure','source':'hypothesis'})
                    trace.record('FALLBACK_REPORT_BUILT',{'evidence_ids':report.evidence_ids,'confidence':report.confidence,'primary_report_failure':asdict(failure)})
                    return report
                if flags.fallback_reporter and state.evidence:
                    report=fallback.build_from_evidence(task.task_id,state.evidence,candidates); state.report_source='fallback'; report.acquired_unreviewed=[{**x,'reason':'reporter_failed'} for x in ready_unreviewed]
                    report,_=apply_reporting_rules(report,evidence=state.evidence,repository_index=index,obligations=obligations,hypothesis=(hypothesis.state if hypothesis is not None else None))
                    if not getattr(reporter,'last_fallback_reason',None):
                        trace.record('REPORTER_FALLBACK_TRIGGERED',{'reason':'primary_failure','source':'evidence_fallback'})
                    trace.record('FALLBACK_REPORT_BUILT',{'evidence_ids':report.evidence_ids,'confidence':report.confidence,'primary_report_failure':asdict(failure),'source':'evidence_fallback'})
                    return report
                raise
            finally:
                _flush_llm(current_stage,before)

        def finalize_now(reason:str):
            state.report=build_report()
            grounded=bool(controller is not None and controller.can_finalize(state.current_hypothesis or {}))
            state.status=('success' if grounded and state.report_source=='llm' else 'partial_success')
            trace.record('FINALIZATION_TRIGGERED',{'reason':reason,'grounded_finalization':grounded,'report_source':state.report_source,'status':state.status})


        def _expand_inspect_sources(parent_obs):
            data=(parent_obs.metadata or {}).get('symbol_context') or {}
            if not parent_obs.ok or not data.get('include_source'): return []
            rows=[]
            def add(relation,row,path,start,end,code,resolution_kind='exact'):
                if not path or not start or not end or not code:return
                rows.append(ToolObservation('read_file',True,code,{
                    'path':path,'start_line':int(start),'end_line':int(end),'requested_end_line':int(end),'truncated':False,'retryable':False,
                    'provenance':{'parent_tool':'inspect_symbol_context','parent_observation_id':parent_obs.observation_id,'relation':relation,'symbol':row.get('symbol') or row.get('qualified_name') or row.get('name'),'resolution_kind':resolution_kind}
                }))
            d=data.get('definition') or {}
            sr=d.get('source_range') or []
            if len(sr)==2:add('definition',d,d.get('path'),sr[0],sr[1],d.get('source_code',''),'exact')
            for c in data.get('callers') or []:
                sr=c.get('source_range') or []
                if len(sr)==2:add('caller',c,c.get('path'),sr[0],sr[1],c.get('source_code',''),c.get('resolution_kind','dynamic'))
            for c in data.get('callees') or []:
                sr=c.get('source_range') or []
                if len(sr)==2:add('callee',c,c.get('definition_path') or c.get('path'),sr[0],sr[1],c.get('source_code',''),c.get('resolution_kind','dynamic'))
            return rows

        def _acquire_observation(obs, *, need=None, mode='', action_tool='', update_need=True):
            """Phase A only: persist immutable observation/evidence facts. Semantic matching is separate."""
            nonlocal pending_route_recovery, rejected_since_last_evidence, pending_reconciliation, force_reflect
            state.observations.append(obs); observations.add(obs); trace.record('TOOL_OBSERVATION',asdict(obs))
            path_resolution=(obs.metadata or {}).get('path_resolution')
            if path_resolution: trace.record('PATH_RESOLVED',path_resolution)
            path_pattern=(obs.metadata or {}).get('path_pattern')
            if path_pattern: trace.record('PATH_PATTERN_NORMALIZED',path_pattern)
            if obs.error_type=='ambiguous_path': trace.record('PATH_AMBIGUOUS',{'message':obs.content,**(obs.metadata or {})})
            elif obs.error_type=='path_rejected': trace.record('PATH_REJECTED',{'message':obs.content,**(obs.metadata or {})})
            elif obs.error_type=='path_not_found': trace.record('PATH_NOT_FOUND',{'message':obs.content,**(obs.metadata or {})})
            if obs.ok and obs.tool=='read_file':
                p=(obs.metadata or {}).get('path'); a=(obs.metadata or {}).get('start_line'); b=(obs.metadata or {}).get('end_line')
                if p and isinstance(a,int) and isinstance(b,int): coverage.add(path=p,start_line=a,end_line=b,observation_id=obs.observation_id)
            if not obs.ok:
                state.errors.append(f"{obs.tool}: {obs.content}")
                return None
            if obs.tool=='code_search': trace.record('RETRIEVAL_DIAGNOSTICS',{'information_need_id':getattr(need,'need_id',None),**(obs.metadata or {})})
            ev=memory.add_observation(obs); gained=bool(ev)
            if ev:
                if need is not None: ev.tags.append('need:'+need.need_id)
                for obligation_id in (obs.metadata or {}).get('obligation_ids', ()):
                    ev.tags.append('obligation:'+str(obligation_id))
                ev.tags.append('step:'+str(state.step)); ev.tags.append('source_class:source' if ev.source=='read_file' else 'source_class:discovery')
                state.evidence.append(ev); rejected_since_last_evidence=False; pending_reconciliation=True; trace.record('EVIDENCE_ADDED',asdict(ev))
                if pending_route_recovery: state.recovered_routes+=1; pending_route_recovery=False; trace.record('ROUTE_RECOVERED',{'step':state.step,'tool':action_tool or obs.tool})
            if update_need and need_tracker is not None and need is not None:
                need_tracker.note_result(need,[ev.evidence_id] if ev else [],gained,mode,step=state.step)
                advisory=need_tracker.advisory(need); trace.record('INFORMATION_NEED_UPDATED',{'need':asdict(need),'new_evidence':gained,'advisory':advisory})
                if advisory: trace.record('RETRIEVAL_STRATEGY_ADVISORY',{'need_id':need.need_id,'reason':'repeated_no_gain','message':advisory,'recommended_mode':'semantic_or_hybrid' if 'semantic or hybrid' in advisory else 'converge'})
            return ev

        def _native_information_need(planned):
            """Build retrieval metadata without making native tool execution depend on it."""
            intent = getattr(planned, 'intent', None)
            active = obligations.open_critical() if obligations is not None else []
            first = active[0] if active else None
            calls = list(getattr(planned, 'tool_calls', ()) or ())
            args = next((call.arguments for call in calls if isinstance(call.arguments, dict)), {})
            target = getattr(intent, 'target', None) or getattr(first, 'target', None)
            if not target:
                target = args.get('path') or args.get('symbol') or args.get('query') or args.get('target')
            question_type = getattr(intent, 'question_type', None) or getattr(first, 'goal_type', None)
            evidence_goal = getattr(intent, 'evidence_goal', None) or getattr(first, 'reason', None)
            text = (getattr(intent, 'information_need', None) or getattr(planned, 'assistant_text', None)
                    or getattr(intent, 'reason', None) or getattr(planned, 'reason', None) or '')
            if not text or text.strip().lower() == 'native tool request':
                text = 'Investigate current required evidence gap'
                if target:
                    text += f': {target}'
            if not evidence_goal:
                evidence_goal = getattr(planned, 'expected_evidence', None) or text
            structured = {
                'target': target or '',
                'question_type': question_type or '',
                'evidence_goal': evidence_goal or '',
            }
            return text.strip(), structured

        try:
            repo=Path(task.repo_path).resolve()
            if not repo.exists(): raise FileNotFoundError(repo)
            fs=SafeRepositoryFS(repo)
            hypothesis=HypothesisManager(repo)
            obligations=EvidenceObligationTracker(max_attempts=2,repo_root=repo) if flags.evidence_obligations else None
            llm=build_llm(cfg.model)
            capabilities = getattr(llm, 'capabilities', None)
            if capabilities is not None:
                trace.record('PROVIDER_CAPABILITIES', asdict(capabilities) if hasattr(capabilities, '__dataclass_fields__') else dict(capabilities))
            current_stage=RuntimeStage.INDEX_BUILD
            if h.build_task_index:
                index_path=Path(h.trace_dir).parent/'indexes'/f'{trace.run_id}.sqlite'
                index=RepositoryIndex(repo,index_path,fs=fs)
                trace.record('INDEX_BUILD_STARTED',{'deadline_remaining_seconds':deadline.remaining()})
                try:
                    stats=index.build(deadline=deadline); trace.record('INDEX_BUILT',stats)
                except IndexDeadlineExceeded as exc:
                    index.close(); index=None
                    trace.record('INDEX_BUILD_DEADLINE_EXCEEDED',{'error_type':type(exc).__name__,'fallback':'safe_filesystem_tools'})
                if obligations is not None and index is not None: obligations.set_symbol_lookup(index.symbols)
                search_engine=RepositorySearchEngine(index,None,rrf_k=h.semantic_search.rrf_k,deadline=deadline) if index is not None else None
                if index is not None and flags.semantic_code_search and h.semantic_search.enabled:
                    try:
                        manifest=build_chunk_manifest(fs,max_embedding_tokens=h.semantic_search.max_embedding_tokens,deadline=deadline)
                        trace.record('CHUNK_MANIFEST_BUILT',{'chunks':len(manifest.chunks),'digest':manifest.digest,'chunker_version':manifest.chunker_version})
                        provider=SiliconFlowEmbeddingProvider(api_key=h.semantic_search.api_key,model=h.semantic_search.model,base_url=h.semantic_search.base_url,dimension=h.semantic_search.dimension,timeout=h.semantic_search.timeout,batch_size=h.semantic_search.batch_size,max_retries=h.semantic_search.max_retries,max_isolation_depth=h.semantic_search.max_isolation_depth)
                        embedding_cache=EmbeddingCache(Path(h.semantic_search.cache_path))
                        semantic_index=SemanticIndex(manifest,provider,embedding_cache)
                        sem_stats=semantic_index.build(deadline=deadline); trace.record('SEMANTIC_INDEX_BUILT',asdict(sem_stats))
                        if sem_stats.status != 'ready':
                            trace.record('SEMANTIC_INDEX_DEGRADED',{'error_type':'EmbeddingError','message':sem_stats.error,'failed_path':sem_stats.failed_path,'failed_symbol':sem_stats.failed_symbol,'failed_start_line':sem_stats.failed_start_line,'failed_end_line':sem_stats.failed_end_line,'fallback':'lexical'})
                        search_engine=RepositorySearchEngine(index,semantic_index,rrf_k=h.semantic_search.rrf_k,deadline=deadline)
                    except Exception as exc:
                        trace.record('SEMANTIC_INDEX_DEGRADED',{'error_type':type(exc).__name__,'message':str(exc),'fallback':'lexical'})
                        search_engine=RepositorySearchEngine(index,None,rrf_k=h.semantic_search.rrf_k,deadline=deadline)
            current_stage=RuntimeStage.SETUP
            tools=ToolRegistry(repo,index=search_engine or index,fs=fs)
            planner=PlannerFacade(llm,tools,cfg.model.planner_model,
                                  native_enabled=flags.native_tool_calling,
                                  max_parallel_actions=h.parallel_max_actions)
            trace.record('PLANNER_MODE_SELECTED', {'mode': planner.mode,
                                                    'tool_calling': bool(getattr(capabilities, 'tool_calling', False))})
            orchestrator=ToolOrchestrator(tools, max_parallel_actions=h.parallel_max_actions,
                                          max_tool_calls=h.max_tool_calls,
                                          read_context_padding=60)
            if hasattr(planner,"compact_prompt"): planner.compact_prompt=flags.compact_prompt_rendering
            critic_model=cfg.model.critic_model or cfg.model.planner_model
            use_typed_reflection=bool(flags.structured_reflection and planner.mode == 'native_tool_calling')
            reflector=TypedReflection(llm,critic_model) if use_typed_reflection else Reflector(llm,critic_model)
            typed_reducer=SemanticReducer(obligations,hypothesis,state.evidence,revision=semantic_revision) if use_typed_reflection and obligations is not None else None
            reporter=Reporter(llm,critic_model)
            reporter.repair_admission=_admit_reporter_repair
            if hasattr(reflector,"compact_prompt"): reflector.compact_prompt=flags.compact_prompt_rendering
            if hasattr(reporter,"compact_prompt"): reporter.compact_prompt=flags.compact_prompt_rendering
            router=RouterGuard(tools); loop=LoopGuard(h.max_repeat_action,h.max_no_progress_steps)
            trace.record('RUN_CONFIG',{
                'task':asdict(task),'policy':'read_only','runtime_version':RUNTIME_VERSION,
                'model':{'provider':cfg.model.provider,'planner':cfg.model.planner_model,'critic':critic_model,'temperature':cfg.model.temperature},
                'harness':{'max_steps':h.max_steps,'max_tool_calls':h.max_tool_calls,'max_context_chars':h.max_context_chars,'max_llm_calls':h.max_llm_calls,'max_total_tokens':h.max_total_tokens,'max_wall_time_seconds':h.max_wall_time_seconds,'finalization_reserve_seconds':h.finalization_reserve_seconds,
                           'planner_start_guard_seconds':h.planner_start_guard_seconds,'reflection_start_guard_seconds':h.reflection_start_guard_seconds,
                           'planner_llm_timeout_seconds':h.planner_llm_timeout_seconds,'reflection_llm_timeout_seconds':h.reflection_llm_timeout_seconds,'reporter_llm_timeout_seconds':h.reporter_llm_timeout_seconds,'llm_cleanup_margin_seconds':h.llm_cleanup_margin_seconds,
                           'obligation_review_min_seconds':h.obligation_review_min_seconds,'max_auto_obligation_presentations_per_reflection':h.max_auto_obligation_presentations_per_reflection,'obligation_presentation_max_chars':h.obligation_presentation_max_chars,
                           'focused_reflection_max_obligations':h.focused_reflection_max_obligations,'focused_reflection_max_chars':h.focused_reflection_max_chars,'focused_reflection_timeout_seconds':h.focused_reflection_timeout_seconds,'evidence_bundle_max_chars':h.evidence_bundle_max_chars,
                           'parallel_max_actions':h.parallel_max_actions,'parallel_group_timeout_seconds':h.parallel_group_timeout_seconds,'tool_retry_attempts':h.tool_retry_attempts,'retry_base_delay_seconds':h.retry_base_delay_seconds,'retry_max_delay_seconds':h.retry_max_delay_seconds,'semantic_no_progress_limit':h.semantic_no_progress_limit,
                           'provider_health_window':h.provider_health_window,'provider_failure_threshold':h.provider_failure_threshold,'provider_consecutive_failures':h.provider_consecutive_failures,'provider_recovery_successes':h.provider_recovery_successes,'provider_degraded_timeout_seconds':h.provider_degraded_timeout_seconds,
                           'max_consecutive_reflection_failures':h.max_consecutive_reflection_failures,'max_consecutive_planner_contract_failures':h.max_consecutive_planner_contract_failures,
                           'max_reporter_context_tokens':h.max_reporter_context_tokens,'max_evidence_per_file':h.max_evidence_per_file,'max_snippet_lines':h.max_snippet_lines,'max_trace_summary_items':h.max_trace_summary_items,
                           'context':asdict(h.context),'features':asdict(flags),'semantic_search':asdict(h.semantic_search)},
            })

            while state.status=='running':
                snap=_budget_snapshot('loop_start')
                exhausted=budget.exhausted(snap)
                if exhausted:
                    state.status='budget_exhausted'; state.errors.append(exhausted+' exceeded'); trace.record('BUDGET_EXHAUSTED',{'reason':exhausted,'snapshot':asdict(snap)}); break
                if controller is not None and flags.cost_aware_convergence:
                    old_mode=controller.state.mode; controller.apply_budget(remaining_ratio=snap.remaining_ratio,hyp=state.current_hypothesis or {})
                    state.convergence_mode=controller.state.mode.value
                    if old_mode != controller.state.mode: trace.record('CONVERGENCE_MODE_CHANGED',{'from':old_mode.value,'to':controller.state.mode.value,'step':state.step,'reason':'cost_aware_budget'})
                terminal_mode=(controller is not None and controller.state.mode in {ConvergenceMode.BUDGET_CRITICAL,ConvergenceMode.FORCE_FINALIZATION})
                if snap.phase=='finalize' or terminal_mode:
                    reason=('finalization_reserve_or_budget_phase' if snap.phase=='finalize' else controller.state.mode.value)
                    if state.evidence:
                        finalize_now(reason); break
                    state.status='budget_exhausted'; state.errors.append(reason); trace.record('BUDGET_EXHAUSTED',{'reason':reason,'snapshot':asdict(snap)}); break
                state.step+=1
                periodic=(state.step>1 and state.step%h.reflect_every==0)
                near_budget=(h.max_steps-state.step <= 2)
                if force_reflect or periodic or near_budget:
                    if pending_reconciliation:
                        _reconcile_pending(boundary='pre_reflection')
                    if _stage_guard_due(RuntimeStage.REFLECTION,h.reflection_start_guard_seconds):
                        finalize_now('reflection_stage_guard'); break
                    reflection_id=f"R{state.step}-{uuid.uuid4().hex[:8]}"
                    bundle=_prepare_obligation_bundle(reflection_id)
                    if bundle and isinstance(bundle,dict) and bundle.get('finalize'):
                        finalize_now('obligation_review_budget_guard'); break
                    current_reflection_signature=_reflection_signature(bundle)
                    if current_reflection_signature != reflection_signature:
                        reflection_signature=current_reflection_signature
                        consecutive_reflection_failures=0
                        compact_reflection_retry_pending=False
                    compact_retry=compact_reflection_retry_pending
                    compact_reflection_retry_pending=False
                    current_stage=RuntimeStage.REFLECTION
                    if bundle is not None:
                        previous_metrics=reflection_context_metrics.get(current_reflection_signature)
                        target_chars=(int(previous_metrics['chars'] * 0.6) if compact_retry and previous_metrics else None)
                        rctx=_focused_reflection_context(bundle,compact=compact_retry,target_chars=target_chars)
                        current_metrics={'chars':len(rctx),'obligation_count':1 if compact_retry else len(active_presentation_plans)}
                        if compact_retry and previous_metrics:
                            trace.record('REFLECTION_RETRY_DEGRADED',{
                                'previous_chars':previous_metrics['chars'],'retry_chars':len(rctx),
                                'previous_obligation_count':previous_metrics['obligation_count'],
                                'retry_obligation_count':current_metrics['obligation_count'],
                                'reason':'same_signature_compact_retry',
                            })
                        reflection_context_metrics[current_reflection_signature]=current_metrics
                    else:
                        rctx=build_ctx(current_stage)
                    reservation,gate_error=budget_gate.admit('focused_reflection' if bundle is not None else 'reflection')
                    if gate_error:
                        trace.record('BUDGET_GATE_REJECTED',{**gate_error,'stage':'reflection','step':state.step})
                        finalize_now('reflection_budget_gate'); break
                    before=len(getattr(llm,'calls',[])); review=None
                    def _reflection_attempt_started(meta):
                        if active_presentation_plans:
                            submitted_reflection_ids.add(reflection_id)
                            for plan in active_presentation_plans:
                                if plan.get('reflection_id')==reflection_id and plan.get('prepared'):
                                    plan['request_submitted']=True; plan['logical_call_id']=meta.get('logical_call_id')
                    try:
                        review=_call_with_optional_timeout(reflector.review,rctx,logical_timeout_seconds=_llm_stage_timeout(RuntimeStage.REFLECTION),on_attempt_started=_reflection_attempt_started)
                    except Exception as exc:
                        _flush_llm(current_stage,before); active_presentation_plans=[]
                        failure=_failure_from_exception(state,current_stage,exc); state.reflection_failure_count+=1; consecutive_reflection_failures+=1
                        failure_count=reflection_failures_by_signature.get(current_reflection_signature,0)+1
                        reflection_failures_by_signature[current_reflection_signature]=failure_count
                        consecutive_reflection_failures=failure_count
                        state.max_consecutive_reflection_failures_observed=max(state.max_consecutive_reflection_failures_observed,consecutive_reflection_failures)
                        trace.record('REFLECTION_FAILED',{**asdict(failure),'consecutive_failures':consecutive_reflection_failures,'max_tolerated':h.max_consecutive_reflection_failures})
                        if not failure.retryable: raise
                        state.errors.append(f"recoverable reflection failure: {failure.error_type}: {failure.message}")
                        trace.record('CRITIC_DEGRADED',{'step':state.step,'consecutive_failures':consecutive_reflection_failures,'focused':bundle is not None,'reason':failure.error_type})
                        if failure_count < max(1,h.max_consecutive_reflection_failures):
                            force_reflect=bool(bundle is not None)
                            compact_reflection_retry_pending=bool(bundle is not None)
                            trace.record('REFLECTION_FAILURE_RECOVERED',{'step':state.step,'consecutive_failures':failure_count,'preserved_hypothesis':bool(state.current_hypothesis),'retry_mode':'compact_focused' if compact_reflection_retry_pending else 'planner'})
                        else:
                            # Critic degradation is not budget exhaustion. Finalize conservatively from acquired facts.
                            if state.evidence:
                                finalize_now('critic_failure_limit'); break
                            state.status='budget_exhausted'; state.errors.append('critic failure limit without source evidence'); break
                        continue
                    finally:
                        budget_gate.release(reservation)
                    if review is not None:
                        _flush_llm(current_stage,before); _record_prompt_breakdown(current_stage,reflector)
                        if getattr(reflector,'last_repair_attempted',False): trace.record('REFLECTION_SCHEMA_REPAIRED',{'step':state.step})
                        state.reflection_count+=1; semantic_ok=True
                        try:
                            if isinstance(review, ReflectionDecision):
                                trace.record('SEMANTIC_REDUCER_STARTED', {'reflection_id': reflection_id, 'typed': True})
                                typed_reducer.evidence=list(state.evidence)
                                candidate=typed_reducer.reduce_and_commit(review, reflection_id=reflection_id, presented_evidence_ids=set())
                                semantic_revision=candidate.revision
                                state.current_hypothesis=asdict(candidate.hypothesis)
                                review={
                                    'decision': review.decision,
                                    'evidence_sufficient': candidate.hypothesis.evidence_sufficient,
                                    'supporting_evidence_ids': candidate.hypothesis.supporting_evidence_ids,
                                    'contradicting_evidence_ids': candidate.hypothesis.contradicting_evidence_ids,
                                }
                            elif flags.hypothesis_state:
                                trace.record('SEMANTIC_REDUCER_STARTED', {'reflection_id': reflection_id, 'typed': False})
                                review=update_hypothesis(review,reflection_id=reflection_id)
                            trace.record('SEMANTIC_REDUCER_RESULT', {'reflection_id': reflection_id, 'revision': semantic_revision, 'active_required_count': len(obligations.open_critical()) if obligations is not None else 0})
                            trace.record('DERIVED_HYPOTHESIS_STATE', {'status': (state.current_hypothesis or {}).get('status'), 'evidence_sufficient': (state.current_hypothesis or {}).get('evidence_sufficient', False), 'required_gap_count': len((state.current_hypothesis or {}).get('required_missing_evidence') or [])})
                        except Exception as exc:
                            semantic_ok=False; state.reflection_failure_count+=1
                            failure_count=reflection_failures_by_signature.get(current_reflection_signature,0)+1
                            reflection_failures_by_signature[current_reflection_signature]=failure_count
                            consecutive_reflection_failures=failure_count
                            state.max_consecutive_reflection_failures_observed=max(state.max_consecutive_reflection_failures_observed,failure_count)
                            failure=_failure_from_exception(state,RuntimeStage.REFLECTION,exc)
                            trace.record('REFLECTION_FAILED',{**asdict(failure),'consecutive_failures':consecutive_reflection_failures,'semantic_transaction':True})
                            state.errors.append(f"semantic reflection failure: {failure.error_type}: {failure.message}")
                        active_presentation_plans=[]
                        if not semantic_ok:
                            trace.record('CRITIC_DEGRADED',{'step':state.step,'consecutive_failures':consecutive_reflection_failures,'focused':bundle is not None,'reason':'semantic_transaction'})
                            if consecutive_reflection_failures >= max(1,h.max_consecutive_reflection_failures):
                                if state.evidence:
                                    finalize_now('critic_failure_limit'); break
                                state.status='budget_exhausted'; state.errors.append('critic failure limit without source evidence'); break
                            force_reflect=bool(bundle is not None)
                            compact_reflection_retry_pending=bool(bundle is not None)
                            continue
                        if consecutive_reflection_failures:
                            trace.record('CRITIC_RECOVERED',{'step':state.step,'previous_consecutive_failures':consecutive_reflection_failures})
                        reflection_failures_by_signature[current_reflection_signature]=0
                        consecutive_reflection_failures=0; critic_compact_retry_used=False
                        trace.record('REFLECTION',review)
                        force_reflect=bool(obligations is not None and obligations.presentation_candidates())
                        # semantic_no_progress_streak counts only successful semantic
                        # transactions. It may trigger conservative finalization, but
                        # never while READY evidence still awaits review.
                        if semantic_no_progress_streak >= max(1,int(h.semantic_no_progress_limit)) and not force_reflect and state.evidence:
                            finalize_now('semantic_no_progress_limit'); break
                        post_reflection_budget=_budget_snapshot('post_reflection')
                        if state.status=='budget_exhausted': break
                        if post_reflection_budget.phase=='finalize' and state.evidence:
                            finalize_now('post_reflection_budget'); break
                        if controller is not None and controller.state.mode in {ConvergenceMode.FORCE_FINALIZATION,ConvergenceMode.BUDGET_CRITICAL}:
                            finalize_now('post_reflection_'+controller.state.mode.value); break
                        if review.get('decision')=='finish' and review.get('evidence_sufficient') and len(state.evidence)>=2 and (controller is None or controller.can_finalize(state.current_hypothesis or {})):
                            state.report=build_report(); state.status='partial_success' if state.report_source=='fallback' else 'success'; break

                if controller is not None and controller.state.mode is ConvergenceMode.BUDGET_CRITICAL:
                    if state.evidence:
                        finalize_now('budget_critical_after_reflection'); break
                    state.status='budget_exhausted'; state.errors.append('budget critical without source evidence'); trace.record('BUDGET_EXHAUSTED',{'step':state.step,'reason':'budget_critical_without_evidence'}); break

                if _stage_guard_due(RuntimeStage.PLANNER,h.planner_start_guard_seconds):
                    finalize_now('planner_stage_guard'); break
                # No-evidence bootstrap Planner is deliberately exempt from the
                # conservative cost-table gate.  It is still bounded by the absolute
                # run deadline in _llm_stage_timeout().  Once any source Evidence exists,
                # all subsequent Planner turns must pass RemainingBudgetGate normally.
                if state.evidence:
                    planner_reservation,planner_gate_error=budget_gate.admit('planner')
                else:
                    planner_reservation,planner_gate_error=None,None
                if planner_gate_error:
                    trace.record('BUDGET_GATE_REJECTED',{**planner_gate_error,'stage':'planner','step':state.step})
                    finalize_now('planner_budget_gate'); break
                current_stage=RuntimeStage.PLANNER; context=build_ctx(current_stage); before=len(getattr(llm,'calls',[]))
                try:
                    planned=_call_with_optional_timeout(planner.propose,state,context,logical_timeout_seconds=_llm_stage_timeout(RuntimeStage.PLANNER))
                except Exception as exc:
                    _flush_llm(current_stage,before)
                    if isinstance(exc, PlannerContractError):
                        metadata=getattr(exc,'metadata',{'error_type':'planner_contract_invalid'})
                        if planner.mode == 'native_tool_calling' and isinstance(exc, NativePlannerContractError):
                            trace.record('NATIVE_TOOL_CALL_INVALID', {
                                'index': metadata.get('index'),
                                'tool': metadata.get('tool'),
                                'error_type': metadata.get('error_type', 'provider_contract_mismatch'),
                            })
                        if metadata.get('repair_rejection_reason'):
                            trace.record('PLANNER_REPAIR_REJECTED',{'reason':metadata['repair_rejection_reason']})
                        trace.record('PLANNER_CONTRACT_INVALID',metadata)
                        consecutive_planner_contract_failures+=1; state.planner_contract_failure_count=consecutive_planner_contract_failures
                        last_planner_contract_feedback={k:metadata.get(k) for k in ('validation_errors','kind','tool','arguments_type','actions_type','actions_count','child_types','output_shape')}
                        trace.record('PLANNER_CONTRACT_REJECTED',{'count':consecutive_planner_contract_failures,'max_count':h.max_consecutive_planner_contract_failures,**last_planner_contract_feedback})
                        state.errors.append('planner contract repeatedly invalid' if consecutive_planner_contract_failures>=max(1,h.max_consecutive_planner_contract_failures) else 'planner contract rejected; retrying next iteration')
                        if consecutive_planner_contract_failures>=max(1,h.max_consecutive_planner_contract_failures):
                            failure=_failure_from_exception(state,current_stage,exc); state.failure=failure
                            if state.evidence:
                                finalize_now('planner_contract_failure_limit'); break
                            state.status='failed'; break
                        continue
                    state.errors.append(str(exc)); failure=_failure_from_exception(state,current_stage,exc); trace.record('PLANNER_FAILED',asdict(failure))
                    consecutive_planner_errors+=1
                    # Contract repair is already bounded inside Planner. Repeating
                    # the same malformed output is not useful blind retry work.
                    limit=1 if type(exc).__name__=='PlannerContractError' else 3
                    if consecutive_planner_errors>=limit: state.failure=_failure_from_exception(state,current_stage,exc); state.status='failed'; break
                    continue
                finally:
                    budget_gate.release(planner_reservation)
                _flush_llm(current_stage,before); _record_prompt_breakdown(current_stage,planner); consecutive_planner_errors=0
                consecutive_planner_contract_failures=0; state.planner_contract_failure_count=0; last_planner_contract_feedback=None
                if isinstance(planned, NativePlannerResult):
                    # Native tool calls enter the typed orchestrator directly. No
                    # synthetic AgentAction/parallel JSON is created on this path.
                    trace.record('NATIVE_PLANNER_RESULT', {
                        'tool_call_count': len(planned.tool_calls),
                        'has_content': bool(planned.assistant_text),
                        'has_structured_metadata': planned.response.structured is not None,
                    })
                    trace.record('NATIVE_TOOL_CALLS_RECEIVED', {
                        'count': len(planned.tool_calls),
                        'tool_names': [call.name for call in planned.tool_calls],
                    })
                    need_text, need_structured = _native_information_need(planned)
                    current_need = need_tracker.get_or_create(need_text, state.step, need_structured) if need_tracker is not None else None
                    if current_need is not None:
                        need_tracker.note_attempt(current_need, 'native_tool_calling', step=state.step)
                        trace.record('INFORMATION_NEED', {'need': asdict(current_need), 'action_tool': 'native_tool_calls', 'mode': 'native_tool_calling', 'structured': need_structured})
                    native_intent = ' '.join(x for x in (
                        planned.information_need, planned.expected_evidence, planned.reason,
                    ) if x)
                    if any(not _check_obligation_scope(call.name, call.arguments, native_intent)
                           for call in planned.tool_calls):
                        force_reflect = True
                        continue
                    requests = [RequestedToolCall(
                        id=call.id, name=call.name, arguments=dict(call.arguments),
                        information_need_id=getattr(current_need, 'need_id', None),
                        obligation_ids=tuple(x for x in planned.obligation_ids if obligations is not None and x in obligations.items),
                        reason=planned.reason, expected_evidence=planned.expected_evidence,
                        retain_context_ids=planned.retain_context_ids,
                    ) for call in planned.tool_calls]
                    orchestrator.max_tool_calls=max(0, h.max_tool_calls-state.tool_calls)
                    try:
                        execution_plan=orchestrator.build_plan(requests)
                    except ToolPlanningError as exc:
                        trace.record('TOOL_EXECUTION_PLAN_REJECTED', {'error_type': exc.error_type, 'tool': exc.tool})
                        state.errors.append(str(exc))
                        if state.evidence:
                            finalize_now('native_tool_plan_rejected'); break
                        state.status='failed'; break
                    trace.record('TOOL_REQUEST_EXPANDED', {
                        'requested_count': len(requests), 'expanded_count': execution_plan.expanded_count,
                        'requests': [
                            {'id': x.request.id, 'tool': x.request.name,
                             'requested_range': x.requested_range, 'expanded_range': x.expanded_range}
                            for x in execution_plan.calls
                        ],
                    })
                    trace.record('TOOL_EXECUTION_PLAN_BUILT', {
                        'parallel_groups': [[x.request.name for x in group] for group in execution_plan.parallel_groups],
                        'serial_calls': [x.request.name for x in execution_plan.serial_calls],
                        'expanded_count': execution_plan.expanded_count,
                    })
                    if not execution_plan.calls:
                        trace.record('NATIVE_PLANNER_NO_TOOL_TURN', {
                            'has_content': bool(planned.assistant_text),
                            'policy_decision': 'reflection',
                        })
                        force_reflect=True
                        continue
                    state.tool_calls += len(execution_plan.calls)
                    current_stage=RuntimeStage.TOOL_EXECUTION
                    try:
                        native_observations=orchestrator.execute(
                            execution_plan, deadline=deadline.absolute_deadline, retry_policy=retry_policy
                        )
                    except Exception as exc:
                        failure=_failure_from_exception(state,current_stage,exc)
                        trace.record('TOOL_EXECUTION_FAILED',asdict(failure)); state.errors.append(str(exc))
                        if state.evidence:
                            finalize_now('native_tool_execution_failure'); break
                        state.failure=failure; state.status='failed'; break
                    acquired_ids=[]; any_gain=False
                    for obs in native_observations:
                        tool=tools.get(obs.tool)
                        if tool is not None:
                            _truncate_observation_content(obs,min(h.max_tool_output_chars,tool.spec.output_limit or h.max_tool_output_chars))
                        ev=_acquire_observation(obs,need=current_need,mode='native_tool_calling',action_tool=obs.tool,update_need=False)
                        if ev: acquired_ids.append(ev.evidence_id); any_gain=True
                    if current_need is not None and need_tracker is not None:
                        need_tracker.note_result(current_need, acquired_ids, any_gain, 'native_tool_calling', step=state.step)
                        trace.record('INFORMATION_NEED_UPDATED', {'need': asdict(current_need), 'new_evidence': any_gain, 'advisory': need_tracker.advisory(current_need)})
                    if pending_reconciliation:
                        _reconcile_pending(boundary='post_native_tool_acquisition')
                    force_reflect=True
                    continue
                action=planned
                if getattr(planner,'last_action_normalization',None):
                    trace.record('PLANNER_ACTION_NORMALIZED',planner.last_action_normalization)
                state.actions.append(action); trace.record('ACTION_PROPOSED',asdict(action))
                if flags.model_context_selection: requested_context_ids=list(action.retain_context_ids or [])

                if action.kind in {ActionKind.TOOL,ActionKind.PARALLEL}:
                    need=(action.information_need or action.expected_evidence or action.reason).strip().lower()
                    if need and need==last_information_need: same_information_need+=1
                    else: same_information_need=0; last_information_need=need
                    current_need=need_tracker.get_or_create(action.information_need or action.expected_evidence or action.reason,state.step,action.information_need_structured) if need_tracker is not None else None
                    if action.kind is ActionKind.PARALLEL: mode='parallel'
                    elif action.tool=='code_search': mode=str(action.arguments.get('mode','lexical')).lower()
                    elif action.tool in {'grep','symbol_search'}: mode='lexical'
                    else: mode=action.tool or ''
                    if need_tracker is not None: need_tracker.note_attempt(current_need,mode,step=state.step)
                    if obligations is not None: obligations.note_attempt_for_need(action.information_need or action.expected_evidence or action.reason)
                    if current_need is not None: trace.record('INFORMATION_NEED',{'need':asdict(current_need),'action_tool':action.tool or ('parallel' if action.kind is ActionKind.PARALLEL else None),'mode':mode,'structured':action.information_need_structured})
                else:
                    same_information_need=0; current_need=None

                obligation_intent = ' '.join(x for x in (
                    action.information_need, action.expected_evidence, action.reason,
                ) if x)
                scoped_actions = action.actions if action.kind is ActionKind.PARALLEL else [
                    {'tool': action.tool, 'arguments': action.arguments}
                ]
                if any(not _check_obligation_scope(
                    str(child.get('tool') or ''), dict(child.get('arguments') or {}), obligation_intent
                ) for child in scoped_actions):
                    force_reflect = True
                    continue

                current_stage=RuntimeStage.ROUTE_VALIDATION; gd=router.validate(action,state)
                if not gd.ok:
                    state.invalid_routes+=1; pending_route_recovery=True; rejected_since_last_evidence=True
                    err=gd.error or {'error_type':'action_rejected','message':gd.reason,'retryable':True}
                    state.errors.append(f"{err.get('error_type')}: {gd.reason}")
                    reject_count=1; reject_exceeded=False; reject_sig=None
                    # V1.3.2.2 rejection-loop accounting is intentionally scoped to TOOL
                    # actions. Premature FINISH is already governed by convergence/no-progress
                    # logic and must not pre-empt safe force-finalization.
                    if action.kind in {ActionKind.TOOL,ActionKind.PARALLEL}:
                        reject_count,reject_exceeded,reject_sig=loop.observe_rejected_action(action,state,err.get('error_type','action_rejected'))
                    trace.record('ACTION_REJECTED',{'reason':gd.reason,'error':err,'action':asdict(action),'rejected_count':reject_count,'rejected_signature':reject_sig})
                    if err.get('error_type')=='parallel_dependency': trace.record('PARALLEL_DEPENDENCY_REJECTED',{'group_action':asdict(action),'reason':gd.reason,'action_index':err.get('action_index')})
                    if action.kind in {ActionKind.TOOL,ActionKind.PARALLEL} and reject_count>1:
                        trace.record('REPEATED_REJECTED_ACTION',{'count':reject_count,'limit':loop.max_repeat,'signature':reject_sig,'action':asdict(action),'error_type':err.get('error_type')})
                    # A rejected TOOL action is never diagnostic progress and must not reset BUDGET_CRITICAL.
                    if action.kind in {ActionKind.TOOL,ActionKind.PARALLEL} and controller is not None and reject_exceeded:
                        old_mode=controller.state.mode
                        # Repeated invalid/rejected actions indicate planning stagnation,
                        # not depleted budget. Narrow the action space and force semantic
                        # review without corrupting BUDGET_CRITICAL telemetry.
                        if controller.state.mode is ConvergenceMode.NORMAL:
                            controller.state.mode=ConvergenceMode.CONVERGENCE_REQUIRED
                        state.convergence_mode=controller.state.mode.value
                        if old_mode is not controller.state.mode:
                            trace.record('CONVERGENCE_MODE_CHANGED',{'from':old_mode.value,'to':controller.state.mode.value,'step':state.step,'reason':'repeated_rejected_action'})
                    force_reflect=gd.force_reflection or action.confidence>=0.8 or (action.kind in {ActionKind.TOOL,ActionKind.PARALLEL} and reject_count>1)
                    continue
                if gd.repair:
                    trace.record('ACTION_ARGUMENT_REPAIRED',{**gd.repair,'step':state.step,'skill':action.skill,'information_need':action.information_need})
                if gd.advisory: trace.record('ROUTE_ADVISORY',{'message':gd.advisory,'skill':action.skill,'tool':action.tool})
                if gd.canonical_arguments is not None: action.arguments=gd.canonical_arguments
                if gd.canonical_actions is not None: action.actions=gd.canonical_actions

                if action.kind==ActionKind.REFLECT: force_reflect=True; continue
                if action.kind==ActionKind.FINISH:
                    # The quality precondition constrains Harness-forced finalization, not a normal
                    # model finish that already passed RouterGuard's grounded-evidence requirement.
                    state.report=build_report(); state.status='partial_success' if state.report_source=='fallback' else 'success'; break

                # Re-check budget after Planner latency. A tool already in progress is allowed to
                # complete, but no new investigation tool starts once finalization becomes due.
                policy_budget=_budget_snapshot('post_planner')
                policy_mode=(controller.state.mode.value if controller is not None else 'normal')
                pd=action_policy.evaluate(action,budget_phase=policy_budget.phase,convergence_mode=policy_mode,evidence=state.evidence)
                if pd.advisory:
                    state.termination_advisory=pd.advisory
                    trace.record('ACTION_POLICY_ADVISORY',{'message':pd.advisory,'phase':policy_budget.phase,'convergence_mode':policy_mode,'action':asdict(action)})
                if not pd.allowed:
                    trace.record('ACTION_POLICY_REJECTED',{'reason':pd.reason,'hard_block':pd.hard_block,'phase':policy_budget.phase,'convergence_mode':policy_mode,'action':asdict(action)})
                    if policy_budget.phase=='finalize' or policy_mode in {'budget_critical','force_finalization'}:
                        if state.evidence:
                            finalize_now('action_policy_finalization'); break
                        state.status='budget_exhausted'; state.errors.append(pd.reason); break
                    force_reflect=True
                    continue

                # Coverage/reuse precedes exact-repeat blocking for read_file. V1.3.1 distinguishes
                # already-visible redundant requests from cold observations that must be rehydrated.
                reused=False
                if flags.observation_reuse and action.tool=='read_file':
                    p=action.arguments.get('path'); sline=action.arguments.get('start_line'); eline=action.arguments.get('end_line')
                    if p and isinstance(sline,int) and isinstance(eline,int):
                        # Canonicalize before coverage lookup so equivalent spellings (./a.py,
                        # Windows separators, unique read-only suffix/basename hints) share the
                        # same immutable observation identity. Ambiguity/not-found is left to the
                        # Tool to return as a structured observation.
                        try:
                            resolved_path=tools.path_resolver.resolve_file(p,mode=ResolutionMode.READ_TOLERANT)
                            if resolved_path.strategy != 'exact_relative' or resolved_path.relative_path != p:
                                trace.record('PATH_RESOLVED',{**resolved_path.metadata(str(p)),'stage':'pre_reuse'})
                            p=resolved_path.relative_path; action.arguments['path']=p
                        except RepositoryPathError:
                            pass
                        hit=coverage.find_covering(path=p,start_line=sline,end_line=eline)
                        if hit:
                            obs=observations.get(hit.observation_id)
                            if obs is not None:
                                need=normalize_target(action.information_need or action.expected_evidence or action.reason)
                                req_key=(normalize_location(p,repo),sline,eline,need)
                                visible=ctxmgr.is_visible_range(hit.path,sline,eline)
                                same_need=(req_key==last_redundant_key)
                                if visible and same_need:
                                    state.redundant_request_count+=1; reused=True
                                    streak=controller.note_redundant() if controller is not None else 1
                                    trace.record('REDUNDANT_CONTEXT_REQUEST',{
                                        'requested':{'path':p,'start_line':sline,'end_line':eline},'covered_by':obs.observation_id,
                                        'information_need':need,'redundant_request_streak':streak,'message':'requested source is already fully visible; reason from current context',
                                    })
                                    if streak>=2: force_reflect=True
                                else:
                                    ctxmgr.rehydrate(obs.observation_id,path=hit.path,start_line=sline,end_line=eline,information_need=need); state.observation_reuse_count+=1; state.rehydration_count+=1; state.rehydration_saved_tool_calls+=1; state.rehydration_saved_chars+=len(obs.content or ''); reused=True
                                    if controller is not None: controller.note_nonredundant_action()
                                    payload={
                                        'requested':{'path':p,'start_line':sline,'end_line':eline},'reused_observation_id':obs.observation_id,
                                        'original_coverage':{'start_line':hit.start_line,'end_line':hit.end_line},
                                        'information_need':need,'information_need_satisfied':True,'was_visible':visible,'saved_tool_calls':1,'saved_chars':len(obs.content or ''),
                                    }
                                    trace.record('OBSERVATION_REHYDRATED',payload)
                                    if sline > hit.start_line or eline < hit.end_line:
                                        trace.record('OBSERVATION_SUBRANGE_REHYDRATED',payload)
                                    trace.record('OBSERVATION_REUSED',payload)  # backward-compatible aggregate event
                                    force_reflect = True
                                last_redundant_key=req_key
                if reused: continue

                if controller is not None:
                    controller.note_nonredundant_action(); last_redundant_key=None
                ok,reason=loop.observe_action(action,state)
                if not ok:
                    trace.record('LOOP_BLOCKED',{'reason':reason}); force_reflect=True
                    if not flags.convergence_control: state.no_progress_count+=1
                    continue

                current_stage=RuntimeStage.TOOL_EXECUTION
                acquired_ids=[]; any_gain=False
                if action.kind is ActionKind.PARALLEL:
                    child_costs=[]
                    for child in action.actions:
                        tn=child.get('tool') or 'tool'; args=child.get('arguments') or {}
                        if tn=='inspect_symbol_context': tn='inspect_symbol_context' if args.get('include_source') else 'inspect_symbol_context_metadata'
                        child_costs.append(budget_gate.estimate(tn if tn in budget_gate.costs else 'tool'))
                    bootstrap_acquisition=not state.evidence
                    if bootstrap_acquisition:
                        reservation,gate_error=None,None
                    else:
                        reservation,gate_error=budget_gate.admit('parallel',child_costs=child_costs,max_workers=h.parallel_max_actions)
                    if gate_error:
                        trace.record('BUDGET_GATE_REJECTED',{**gate_error,'stage':'parallel_tool_execution','step':state.step})
                        finalize_now('parallel_budget_gate'); break
                    group_id=f'PG-{state.step}-{uuid.uuid4().hex[:6]}'
                    trace.record('PARALLEL_ACTION_GROUP_STARTED',{'group_id':group_id,'action_count':len(action.actions),'information_need_id':getattr(current_need,'need_id',None)})
                    try:
                        state.tool_calls+=len(action.actions)
                        hard_remaining=max(0.001,deadline.remaining())
                        effective_group_timeout=min(float(h.parallel_group_timeout_seconds),hard_remaining)
                        group=execute_parallel_group(tools,action.actions,group_id=group_id,max_workers=h.parallel_max_actions,group_timeout_seconds=effective_group_timeout,retry_policy=retry_policy,on_retry=lambda idx,n,o:trace.record('TOOL_RETRY',{'group_id':group_id,'action_index':idx,'attempt':n,'tool':action.actions[idx].get('tool'),'error':o.content[:500]}))
                    finally:
                        budget_gate.release(reservation)
                    for child in group.children:
                        obs=child.observation; tool=tools.get(child.tool)
                        effective_limit=min(h.max_tool_output_chars,tool.spec.output_limit or h.max_tool_output_chars); _truncate_observation_content(obs,effective_limit)
                        ev=_acquire_observation(obs,need=current_need,mode='parallel',action_tool=child.tool,update_need=False)
                        if ev: acquired_ids.append(ev.evidence_id); any_gain=True
                        for derived in _expand_inspect_sources(obs):
                            dev=_acquire_observation(derived,need=current_need,mode='parallel',action_tool=child.tool,update_need=False)
                            if dev: acquired_ids.append(dev.evidence_id); any_gain=True
                    trace.record('PARALLEL_ACTION_GROUP_FINISHED',{'group_id':group_id,'status':group.status,'elapsed_ms':group.elapsed_ms,'children':[{'action_index':x.action_index,'action_id':x.action_id,'tool':x.tool,'ok':x.observation.ok,'error_type':x.observation.error_type} for x in group.children]})
                    if need_tracker is not None and current_need is not None:
                        need_tracker.note_result(current_need,acquired_ids,any_gain,'parallel',step=state.step)
                        advisory=need_tracker.advisory(current_need); trace.record('INFORMATION_NEED_UPDATED',{'need':asdict(current_need),'new_evidence':any_gain,'advisory':advisory})
                    if pending_reconciliation:_reconcile_pending(boundary='post_parallel_acquisition')
                else:
                    tool=tools.get(action.tool)
                    action_type='inspect_symbol_context' if action.tool=='inspect_symbol_context' and action.arguments.get('include_source') else 'inspect_symbol_context_metadata' if action.tool=='inspect_symbol_context' else action.tool if action.tool in budget_gate.costs else 'tool'
                    bootstrap_acquisition=not state.evidence
                    if bootstrap_acquisition:
                        reservation,gate_error=None,None
                    else:
                        reservation,gate_error=budget_gate.admit(action_type)
                    if gate_error:
                        trace.record('BUDGET_GATE_REJECTED',{**gate_error,'stage':'tool_execution','tool':action.tool,'step':state.step})
                        finalize_now('tool_budget_gate'); break
                    state.tool_calls+=1
                    try:
                        hard_remaining=max(0.001,deadline.remaining())
                        attempt_budget=max(0.1,reservation.seconds if reservation is not None else hard_remaining)
                        obs=execute_with_retry(tool,action.arguments,policy=retry_policy,absolute_deadline=time.monotonic()+min(attempt_budget,hard_remaining),on_retry=lambda n,o: trace.record('TOOL_RETRY',{'attempt':n,'tool':action.tool,'error':o.content[:500]}))
                    finally:
                        budget_gate.release(reservation)
                    effective_limit=min(h.max_tool_output_chars,tool.spec.output_limit or h.max_tool_output_chars); _truncate_observation_content(obs,effective_limit)
                    # Record retrieval telemetry only after all source observations derived
                    # from this single Action have been acquired.  In particular,
                    # inspect_symbol_context(include_source=true) is one Action that
                    # yields multiple provenance-backed source Evidence items.
                    ev=_acquire_observation(obs,need=current_need,mode=mode,action_tool=action.tool,update_need=False)
                    if ev: acquired_ids.append(ev.evidence_id); any_gain=True
                    for derived in _expand_inspect_sources(obs):
                        dev=_acquire_observation(derived,need=current_need,mode=mode,action_tool=action.tool,update_need=False)
                        if dev: acquired_ids.append(dev.evidence_id); any_gain=True
                    if need_tracker is not None and current_need is not None:
                        need_tracker.note_result(current_need,acquired_ids,any_gain,mode,step=state.step)
                        advisory=need_tracker.advisory(current_need); trace.record('INFORMATION_NEED_UPDATED',{'need':asdict(current_need),'new_evidence':any_gain,'advisory':advisory})
                        if advisory: trace.record('RETRIEVAL_STRATEGY_ADVISORY',{'need_id':current_need.need_id,'reason':'repeated_no_gain','message':advisory,'recommended_mode':'semantic_or_hybrid' if 'semantic or hybrid' in advisory else 'converge'})
                    if not obs.ok:
                        if obs.error_type in {'ambiguous_path','path_not_found'} and bool((obs.metadata or {}).get('planner_retryable')):
                            force_reflect=False; trace.record('PATH_REPLAN_REQUESTED',{'error_type':obs.error_type,'tool':obs.tool,'metadata':obs.metadata})
                        else: force_reflect=bool(obs.error_type)
                    if pending_reconciliation:_reconcile_pending(boundary='post_tool_acquisition')

                loop.observe_progress(state)
                if controller is not None and controller.state.mode in {ConvergenceMode.CONVERGENCE_REQUIRED,ConvergenceMode.BUDGET_CRITICAL}:
                    # Once convergence begins, assess each exploration cycle rather than waiting for the periodic reflector.
                    force_reflect=True
                if not flags.convergence_control and loop.no_progress >= h.max_no_progress_steps:
                    force_reflect=True; state.no_progress_count+=1; trace.record('NO_PROGRESS',{'steps':loop.no_progress,'reason':'legacy_no_new_evidence'})

            if state.report is None and state.evidence and state.status not in ('failed','partial_success'):
                try:
                    state.report=build_report()
                    if state.report_source=='fallback': state.status='partial_success'
                except Exception:
                    if state.status=='budget_exhausted': raise
                    raise

            current_stage=RuntimeStage.SERIALIZATION
            result=redact_sensitive({'state':state.to_summary(),'report':asdict(state.report) if state.report else None,'trace':trace.export_meta(),
                    'failure':asdict(state.failure) if state.failure else None,'report_source':state.report_source or None})
            if output_dir:
                d=Path(output_dir); d.mkdir(parents=True,exist_ok=True)
                (d/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
                if state.report: (d/'report.md').write_text(render_report_md(state.report),encoding='utf-8')

        except Exception as exc:
            if state.failure is None: state.failure=_failure_from_exception(state,current_stage,exc)
            state.status='failed'; state.errors.append(f"{state.failure.error_type}: {state.failure.message}")
        finally:
            current_stage=RuntimeStage.FINALIZATION
            try:
                if llm is not None:
                    _record_new_llm_usage(trace,llm,usage_cursor,current_stage)
                    _record_new_llm_events(trace,llm,provider_event_cursor,current_stage)
            except Exception as final_exc:
                try: trace.record('FINALIZATION_ERROR',{'component':'per_call_usage','error_type':type(final_exc).__name__,'message':str(final_exc)})
                except Exception: pass
            try: trace.record('LLM_USAGE',_usage_snapshot(llm))
            except Exception as final_exc:
                try: trace.record('FINALIZATION_ERROR',{'component':'usage_snapshot','error_type':type(final_exc).__name__,'message':str(final_exc)})
                except Exception: pass
            try:
                if state.status=='failed':
                    payload=asdict(state.failure) if state.failure else {'stage':'unknown','error_type':'unexpected_error','exception_type':'Unknown','message':'run failed without failure metadata','retryable':None,'step':state.step,'tool_calls':state.tool_calls,'evidence_count':len(state.evidence),'last_action':None}
                    payload['summary']=state.to_summary(); trace.record('RUN_FAILED',payload)
                else:
                    trace.record('RUN_END',{'summary':state.to_summary(),'report':asdict(state.report) if state.report else None,'report_source':state.report_source or None})
            except Exception as final_exc:
                try: trace.record('FINALIZATION_ERROR',{'component':'run_terminal_event','error_type':type(final_exc).__name__,'message':str(final_exc)})
                except Exception: pass
            try:
                if index is not None: index.close()
                if embedding_cache is not None: embedding_cache.close()
                if index_path is not None and not h.keep_task_index:
                    try: index_path.unlink()
                    except OSError: pass
            except Exception as cleanup_exc:
                try: trace.record('CLEANUP_ERROR',{'error_type':type(cleanup_exc).__name__,'message':str(cleanup_exc)})
                except Exception: pass

        result=redact_sensitive({'state':state.to_summary(),'report':asdict(state.report) if state.report else None,'trace':trace.export_meta(),
                'failure':asdict(state.failure) if state.failure else None,'report_source':state.report_source or None})
        if output_dir:
            try:
                d=Path(output_dir); d.mkdir(parents=True,exist_ok=True); (d/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
                if state.report: (d/'report.md').write_text(render_report_md(state.report),encoding='utf-8')
            except Exception: pass
        return result


def render_report_md(r):
    files='\n'.join(f"- `{x}`" for x in r.likely_files) or '- none'; syms='\n'.join(f"- `{x}`" for x in r.likely_symbols) or '- none'
    ev='\n'.join(f"- **{e['evidence_id']}** {e.get('file') or e['source']}: {e['summary']}" for e in r.evidence)
    changes='\n'.join(f"- `{x.get('file','')}` / `{x.get('symbol','')}`: {x.get('reason','')}" for x in r.recommended_change_points) or '- none'
    unreviewed=[]
    for x in getattr(r,'acquired_unreviewed',[]) or []:
        loc=x.get('file') or 'unknown source'; a=x.get('line_start'); b=x.get('line_end')
        if a is not None and b is not None: loc+=f':{a}-{b}'
        sym=f" / `{x.get('symbol')}`" if x.get('symbol') else ''
        unreviewed.append(f"- `{loc}`{sym} — obligation `{x.get('obligation_id')}`; {x.get('reason','not semantically reviewed')}")
    unreviewed_text='\n'.join(unreviewed) or '- none'
    return f"""# Debug-Assistant Diagnosis\n\n**Status source:** {r.report_source}\n**Confidence:** {r.confidence:.2f}\n\n## Summary\n{r.summary}\n\n## Root cause\n{r.root_cause}\n\n## Likely files\n{files}\n\n## Likely symbols\n{syms}\n\n## Impact scope\n"""+'\n'.join(f'- {x}' for x in r.impact_scope)+f"\n\n## Evidence\n{ev}\n\n## Acquired but not semantically reviewed\n{unreviewed_text}\n\n## Recommended change points (no edits executed)\n{changes}\n\n## Uncertainties\n"+'\n'.join(f'- {x}' for x in r.uncertainties)+"\n\n## Next checks\n"+'\n'.join(f'- {x}' for x in r.next_checks)+f"\n\n> {r.policy_note}\n"
