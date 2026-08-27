from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json, re
from debug_assistant.models import AgentState, ActionKind, TaskSpec, RuntimeStage, RuntimeFailure
from debug_assistant.llm.factory import build_llm
from debug_assistant.llm.base import LLMError
from debug_assistant.tools.registry import ToolRegistry
from debug_assistant.repository.index import RepositoryIndex
from debug_assistant.memory.evidence_memory import EvidenceMemory
from debug_assistant.memory.observation_store import ObservationStore
from debug_assistant.memory.coverage import ReadCoverageIndex
from debug_assistant.memory.hypothesis import HypothesisManager, normalize_target, normalize_location
from debug_assistant.context.manager import ContextManager
from debug_assistant.agent.planner import Planner
from debug_assistant.agent.reflection import Reflector
from debug_assistant.agent.reporter import Reporter
from debug_assistant.reporting.fallback import FallbackReportBuilder
from .guards import RouterGuard, LoopGuard
from .trace import TraceRecorder
from .tool_executor import execute_with_retry
from .convergence import ConvergenceController, ConvergenceMode, ProgressKind

RUNTIME_VERSION="1.3.1.1"


def _classify_error(exc: Exception, stage: RuntimeStage) -> tuple[str, bool | None]:
    name=type(exc).__name__.lower(); msg=str(exc).lower()
    if 'timeout' in name or 'timeout' in msg:
        return ('llm_timeout' if stage in {RuntimeStage.PLANNER,RuntimeStage.REFLECTION,RuntimeStage.REPORTER} else 'timeout'), True
    if isinstance(exc, LLMError):
        if 'json' in msg or 'parse' in msg: return 'llm_parse_error', True
        if 'http' in msg or 'request' in msg or 'network' in msg: return 'llm_http_error', True
        return 'llm_error', True
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


def _usage_snapshot(llm) -> dict:
    calls=[dict(x) for x in getattr(llm,'calls',[])] if llm is not None else []
    return {'calls':calls,'totals':{
        'calls':len(calls),
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


def _validate_evidence_ids(ids: list[str], evidence) -> list[str]:
    available={e.evidence_id for e in evidence}
    return [x for x in ids if x in available]


class AgentHarness:
    def __init__(self,config):
        self.config=config
        self.config.harness.features.validate()

    def run(self,task:TaskSpec,output_dir:str|None=None):
        cfg=self.config; h=cfg.harness; flags=h.features
        trace=TraceRecorder(h.trace_dir,task.task_id); state=AgentState(task=task)
        current_stage=RuntimeStage.SETUP
        llm=None; index=None; index_path=None
        memory=EvidenceMemory(); observations=ObservationStore(); coverage=ReadCoverageIndex(); hypothesis=None
        ctxmgr=ContextManager(h.context,enable_catalog=flags.context_catalog,enable_model_selection=flags.model_context_selection,enable_budget_packing=flags.context_budget_packing)
        fallback=FallbackReportBuilder()
        tools=planner=reflector=reporter=router=loop=None
        pending_route_recovery=False; force_reflect=False; consecutive_planner_errors=0; consecutive_reflection_failures=0; usage_cursor=0
        requested_context_ids=[]; last_information_need=""; same_information_need=0
        controller=ConvergenceController(no_progress_limit=2) if flags.convergence_control else None
        last_redundant_key=None

        def build_ctx(stage: RuntimeStage):
            nonlocal requested_context_ids
            result=ctxmgr.build(state,memory,observations,max_context_chars=h.max_context_chars,max_steps=h.max_steps,max_tool_calls=h.max_tool_calls,requested_ids=requested_context_ids)
            trace.record('CONTEXT_BUILT',{
                'stage':stage.value,'budget_chars':result.budget_chars,'used_chars':result.used_chars,'catalog_size':result.catalog_size,
                'working_set_size':result.working_set_size,'selected':result.selected,'dropped':result.dropped,
                'invalid_requested_ids':result.invalid_requested_ids,'breakdown':result.breakdown,
            })
            if result.invalid_requested_ids: trace.record('CONTEXT_REQUEST_INVALID',{'ids':result.invalid_requested_ids,'stage':stage.value})
            requested_context_ids=[]
            return result.text

        def _current_total_tokens() -> int:
            return int((_usage_snapshot(llm).get('totals') or {}).get('tokens',0) or 0)

        def _record_prompt_breakdown(stage: RuntimeStage, actor) -> None:
            payload=dict(getattr(actor,'last_prompt_breakdown',{}) or {})
            if payload:
                payload.update({'stage':stage.value,'step':state.step})
                trace.record('PROMPT_BREAKDOWN',payload)

        def update_hypothesis(review: dict):
            valid_support=_validate_evidence_ids(review.get('supporting_evidence_ids') or [],state.evidence)
            valid_contra=_validate_evidence_ids(review.get('contradicting_evidence_ids') or [],state.evidence)
            if len(valid_support) != len(review.get('supporting_evidence_ids') or []):
                trace.record('REFLECTION_EVIDENCE_ID_INVALID',{'kind':'support','requested':review.get('supporting_evidence_ids'),'accepted':valid_support})
            if len(valid_contra) != len(review.get('contradicting_evidence_ids') or []):
                trace.record('REFLECTION_EVIDENCE_ID_INVALID',{'kind':'contradiction','requested':review.get('contradicting_evidence_ids'),'accepted':valid_contra})
            review=dict(review); review['supporting_evidence_ids']=valid_support; review['contradicting_evidence_ids']=valid_contra
            hs=hypothesis.update(review,state.step); state.current_hypothesis=asdict(hs)
            trace.record('HYPOTHESIS_UPDATED',state.current_hypothesis)
            if controller is not None:
                old_mode=controller.state.mode
                assessment=controller.assess_reflection(state.current_hypothesis,usage_totals=(_usage_snapshot(llm).get('totals') or {}))
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
                    state.termination_advisory=("BUDGET_CRITICAL: one final tool attempt is allowed only to resolve a required_missing_evidence item. "
                                                "Do not perform optional validation.")
                elif review.get('evidence_sufficient'):
                    state.termination_advisory=("Current diagnosis is sufficiently supported and no unresolved contradiction is recorded. "
                                                "Prefer finish unless a specific unresolved information_need requires another tool call.")
                else:
                    state.termination_advisory=""
                if state.termination_advisory:
                    trace.record('TERMINATION_ADVISORY',{'step':state.step,'hypothesis_status':hs.status,'stable_diagnosis_transitions':hs.stable_diagnosis_transitions,'mode':state.convergence_mode})
            else: state.termination_advisory=""
            return review

        def build_report(context: str):
            nonlocal usage_cursor,current_stage
            current_stage=RuntimeStage.REPORTER; before=len(getattr(llm,'calls',[]))
            try:
                report=reporter.build(task.task_id,context,state.evidence); _record_prompt_breakdown(current_stage,reporter); state.report_source='llm'; return report
            except Exception as exc:
                failure=_failure_from_exception(state,current_stage,exc)
                trace.record('REPORTER_FAILED',asdict(failure))
                if flags.fallback_reporter and hypothesis is not None and hypothesis.state.status in ('supported','confirmed') and hypothesis.state.description and hypothesis.state.supporting_evidence_ids and not hypothesis.state.contradicting_evidence_ids and not hypothesis.state.required_missing_evidence:
                    report=fallback.build(task.task_id,hypothesis.state,state.evidence); state.report_source='fallback'
                    trace.record('FALLBACK_REPORT_BUILT',{'evidence_ids':report.evidence_ids,'confidence':report.confidence,'primary_report_failure':asdict(failure)})
                    return report
                raise
            finally:
                usage_cursor=_record_new_llm_usage(trace,llm,before,current_stage)

        try:
            repo=Path(task.repo_path).resolve()
            if not repo.exists(): raise FileNotFoundError(repo)
            hypothesis=HypothesisManager(repo)
            llm=build_llm(cfg.model)
            current_stage=RuntimeStage.INDEX_BUILD
            if h.build_task_index:
                index_path=Path(h.trace_dir).parent/'indexes'/f'{trace.run_id}.sqlite'
                index=RepositoryIndex(repo,index_path); stats=index.build(); trace.record('INDEX_BUILT',stats)
            current_stage=RuntimeStage.SETUP
            tools=ToolRegistry(repo,index=index); planner=Planner(llm,tools,cfg.model.planner_model)
            critic_model=cfg.model.critic_model or cfg.model.planner_model
            reflector=Reflector(llm,critic_model); reporter=Reporter(llm,critic_model)
            router=RouterGuard(tools); loop=LoopGuard(h.max_repeat_action,h.max_no_progress_steps)
            trace.record('RUN_START',{
                'task':asdict(task),'policy':'read_only','runtime_version':RUNTIME_VERSION,
                'model':{'provider':cfg.model.provider,'planner':cfg.model.planner_model,'critic':critic_model,'temperature':cfg.model.temperature},
                'harness':{'max_steps':h.max_steps,'max_tool_calls':h.max_tool_calls,'max_context_chars':h.max_context_chars,
                           'max_consecutive_reflection_failures':h.max_consecutive_reflection_failures,
                           'context':asdict(h.context),'features':asdict(flags)},
            })

            while state.status=='running':
                if state.step>=h.max_steps: state.status='budget_exhausted'; state.errors.append('max steps exceeded'); break
                if state.tool_calls>=h.max_tool_calls: state.status='budget_exhausted'; state.errors.append('max tool calls exceeded'); break
                state.step+=1
                periodic=(state.step>1 and state.step%h.reflect_every==0)
                near_budget=(h.max_steps-state.step <= 2)
                if force_reflect or periodic or near_budget:
                    current_stage=RuntimeStage.REFLECTION; rctx=build_ctx(current_stage); before=len(getattr(llm,'calls',[]))
                    review=None
                    try:
                        review=reflector.review(rctx)
                    except Exception as exc:
                        usage_cursor=_record_new_llm_usage(trace,llm,before,current_stage)
                        failure=_failure_from_exception(state,current_stage,exc)
                        state.reflection_failure_count += 1
                        consecutive_reflection_failures += 1
                        state.max_consecutive_reflection_failures_observed=max(
                            state.max_consecutive_reflection_failures_observed,consecutive_reflection_failures
                        )
                        trace.record('REFLECTION_FAILED',{**asdict(failure),'consecutive_failures':consecutive_reflection_failures,
                                                          'max_tolerated':h.max_consecutive_reflection_failures})
                        # Only retryable transport/model failures are recoverable. Schema/programming
                        # errors must remain visible rather than being silently masked.
                        if not failure.retryable:
                            raise
                        state.errors.append(f"recoverable reflection failure: {failure.error_type}: {failure.message}")
                        if consecutive_reflection_failures < max(1,h.max_consecutive_reflection_failures):
                            # Preserve the last valid Hypothesis and allow one normal planner cycle.
                            # A later successful reflection resets this streak.
                            force_reflect=False
                            trace.record('REFLECTION_FAILURE_RECOVERED',{
                                'step':state.step,'consecutive_failures':consecutive_reflection_failures,
                                'preserved_hypothesis':bool(state.current_hypothesis),
                            })
                        else:
                            # Reflection has failed repeatedly. If the previous structured state is
                            # already safe to finalize, stop exploration and report from that state.
                            if controller is not None and controller.can_finalize(state.current_hypothesis or {}):
                                old_mode=controller.state.mode
                                controller.state.mode=ConvergenceMode.FORCE_FINALIZATION
                                controller.state.forced_finalization=True
                                state.convergence_mode=controller.state.mode.value; state.forced_finalization=True
                                if old_mode is not controller.state.mode:
                                    trace.record('CONVERGENCE_MODE_CHANGED',{'from':old_mode.value,'to':controller.state.mode.value,'step':state.step,'reason':'reflection_failure_limit'})
                                trace.record('FORCE_FINALIZATION',{'step':state.step,'reason':'reflection_failure_limit',
                                                                   'hypothesis_status':(state.current_hypothesis or {}).get('status'),
                                                                   'supporting_evidence_ids':(state.current_hypothesis or {}).get('supporting_evidence_ids',[])})
                                state.report=build_report(build_ctx(RuntimeStage.REPORTER))
                                state.status='partial_success' if state.report_source=='fallback' else 'success'
                                break
                            if controller is not None:
                                old_mode=controller.state.mode
                                if controller.state.mode is ConvergenceMode.BUDGET_CRITICAL and controller.state.critical_attempt_used:
                                    state.status='budget_exhausted'
                                    state.errors.append('reflection failure limit exceeded after budget-critical attempt')
                                    trace.record('BUDGET_EXHAUSTED',{'step':state.step,'reason':'reflection_failure_limit_after_critical_attempt'})
                                    break
                                controller.state.mode=ConvergenceMode.BUDGET_CRITICAL
                                controller.state.budget_critical_entered=True
                                controller.state.critical_attempt_used=False
                                state.convergence_mode=controller.state.mode.value; state.budget_critical_entered=True
                                if old_mode is not controller.state.mode:
                                    trace.record('CONVERGENCE_MODE_CHANGED',{'from':old_mode.value,'to':controller.state.mode.value,'step':state.step,'reason':'reflection_failure_limit'})
                                trace.record('BUDGET_CRITICAL',{'step':state.step,'reason':'reflection_failure_limit',
                                                                 'required_missing_evidence':(state.current_hypothesis or {}).get('required_missing_evidence',[])})
                                force_reflect=False
                            else:
                                state.status='budget_exhausted'
                                state.errors.append('reflection failure tolerance exhausted')
                                trace.record('BUDGET_EXHAUSTED',{'step':state.step,'reason':'reflection_failure_limit'})
                                break
                    if review is not None:
                        usage_cursor=_record_new_llm_usage(trace,llm,before,current_stage)
                        _record_prompt_breakdown(current_stage,reflector)
                        consecutive_reflection_failures=0
                        state.reflection_count+=1
                        if flags.hypothesis_state: review=update_hypothesis(review)
                        trace.record('REFLECTION',review); force_reflect=False
                        if state.status=='budget_exhausted': break
                        if controller is not None and controller.state.mode is ConvergenceMode.FORCE_FINALIZATION:
                            state.report=build_report(build_ctx(RuntimeStage.REPORTER)); state.status='partial_success' if state.report_source=='fallback' else 'success'; break
                        if review.get('decision')=='finish' and review.get('evidence_sufficient') and len(state.evidence)>=2 and (controller is None or controller.can_finalize(state.current_hypothesis or {})):
                            state.report=build_report(build_ctx(RuntimeStage.REPORTER)); state.status='partial_success' if state.report_source=='fallback' else 'success'; break

                current_stage=RuntimeStage.PLANNER; context=build_ctx(current_stage); before=len(getattr(llm,'calls',[]))
                try: action=planner.propose(state,context)
                except Exception as exc:
                    usage_cursor=_record_new_llm_usage(trace,llm,before,current_stage)
                    state.errors.append(str(exc)); trace.record('ACTION_VALIDATION_ERROR',{'error':str(exc),'retryable':True,'stage':current_stage.value})
                    consecutive_planner_errors+=1
                    if consecutive_planner_errors>=3: state.failure=_failure_from_exception(state,current_stage,exc); state.status='failed'; break
                    continue
                usage_cursor=_record_new_llm_usage(trace,llm,before,current_stage); _record_prompt_breakdown(current_stage,planner); consecutive_planner_errors=0
                state.actions.append(action); trace.record('ACTION_PROPOSED',asdict(action))
                if flags.model_context_selection: requested_context_ids=list(action.retain_context_ids or [])

                if action.kind==ActionKind.TOOL:
                    need=(action.information_need or action.expected_evidence or action.reason).strip().lower()
                    if need and need==last_information_need: same_information_need+=1
                    else: same_information_need=0; last_information_need=need
                else: same_information_need=0

                current_stage=RuntimeStage.ROUTE_VALIDATION; gd=router.validate(action,state)
                if not gd.ok:
                    state.invalid_routes+=1; pending_route_recovery=True; err=gd.error or {'error_type':'action_rejected','message':gd.reason,'retryable':True}
                    state.errors.append(f"{err.get('error_type')}: {gd.reason}"); trace.record('ACTION_REJECTED',{'reason':gd.reason,'error':err,'action':asdict(action)})
                    force_reflect=gd.force_reflection or action.confidence>=0.8; continue
                if gd.advisory: trace.record('ROUTE_ADVISORY',{'message':gd.advisory,'skill':action.skill,'tool':action.tool})
                if gd.canonical_arguments is not None: action.arguments=gd.canonical_arguments

                if action.kind==ActionKind.REFLECT: force_reflect=True; continue
                if action.kind==ActionKind.FINISH:
                    # The quality precondition constrains Harness-forced finalization, not a normal
                    # model finish that already passed RouterGuard's grounded-evidence requirement.
                    state.report=build_report(build_ctx(RuntimeStage.REPORTER)); state.status='partial_success' if state.report_source=='fallback' else 'success'; break

                # Coverage/reuse precedes exact-repeat blocking for read_file. V1.3.1 distinguishes
                # already-visible redundant requests from cold observations that must be rehydrated.
                reused=False
                if flags.observation_reuse and action.tool=='read_file':
                    p=action.arguments.get('path'); sline=action.arguments.get('start_line'); eline=action.arguments.get('end_line')
                    if p and isinstance(sline,int) and isinstance(eline,int):
                        hit=coverage.find_covering(path=p,start_line=sline,end_line=eline)
                        if hit:
                            obs=observations.get(hit.observation_id)
                            if obs is not None:
                                need=normalize_target(action.information_need or action.expected_evidence or action.reason)
                                req_key=(normalize_location(p,repo),sline,eline,need)
                                visible=ctxmgr.is_visible(obs.observation_id)
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
                                    ctxmgr.rehydrate(obs.observation_id); state.observation_reuse_count+=1; state.rehydration_count+=1; reused=True
                                    if controller is not None: controller.note_nonredundant_action()
                                    payload={
                                        'requested':{'path':p,'start_line':sline,'end_line':eline},'reused_observation_id':obs.observation_id,
                                        'original_coverage':{'start_line':hit.start_line,'end_line':hit.end_line},
                                        'information_need':need,'information_need_satisfied':True,'was_visible':visible,
                                    }
                                    trace.record('OBSERVATION_REHYDRATED',payload)
                                    trace.record('OBSERVATION_REUSED',payload)  # backward-compatible aggregate event
                                    force_reflect = True
                                last_redundant_key=req_key
                if reused: continue

                if controller is not None:
                    controller.note_nonredundant_action(); last_redundant_key=None
                    if controller.state.mode is ConvergenceMode.BUDGET_CRITICAL and action.kind==ActionKind.TOOL:
                        if not controller.allow_critical_tool_attempt(action.information_need,state.current_hypothesis or {}):
                            trace.record('ACTION_REJECTED',{'reason':'budget critical permits only one final required-information tool attempt','action':asdict(action)})
                            state.status='budget_exhausted'; state.errors.append('budget critical exploration exhausted');
                            trace.record('BUDGET_EXHAUSTED',{'step':state.step,'reason':'budget_critical_invalid_or_extra_attempt'}); break

                ok,reason=loop.observe_action(action,state)
                if not ok:
                    trace.record('LOOP_BLOCKED',{'reason':reason}); force_reflect=True
                    if not flags.convergence_control: state.no_progress_count+=1
                    continue

                current_stage=RuntimeStage.TOOL_EXECUTION; tool=tools.get(action.tool); state.tool_calls+=1
                obs=execute_with_retry(tool,action.arguments,attempts=2,on_retry=lambda n,o: trace.record('TOOL_RETRY',{'attempt':n,'tool':action.tool,'error':o.content[:500]}))
                effective_limit=min(h.max_tool_output_chars,tool.spec.output_limit or h.max_tool_output_chars); _truncate_observation_content(obs,effective_limit)
                state.observations.append(obs); observations.add(obs); trace.record('TOOL_OBSERVATION',asdict(obs))
                if obs.ok and obs.tool=='read_file':
                    p=obs.metadata.get('path'); s=obs.metadata.get('start_line'); e=obs.metadata.get('end_line')
                    if p and isinstance(s,int) and isinstance(e,int): coverage.add(path=p,start_line=s,end_line=e,observation_id=obs.observation_id)
                if not obs.ok:
                    state.errors.append(f"{obs.tool}: {obs.content}"); force_reflect=bool(obs.error_type); continue

                current_stage=RuntimeStage.MEMORY_INGESTION; ev=memory.add_observation(obs)
                if ev:
                    state.evidence.append(ev); trace.record('EVIDENCE_ADDED',asdict(ev))
                    if pending_route_recovery: state.recovered_routes+=1; pending_route_recovery=False; trace.record('ROUTE_RECOVERED',{'step':state.step,'tool':action.tool})
                loop.observe_progress(state)
                if controller is not None and controller.state.mode in {ConvergenceMode.CONVERGENCE_REQUIRED,ConvergenceMode.BUDGET_CRITICAL}:
                    # Once convergence begins, assess each exploration cycle rather than waiting for the periodic reflector.
                    force_reflect=True
                if not flags.convergence_control and loop.no_progress >= h.max_no_progress_steps:
                    force_reflect=True; state.no_progress_count+=1; trace.record('NO_PROGRESS',{'steps':loop.no_progress,'reason':'legacy_no_new_evidence'})

            if state.report is None and state.evidence and state.status not in ('failed','partial_success'):
                try:
                    state.report=build_report(build_ctx(RuntimeStage.REPORTER))
                    if state.report_source=='fallback': state.status='partial_success'
                except Exception:
                    if state.status=='budget_exhausted': raise
                    raise

            current_stage=RuntimeStage.SERIALIZATION
            result={'state':state.to_summary(),'report':asdict(state.report) if state.report else None,'trace':trace.export_meta(),
                    'failure':asdict(state.failure) if state.failure else None,'report_source':state.report_source or None}
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
                if llm is not None: _record_new_llm_usage(trace,llm,usage_cursor,current_stage)
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
                if index_path is not None and not h.keep_task_index:
                    try: index_path.unlink()
                    except OSError: pass
            except Exception as cleanup_exc:
                try: trace.record('CLEANUP_ERROR',{'error_type':type(cleanup_exc).__name__,'message':str(cleanup_exc)})
                except Exception: pass

        result={'state':state.to_summary(),'report':asdict(state.report) if state.report else None,'trace':trace.export_meta(),
                'failure':asdict(state.failure) if state.failure else None,'report_source':state.report_source or None}
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
    return f"""# Debug-Assistant Diagnosis\n\n**Status source:** {r.report_source}\n**Confidence:** {r.confidence:.2f}\n\n## Summary\n{r.summary}\n\n## Root cause\n{r.root_cause}\n\n## Likely files\n{files}\n\n## Likely symbols\n{syms}\n\n## Impact scope\n"""+'\n'.join(f'- {x}' for x in r.impact_scope)+f"\n\n## Evidence\n{ev}\n\n## Recommended change points (no edits executed)\n{changes}\n\n## Uncertainties\n"+'\n'.join(f'- {x}' for x in r.uncertainties)+"\n\n## Next checks\n"+'\n'.join(f'- {x}' for x in r.next_checks)+f"\n\n> {r.policy_note}\n"
