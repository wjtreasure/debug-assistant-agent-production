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
from debug_assistant.memory.hypothesis import HypothesisManager
from debug_assistant.context.manager import ContextManager
from debug_assistant.agent.planner import Planner
from debug_assistant.agent.reflection import Reflector
from debug_assistant.agent.reporter import Reporter
from debug_assistant.reporting.fallback import FallbackReportBuilder
from .guards import RouterGuard, LoopGuard
from .trace import TraceRecorder
from .tool_executor import execute_with_retry

RUNTIME_VERSION="1.3"


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
        memory=EvidenceMemory(); observations=ObservationStore(); coverage=ReadCoverageIndex(); hypothesis=HypothesisManager()
        ctxmgr=ContextManager(h.context,enable_catalog=flags.context_catalog,enable_model_selection=flags.model_context_selection,enable_budget_packing=flags.context_budget_packing)
        fallback=FallbackReportBuilder()
        tools=planner=reflector=reporter=router=loop=None
        pending_route_recovery=False; force_reflect=False; consecutive_planner_errors=0; usage_cursor=0
        requested_context_ids=[]; last_information_need=""; same_information_need=0

        def build_ctx(stage: RuntimeStage):
            nonlocal requested_context_ids
            result=ctxmgr.build(state,memory,observations,max_context_chars=h.max_context_chars,max_steps=h.max_steps,max_tool_calls=h.max_tool_calls,requested_ids=requested_context_ids)
            trace.record('CONTEXT_BUILT',{
                'stage':stage.value,'budget_chars':result.budget_chars,'used_chars':result.used_chars,'catalog_size':result.catalog_size,
                'working_set_size':result.working_set_size,'selected':result.selected,'dropped':result.dropped,
                'invalid_requested_ids':result.invalid_requested_ids,
            })
            if result.invalid_requested_ids: trace.record('CONTEXT_REQUEST_INVALID',{'ids':result.invalid_requested_ids,'stage':stage.value})
            requested_context_ids=[]
            return result.text

        def update_hypothesis(review: dict):
            valid_support=_validate_evidence_ids(review.get('supporting_evidence_ids') or [],state.evidence)
            valid_contra=_validate_evidence_ids(review.get('contradicting_evidence_ids') or [],state.evidence)
            if len(valid_support) != len(review.get('supporting_evidence_ids') or []):
                trace.record('REFLECTION_EVIDENCE_ID_INVALID',{'kind':'support','requested':review.get('supporting_evidence_ids'),'accepted':valid_support})
            review=dict(review); review['supporting_evidence_ids']=valid_support; review['contradicting_evidence_ids']=valid_contra
            hs=hypothesis.update(review,state.step); state.current_hypothesis=asdict(hs)
            trace.record('HYPOTHESIS_UPDATED',state.current_hypothesis)
            if flags.termination_advisory and review.get('evidence_sufficient') and not hs.contradicting_evidence_ids and hs.status in ('supported','confirmed'):
                state.termination_advisory=("Current diagnosis is sufficiently supported and no unresolved contradiction is recorded. "
                                            "Prefer finish unless a specific unresolved information_need requires another tool call.")
                trace.record('TERMINATION_ADVISORY',{'step':state.step,'hypothesis_status':hs.status,'stable_reflections':hs.stable_reflections})
            else: state.termination_advisory=""
            return review

        def build_report(context: str):
            nonlocal usage_cursor,current_stage
            current_stage=RuntimeStage.REPORTER; before=len(getattr(llm,'calls',[]))
            try:
                report=reporter.build(task.task_id,context,state.evidence); state.report_source='llm'; return report
            except Exception as exc:
                failure=_failure_from_exception(state,current_stage,exc)
                trace.record('REPORTER_FAILED',asdict(failure))
                if flags.fallback_reporter and hypothesis.state.description and hypothesis.state.supporting_evidence_ids:
                    report=fallback.build(task.task_id,hypothesis.state,state.evidence); state.report_source='fallback'
                    trace.record('FALLBACK_REPORT_BUILT',{'evidence_ids':report.evidence_ids,'confidence':report.confidence,'primary_report_failure':asdict(failure)})
                    return report
                raise
            finally:
                usage_cursor=_record_new_llm_usage(trace,llm,before,current_stage)

        try:
            repo=Path(task.repo_path).resolve()
            if not repo.exists(): raise FileNotFoundError(repo)
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
                    try: review=reflector.review(rctx)
                    except Exception:
                        usage_cursor=_record_new_llm_usage(trace,llm,before,current_stage); raise
                    usage_cursor=_record_new_llm_usage(trace,llm,before,current_stage)
                    state.reflection_count+=1
                    if flags.hypothesis_state: review=update_hypothesis(review)
                    trace.record('REFLECTION',review); force_reflect=False
                    if review.get('decision')=='finish' and review.get('evidence_sufficient') and len(state.evidence)>=2:
                        state.report=build_report(build_ctx(RuntimeStage.REPORTER)); state.status='partial_success' if state.report_source=='fallback' else 'success'; break

                current_stage=RuntimeStage.PLANNER; context=build_ctx(current_stage); before=len(getattr(llm,'calls',[]))
                try: action=planner.propose(state,context)
                except Exception as exc:
                    usage_cursor=_record_new_llm_usage(trace,llm,before,current_stage)
                    state.errors.append(str(exc)); trace.record('ACTION_VALIDATION_ERROR',{'error':str(exc),'retryable':True,'stage':current_stage.value})
                    consecutive_planner_errors+=1
                    if consecutive_planner_errors>=3: state.failure=_failure_from_exception(state,current_stage,exc); state.status='failed'; break
                    continue
                usage_cursor=_record_new_llm_usage(trace,llm,before,current_stage); consecutive_planner_errors=0
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
                    state.report=build_report(build_ctx(RuntimeStage.REPORTER)); state.status='partial_success' if state.report_source=='fallback' else 'success'; break

                # Coverage/reuse precedes exact-repeat blocking for read_file. Re-reading known facts should rehydrate context, not become a blocked loop.
                reused=False
                if flags.observation_reuse and action.tool=='read_file':
                    p=action.arguments.get('path'); s=action.arguments.get('start_line'); e=action.arguments.get('end_line')
                    if p and isinstance(s,int) and isinstance(e,int):
                        hit=coverage.find_covering(path=p,start_line=s,end_line=e)
                        if hit:
                            obs=observations.get(hit.observation_id)
                            if obs is not None:
                                ctxmgr.rehydrate(obs.observation_id); state.observation_reuse_count+=1; reused=True
                                trace.record('OBSERVATION_REUSED',{'requested':{'path':p,'start_line':s,'end_line':e},'reused_observation_id':obs.observation_id,
                                                                  'original_coverage':{'start_line':hit.start_line,'end_line':hit.end_line}})
                                force_reflect = same_information_need >= 1 or loop.no_progress >= max(1,h.max_no_progress_steps-2)
                                if force_reflect: state.no_progress_count+=1; trace.record('NO_PROGRESS',{'reason':'reused_observation_same_need','same_information_need':same_information_need})
                if reused: continue

                ok,reason=loop.observe_action(action,state)
                if not ok: trace.record('LOOP_BLOCKED',{'reason':reason}); force_reflect=True; state.no_progress_count+=1; continue

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
                if not loop.observe_progress(state):
                    force_reflect=True; state.no_progress_count+=1; trace.record('NO_PROGRESS',{'steps':loop.no_progress,'reason':'no_new_evidence'})

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
