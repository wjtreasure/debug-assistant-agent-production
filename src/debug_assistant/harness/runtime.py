from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json, time
from debug_assistant.models import AgentState, ActionKind, TaskSpec
from debug_assistant.llm.factory import build_llm
from debug_assistant.tools.registry import ToolRegistry
from debug_assistant.repository.index import RepositoryIndex
from debug_assistant.memory.evidence_memory import EvidenceMemory
from debug_assistant.agent.planner import Planner
from debug_assistant.agent.reflection import Reflector
from debug_assistant.agent.reporter import Reporter
from .guards import RouterGuard, LoopGuard
from .context import build_context
from .trace import TraceRecorder
from .tool_executor import execute_with_retry

class AgentHarness:
    def __init__(self,config): self.config=config

    def run(self,task:TaskSpec,output_dir:str|None=None):
        repo=Path(task.repo_path).resolve()
        if not repo.exists(): raise FileNotFoundError(repo)
        cfg=self.config; llm=build_llm(cfg.model)
        trace=TraceRecorder(cfg.harness.trace_dir,task.task_id)
        index=None; index_path=None
        if cfg.harness.build_task_index:
            index_path=Path(cfg.harness.trace_dir).parent/'indexes'/f'{trace.run_id}.sqlite'
            index=RepositoryIndex(repo,index_path); stats=index.build(); trace.record('INDEX_BUILT',stats)
        tools=ToolRegistry(repo,index=index)
        planner=Planner(llm,tools,cfg.model.planner_model)
        critic_model=cfg.model.critic_model or cfg.model.planner_model
        reflector=Reflector(llm,critic_model); reporter=Reporter(llm,critic_model)
        memory=EvidenceMemory(); state=AgentState(task=task)
        router=RouterGuard(tools); loop=LoopGuard(cfg.harness.max_repeat_action,cfg.harness.max_no_progress_steps)
        trace.record('RUN_START',{"task":asdict(task),"policy":"read_only"})
        force_reflect=False
        pending_route_recovery=False

        while state.status=='running':
            if state.step>=cfg.harness.max_steps:
                state.status='budget_exhausted'; state.errors.append('max steps exceeded'); break
            if state.tool_calls>=cfg.harness.max_tool_calls:
                state.status='budget_exhausted'; state.errors.append('max tool calls exceeded'); break
            state.step+=1
            context=build_context(state,memory,cfg.harness.max_context_chars)

            periodic=(state.step>1 and state.step%cfg.harness.reflect_every==0)
            if force_reflect or periodic:
                review=reflector.review(context); state.reflection_count+=1; trace.record('REFLECTION',review); force_reflect=False
                if review.get('decision')=='finish' and len(state.evidence)>=2:
                    state.report=reporter.build(task.task_id,context,state.evidence); state.status='success'; break

            try: action=planner.propose(state,context)
            except Exception as exc:
                state.errors.append(str(exc)); trace.record('PLANNER_ERROR',{"error":str(exc)})
                if len(state.errors)>=3: state.status='failed'; break
                continue
            state.actions.append(action); trace.record('ACTION_PROPOSED',asdict(action))

            gd=router.validate(action,state)
            if not gd.ok:
                state.invalid_routes+=1; pending_route_recovery=True; trace.record('ROUTE_REJECTED',{"reason":gd.reason,"action":asdict(action)})
                force_reflect=gd.force_reflection or action.confidence>=0.8
                continue
            ok,reason=loop.observe_action(action,state)
            if not ok:
                trace.record('LOOP_BLOCKED',{"reason":reason}); force_reflect=True; continue

            if action.kind==ActionKind.REFLECT:
                force_reflect=True; continue
            if action.kind==ActionKind.FINISH:
                state.report=reporter.build(task.task_id,context,state.evidence); state.status='success'; break

            tool=tools.get(action.tool); state.tool_calls+=1
            obs=execute_with_retry(tool,action.arguments,attempts=2,on_retry=lambda n,o: trace.record('TOOL_RETRY',{'attempt':n,'tool':action.tool,'error':o.content[:500]}))
            if len(obs.content)>cfg.harness.max_tool_output_chars:
                obs.content=obs.content[:cfg.harness.max_tool_output_chars]+"\n...[truncated by harness]"
                obs.metadata['truncated']=True
            state.observations.append(obs); trace.record('TOOL_OBSERVATION',asdict(obs))
            if not obs.ok:
                state.errors.append(f"{obs.tool}: {obs.content}")
                # Tool failures return to planner; they are observations, not fatal exceptions.
                continue
            ev=memory.add_observation(obs)
            if ev:
                state.evidence.append(ev); trace.record('EVIDENCE_ADDED',asdict(ev))
                if pending_route_recovery:
                    state.recovered_routes+=1; pending_route_recovery=False; trace.record('ROUTE_RECOVERED',{'step':state.step,'tool':action.tool})
            if not loop.observe_progress(state):
                force_reflect=True; trace.record('NO_PROGRESS',{"steps":loop.no_progress})

        if state.report is None and state.evidence:
            # Best-effort report on budget exhaustion/failure; status remains non-success.
            try: state.report=reporter.build(task.task_id,build_context(state,memory,cfg.harness.max_context_chars),state.evidence)
            except Exception as exc: state.errors.append(f"report: {exc}")
        trace.record('LLM_USAGE',{'calls':getattr(llm,'calls',[]),'totals':{'calls':len(getattr(llm,'calls',[])),'tokens':sum(x.get('total_tokens',0) for x in getattr(llm,'calls',[]))}})
        trace.record('RUN_END',{"summary":state.to_summary(),"report":asdict(state.report) if state.report else None})
        if index is not None:
            index.close()
            if index_path is not None and not cfg.harness.keep_task_index:
                try: index_path.unlink()
                except OSError: pass
        result={"state":state.to_summary(),"report":asdict(state.report) if state.report else None,"trace":trace.export_meta()}
        if output_dir:
            d=Path(output_dir); d.mkdir(parents=True,exist_ok=True)
            (d/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            if state.report:
                (d/'report.md').write_text(render_report_md(state.report),encoding='utf-8')
        return result

def render_report_md(r):
    files='\n'.join(f"- `{x}`" for x in r.likely_files) or '- none'
    syms='\n'.join(f"- `{x}`" for x in r.likely_symbols) or '- none'
    ev='\n'.join(f"- **{e['evidence_id']}** {e.get('file') or e['source']}: {e['summary']}" for e in r.evidence)
    changes='\n'.join(f"- `{x.get('file','')}` / `{x.get('symbol','')}`: {x.get('reason','')}" for x in r.recommended_change_points) or '- none'
    return f"""# Debug-Assistant Diagnosis\n\n**Confidence:** {r.confidence:.2f}\n\n## Summary\n{r.summary}\n\n## Root cause\n{r.root_cause}\n\n## Likely files\n{files}\n\n## Likely symbols\n{syms}\n\n## Impact scope\n"""+'\n'.join(f'- {x}' for x in r.impact_scope)+f"\n\n## Evidence\n{ev}\n\n## Recommended change points (no edits executed)\n{changes}\n\n## Uncertainties\n"+'\n'.join(f'- {x}' for x in r.uncertainties)+"\n\n## Next checks\n"+'\n'.join(f'- {x}' for x in r.next_checks)+f"\n\n> {r.policy_note}\n"
