from __future__ import annotations
from dataclasses import asdict
from debug_assistant.context.models import ContextItem, ContextBuildResult


def _line_safe_truncate(text: str, max_chars: int) -> tuple[str,bool]:
    if len(text) <= max_chars: return text,False
    kept=[]; used=0
    for line in text.splitlines():
        add=len(line)+(1 if kept else 0)
        if used+add > max_chars: break
        kept.append(line); used+=add
    if kept: return '\n'.join(kept),True
    return text[:max_chars],True

def _evidence_reference(x: ContextItem) -> str:
    return f"[{x.context_id}] evidence metadata: {x.title}; raw_observation_id={x.raw_observation_id or 'none'}"

class ContextManager:
    """Harness-managed working set. Model-selected ids are optional priority hints."""
    def __init__(self, cfg, *, enable_catalog=True, enable_model_selection=False, enable_budget_packing=True):
        self.cfg=cfg
        self.enable_catalog=enable_catalog
        self.enable_model_selection=enable_model_selection
        self.enable_budget_packing=enable_budget_packing
        self._rehydrated: set[str]=set()
        self._last_selected_ids: set[str]=set()

    def rehydrate(self, observation_id: str) -> None:
        self._rehydrated.add(observation_id)

    def is_visible(self, context_id: str) -> bool:
        return context_id in self._last_selected_ids

    def _obs_item(self, obs, step: int, reason: str, priority: int) -> ContextItem:
        full=obs.content
        display,trunc=_line_safe_truncate(full,self.cfg.max_item_chars)
        meta=dict(obs.metadata or {})
        meta.update({"ok":obs.ok,"error_type":obs.error_type,"context_truncated":trunc,
                     "raw_chars":len(full),"display_chars":len(display)})
        title=f"{obs.tool} {meta.get('path','')}".strip()
        compact=f"{obs.observation_id} tool={obs.tool} ok={obs.ok} metadata={meta}"
        return ContextItem(obs.observation_id,"observation",title,compact,display,len(display),priority,step,obs.observation_id,{**meta,"selection_reason":reason})

    def _ev_item(self, ev, step: int, priority: int, reason: str) -> ContextItem:
        loc=ev.file or ev.source
        if ev.source_start_line is not None and ev.source_end_line is not None:
            loc=f"{loc}:{ev.source_start_line}-{ev.source_end_line}"
        compact=f"[{ev.evidence_id}] {ev.kind} {loc}: {ev.summary}"
        return ContextItem(ev.evidence_id,"evidence",loc,compact,ev.excerpt,len(ev.excerpt),priority,step,ev.raw_observation_id,{"selection_reason":reason})

    def catalog(self, state, memory, observation_store) -> list[ContextItem]:
        items=[]
        latest=state.observations[-1].observation_id if state.observations else None
        hyp=state.current_hypothesis or {}
        support=set(hyp.get('supporting_evidence_ids') or [])
        contradict=set(hyp.get('contradicting_evidence_ids') or [])
        recent_ids={o.observation_id for o in observation_store.recent(self.cfg.fallback_recent_count)}
        for obs in observation_store.all():
            if obs.observation_id in self._rehydrated: p,reason=10,"observation_reused"
            elif obs.observation_id == latest: p,reason=12,"latest_observation"
            elif not obs.ok: p,reason=25,"tool_error"
            elif obs.observation_id in recent_ids: p,reason=50,"fallback_recent"
            else: p,reason=90,"historical_observation"
            items.append(self._obs_item(obs,state.step,reason,p))
        for ev in memory.pinned:
            if ev.evidence_id in contradict: p,reason=15,"contradiction"
            elif ev.evidence_id in support: p,reason=30,"hypothesis_support"
            else: p,reason=80,"historical_evidence"
            items.append(self._ev_item(ev,state.step,p,reason))
        return items

    def catalog_text(self, items: list[ContextItem], max_chars: int=7000) -> str:
        if not self.enable_catalog: return '(catalog disabled)'
        rows=[]; used=0
        for x in sorted(items,key=lambda i:(i.created_step,i.context_id)):
            if x.source_kind=='observation':
                m=x.metadata
                loc=m.get('path','')
                if m.get('start_line') is not None: loc+=f":{m.get('start_line')}-{m.get('end_line')}"
                row=f"- {x.context_id} type=observation tool={x.title.split(' ',1)[0]} location={loc or '-'} chars={m.get('raw_chars',x.chars)} priority={x.priority}\n"
            else:
                row=f"- {x.context_id} type=evidence location={x.title} chars={x.chars} priority={x.priority}\n"
            if used+len(row)>max_chars: break
            rows.append(row); used+=len(row)
        return ''.join(rows) or '(none)'

    def build(self, state, memory, observation_store, *, max_context_chars: int, max_steps=None, max_tool_calls=None, requested_ids=None) -> ContextBuildResult:
        requested_ids=list(requested_ids or []) if self.enable_model_selection else []
        items=self.catalog(state,memory,observation_store)
        by_id={x.context_id:x for x in items}
        invalid=[x for x in requested_ids if x not in by_id]
        requested={x for x in requested_ids if x in by_id}
        issue_budget=max(2000,min(max_context_chars//3,18000))
        issue=state.task.issue[:issue_budget]
        recent_actions='\n'.join(f"- {a.skill}/{a.tool or a.kind.value}: {a.reason}" for a in state.actions[-6:]) or '(none)'
        budget=[]
        if max_steps is not None:
            budget += [f"step={state.step}/{max_steps}",f"remaining_steps={max(0,max_steps-state.step)}"]
        if max_tool_calls is not None:
            budget += [f"tool_calls={state.tool_calls}/{max_tool_calls}",f"remaining_tool_calls={max(0,max_tool_calls-state.tool_calls)}"]
        hyp=state.current_hypothesis or {}
        advisory=(state.termination_advisory or '').strip()
        fixed=(f"TASK_ID: {state.task.task_id}\nISSUE:\n{issue}\n\nRECENT_ACTIONS:\n{recent_actions}\n\n"
               f"RUNTIME_BUDGET: {', '.join(budget) or 'not configured'}\n"
               f"CURRENT_HYPOTHESIS: {hyp if hyp else '(none)'}\n"
               f"TERMINATION_ADVISORY: {advisory or '(none)'}\nSTATE: {state.to_summary()}\n\n")
        catalog=self.catalog_text(items)
        catalog_section=f"CONTEXT_CATALOG (metadata only):\n{catalog}\n"
        available=max(0,max_context_chars-len(fixed)-len(catalog_section)-self.cfg.safety_margin_chars)
        ranked=[]; dropped=[]
        for x in items:
            # Old raw observations live in the catalog/store, not automatically in every prompt.
            # They re-enter the working set only through recency, reuse, errors, or an optional model request.
            reason=x.metadata.get('selection_reason','')
            if x.source_kind=='observation' and reason=='historical_observation' and x.context_id not in requested:
                dropped.append({'id':x.context_id,'reason':'catalog_only','chars':x.chars})
                continue
            boost=-18 if x.context_id in requested else 0
            ranked.append((x.priority+boost,-x.created_step,x.context_id,x))
        ranked.sort()
        selected=[]; used=0; selected_raw=set()
        for _,__,___,x in ranked:
            # Suppress evidence excerpt duplication if its raw observation is already selected.
            if x.source_kind=='evidence' and x.raw_observation_id and x.raw_observation_id in selected_raw:
                content=_evidence_reference(x)
            elif x.source_kind=='observation':
                content=(f"OBSERVATION {x.context_id}\n{x.compact_content}\n{x.full_content}\nEND OBSERVATION {x.context_id}")
            else:
                content=x.compact_content + (f"\n{x.full_content}" if x.full_content else '')
            size=len(content)+2
            if not self.enable_budget_packing or used+size<=available:
                selected.append((x,content)); used+=size
                if x.source_kind=='observation': selected_raw.add(x.context_id)
            else:
                dropped.append({"id":x.context_id,"reason":"budget","chars":size})
        # second-pass dedup: evidence selected before raw due higher priority can still duplicate; compact it.
        rendered=[]; used2=0; selected_meta=[]
        for x,content in selected:
            if x.source_kind=='evidence' and x.raw_observation_id in selected_raw:
                content=_evidence_reference(x)
            rendered.append(content); used2+=len(content)+2
            selected_meta.append({"id":x.context_id,"reason":("model_requested" if x.context_id in requested else x.metadata.get('selection_reason','priority')),"chars":len(content),"kind":x.source_kind})
        working='\n\n'.join(rendered) or '(no working context yet)'
        text=f"{fixed}{catalog_section}\nWORKING_CONTEXT:\n{working}"
        if len(text)>max_context_chars:
            text=text[:max_context_chars]  # defensive only; packing should normally prevent this.
        self._last_selected_ids={x["id"] for x in selected_meta}
        breakdown={
            "issue_chars":len(issue),
            "recent_actions_chars":len(recent_actions),
            "hypothesis_chars":len(str(hyp if hyp else '(none)')),
            "advisory_chars":len(advisory),
            "state_chars":len(str(state.to_summary())),
            "catalog_chars":len(catalog_section),
            "observation_chars":sum(m["chars"] for m in selected_meta if m.get("kind")=="observation"),
            "evidence_chars":sum(m["chars"] for m in selected_meta if m.get("kind")=="evidence"),
        }
        self._rehydrated.clear()
        return ContextBuildResult(text,max_context_chars,len(text),len(items),len(selected_meta),selected_meta,dropped,invalid,breakdown)
