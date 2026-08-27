from __future__ import annotations
from hashlib import sha1
from debug_assistant.context.models import ContextItem, ContextBuildResult, ContextProjection
from debug_assistant.context.indexes import DisplayCoverageIndex, KnownContextIndex, extract_numbered_range, merge_ranges


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
    """V1.3.2 task-local context lifecycle manager.

    Hard invariants:
    - Raw observations/evidence are never deleted by eviction.
    - KnownContextIndex preserves model-visible pointers to previously acquired source.
    - DisplayCoverageIndex reflects only source lines that made it into the final rendered prompt.
    - Rehydration projects the requested source range from raw immutable observations.
    """
    def __init__(self, cfg, *, enable_catalog=True, enable_model_selection=False, enable_budget_packing=True,
                 enable_lifecycle=True, enable_projection=True, compact_known_index=True):
        self.cfg=cfg
        self.enable_catalog=enable_catalog
        self.enable_model_selection=enable_model_selection
        self.enable_budget_packing=enable_budget_packing
        self.enable_lifecycle=enable_lifecycle
        self.enable_projection=enable_projection
        self.compact_known_index=compact_known_index
        self.display_coverage=DisplayCoverageIndex()
        self.known_index=KnownContextIndex()
        self._rehydrate_requests: dict[str,list[tuple[int,int,str]]] = {}
        self._last_selected_ids: set[str]=set()
        self._last_active_ids: set[str]=set()
        self._last_lifecycle: dict[str,str]={}
        self._eviction_total=0
        self._projection_total=0

    def rehydrate(self, observation_id: str, *, path: str|None=None, start_line:int|None=None,
                  end_line:int|None=None, information_need:str="") -> None:
        if path and isinstance(start_line,int) and isinstance(end_line,int):
            reqs=self._rehydrate_requests.setdefault(observation_id,[])
            reqs.append((start_line,end_line,information_need))
            # Coalesce only source ranges; information_need is telemetry, not identity here.
            merged=merge_ranges([(a,b) for a,b,_ in reqs])
            self._rehydrate_requests[observation_id]=[(a,b,information_need) for a,b in merged]
        else:
            self._rehydrate_requests.setdefault(observation_id,[])

    def is_visible(self, context_id: str) -> bool:
        return context_id in self._last_selected_ids

    def is_visible_range(self, path:str,start_line:int,end_line:int) -> bool:
        return self.display_coverage.covers(path,start_line,end_line)

    def _obs_item(self, obs, step: int, reason: str, priority: int, lifecycle:str='active', pinned:bool=False) -> ContextItem:
        full=obs.content
        superseded_count=0
        if obs.tool in {'grep','code_search','symbol_search'}:
            full,superseded_count=self.known_index.filter_search_content(full)
        display,trunc=_line_safe_truncate(full,self.cfg.max_item_chars)
        meta=dict(obs.metadata or {})
        meta.update({"ok":obs.ok,"error_type":obs.error_type,"context_truncated":trunc,
                     "raw_chars":len(obs.content),"display_chars":len(display),"selection_reason":reason,
                     "range_superseded_hits":superseded_count})
        title=f"{obs.tool} {meta.get('path','')}".strip()
        compact=f"{obs.observation_id} tool={obs.tool} ok={obs.ok} metadata={meta}"
        return ContextItem(obs.observation_id,"observation",title,compact,display,len(display),priority,step,
                           obs.observation_id,meta,lifecycle,pinned,step if lifecycle=='active' else 0)

    def _ev_item(self, ev, step: int, priority: int, reason: str, lifecycle:str='active', pinned:bool=False) -> ContextItem:
        loc=ev.file or ev.source
        if ev.source_start_line is not None and ev.source_end_line is not None:
            loc=f"{loc}:{ev.source_start_line}-{ev.source_end_line}"
        compact=f"[{ev.evidence_id}] {ev.kind} {loc}: {ev.summary}"
        return ContextItem(ev.evidence_id,"evidence",loc,compact,ev.excerpt,len(ev.excerpt),priority,step,
                           ev.raw_observation_id,{"selection_reason":reason},lifecycle,pinned,
                           step if lifecycle=='active' else 0)

    def catalog(self, state, memory, observation_store) -> list[ContextItem]:
        # Rebuild before item rendering so search observations can be superseded at hit/range level.
        self.known_index.rebuild(observation_store)
        items=[]
        latest=state.observations[-1].observation_id if state.observations else None
        hyp=state.current_hypothesis or {}
        support=set(hyp.get('supporting_evidence_ids') or [])
        contradict=set(hyp.get('contradicting_evidence_ids') or [])
        recent_ids={o.observation_id for o in observation_store.recent(self.cfg.fallback_recent_count)}
        support_raw={e.raw_observation_id for e in memory.pinned if e.evidence_id in support and e.raw_observation_id}
        contradiction_raw={e.raw_observation_id for e in memory.pinned if e.evidence_id in contradict and e.raw_observation_id}
        rehydrate_ids=set(self._rehydrate_requests)

        for obs in observation_store.all():
            if obs.observation_id in rehydrate_ids:
                p,reason,lifecycle,pinned=5,"observation_reused",'active',True
            elif obs.observation_id == latest:
                p,reason,lifecycle,pinned=10,"latest_observation",'active',True
            elif obs.observation_id in contradiction_raw:
                p,reason,lifecycle,pinned=12,"contradiction_source",'active',True
            elif obs.observation_id in support_raw:
                # Keep compact evidence pinned; raw source itself can age unless it is recent.
                if obs.observation_id in recent_ids: p,reason,lifecycle,pinned=25,"recent_support_source",'active',False
                else: p,reason,lifecycle,pinned=65,"support_source_cold",'cold',False
            elif not obs.ok:
                p,reason,lifecycle,pinned=30,"tool_error",'active',False
            elif obs.observation_id in recent_ids:
                p,reason,lifecycle,pinned=45,"recent_observation",'active',False
            else:
                p,reason,lifecycle,pinned=90,"historical_observation",'cold',False
            if not self.enable_lifecycle: lifecycle='active'
            items.append(self._obs_item(obs,state.step,reason,p,lifecycle,pinned))

        active_raw_ids={x.context_id for x in items if x.source_kind=="observation" and x.lifecycle=="active"}
        for ev in memory.pinned:
            if ev.evidence_id in contradict:
                p,reason,lifecycle,pinned=8,"contradiction",'active',True
            elif ev.evidence_id in support:
                p,reason,lifecycle,pinned=18,"hypothesis_support",'active',True
            else:
                if ev.raw_observation_id and ev.raw_observation_id in active_raw_ids:
                    p,reason,lifecycle,pinned=55,"active_raw_reference",'active',False
                else:
                    p,reason,lifecycle,pinned=75,"historical_evidence",('cold' if self.enable_lifecycle else 'active'),False
            items.append(self._ev_item(ev,state.step,p,reason,lifecycle,pinned))
        return items

    def known_context_text(self, observation_store, max_chars:int|None=None) -> str:
        self.known_index.rebuild(observation_store)
        return self.known_index.render(max_chars=max_chars or getattr(self.cfg,'known_index_max_chars',3500))

    # Backward compatibility: V1.3 callers/tests use catalog_text.
    def catalog_text(self, items: list[ContextItem], max_chars: int=7000) -> str:
        rows=[]; used=0
        for x in sorted(items,key=lambda i:(i.created_step,i.context_id)):
            loc=x.title
            row=f"- {x.context_id} type={x.source_kind} location={loc} lifecycle={x.lifecycle} priority={x.priority}\n"
            if used+len(row)>max_chars: break
            rows.append(row); used+=len(row)
        return ''.join(rows) or '(none)'

    def _projection_for(self, obs, item:ContextItem, step:int) -> ContextProjection|None:
        reqs=self._rehydrate_requests.get(obs.observation_id) or []
        path=(obs.metadata or {}).get('path')
        source_start=(obs.metadata or {}).get('start_line')
        source_end=(obs.metadata or {}).get('end_line')
        if self.enable_projection and obs.tool=='read_file' and reqs and path:
            # Requests are already coalesced. Build one projection spanning the requested union;
            # if several disjoint ranges exist, keep their exact lines in one projection body.
            parts=[]; visible=[]
            for a,b,_ in reqs:
                text,va,vb=extract_numbered_range(obs.content,a,b)
                if text:
                    parts.append(text); visible.append((va,vb))
            if parts:
                content='\n'.join(parts)
                # A projection may contain disjoint ranges; metadata stores min/max while the
                # DisplayCoverageIndex is populated from exact numbered content after rendering.
                va=min(a for a,b in visible); vb=max(b for a,b in visible)
                pid='proj-'+sha1(f"{obs.observation_id}|{visible}".encode()).hexdigest()[:10]
                return ContextProjection(pid,obs.observation_id,path,source_start,source_end,va,vb,content,
                                         item.priority,'active',True,step,'rehydrated_exact_range')
        # Default bounded projection mirrors current active item.
        display=item.full_content
        if obs.tool=='read_file' and path:
            nums=[]
            for line in display.splitlines():
                if '|' not in line: continue
                head=line.split('|',1)[0].strip()
                if head.isdigit(): nums.append(int(head))
            va=nums[0] if nums else None; vb=nums[-1] if nums else None
        else: va=vb=None
        pid='proj-'+sha1(f"{obs.observation_id}|default|{len(display)}".encode()).hexdigest()[:10]
        return ContextProjection(pid,obs.observation_id,path,source_start,source_end,va,vb,display,
                                 item.priority,item.lifecycle,item.pinned,step,item.metadata.get('selection_reason',''))

    def build(self, state, memory, observation_store, *, max_context_chars: int, max_steps=None,
              max_tool_calls=None, requested_ids=None) -> ContextBuildResult:
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

        known=self.known_context_text(observation_store)
        known_section=("KNOWN_CONTEXT_INDEX (compact pointers; content may be cold):\n"
                       f"{known}\nIf details are needed from a known range, request read_file for the exact range; the Harness can rehydrate it without repository I/O.\n")
        available=max(0,max_context_chars-len(fixed)-len(known_section)-self.cfg.safety_margin_chars)

        ranked=[]; dropped=[]; active_count=0; cold_count=0
        for x in items:
            if x.lifecycle=='cold' and x.context_id not in requested:
                cold_count+=1
                dropped.append({'id':x.context_id,'reason':'cold','chars':x.chars,'kind':x.source_kind})
                continue
            active_count+=1
            boost=-15 if x.context_id in requested else 0
            ranked.append((x.priority+boost,0 if x.pinned else 1,-x.last_used_step,x.context_id,x))
        ranked.sort()

        # Soft target only: never evict pinned P0/P1 items, and never use it as a correctness rule.
        target=max(1,int(getattr(self.cfg,'target_active_items',8)))
        hard=max(target,int(getattr(self.cfg,'hard_active_items',12)))
        if self.enable_lifecycle and len(ranked)>target:
            must=[r for r in ranked if r[-1].pinned]
            optional=[r for r in ranked if not r[-1].pinned]
            keep_optional=max(0,max(target-len(must), min(len(optional),hard-len(must))))
            keep_ids={r[-1].context_id for r in must+optional[:keep_optional]}
            new=[]
            for r in ranked:
                if r[-1].context_id in keep_ids: new.append(r)
                else:
                    dropped.append({'id':r[-1].context_id,'reason':'soft_eviction','chars':r[-1].chars,'kind':r[-1].source_kind})
                    cold_count+=1; active_count=max(0,active_count-1)
            ranked=new

        selected=[]; used=0; selected_raw=set(); projections=[]
        obs_by_id={o.observation_id:o for o in observation_store.all()}
        for *_,x in ranked:
            projection=None
            if x.source_kind=='observation':
                obs=obs_by_id.get(x.context_id)
                projection=self._projection_for(obs,x,state.step) if obs is not None else None
                if projection is not None:
                    content=(f"OBSERVATION {x.context_id} projection={projection.projection_id}\n{x.compact_content}\n"
                             f"{projection.content}\nEND OBSERVATION {x.context_id}")
                else:
                    content=(f"OBSERVATION {x.context_id}\n{x.compact_content}\n{x.full_content}\nEND OBSERVATION {x.context_id}")
            elif x.source_kind=='evidence' and x.raw_observation_id and x.raw_observation_id in selected_raw:
                content=_evidence_reference(x)
            else:
                content=x.compact_content + (f"\n{x.full_content}" if x.full_content else '')
            size=len(content)+2
            if not self.enable_budget_packing or used+size<=available:
                selected.append((x,content,projection)); used+=size
                if x.source_kind=='observation': selected_raw.add(x.context_id)
                if projection: projections.append(projection)
            else:
                dropped.append({'id':x.context_id,'reason':'budget','chars':size,'kind':x.source_kind})

        rendered=[]; selected_meta=[]
        for x,content,projection in selected:
            if x.source_kind=='evidence' and x.raw_observation_id in selected_raw:
                content=_evidence_reference(x)
            rendered.append(content)
            meta={'id':x.context_id,'reason':('model_requested' if x.context_id in requested else x.metadata.get('selection_reason','priority')),
                  'chars':len(content),'kind':x.source_kind,'lifecycle':'active','pinned':x.pinned}
            if projection:
                meta.update({'projection_id':projection.projection_id,'display_start_line':projection.display_start_line,
                             'display_end_line':projection.display_end_line,'path':projection.path})
            selected_meta.append(meta)

        working='\n\n'.join(rendered) or '(no active working context yet)'
        text=f"{fixed}{known_section}\nWORKING_CONTEXT:\n{working}"
        # Packing is exact. If fixed text alone exceeds budget, trim the issue before corrupting display metadata.
        if len(text)>max_context_chars:
            overflow=len(text)-max_context_chars
            if overflow>0 and len(issue)>2000:
                reduced=max(2000,len(issue)-overflow-64)
                issue2=issue[:reduced]
                fixed=fixed.replace(issue,issue2,1)
                text=f"{fixed}{known_section}\nWORKING_CONTEXT:\n{working}"
        if len(text)>max_context_chars:
            # Last-resort: drop selected items from the end; never blind-slice a source projection.
            while selected_meta and len(text)>max_context_chars:
                dropped_id=selected_meta[-1]['id']; selected_meta.pop(); rendered.pop()
                dropped.append({'id':dropped_id,'reason':'defensive_budget_drop','chars':0})
                working='\n\n'.join(rendered) or '(no active working context yet)'
                text=f"{fixed}{known_section}\nWORKING_CONTEXT:\n{working}"

        # HARD REQUIREMENT: update display coverage only after final render/drop decisions.
        self.display_coverage.clear()
        kept_projection_ids={m.get('projection_id') for m in selected_meta if m.get('projection_id')}
        for p in projections:
            if p.projection_id not in kept_projection_ids or not p.path: continue
            # Parse exact numbered content; min/max metadata alone can hide gaps.
            ranges=[]; current=[]; prev=None
            for line in p.content.splitlines():
                if '|' not in line: continue
                head=line.split('|',1)[0].strip()
                if not head.isdigit(): continue
                n=int(head)
                if prev is None or n==prev+1: current.append(n)
                else:
                    if current: ranges.append((current[0],current[-1]))
                    current=[n]
                prev=n
            if current: ranges.append((current[0],current[-1]))
            for a,b in ranges: self.display_coverage.add(p.path,a,b,p.projection_id)

        current_ids={m['id'] for m in selected_meta}
        evicted=len(self._last_active_ids-current_ids) if self._last_active_ids else len([d for d in dropped if d.get('reason')=='soft_eviction'])
        self._eviction_total += evicted
        self._projection_total += len([p for p in projections if p.projection_id in kept_projection_ids])
        self._last_active_ids=current_ids
        self._last_selected_ids=current_ids
        self._last_lifecycle={x.context_id:('active' if x.context_id in current_ids else 'cold') for x in items}

        breakdown={
            'issue_chars':len(issue),
            'recent_actions_chars':len(recent_actions),
            'hypothesis_chars':len(str(hyp if hyp else '(none)')),
            'advisory_chars':len(advisory),
            'state_chars':len(str(state.to_summary())),
            'known_context_chars':len(known_section),
            'observation_chars':sum(m['chars'] for m in selected_meta if m.get('kind')=='observation'),
            'evidence_chars':sum(m['chars'] for m in selected_meta if m.get('kind')=='evidence'),
        }
        display=self.display_coverage.export()
        projection_count=sum(1 for m in selected_meta if m.get('projection_id'))
        self._rehydrate_requests.clear()
        return ContextBuildResult(text,max_context_chars,len(text),len(items),len(selected_meta),selected_meta,dropped,invalid,breakdown,
                                  len(known_section),active_count,cold_count,evicted,projection_count,display)
