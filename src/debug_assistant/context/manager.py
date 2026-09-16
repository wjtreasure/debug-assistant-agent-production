from __future__ import annotations
import re
from debug_assistant.context.models import ContextItem, ContextBuildResult, ContextProjection
from debug_assistant.context.indexes import DisplayCoverageIndex, KnownContextIndex, merge_ranges
from debug_assistant.context.packing import line_safe_truncate
from debug_assistant.context.projection import CodeProjectionPolicy
from debug_assistant.llm.base import estimate_tokens_char4


def _terms(value) -> set[str]:
    """Small deterministic relevance signal; never interprets benchmark labels."""
    return {
        token.lower() for token in re.findall(r"[A-Za-z0-9_.:/-]+", str(value or ""))
        if len(token) >= 3
    }


def _model_metadata(value):
    """Remove internal provenance identifiers from model-visible tool metadata."""
    if isinstance(value, dict):
        return {
            key: _model_metadata(item) for key, item in value.items()
            if "observation_id" not in str(key).lower() and str(key).lower() != "provenance"
        }
    if isinstance(value, list):
        return [_model_metadata(item) for item in value]
    return value


class ContextManager:
    """V1.3.2 task-local context lifecycle manager.

    Hard invariants:
    - Raw observations/evidence are never deleted by eviction.
    - KnownContextIndex preserves model-visible pointers to previously acquired source.
    - DisplayCoverageIndex reflects only source lines that made it into the final rendered prompt.
    - Rehydration projects the requested source range from raw immutable observations.
    """
    def __init__(self, cfg, *, enable_catalog=True, enable_model_selection=False, enable_budget_packing=True,
                 enable_lifecycle=True, enable_projection=True,
                 projection_policy=None):
        self.cfg=cfg
        self.enable_catalog=enable_catalog
        self.enable_model_selection=enable_model_selection
        self.enable_budget_packing=enable_budget_packing
        self.enable_lifecycle=enable_lifecycle
        self.enable_projection=enable_projection
        self.projection_policy=projection_policy or CodeProjectionPolicy()
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

    def _recent_observation_ids(self, observation_store) -> set[str]:
        """Apply the one shared recency window before relevance/lifecycle ranking."""
        count=max(0,int(getattr(self.cfg,'fallback_recent_count',2)))
        char_budget=max(0,int(getattr(self.cfg,'fallback_recent_chars',16000)))
        selected=[]; used=0
        for observation in reversed(observation_store.all()):
            if len(selected)>=count:
                break
            size=len(observation.content or '')
            if selected and used+size>char_budget:
                continue
            if not selected or used+size<=char_budget:
                selected.append(observation.observation_id); used+=size
        return set(selected)

    def _obs_item(self, obs, step: int, reason: str, priority: int, lifecycle:str='active', pinned:bool=False,
                  evidence=None) -> ContextItem:
        full=obs.content
        superseded_count=0
        if obs.tool in {'grep','code_search','symbol_search'}:
            full,superseded_count=self.known_index.filter_search_content(full)
        display,trunc=line_safe_truncate(full,self.cfg.max_item_chars)
        meta=dict(obs.metadata or {})
        meta.update({"ok":obs.ok,"error_type":obs.error_type,"context_truncated":trunc,
                     "raw_chars":len(obs.content),"display_chars":len(display),"selection_reason":reason,
                     "range_superseded_hits":superseded_count})
        title=(evidence.file or evidence.target or evidence.source) if evidence is not None else f"{obs.tool} {meta.get('path','')}".strip()
        context_id=evidence.evidence_id if evidence is not None else obs.observation_id
        source_kind="evidence" if evidence is not None else "tool_result"
        compact=(f"[{context_id}] kind={evidence.kind} source={evidence.source} location={title}"
                 if evidence is not None else
                 f"tool={obs.tool} ok={obs.ok} error_type={obs.error_type or 'none'} metadata={_model_metadata(meta)}")
        level = (
            "L1" if evidence is not None and reason in {
                "hypothesis_support", "contradiction_source", "observation_reused",
            } else "L1" if evidence is not None else "L3" if lifecycle == "cold" else "L2"
        )
        return ContextItem(context_id,source_kind,title,compact,display,len(display),priority,step,
                           obs.observation_id,meta,lifecycle,pinned,step if lifecycle=='active' else 0,
                           level)

    def _ev_item(self, ev, step: int, priority: int, reason: str, lifecycle:str='active', pinned:bool=False) -> ContextItem:
        loc=ev.file or ev.target or ev.source
        if ev.source_start_line is not None and ev.source_end_line is not None:
            loc=f"{loc}:{ev.source_start_line}-{ev.source_end_line}"
        compact=f"[{ev.evidence_id}] {ev.kind} {loc}: {ev.summary}"
        level = "L1" if reason in {"hypothesis_support", "contradiction", "contradiction_source"} else "L3" if lifecycle == "cold" else "L2"
        return ContextItem(ev.evidence_id,"evidence",loc,compact,ev.excerpt,len(ev.excerpt),priority,step,
                           ev.raw_observation_id,{"selection_reason":reason},lifecycle,pinned,
                           step if lifecycle=='active' else 0, level)

    def catalog(self, state, memory, observation_store) -> list[ContextItem]:
        # Rebuild before item rendering so search observations can be superseded at hit/range level.
        self.known_index.rebuild(observation_store)
        items=[]
        latest=state.observations[-1].observation_id if state.observations else None
        hyp=state.current_hypothesis or {}
        support=set(hyp.get('supporting_evidence_ids') or [])
        contradict=set(hyp.get('contradicting_evidence_ids') or [])
        recent_ids=self._recent_observation_ids(observation_store)
        recent_evidence_ids={
            mapped.evidence_id for observation_id in recent_ids
            if (mapped := memory.evidence_for_observation(observation_id)) is not None
        }
        latest_evidence=memory.evidence_for_observation(latest) if latest else None
        rehydrate_ids=set(self._rehydrate_requests)
        relevance_terms=_terms(hyp)
        represented_evidence_ids=set()

        for obs in observation_store.all():
            ev=memory.evidence_for_observation(obs.observation_id)
            evidence_id=ev.evidence_id if ev is not None else None
            if evidence_id is not None and evidence_id in represented_evidence_ids:
                # Repeated tool results may point at the same canonical fact. Keep
                # all raw Observations in the Store, but spend prompt budget once.
                continue
            if obs.observation_id in rehydrate_ids:
                p,reason,lifecycle,pinned=5,"observation_reused",'active',True
            elif obs.observation_id == latest or (
                latest_evidence is not None and evidence_id == latest_evidence.evidence_id
            ):
                p,reason,lifecycle,pinned=10,"latest_observation",'active',True
            elif evidence_id in contradict:
                p,reason,lifecycle,pinned=12,"contradiction_source",'active',True
            elif evidence_id in support:
                p,reason,lifecycle,pinned=18,"hypothesis_support",'active',True
            elif not obs.ok:
                p,reason,lifecycle,pinned=30,"tool_error",'active',False
            elif ev is not None and relevance_terms & _terms((ev.target, ev.kind, ev.summary)):
                p,reason,lifecycle,pinned=38,"hypothesis_relevant",'active',False
            elif obs.observation_id in recent_ids or evidence_id in recent_evidence_ids:
                p,reason,lifecycle,pinned=45,"recent_observation",'active',False
            else:
                p,reason,lifecycle,pinned=90,"historical_observation",'cold',False
            if not self.enable_lifecycle: lifecycle='active'
            items.append(self._obs_item(obs,state.step,reason,p,lifecycle,pinned,evidence=ev))
            if ev is not None:
                represented_evidence_ids.add(ev.evidence_id)

        # Evidence without a retained raw Observation remains available as compact
        # provenance.  Evidence backed by the Store is already represented exactly once.
        for ev in memory.pinned:
            if ev.evidence_id in represented_evidence_ids:
                continue
            if ev.evidence_id in contradict:
                p,reason,lifecycle,pinned=8,"contradiction",'active',True
            elif ev.evidence_id in support:
                p,reason,lifecycle,pinned=18,"hypothesis_support",'active',True
            elif relevance_terms & _terms((ev.target, ev.kind, ev.summary)):
                p,reason,lifecycle,pinned=40,"hypothesis_relevant",'active',False
            else:
                p,reason,lifecycle,pinned=75,"historical_evidence",('cold' if self.enable_lifecycle else 'active'),False
            items.append(self._ev_item(ev,state.step,p,reason,lifecycle,pinned))
        return items

    def known_context_text(self, observation_store, max_chars:int|None=None) -> str:
        self.known_index.rebuild(observation_store)
        return self.known_index.render(max_chars=max_chars or getattr(self.cfg,'known_index_max_chars',3500))

    def read_ledger_text(self, memory, observation_store, max_chars: int | None = None) -> tuple[str, int]:
        """Render a bounded ledger of source ranges already read by ``read_file``.

        The ledger is deliberately metadata-only.  CODE Evidence may be evicted from
        the working context to satisfy the token budget, but the planner must still
        know which immutable source ranges were acquired so it does not rediscover
        and reread them solely because compaction hid their contents.
        """
        limit = max(0, int(max_chars if max_chars is not None else
                            getattr(self.cfg, 'known_index_max_chars', 3500)))
        rows = []
        seen = set()
        for observation in observation_store.all():
            if not observation.ok or observation.tool != 'read_file':
                continue
            metadata = observation.metadata or {}
            if str(metadata.get('context_kind') or '').upper() != 'CODE':
                continue
            path = str(metadata.get('path') or '').strip()
            start = metadata.get('start_line')
            end = metadata.get('end_line')
            count = metadata.get('requested_line_count')
            if not path or not isinstance(start, int) or not isinstance(end, int):
                continue
            if not isinstance(count, int) or count < 1:
                count = end - start + 1
            evidence = memory.evidence_for_observation(observation.observation_id)
            evidence_id = evidence.evidence_id if evidence is not None else 'unpromoted'
            key = (path, start, count)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                f"- {path} L{start}-{end} already read; Evidence={evidence_id}\n"
            )
        if not rows:
            return '', 0
        rendered = ''.join(rows)
        if limit:
            rendered, _ = line_safe_truncate(rendered, limit)
        else:
            rendered = ''
        return rendered, len(rows)

    def catalog_text(self, items: list[ContextItem], max_chars: int=7000) -> str:
        rows=[]; used=0
        for x in sorted(items,key=lambda i:(i.created_step,i.context_id)):
            if x.source_kind == 'tool_result':
                continue
            loc=x.title
            row=f"- {x.context_id} type={x.source_kind} location={loc} lifecycle={x.lifecycle} priority={x.priority}\n"
            if used+len(row)>max_chars: break
            rows.append(row); used+=len(row)
        return ''.join(rows) or '(none)'

    def _projection_for(self, obs, item:ContextItem, step:int) -> ContextProjection|None:
        if not self.enable_projection:
            return None
        return self.projection_policy.project(
            obs, item, step,
            requests=self._rehydrate_requests.get(obs.observation_id) or [],
            rehydrate_requested=obs.observation_id in self._rehydrate_requests,
        )

    def build(self, state, memory, observation_store, *, max_context_chars: int | None = None,
              max_context_tokens: int | None = None, token_estimator=None,
              max_steps=None, max_tool_calls=None, requested_ids=None,
              include_agent_control_state: bool=True,
              external_context_chars: int=0, pressure_state: str = "NORMAL") -> ContextBuildResult:
        if external_context_chars < 0:
            raise ValueError("external_context_chars must be non-negative")
        if max_context_chars is None and max_context_tokens is None:
            raise ValueError("one of max_context_chars or max_context_tokens is required")
        estimator = token_estimator or estimate_tokens_char4
        total_token_budget = max(1, int(max_context_tokens)) if max_context_tokens is not None else None
        external_tokens = max(0, int(estimator("x" * external_context_chars)))
        if total_token_budget is not None:
            diagnostic_token_budget = max(0, total_token_budget - external_tokens)
            diagnostic_budget = diagnostic_token_budget * 4
            budget_chars = diagnostic_budget + external_context_chars
        else:
            budget_chars = int(max_context_chars)
            diagnostic_budget=max(0,budget_chars-external_context_chars)
            diagnostic_token_budget = None
        requested_ids=list(requested_ids or []) if self.enable_model_selection else []
        # Evidence-aware projection: compact Evidence excerpts may truthfully represent only
        # the beginning of a larger read_file observation. If a truncated read is currently
        # hypothesis support, re-project its immutable raw source range instead of letting the
        # model infer that unseen tail lines were never read. This is still bounded by the
        # normal context packer and never performs repository I/O.
        hyp0=state.current_hypothesis or {}
        support0=set(hyp0.get('supporting_evidence_ids') or [])
        if (self.enable_projection and support0
                and getattr(self.projection_policy, 'supports_source_ranges', False)):
            for ev in memory.pinned:
                if (ev.evidence_id in support0 and ev.source == 'read_file' and ev.excerpt_truncated
                        and ev.raw_observation_id and ev.file
                        and isinstance(ev.source_start_line,int) and isinstance(ev.source_end_line,int)):
                    self.rehydrate(ev.raw_observation_id,path=ev.file,start_line=ev.source_start_line,
                                   end_line=ev.source_end_line,information_need='hypothesis_support_projection')
        items=self.catalog(state,memory,observation_store)
        by_id={x.context_id:x for x in items}
        invalid=[x for x in requested_ids if x not in by_id]
        requested={x for x in requested_ids if x in by_id}
        issue_budget=max(2000,min(diagnostic_budget//3,18000))
        issue=state.task.issue[:issue_budget] if include_agent_control_state else ''
        recent_actions=('\n'.join(f"- {a.skill}/{a.tool or a.kind.value}: {a.reason}" for a in state.actions[-6:]) or '(none)') if include_agent_control_state else ''
        budget=[]
        if include_agent_control_state and max_steps is not None:
            budget += [f"step={state.step}/{max_steps}",f"remaining_steps={max(0,max_steps-state.step)}"]
        if include_agent_control_state and max_tool_calls is not None:
            budget += [f"tool_calls={state.tool_calls}/{max_tool_calls}",f"remaining_tool_calls={max(0,max_tool_calls-state.tool_calls)}"]
        hyp=state.current_hypothesis or {}
        advisory=(state.termination_advisory or '').strip() if include_agent_control_state else ''
        fixed=((f"TASK_ID: {state.task.task_id}\nISSUE:\n{issue}\n\nRECENT_ACTIONS:\n{recent_actions}\n\n"
               f"RUNTIME_BUDGET: {', '.join(budget) or 'not configured'}\n"
               f"CURRENT_HYPOTHESIS: {hyp if hyp else '(none)'}\n"
               f"TERMINATION_ADVISORY: {advisory or '(none)'}\nSTATE: {state.to_summary()}\n\n")
               if include_agent_control_state else '')

        if not self.enable_catalog:
            known_section=""
        elif getattr(self.projection_policy, 'catalog_mode', 'source_ranges') == 'items':
            known=self.catalog_text(items,max_chars=getattr(self.cfg,'known_index_max_chars',3500))
            read_ledger, read_ledger_count = self.read_ledger_text(
                memory, observation_store,
                max_chars=getattr(self.cfg, 'known_index_max_chars', 3500),
            )
            read_ledger_section = (
                "READ_FILE_LEDGER (bounded metadata; source ranges already acquired; "
                "do not repeat an equivalent read solely because Evidence is cold):\n"
                f"{read_ledger or '(none)'}"
            )
            known_section=("KNOWN_CONTEXT_INDEX (diagnostic pointers; content may be cold):\n"
                           f"{known}\nOnly ev-* identifiers are citable. Raw observations remain internal and can be rehydrated by the Harness.\n")
            known_section += read_ledger_section
        else:
            known=self.known_context_text(observation_store)
            evidence_catalog=self.catalog_text(items,max_chars=getattr(self.cfg,'known_index_max_chars',3500))
            read_ledger, read_ledger_count = self.read_ledger_text(
                memory, observation_store,
                max_chars=getattr(self.cfg, 'known_index_max_chars', 3500),
            )
            read_ledger_section = (
                "READ_FILE_LEDGER (bounded metadata; source ranges already acquired; "
                "do not repeat an equivalent read solely because Evidence is cold):\n"
                f"{read_ledger or '(none)'}"
            )
            known_section=("KNOWN_CONTEXT_INDEX (compact pointers; content may be cold):\n"
                           f"{known}\nEVIDENCE_CATALOG (only ev-* identifiers are citable):\n{evidence_catalog}"
                           f"{read_ledger_section}"
                           "If details are needed from a known range, request read_file for the exact range; the Harness can rehydrate it without repository I/O.\n")
        available=max(0,diagnostic_budget-len(fixed)-len(known_section)-self.cfg.safety_margin_chars)

        ranked=[]; dropped=[]; active_count=0; cold_count=0
        for x in items:
            if x.lifecycle=='cold' and x.context_id not in requested:
                cold_count+=1
                dropped.append({'id':x.context_id,'reason':'cold','chars':x.chars,'kind':x.source_kind})
                continue
            if pressure_state == "HARD_PRESSURE" and x.context_level in {"L2", "L3"} and not x.pinned and x.context_id not in requested:
                cold_count += 1
                dropped.append({'id':x.context_id,'reason':'hard_pressure','chars':x.chars,'kind':x.source_kind})
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

        selected=[]; used=0; projections=[]
        obs_by_id={o.observation_id:o for o in observation_store.all()}
        for *_,x in ranked:
            projection=None
            if x.raw_observation_id:
                obs=obs_by_id.get(x.raw_observation_id)
                projection=self._projection_for(obs,x,state.step) if obs is not None else None
                body=projection.content if projection is not None else x.full_content
                if x.source_kind == 'evidence':
                    content=(f"EVIDENCE {x.context_id} projection={projection.projection_id if projection else 'none'}\n"
                             f"{x.compact_content}\n{body}\nEND EVIDENCE {x.context_id}")
                else:
                    content=("TOOL_RESULT (not Evidence; do not cite)\n"
                             f"{x.compact_content}\n{body}\nEND TOOL_RESULT")
            else:
                content=x.compact_content + (f"\n{x.full_content}" if x.full_content else '')
            size=len(content)+2
            if not self.enable_budget_packing or used+size<=available:
                selected.append((x,content,projection)); used+=size
                if projection: projections.append(projection)
            else:
                dropped.append({'id':x.context_id,'reason':'budget','chars':size,'kind':x.source_kind})

        rendered=[]; selected_meta=[]
        for x,content,projection in selected:
            rendered.append(content)
            meta={'id':x.context_id,'reason':('model_requested' if x.context_id in requested else x.metadata.get('selection_reason','priority')),
                  'chars':len(content),'kind':x.source_kind,'lifecycle':'active','pinned':x.pinned,
                  'citable':x.source_kind == 'evidence','context_level':x.context_level}
            if x.raw_observation_id:
                meta['provenance_observation_id']=x.raw_observation_id
            if projection:
                meta.update({'projection_id':projection.projection_id,'display_start_line':projection.display_start_line,
                             'display_end_line':projection.display_end_line,'path':projection.path,
                             'projection_reason':projection.reason})
            selected_meta.append(meta)

        working='\n\n'.join(rendered) or '(no active working context yet)'
        text=f"{fixed}{known_section}\nWORKING_CONTEXT:\n{working}"
        # Packing is exact. If fixed text alone exceeds budget, trim the issue before corrupting display metadata.
        if len(text)>diagnostic_budget:
            overflow=len(text)-diagnostic_budget
            if overflow>0 and len(issue)>2000:
                reduced=max(2000,len(issue)-overflow-64)
                issue2=issue[:reduced]
                fixed=fixed.replace(issue,issue2,1)
                text=f"{fixed}{known_section}\nWORKING_CONTEXT:\n{working}"
        scaffolding=f"\nWORKING_CONTEXT:\n{working}"
        # Character packing is the compatibility fast path.  A physical model
        # capability is authoritative, so apply the token bound after the full
        # diagnostic text has been rendered as well.
        if diagnostic_token_budget is not None:
            def total_tokens(value: str) -> int:
                return max(0, int(estimator(value))) + external_tokens
            while selected_meta and total_tokens(text) > total_token_budget:
                removable = next(
                    (index for index in range(len(selected_meta) - 1, -1, -1)
                     if not selected_meta[index].get('pinned')),
                    None,
                )
                if removable is None:
                    break
                dropped_id = selected_meta[removable]['id']
                selected_meta.pop(removable)
                rendered.pop(removable)
                dropped.append({'id': dropped_id, 'reason': 'token_budget', 'chars': 0})
                working = '\n\n'.join(rendered) or '(no active working context yet)'
                text = f"{fixed}{known_section}\nWORKING_CONTEXT:\n{working}"
                scaffolding = f"\nWORKING_CONTEXT:\n{working}"
            if total_tokens(text) > total_token_budget and known_section:
                known_allow = max(0, diagnostic_budget - len(fixed) - len(scaffolding))
                known_section, _ = line_safe_truncate(known_section, known_allow)
                text = f"{fixed}{known_section}{scaffolding}"
        if len(text)>diagnostic_budget:
            # Last-resort: drop selected items from the end; never blind-slice a source projection.
            while selected_meta and len(text)>diagnostic_budget:
                dropped_id=selected_meta[-1]['id']; selected_meta.pop(); rendered.pop()
                dropped.append({'id':dropped_id,'reason':'defensive_budget_drop','chars':0})
                working='\n\n'.join(rendered) or '(no active working context yet)'
                text=f"{fixed}{known_section}\nWORKING_CONTEXT:\n{working}"
        scaffolding=f"\nWORKING_CONTEXT:\n{working}"
        if len(text)>diagnostic_budget and known_section:
            known_allow=max(0,diagnostic_budget-len(fixed)-len(scaffolding))
            known_section,_=line_safe_truncate(known_section,known_allow)
            text=f"{fixed}{known_section}{scaffolding}"
        control_state_truncated=False
        if len(text)>diagnostic_budget:
            # This can occur only when external/control state consumes almost the
            # entire global budget. Source projections were already dropped whole;
            # compact the manager-owned control scaffold without slicing Evidence.
            fixed_allow=max(0,diagnostic_budget-len(scaffolding))
            fixed,control_state_truncated=line_safe_truncate(fixed,fixed_allow)
            text=f"{fixed}{scaffolding}"
        if len(text)>diagnostic_budget:
            text="(diagnostic context omitted: global budget reserved for agent control)"[:diagnostic_budget]
            control_state_truncated=True

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

        # Quality telemetry is intentionally separate from token reduction. A
        # shorter context is not an improvement if critical Evidence or an
        # open verification obligation disappears. The Runtime control prefix
        # is counted as the authoritative external state when the manager is
        # called in its split-control mode.
        current_hypothesis = state.current_hypothesis or {}
        critical_evidence_ids = set(current_hypothesis.get('supporting_evidence_ids') or [])
        critical_evidence_ids.update(current_hypothesis.get('contradicting_evidence_ids') or [])
        for obligation in current_hypothesis.get('verification_obligations') or ():
            if isinstance(obligation, dict):
                critical_evidence_ids.update(obligation.get('supporting_evidence_ids') or [])
        selected_evidence_ids = {
            str(item.get('id')) for item in selected_meta if item.get('citable')
        }
        critical_retained = (
            len(critical_evidence_ids.intersection(selected_evidence_ids))
            / len(critical_evidence_ids)
            if critical_evidence_ids else 1.0
        )
        open_obligation_ids = {
            str(item.get('id')) for item in current_hypothesis.get('verification_obligations') or ()
            if isinstance(item, dict) and item.get('status') == 'OPEN'
            and item.get('blocks_finalization')
        }
        blocking_contradiction_ids = {
            str(item.get('evidence_id')) for item in current_hypothesis.get('contradictions') or ()
            if isinstance(item, dict) and item.get('status') == 'OPEN'
            and item.get('blocks_finalization')
        }
        # In split-control mode these IDs are rendered by the external
        # AGENT_CONTROL_STATE/PLANNER_STATE prefix, so retention is evaluated
        # against the complete planner input rather than manager text alone.
        planner_input_has_external_state = external_context_chars > 0 and not include_agent_control_state
        obligation_retention = (
            sum(1 for oid in open_obligation_ids if oid in text or planner_input_has_external_state)
            / len(open_obligation_ids) if open_obligation_ids else 1.0
        )
        contradiction_retention = (
            sum(1 for cid in blocking_contradiction_ids if cid in text or planner_input_has_external_state)
            / len(blocking_contradiction_ids) if blocking_contradiction_ids else 1.0
        )
        rehydrate_requested_count = len(self._rehydrate_requests)
        rehydrated_success_count = sum(
            1 for item in selected_meta
            if str(item.get('projection_reason', '')).startswith('rehydrated_')
        )
        duplicate_observation_count = sum(
            1 for observation in observation_store.all()
            if (memory.evidence_for_observation(observation.observation_id) is not None
                and memory.evidence_for_observation(observation.observation_id).raw_observation_id
                != observation.observation_id)
        )

        breakdown={
            'issue_chars':len(issue),
            'recent_actions_chars':len(recent_actions),
            'hypothesis_chars':len(str(hyp if hyp else '(none)')),
            'advisory_chars':len(advisory),
            'state_chars':len(str(state.to_summary())),
            'known_context_chars':len(known_section),
            'read_ledger_count': read_ledger_count if 'read_ledger_count' in locals() else 0,
            'tool_result_chars':sum(m['chars'] for m in selected_meta if m.get('kind')=='tool_result'),
            'evidence_chars':sum(m['chars'] for m in selected_meta if m.get('kind')=='evidence'),
            'diagnostic_context_chars':len(text),
            'external_context_chars':external_context_chars,
            'total_context_chars':len(text)+external_context_chars,
            'selected_evidence_count':sum(1 for m in selected_meta if m.get('citable')),
            'dropped_evidence_count':sum(1 for d in dropped if d.get('kind')=='evidence'),
            'rehydrated_item_count':sum(
                1 for m in selected_meta if m.get('projection_reason','').startswith('rehydrated_')
            ),
            'duplicate_observations_collapsed':sum(
                1 for _ in range(duplicate_observation_count)
            ),
            'manager_control_state_truncated':int(control_state_truncated),
            'critical_evidence_retention': critical_retained,
            'open_obligation_retention': obligation_retention,
            'blocking_contradiction_retention': contradiction_retention,
            'duplicate_context_ratio': (
                duplicate_observation_count / max(1, len(observation_store.all()))
            ),
            'rehydration_requested_count': rehydrate_requested_count,
            'rehydration_success_count': rehydrated_success_count,
            'rehydration_success_rate': (
                rehydrated_success_count / rehydrate_requested_count
                if rehydrate_requested_count else 1.0
            ),
        }
        diagnostic_tokens = max(0, int(estimator(text)))
        breakdown.update({
            'diagnostic_context_tokens': diagnostic_tokens,
            'external_context_tokens': external_tokens,
            'total_context_tokens': diagnostic_tokens + external_tokens,
            'budget_tokens': total_token_budget if total_token_budget is not None else 0,
        })
        display=self.display_coverage.export()
        projection_count=sum(1 for m in selected_meta if m.get('projection_id'))
        self._rehydrate_requests.clear()
        return ContextBuildResult(
            text, budget_chars, len(text) + external_context_chars, len(items),
            len(selected_meta), selected_meta, dropped, invalid, breakdown,
            len(known_section), active_count, cold_count, evicted, projection_count,
            display, total_token_budget, diagnostic_tokens + external_tokens,
            getattr(getattr(self.cfg, "tokenizer", None), "name", "char4")
            if not callable(token_estimator) else "custom",
        )
