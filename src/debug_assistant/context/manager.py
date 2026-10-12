from __future__ import annotations
import re
import json
import hashlib
from debug_assistant.context.models import (
    ContextItem, ContextBuildResult, ContextProjection, PromptSnapshot, ContextView,
)
from debug_assistant.context.indexes import (DisplayCoverageIndex, KnownContextIndex, merge_ranges,
                                            retained_source_lines)
from debug_assistant.context.requests import ContextRequest, ContextRequestBatch, ContextResolutionError, LineRange, LocalResolutionFeedback
from debug_assistant.context.packing import line_safe_truncate
from debug_assistant.context.projection import CodeProjectionPolicy
from debug_assistant.context.task_projection import project_verification_tasks
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
                  end_line:int|None=None, information_need:str="", observation_store=None) -> str:
        """Queue an immutable Observation projection and report the request state.

        ``REQUESTED`` means the next build may re-project the Observation;
        ``NOT_FOUND`` and ``DENIED`` are explicit fail-closed outcomes. The
        active Runtime passes the immutable ObservationStore when available.
        """
        if not str(observation_id or "").strip() or not self.enable_projection:
            return "DENIED"
        if observation_store is not None and observation_store.get(observation_id) is None:
            return "NOT_FOUND"
        if path and isinstance(start_line,int) and isinstance(end_line,int):
            reqs=self._rehydrate_requests.setdefault(observation_id,[])
            reqs.append((start_line,end_line,information_need))
            # Coalesce only source ranges; information_need is telemetry, not identity here.
            merged=merge_ranges([(a,b) for a,b,_ in reqs])
            self._rehydrate_requests[observation_id]=[(a,b,information_need) for a,b in merged]
        else:
            self._rehydrate_requests.setdefault(observation_id,[])
        return "REQUESTED"

    def is_visible(self, context_id: str) -> bool:
        return context_id in self._last_selected_ids

    def rehydrate_evidence(self, evidence_id: str, *, evidence_memory, observation_store,
                           start_line: int | None = None, end_line: int | None = None) -> str:
        """Resolve exact canonical provenance into the existing bounded projector."""
        evidence = evidence_memory.get(evidence_id)
        if evidence is None:
            return "NOT_FOUND"
        return self.rehydrate(evidence.raw_observation_id, path=evidence.file,
                              start_line=start_line, end_line=end_line,
                              observation_store=observation_store)

    def is_visible_range(self, path:str,start_line:int,end_line:int) -> bool:
        return self.display_coverage.covers(path,start_line,end_line)

    def _local_target(self, request, memory, observation_store):
        observation_id = request.observation_id
        if request.evidence_ref is not None:
            evidence = memory.get(request.evidence_ref)
            if evidence is None:
                raise ContextResolutionError('context_unknown_evidence_ref')
            observation_id = evidence.raw_observation_id
            observation = observation_store.get(observation_id) if observation_id else None
            provenance = evidence.provenance or {}
            if (observation is None or evidence.source != observation.tool
                    or provenance.get('observation_id', observation_id) != observation_id
                    or any(provenance[key] != (observation.metadata or {}).get(key)
                           for key in ('path', 'start_line', 'end_line', 'context_kind', 'information_source')
                           if key in provenance)
                    or evidence.file is not None and evidence.file != (observation.metadata or {}).get('path')):
                raise ContextResolutionError('context_provenance_invalid')
        else:
            observation = observation_store.get(observation_id)
            if observation is None:
                raise ContextResolutionError('context_unknown_observation_id')
        return observation

    @staticmethod
    def _local_content(observation, selector):
        if selector.kind == 'FIELD_PATH':
            # ToolObservation has canonical text + metadata, no structured body
            # field. Do not parse display JSON/YAML into a second payload model.
            raise ContextResolutionError('context_selector_unsupported')
        if not observation.content:
            raise ContextResolutionError('context_content_not_retained')
        source = observation.tool in {'read_file', 'get_source_code'}
        lines = retained_source_lines(observation) if source else list(enumerate(observation.content.splitlines(), 1))
        if source and not lines:
            raise ContextResolutionError('context_content_not_retained')
        if selector.kind == 'FULL_STORED_BODY':
            return ('\n'.join(text for _, text in lines) if source else observation.content), [(lines[0][0], lines[-1][0])]
        chosen = [(number, text) for number, text in lines if selector.start_line <= number <= selector.end_line]
        if len(chosen) != selector.end_line - selector.start_line + 1:
            raise ContextResolutionError('context_content_not_retained')
        return '\n'.join(text for _, text in chosen), [(selector.start_line, selector.end_line)]

    @staticmethod
    def _request_visible(view, observation_id, selector, ranges):
        coverage = json.loads(view.body_coverage_json).get(observation_id, {})
        if selector.kind == 'FULL_STORED_BODY':
            return bool(coverage.get('full'))
        return all(any(a <= start and b >= end for a, b in coverage.get('ranges', ()))
                   for start, end in ranges)

    def retained_local_requests(self, view, memory, observation_store):
        """Carry bounded visible selections, never bodies or visibility claims."""
        requests = {}
        for receipt in view.reference_metadata.get('resolved_local_requests', ()):
            request = ContextRequest.model_validate(receipt)
            observation = self._local_target(request, memory, observation_store)
            _, ranges = self._local_content(observation, request.selector)
            if self._request_visible(view, observation.observation_id, request.selector, ranges):
                requests.setdefault(observation.observation_id, []).append(request.selector)
        # The existing per-batch reference bound also bounds carried selections.
        return {key: requests[key] for key in list(requests)[-8:]}

    @staticmethod
    def _body_wrapper(item, body, projection_id):
        if item.source_kind == 'evidence':
            return (f"EVIDENCE {item.context_id} projection={projection_id}\n"
                    f"observation_ref={item.raw_observation_id}\n{item.compact_content}\n{body}\nEND EVIDENCE {item.context_id}")
        return ("TOOL_RESULT (not Evidence; do not cite)\n"
                f"observation_ref={item.raw_observation_id}\n{item.compact_content}\n{body}\nEND TOOL_RESULT")

    def resolve_context_requests(self, batch: ContextRequestBatch, *, view, state, memory,
                                 observation_store, build_kwargs):
        """Validate/extract locally, use the sole packer, then prove display.

        Requests are transient build inputs, never a body cache or store write.
        Any partial packing failure is an insufficiency, not recovery success.
        """
        # Detached selector receipts belong to this role view only. Preserve
        # successful recovery in the same logical turn; a fresh build expires
        # them. Immutable stores remain the only body sources.
        requests = {}
        targets = []
        for receipt in view.reference_metadata.get('resolved_local_requests', ()):
            previous = ContextRequest.model_validate(receipt)
            observation = self._local_target(previous, memory, observation_store)
            _, ranges = self._local_content(observation, previous.selector)
            requests.setdefault(observation.observation_id, []).append(previous.selector)
            targets.append((observation.observation_id, previous.selector, ranges))
        all_visible = True
        for request in batch.requests:
            observation = self._local_target(request, memory, observation_store)
            _, ranges = self._local_content(observation, request.selector)
            targets.append((observation.observation_id, request.selector, ranges))
            all_visible &= self._request_visible(view, observation.observation_id, request.selector, ranges)
            requests.setdefault(observation.observation_id, []).append(request.selector)
        if all_visible:
            raise ContextResolutionError('context_already_visible')
        result = self.build(state, memory, observation_store, local_requests=requests,
                            **{key: value for key, value in build_kwargs.items() if key != 'local_requests'})
        if not all(self._request_visible(result.context_view, oid, selector, ranges)
                   for oid, selector, ranges in targets):
            catalog = {row['observation_id']: row for row in result.context_view.reference_metadata['local_context_references']}
            feedback_rows = []
            for oid, selector, _ in targets:
                observation = observation_store.get(oid)
                row = catalog.get(oid)
                if row is None:
                    lines = (retained_source_lines(observation) if observation.tool in {'read_file', 'get_source_code'}
                             else list(enumerate(observation.content.splitlines(), 1)))
                    # A bounded catalog may omit the target. Still report canonical
                    # size/ranges; zero is a safe lower capacity bound, not a fit claim.
                    row = {'content_sha256': hashlib.sha256(observation.content.encode()).hexdigest(),
                           'tool': observation.tool, 'path': (observation.metadata or {}).get('path'),
                           'content_kind': (observation.metadata or {}).get('context_kind', 'TEXT'),
                           'retained_lines': [lines[0][0], lines[-1][0]],
                           'retrievable_capacity_chars': 0, 'capacity_basis': 'catalog_omitted_conservative_bound',
                           'chunks': [{'start_line': lines[0][0], 'end_line': lines[0][0],
                                       'normalized_chars': len(lines[0][1]), 'fits': False}]}
                feedback_rows.append({'observation_id': oid, **row, 'selector': selector.model_dump(mode='json'),
                    'requested_normalized_chars': len(self._local_content(observation, selector)[0])})
            feedback = LocalResolutionFeedback(reason='context_packing_insufficient', requests=tuple(feedback_rows))
            raise ContextResolutionError('context_packing_insufficient', feedback=feedback.model_dump(mode='json'))
        return result

    @staticmethod
    def legal_recovery_choices(view):
        """Detached, owner-derived selectors; never infer retention from a ref."""
        coverage = json.loads(view.body_coverage_json)
        result = {}
        for row in view.reference_metadata.get('local_context_references', ()):
            oid = row['observation_id']
            displayed = coverage.get(oid, {})
            if displayed.get('full'):
                continue
            selectors = []
            # A partial body requests only missing lines, never the full body.
            if not displayed.get('ranges') and row.get('full_body_fits'):
                selectors.append({'kind': 'FULL_STORED_BODY'})
            for chunk in row.get('chunks', ()):
                if not chunk.get('fits'):
                    continue
                missing = [(chunk['start_line'], chunk['end_line'])]
                for a, b in displayed.get('ranges', ()):
                    missing = [(x, y) for start, end in missing for x, y in
                               ((start, min(end, a-1)), (max(start, b+1), end)) if x <= y]
                selectors.extend({'kind': 'LINE_RANGE', 'start_line': a, 'end_line': b}
                                 for a, b in missing)
            if selectors:
                result[oid] = selectors
        return result

    def has_local_recovery(self, view, observation_store):
        """Availability only; Runtime never selects a target or selector."""
        coverage = json.loads(view.body_coverage_json)
        return any(observation.content
                   and not coverage.get(observation.observation_id, {}).get('full')
                   and (observation.tool not in {'read_file', 'get_source_code'}
                        or retained_source_lines(observation))
                   for observation in observation_store.all())

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

    def catalog(self, state, memory, observation_store, *, semantic_projection=None, local_ids=()) -> list[ContextItem]:
        # Rebuild before item rendering so search observations can be superseded at hit/range level.
        self.known_index.rebuild(observation_store)
        items=[]
        latest=state.observations[-1].observation_id if state.observations else None
        hyp=(semantic_projection or {}).get("active_hypothesis") or {}
        hyp=hyp or {}
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

        for obs in sorted(observation_store.all(), key=lambda item: item.observation_id not in local_ids):
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
              pressure_state: str = "NORMAL",
              verification_dag_state=None, rca_state=None, role="interpretation",
              evidence_references=(), request_renderer=None, max_request_chars=None,
              local_requests=None, reference_ids=()) -> ContextBuildResult:
        if max_context_chars is None and max_context_tokens is None:
            raise ValueError("one of max_context_chars or max_context_tokens is required")
        estimator = token_estimator or estimate_tokens_char4
        total_token_budget = max(1, int(max_context_tokens)) if max_context_tokens is not None else None
        diagnostic_token_budget = total_token_budget
        budget_chars = (total_token_budget * 4 if total_token_budget is not None else int(max_context_chars))
        diagnostic_budget = budget_chars
        semantic_projection = None
        if rca_state is not None:
            projector = getattr(rca_state, "context_projection", None)
            if not callable(projector):
                raise TypeError("rca_state must expose context_projection()")
            semantic_projection = projector()
            if verification_dag_state is None:
                verification_dag_state = semantic_projection
        task_context = project_verification_tasks(
            verification_dag_state,
            max_chars=getattr(self.cfg, 'known_index_max_chars', 3_500),
        )
        requested_ids=list(requested_ids or []) if self.enable_model_selection else []
        requested_ids.extend(task_context.linked_evidence_ids)
        requested_ids=list(dict.fromkeys(requested_ids))
        # Evidence-aware projection: compact Evidence excerpts may truthfully represent only
        # the beginning of a larger read_file observation. If a truncated read is currently
        # hypothesis support, re-project its immutable raw source range instead of letting the
        # model infer that unseen tail lines were never read. This is still bounded by the
        # normal context packer and never performs repository I/O.
        hyp0=(semantic_projection or {}).get("active_hypothesis") or {}
        hyp0=hyp0 or {}
        support0=set(hyp0.get('supporting_evidence_ids') or [])
        source_projection_ids = support0 | set(task_context.linked_evidence_ids)
        if (self.enable_projection and source_projection_ids
                and getattr(self.projection_policy, 'supports_source_ranges', False)):
            for ev in memory.pinned:
                if (ev.evidence_id in source_projection_ids and ev.source == 'read_file' and ev.excerpt_truncated
                        and ev.raw_observation_id and ev.file
                        and isinstance(ev.source_start_line,int) and isinstance(ev.source_end_line,int)):
                    self.rehydrate(ev.raw_observation_id,path=ev.file,start_line=ev.source_start_line,
                                   end_line=ev.source_end_line,
                                   information_need=(
                                       'verification_task_projection'
                                       if ev.evidence_id in task_context.linked_evidence_ids
                                       else 'hypothesis_support_projection'
                                   ))
        local_requests = local_requests or {}
        items=self.catalog(
            state, memory, observation_store,
            semantic_projection=semantic_projection, local_ids=local_requests,
        )
        for item in items:
            if item.raw_observation_id in local_requests:
                item.priority, item.lifecycle, item.pinned = 0, 'active', True
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
        hyp=(semantic_projection or {}).get("active_hypothesis") or {}
        hyp=hyp or {}
        if verification_dag_state is not None:
            # DAG task state owns the active verification question. Do not
            # spend planner context on duplicate gap projections.
            hyp = dict(hyp)
            for key in ("evidence_gap", "required_gaps", "verification_obligations"):
                hyp.pop(key, None)
        state_summary = state.to_summary()
        if semantic_projection is not None:
            state_summary["current_hypothesis"] = semantic_projection.get("active_hypothesis") or {}
        if verification_dag_state is not None:
            summary_hypothesis = dict(state_summary.get("current_hypothesis") or {})
            for key in ("evidence_gap", "required_gaps", "verification_obligations"):
                summary_hypothesis.pop(key, None)
            state_summary["current_hypothesis"] = summary_hypothesis
        advisory=(state.termination_advisory or '').strip() if include_agent_control_state else ''
        fixed=((f"TASK_ID: {state.task.task_id}\nISSUE:\n{issue}\n\nRECENT_ACTIONS:\n{recent_actions}\n\n"
               f"RUNTIME_BUDGET: {', '.join(budget) or 'not configured'}\n"
               f"CURRENT_HYPOTHESIS: {hyp if hyp else '(none)'}\n"
               f"TERMINATION_ADVISORY: {advisory or '(none)'}\nSTATE: {state_summary}\n\n")
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
                           f"{known}\nOnly ev-* identifiers are citable. Use CONTEXT_REQUEST to recover retained local bodies; raw references are citable only after display.\n")
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
                           "Request CONTEXT_REQUEST for retained local details; this is not read_file or external acquisition.\n")
        # Keep Evidence/index context ahead of task scheduling metadata.
        verification_section=(task_context.text + "\n\n") if task_context.text else ""
        known_section=known_section + verification_section
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
            must=[r for r in ranked if r[-1].pinned or r[-1].context_id in requested]
            optional=[r for r in ranked if not (r[-1].pinned or r[-1].context_id in requested)]
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
        # Explicit closure items are correctness-critical: source evidence and
        # the current hypothesis must remain visible during SOURCE_VERIFY/
        # CONVERGE. Pack them first and permit them to exceed the optional-item
        # budget; ordinary pinned Evidence still follows the normal packer.
        ordered = sorted(
            ranked,
            key=lambda row: (0 if row[-1].context_id in requested else 1,
                             row[0], row[1], row[2], row[3]),
        )
        for *_,x in ordered:
            projection=None
            if x.raw_observation_id:
                obs=obs_by_id.get(x.raw_observation_id)
                projection=self._projection_for(obs,x,state.step) if obs is not None else None
                if obs is not None and obs.observation_id in local_requests:
                    selectors = local_requests[obs.observation_id]
                    full = next((selector for selector in selectors if selector.kind == 'FULL_STORED_BODY'), None)
                    parts = [self._local_content(obs, selector)[0] for selector in ([full] if full else selectors)]
                    # Range batches coalesce without rendering the same line twice.
                    if full is None:
                        ranges = merge_ranges([(selector.start_line, selector.end_line) for selector in selectors])
                        parts = [self._local_content(obs, LineRange(kind='LINE_RANGE', start_line=a, end_line=b))[0]
                                 for a, b in ranges]
                    projection = ContextProjection('local-' + obs.observation_id, obs.observation_id,
                        (obs.metadata or {}).get('path') if obs.tool in {'read_file', 'get_source_code'} else None,
                        None, None, None, None, '\n'.join(parts), 0, 'active', True,
                        state.step, 'rehydrated_local_context')
                    if full is not None and obs.tool in {'read_file', 'get_source_code'}:
                        projection.content = '\n'.join(text for _, text in retained_source_lines(obs))
                body=projection.content if projection is not None else x.full_content
                content = self._body_wrapper(x, body, projection.projection_id if projection else "none")
            else:
                content=x.compact_content + (f"\n{x.full_content}" if x.full_content else '')
            size=len(content)+2
            closure_item = x.context_id in requested
            if not self.enable_budget_packing or used+size<=available or closure_item:
                selected.append((x,content,projection)); used+=size
                if projection: projections.append(projection)
            else:
                dropped.append({'id':x.context_id,'reason':'budget','chars':size,'kind':x.source_kind})

        rendered=[]; selected_meta=[]
        for x,content,projection in selected:
            rendered.append(content)
            meta={'id':x.context_id,'reason':('model_requested' if x.context_id in requested else x.metadata.get('selection_reason','priority')),
                  'chars':len(content),'kind':x.source_kind,'lifecycle':'active','pinned':x.pinned,
                  'citable':x.source_kind == 'evidence','context_level':x.context_level,
                  'body_chars':len(projection.content if projection else x.full_content)}
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
        # Character packing is the deterministic budget path. A physical model
        # capability is authoritative, so apply the token bound after the full
        # diagnostic text has been rendered as well.
        if diagnostic_token_budget is not None:
            def total_tokens(value: str) -> int:
                return max(0, int(estimator(value)))
            while selected_meta and total_tokens(text) > total_token_budget:
                removable = next(
                    (index for index in range(len(selected_meta) - 1, -1, -1)
                     if not selected_meta[index].get('pinned')
                     and selected_meta[index].get('id') not in requested),
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
                removable = next(
                    (index for index in range(len(selected_meta) - 1, -1, -1)
                     if not selected_meta[index].get('pinned')
                     and selected_meta[index].get('id') not in requested),
                    None,
                )
                if removable is None:
                    break
                dropped_id=selected_meta[removable]['id']
                selected_meta.pop(removable); rendered.pop(removable)
                dropped.append({'id':dropped_id,'reason':'defensive_budget_drop','chars':0})
                working='\n\n'.join(rendered) or '(no active working context yet)'
                text=f"{fixed}{known_section}\nWORKING_CONTEXT:\n{working}"
        scaffolding=f"\nWORKING_CONTEXT:\n{working}"
        if len(text)>diagnostic_budget and known_section:
            known_allow=max(0,diagnostic_budget-len(fixed)-len(scaffolding))
            known_section,_=line_safe_truncate(known_section,known_allow)
            text=f"{fixed}{known_section}{scaffolding}"
        control_state_truncated=False
        verification_task_context_retained = all(
            f"task_id={task_id}" in text for task_id in task_context.task_ids
        )
        verification_task_context_fail_closed = bool(
            task_context.task_ids and not verification_task_context_retained
        )
        if verification_task_context_fail_closed:
            # A critical graph-state omission must be visible to the caller;
            # never let a budget cut look like a valid context projection.  Do
            # not erase selected Evidence: canonical scheduling metadata is
            # supplied separately in the role envelope.
            marker = (
                "VERIFICATION_TASK_CONTEXT_UNAVAILABLE: graph task state exceeded the "
                "context budget; do not finalize until Runtime reprojections succeed."
            )
            known_section = f"{marker}\n\n{known_section}"
            text = f"{fixed}{known_section}{scaffolding}"
            control_state_truncated = True
        if len(text)>diagnostic_budget:
            # This can occur when manager scaffolding consumes the budget.
            # Source projections were already dropped whole;
            # compact the manager-owned control scaffold without slicing Evidence.
            fixed_allow=max(0,diagnostic_budget-len(scaffolding))
            fixed,control_state_truncated=line_safe_truncate(fixed,fixed_allow)
            text=f"{fixed}{scaffolding}"
        if len(text)>diagnostic_budget:
            text="(diagnostic context omitted: packing budget exhausted)"[:diagnostic_budget]
            control_state_truncated=True
            selected_meta.clear()
            rendered.clear()

        retained_ranges = []
        local_references = []
        for obs in sorted(observation_store.all(), key=lambda item: item.observation_id not in reference_ids):
            source_lines = retained_source_lines(obs)
            path = (obs.metadata or {}).get('path')
            if source_lines and path:
                retained_ranges.append((path, source_lines[0][0], source_lines[-1][0]))
            lines = source_lines or (list(enumerate(obs.content.splitlines(), 1))
                                     if obs.tool not in {'read_file', 'get_source_code'} else [])
            normalized = '\n'.join(text for _, text in lines) if source_lines else obs.content
            row = {'observation_id': obs.observation_id, 'tool': obs.tool,
                   'path': path, 'content_kind': (obs.metadata or {}).get('context_kind', 'TEXT'),
                   'content_sha256': hashlib.sha256(obs.content.encode()).hexdigest(),
                   'retained_lines': [lines[0][0], lines[-1][0]] if lines else None,
                   'retained_normalized_chars': len(normalized),
                   'selector_applicability': ['FULL_STORED_BODY', 'LINE_RANGE'] if lines else [],
                   'retrievable_capacity_chars': 999999999999999999,
                   'capacity_basis': 'conservative_real_wrapper_renderer_with_pinned_bodies',
                   'full_body_fits': False, 'chunks': []}
            # Navigation only: consecutive retained lines, at most eight choices.
            # Size each candidate conservatively before the exact renderer check.
            chunk_limit = max(1, min(100, len(lines)))
            start = 0
            while start < len(lines) and len(row['chunks']) < 8:
                end = start
                size = 0
                while end < len(lines) and end-start < chunk_limit:
                    addition = len(lines[end][1]) + bool(end-start)
                    if size + addition > max(1, available // 3):
                        break
                    size += addition
                    end += 1
                if end == start:
                    end += 1
                    size = len(lines[start][1])
                row['chunks'].append({'start_line': lines[start][0], 'end_line': lines[end-1][0],
                                      'normalized_chars': size, 'fits': False})
                start = end
            evidence = memory.evidence_for_observation(obs.observation_id)
            if evidence is not None:
                row['evidence_ref'] = evidence.evidence_id
            if len(json.dumps(local_references + [row])) <= self.cfg.known_index_max_chars:
                local_references.append(row)
        body_coverage = {}

        # The role renderer serializes this same view in the real request envelope.
        # Account for JSON escaping and reference metadata, not a retired control prefix.
        def observation_references():
            rows = []
            for meta in selected_meta:
                observation = obs_by_id.get(meta.get('provenance_observation_id'))
                if observation is not None and meta.get('body_chars'):
                    metadata = _model_metadata(dict(observation.metadata or {}))
                    rows.append({'observation_id': observation.observation_id,
                                 'tool': observation.tool, 'ok': observation.ok, 'metadata': metadata,
                                 'hypothesis_scopes': metadata.get('acquisition_hypothesis_scopes', ())})
            return rows

        def role_view(body, *, ranges=(), acquired=(), display=None):
            observations = observation_references()
            bodies = tuple(meta['id'] for meta in selected_meta if meta.get('citable') and meta.get('body_chars'))
            references = list(evidence_references)
            known = {row['evidence_ref'] for row in references}
            for evidence_id in bodies:
                if evidence_id not in known:
                    evidence = memory.get(evidence_id)
                    if evidence is not None:
                        references.append({'evidence_ref': evidence_id, 'claim': evidence.summary[:500],
                            'source': evidence.source, 'kind': evidence.kind,
                            'observation_refs': [evidence.raw_observation_id] if evidence.raw_observation_id else [],
                            'relations': []})
            refs = tuple(row['evidence_ref'] for row in references)
            return ContextView(role, body, tuple(row['observation_id'] for row in observations),
                refs, bodies, acquired, ranges,
                json.dumps({'visible_observations': observations,
                            'existing_evidence': references,
                            'local_context_references': local_references,
                            'resolved_local_requests': [
                                {'observation_id': oid, 'selector': selector.model_dump(mode='json')}
                                for oid, selectors in local_requests.items() for selector in selectors
                                if display is None or (
                                    body_coverage.get(oid, {}).get('full') if selector.kind == 'FULL_STORED_BODY'
                                    else any(a <= selector.start_line and b >= selector.end_line
                                             for a, b in body_coverage.get(oid, {}).get('ranges', ())))]}, ensure_ascii=False, sort_keys=True),
                json.dumps(display or {}, sort_keys=True),
                tuple(retained_ranges), json.dumps(body_coverage if display is not None else {}, sort_keys=True))

        # Dry rendering never changes stores, canonical state or published coverage.
        # Preserve pinned bodies, discard optional bodies; six-character escaping
        # is the worst JSON serialization cost for one text character.
        original_meta = selected_meta
        feasibility_checks = []
        for row in local_references:
            obs = obs_by_id[row['observation_id']]
            item = next((item for item in items if item.raw_observation_id == obs.observation_id), None)
            if item is None:
                row['retrievable_capacity_chars'] = 0
                continue
            pinned = [(meta, body) for meta, body in zip(original_meta, rendered)
                      if (meta.get('pinned') or meta['id'] in requested)
                      and meta.get('provenance_observation_id') != obs.observation_id]
            overhead = self._body_wrapper(item, '', 'local-' + obs.observation_id)
            upper = max(0, available - sum(len(body)+2 for _, body in pinned) - len(overhead)-2)
            def fits_body(body, item=item, obs=obs, pinned=pinned, upper=upper):
                nonlocal selected_meta
                content = self._body_wrapper(item, body, 'local-' + obs.observation_id)
                candidate = '\n\n'.join([part for _, part in pinned] + [content])
                candidate_text = f"{fixed}{known_section}\nWORKING_CONTEXT:\n{candidate}"
                if len(candidate_text) > diagnostic_budget or len(body) > upper:
                    return False
                selected_meta = [meta for meta, _ in pinned] + [{
                    'id': item.context_id, 'citable': item.source_kind == 'evidence',
                    'body_chars': len(body), 'provenance_observation_id': obs.observation_id}]
                try:
                    system, user = request_renderer(role_view(candidate_text)) if request_renderer else ('', candidate_text)
                    return (int(estimator(system + "\n\n" + user)) <= total_token_budget
                            if total_token_budget is not None else max_request_chars is None
                            or len(system + "\n\n" + user) <= max_request_chars)
                finally:
                    selected_meta = original_meta
            low, high = 0, upper
            while low < high:
                middle = (low+high+1)//2
                if fits_body('\x00'*middle):
                    low = middle
                else:
                    high = middle-1
            row['retrievable_capacity_chars'] = low
            # Actual selector feasibility uses actual content, including escaping.
            lines = retained_source_lines(obs) if obs.tool in {'read_file', 'get_source_code'} else list(enumerate(obs.content.splitlines(), 1))
            row['full_body_fits'] = bool(lines) and fits_body(
                '\n'.join(value for _, value in lines) if obs.tool in {'read_file', 'get_source_code'} else obs.content)
            chunk_bodies = ['\n'.join(value for number, value in lines
                if chunk['start_line'] <= number <= chunk['end_line']) for chunk in row['chunks']]
            for chunk, body in zip(row['chunks'], chunk_bodies):
                chunk['fits'] = fits_body(body)
            feasibility_checks.append((row, fits_body,
                '\n'.join(value for _, value in lines) if obs.tool in {'read_file', 'get_source_code'} else obs.content,
                chunk_bodies))

        # Fit flags themselves change the dynamic decision schema. Publish only
        # a stable, conservative catalog, never a fit measured against an earlier
        # smaller surface. Clearing choices is monotone and finitely bounded.
        changed = True
        while changed:
            changed = False
            for row, fits_body, full_body, chunk_bodies in feasibility_checks:
                if row['full_body_fits'] and not fits_body(full_body):
                    row['full_body_fits'] = False
                    changed = True
                for chunk, body in zip(row['chunks'], chunk_bodies):
                    if chunk['fits'] and not fits_body(body):
                        chunk['fits'] = False
                        changed = True
        for row, fits_body, _, _ in feasibility_checks:
            low, high = 0, row['retrievable_capacity_chars']
            while low < high:
                middle = (low + high + 1) // 2
                if fits_body('\x00' * middle):
                    low = middle
                else:
                    high = middle - 1
            row['retrievable_capacity_chars'] = low

        if request_renderer is not None:
            while True:
                system, user = request_renderer(role_view(text))
                fits = (int(estimator(system + "\n\n" + user)) <= total_token_budget
                        if total_token_budget is not None else (max_request_chars is None
                        or len(system + "\n\n" + user) <= max_request_chars))
                if fits or not selected_meta:
                    break
                index = next((i for i in range(len(selected_meta)-1, -1, -1)
                              if not selected_meta[i].get('pinned')
                              and selected_meta[i]['id'] not in requested), len(selected_meta)-1)
                removed = selected_meta.pop(index)
                rendered.pop(index)
                dropped.append({'id': removed['id'], 'reason': 'role_request_budget', 'chars': removed['chars']})
                working = '\n\n'.join(rendered) or '(no active working context yet)'
                # Metadata can be omitted without implying that its body was displayed.
                text = f"{fixed}{known_section}\nWORKING_CONTEXT:\n{working}"
            if not fits:
                text = ""
                selected_meta.clear()
                rendered.clear()

        # HARD REQUIREMENT: coverage is rebuilt after every final packing drop.
        self.display_coverage.clear()
        kept_projection_ids={m.get('projection_id') for m in selected_meta if m.get('projection_id')}
        body_coverage = {}
        for meta in selected_meta:
            obs = obs_by_id.get(meta.get('provenance_observation_id'))
            if obs is None or not meta.get('body_chars'):
                continue
            projection = next((p for p in projections if p.projection_id == meta.get('projection_id')), None)
            item = by_id[meta['id']]
            body = projection.content if projection is not None else item.full_content
            source_lines = retained_source_lines(obs)
            full = body == obs.content or bool(source_lines and body == '\n'.join(text for _, text in source_lines))
            if obs.tool in {'read_file', 'get_source_code'}:
                retained = {number: text for number, text in source_lines}
                # Match actual canonical line records, not incidental numbers in prose.
                displayed = [number for number, record in retained.items() if record in body.splitlines()]
                ranges = merge_ranges([(n, n) for n in displayed])
                path = (obs.metadata or {}).get('path')
                if path:
                    for start, end in ranges:
                        self.display_coverage.add(path, start, end, meta.get('projection_id', obs.observation_id))
            elif full:
                ranges = [(1, len(obs.content.splitlines()))]
            elif obs.observation_id in local_requests:
                ranges = merge_ranges([(selector.start_line, selector.end_line)
                                       for selector in local_requests[obs.observation_id]
                                       if selector.kind == 'LINE_RANGE'])
            else:
                ranges = []
            body_coverage[obs.observation_id] = {'full': full, 'ranges': ranges}

        current_ids={m['id'] for m in selected_meta}
        evicted=len(self._last_active_ids-current_ids) if self._last_active_ids else len([d for d in dropped if d.get('reason')=='soft_eviction'])
        self._eviction_total += evicted
        self._projection_total += len([p for p in projections if p.projection_id in kept_projection_ids])
        self._last_active_ids=current_ids
        self._last_selected_ids=current_ids
        self._last_lifecycle={x.context_id:('active' if x.context_id in current_ids else 'cold') for x in items}

        # Role-envelope telemetry measures the same serialized request as packing.
        provider_visible_text = text
        if request_renderer is not None:
            system, user = request_renderer(role_view(text))
            provider_visible_text = system + "\n\n" + user
        current_hypothesis=(semantic_projection or {}).get("active_hypothesis") or {}
        current_hypothesis=current_hypothesis or {}
        critical_evidence_ids = set(current_hypothesis.get('supporting_evidence_ids') or [])
        critical_evidence_ids.update(current_hypothesis.get('contradicting_evidence_ids') or [])
        task_linked_evidence_ids = set(task_context.linked_evidence_ids)
        critical_evidence_ids.update(task_linked_evidence_ids)
        for obligation in current_hypothesis.get('verification_obligations') or ():
            if isinstance(obligation, dict):
                critical_evidence_ids.update(obligation.get('supporting_evidence_ids') or [])
        selected_evidence_ids = {
            str(item.get('id')) for item in selected_meta if item.get('citable')
        }
        final_visible_evidence_ids = set(
            re.findall(r"\bev-[0-9a-f]+\b", provider_visible_text)
        )
        critical_retained = (
            len(critical_evidence_ids.intersection(final_visible_evidence_ids))
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
        obligation_retention = (
            sum(1 for oid in open_obligation_ids if oid in provider_visible_text)
            / len(open_obligation_ids) if open_obligation_ids else 1.0
        )
        contradiction_retention = (
            sum(1 for cid in blocking_contradiction_ids if cid in provider_visible_text)
            / len(blocking_contradiction_ids) if blocking_contradiction_ids else 1.0
        )
        task_linked_retention = (
            len(task_linked_evidence_ids.intersection(selected_evidence_ids))
            / len(task_linked_evidence_ids)
            if task_linked_evidence_ids else 1.0
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
            'role_envelope_chars':max(0, len(provider_visible_text)-len(text)),
            'total_context_chars':len(provider_visible_text),
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
            'final_prompt_visible_evidence_count': len(final_visible_evidence_ids),
            'open_obligation_retention': obligation_retention,
            'blocking_contradiction_retention': contradiction_retention,
            'verification_task_count': task_context.task_count,
            'verification_task_duplicate_count': task_context.duplicate_task_count,
            'verification_ready_task_count': task_context.ready_task_count,
            'verification_critical_open_task_count': task_context.critical_open_task_count,
            'verification_task_context_chars': len(task_context.text),
            'verification_task_context_retained': int(verification_task_context_retained),
            'verification_task_context_fail_closed': int(verification_task_context_fail_closed),
            'task_linked_evidence_count': len(task_linked_evidence_ids),
            'task_linked_evidence_retention': task_linked_retention,
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
            'role_request_tokens': max(0, int(estimator(provider_visible_text))),
            'total_context_tokens': max(0, int(estimator(provider_visible_text))),
            'budget_tokens': total_token_budget if total_token_budget is not None else 0,
        })
        display=self.display_coverage.export()
        projection_count=sum(1 for m in selected_meta if m.get('projection_id'))
        acquired_ranges = set()
        for observation in observation_store.all():
            if observation.tool not in {'read_file', 'get_source_code'} or not observation.ok:
                continue
            metadata = observation.metadata or {}
            path = str(metadata.get('path') or '').strip()
            start = metadata.get('acquired_start_line', metadata.get('start_line'))
            end = metadata.get('acquired_end_line', metadata.get('end_line'))
            if path and isinstance(start, int) and isinstance(end, int):
                acquired_ranges.add((path, start, end))
        visible_ranges = tuple(
            (path, int(start), int(end))
            for path, ranges in sorted(display.items())
            for start, end in ranges
        )
        context_view = role_view(text, ranges=visible_ranges, acquired=tuple(sorted(acquired_ranges)), display=display)
        if request_renderer is not None:
            system, user = request_renderer(context_view)
            provider_visible_text = system + "\n\n" + user
        prompt_snapshot = PromptSnapshot.from_render(
            provider_visible_text,
            token_count=estimator(provider_visible_text),
            acquired_source_ranges=tuple(sorted(acquired_ranges)),
            visible_source_ranges=visible_ranges,
            context_view=context_view,
        )
        self._rehydrate_requests.clear()
        return ContextBuildResult(
            text, budget_chars, len(text), len(items),
            len(selected_meta), selected_meta, dropped, invalid, breakdown,
            len(known_section), active_count, cold_count, evicted, projection_count,
            display, total_token_budget, int(estimator(provider_visible_text)),
            getattr(getattr(self.cfg, "tokenizer", None), "name", "char4")
            if not callable(token_estimator) else "custom",
            prompt_snapshot, context_view,
        )
