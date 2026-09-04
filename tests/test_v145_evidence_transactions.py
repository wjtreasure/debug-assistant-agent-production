import asyncio
import json
import re
import time
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from debug_assistant.config import AppConfig, HarnessConfig, ModelConfig
from debug_assistant.contracts import ReflectionContract
from debug_assistant.harness.obligations import EvidenceObligationTracker, ObligationStatus
from debug_assistant.harness.provider_health import ProviderCircuitBreaker, ProviderHealthSample
from debug_assistant.harness.runtime import AgentHarness
from debug_assistant.llm.base import LLMDeadlineExceeded
from debug_assistant.llm.openai_compatible import OpenAICompatibleClient
from debug_assistant.models import Evidence, TaskSpec


def _source(eid='ev-1', path='a.py', start=1, end=20):
    return Evidence(eid,'read_file','read_file','source',file=path,
                    source_start_line=start,source_end_line=end,line_start=start,line_end=end,
                    raw_observation_id='obs-1')


def _symbol_lookup(query, limit=40):
    if query == 'foo':
        return [{'path':'a.py','name':'foo','qualified_name':'foo','kind':'FunctionDef','start_line':1,'end_line':3}]
    return []


def test_semantic_obligation_requires_same_reflection_presentation_before_resolution(tmp_path):
    tr=EvidenceObligationTracker(repo_root=tmp_path,symbol_lookup=_symbol_lookup)
    tr.sync([{'target':'foo behavior','location':'a.py','goal_type':'behavior','reason':'confirm behavior'}])
    obj=next(iter(tr.items.values()))
    assert tr.note_evidence(_source(), 'def foo():\n    return 1') == []
    assert obj.evidence_ready is True
    assert obj.status is ObligationStatus.ATTEMPTED

    # A model claim cannot close semantic state if Harness cannot prove source presentation.
    ok,_=tr.apply_explicit_review({'obligation_id':obj.obligation_id,'decision':'resolved','reason':'looks fixed'},reflection_id='R1')
    assert ok is False
    assert obj.status is ObligationStatus.ATTEMPTED

    fp=tr.evidence_fingerprint(obj)
    tr.mark_presented(obj.obligation_id,reflection_id='R2',projection_id='P2',evidence_fingerprint=fp)
    ok,_=tr.apply_explicit_review({'obligation_id':obj.obligation_id,'decision':'resolved','reason':'shown function body proves it'},reflection_id='R2')
    assert ok is True
    assert obj.status is ObligationStatus.SATISFIED
    assert obj.last_reviewed_reflection_id == 'R2'
    assert obj.review_decision_source == 'explicit'


def test_implicit_resolution_requires_presentation_in_same_reflection(tmp_path):
    tr=EvidenceObligationTracker(repo_root=tmp_path,symbol_lookup=_symbol_lookup)
    tr.sync([{'target':'foo behavior','location':'a.py','goal_type':'behavior','reason':'confirm behavior'}])
    obj=next(iter(tr.items.values())); tr.note_evidence(_source(), 'def foo(): pass')
    assert tr.apply_implicit_resolution(obj.obligation_id,reflection_id='R1') is False
    tr.mark_presented(obj.obligation_id,reflection_id='R2',projection_id='P2',evidence_fingerprint=tr.evidence_fingerprint(obj))
    assert tr.apply_implicit_resolution(obj.obligation_id,reflection_id='R1') is False
    assert tr.apply_implicit_resolution(obj.obligation_id,reflection_id='R2') is True


def test_reflection_contract_rejects_duplicate_obligation_reviews():
    with pytest.raises(ValidationError):
        ReflectionContract.model_validate({
            'decision':'continue','reason':'review',
            'obligation_reviews':[
                {'obligation_id':'O1','decision':'still_open','reason':'need more'},
                {'obligation_id':'O1','decision':'resolved','reason':'done'},
            ],
        })


def test_refine_is_atomic_parent_supersession_plus_child_creation(tmp_path):
    tr=EvidenceObligationTracker(repo_root=tmp_path,symbol_lookup=_symbol_lookup)
    tr.sync([{'target':'foo behavior','location':'a.py','goal_type':'behavior','reason':'broad'}])
    parent=next(iter(tr.items.values())); tr.note_evidence(_source(),'def foo(): pass')
    tr.mark_presented(parent.obligation_id,reflection_id='R1',projection_id='P1',evidence_fingerprint=tr.evidence_fingerprint(parent))
    ok,transitions=tr.apply_explicit_review({
        'obligation_id':parent.obligation_id,'decision':'refine','reason':'real cause is delegated',
        'refined_requirement':{'target':'bar behavior','location':'a.py','goal_type':'behavior','reason':'inspect delegated behavior'},
    },reflection_id='R1')
    assert ok is True
    assert parent.status is ObligationStatus.SUPERSEDED
    assert parent.superseded_by
    child=tr.items[parent.superseded_by]
    assert child.refined_from == parent.obligation_id
    assert child.status is ObligationStatus.OPEN
    assert transitions


class _SlowTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request):
        await asyncio.sleep(0.2)
        return httpx.Response(200,request=request,json={'choices':[{'message':{'content':'{"ok":true}'}}]})


@pytest.mark.asyncio
async def test_presentation_submission_callback_fires_only_after_attempt_starts():
    client=OpenAICompatibleClient('https://example.invalid/v1','test-key','fake',timeout=1.0,async_transport=_SlowTransport(),max_attempts=1)
    submitted=[]
    with pytest.raises(LLMDeadlineExceeded):
        await client.acomplete_json('s','u',logical_timeout_seconds=0.05,on_attempt_started=lambda meta: submitted.append(meta))
    assert len(submitted)==1
    assert submitted[0]['attempt_index']==1

    submitted.clear()
    with pytest.raises(LLMDeadlineExceeded):
        await client.acomplete_json('s','u',logical_timeout_seconds=0,on_attempt_started=lambda meta: submitted.append(meta))
    assert submitted==[]


def _sample(call_id, success, *, elapsed=10, allowed=90, failure=None):
    return ProviderHealthSample(call_id,'planner',success,not success if failure else False,
                                error_type=failure or '',elapsed_seconds=elapsed,allowed_seconds=allowed)


def test_provider_circuit_breaker_degrades_and_recovers_from_logical_provider_health():
    cb=ProviderCircuitBreaker(window=5,failure_threshold=3,consecutive_failures=2,recovery_successes=2,degraded_timeout_seconds=60)
    seq=[
        _sample('1',True),
        _sample('2',False,failure='logical_deadline'),
        _sample('3',True),
        _sample('4',False,failure='network_error'),
        _sample('5',False,failure='logical_deadline'),
    ]
    transitions=[cb.observe(x) for x in seq]
    assert transitions[-1]=='degraded'
    assert cb.degraded is True
    assert cb.cap(90,'planner')==60
    # Model/schema failures are provider successes and do not count against health.
    assert cb.observe(ProviderHealthSample('6','reflection',True,False,error_type='schema_failure',elapsed_seconds=20,allowed_seconds=60)) is None
    assert cb.degraded is True
    assert cb.observe(ProviderHealthSample('7','reflection',True,False,elapsed_seconds=20,allowed_seconds=60))=='recovered'
    assert cb.cap(90,'planner')==90


class _LifecycleLLM:
    """Small runtime fake with provider lifecycle events and two Reflection phases."""
    def __init__(self):
        self.calls=[]; self.events=[]; self.planner_n=0; self.reflection_n=0; self.call_n=0

    def _lifecycle(self, system, user, on_attempt_started=None):
        self.call_n+=1; cid=f'c{self.call_n}'
        self.events.append({'type':'LLM_LOGICAL_CALL_STARTED','payload':{'logical_call_id':cid,'logical_timeout_seconds':90.0}})
        meta={'logical_call_id':cid,'attempt_index':1,'attempt_timeout_seconds':90.0,'logical_remaining_seconds':90.0}
        self.events.append({'type':'LLM_ATTEMPT_STARTED','payload':meta})
        if on_attempt_started: on_attempt_started(dict(meta))
        self.events.append({'type':'LLM_ATTEMPT_FINISHED','payload':{'logical_call_id':cid,'attempt_index':1,'attempt_elapsed_ms':1.0,'success':True}})
        self.events.append({'type':'LLM_LOGICAL_CALL_FINISHED','payload':{'logical_call_id':cid,'provider_attempts':1,'logical_elapsed_ms':1.0,'success':True,'provider_success':True,'provider_failure':False,'logical_timeout_seconds':90.0}})
        self.calls.append({'model':'fake','prompt_tokens':10,'completion_tokens':2,'total_tokens':12,'input_tokens':10,'output_tokens':2,'prompt_chars':len(system)+len(user),'completion_chars':20,'latency_ms':1.0,'provider_attempts':1,'logical_call_id':cid})

    def complete_json(self,system,user,model=None,logical_timeout_seconds=None,on_attempt_started=None):
        self._lifecycle(system,user,on_attempt_started)
        if 'FINAL_REPORT_SCHEMA' in user:
            ids=re.findall(r'ev-[A-Za-z0-9-]+',user)
            return {'summary':'foo bug','root_cause':'foo behavior is wrong','likely_files':['a.py'],'likely_symbols':['foo'],
                    'impact_scope':[],'recommended_change_points':[],'uncertainties':[],'next_checks':[],
                    'evidence_ids':list(dict.fromkeys(ids))[:1],'confidence':.9}
        if 'REFLECTION_SCHEMA' in user:
            self.reflection_n+=1
            ids=list(dict.fromkeys(re.findall(r'ev-[A-Za-z0-9-]+',user)))
            if self.reflection_n==1:
                return {'decision':'continue','reason':'need semantic review','current_diagnosis':'foo likely wrong',
                        'root_cause_target':None,'root_cause_location':'a.py','root_cause_mechanism':None,
                        'evidence_sufficient':False,'supporting_evidence_ids':ids[:1],'contradicting_evidence_ids':[],
                        'required_missing_evidence':[{'target':'foo behavior','location':'a.py','goal_type':'behavior','reason':'review function body'}],
                        'optional_validation':[],'obligation_reviews':[],'recommended_next_goal':'review foo','confidence':.4,'hypothesis_changed':True}
            oid=re.search(r'"obligation_id"\s*:\s*"(O[0-9a-f]+)"',user)
            assert oid, user
            return {'decision':'finish','reason':'presented source resolves the gap','current_diagnosis':'foo returns wrong boundary value',
                    'root_cause_target':'foo','root_cause_location':'a.py','root_cause_mechanism':'foo returns the wrong boundary value',
                    'evidence_sufficient':True,'supporting_evidence_ids':ids[:1],'contradicting_evidence_ids':[],
                    'required_missing_evidence':[],'optional_validation':[],
                    'obligation_reviews':[{'obligation_id':oid.group(1),'decision':'resolved','reason':'pinned foo body proves the behavior'}],
                    'recommended_next_goal':'finish','confidence':.9,'hypothesis_changed':True}
        self.planner_n+=1
        if self.planner_n==1:
            return {'kind':'tool','skill':'repository_exploration','reason':'read foo','confidence':.8,'tool':'read_file',
                    'arguments':{'path':'a.py','start_line':1,'end_line':10},'expected_evidence':'foo source','information_need':'inspect foo behavior',
                    'information_need_structured':{'target':'foo behavior','question_type':'behavior','evidence_goal':'read foo source'}}
        return {'kind':'finish','skill':'hypothesis_validation','reason':'done','confidence':.9,'tool':None,'arguments':{},'expected_evidence':'','information_need':''}


def _trace_events(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def test_runtime_context_guard_presents_ready_evidence_before_semantic_resolution(monkeypatch,tmp_path):
    repo=tmp_path/'repo'; repo.mkdir(); (repo/'a.py').write_text('def foo():\n    return 0\n\nx=1\n'+'x=2\n'*10)
    fake=_LifecycleLLM(); monkeypatch.setattr('debug_assistant.harness.runtime.build_llm',lambda cfg: fake)
    cfg=AppConfig(model=ModelConfig(provider='mock',planner_model='mock',critic_model='mock'),
                  harness=HarnessConfig(build_task_index=True,trace_dir=str(tmp_path/'traces'),reflect_every=2,max_steps=6,
                                        obligation_review_min_seconds=0,planner_start_guard_seconds=0,reflection_start_guard_seconds=0))
    result=AgentHarness(cfg).run(TaskSpec('v145-present','foo boundary issue',str(repo)))
    assert result['state']['status']=='success'
    events=_trace_events(result['trace']['trace_path'])
    types=[e['type'] for e in events]
    assert 'EVIDENCE_OBLIGATIONS_READY' in types
    assert 'OBLIGATION_PRESENTATION_PREPARED' in types
    assert 'OBLIGATION_PRESENTATION_CONTEXT_CONFIRMED' in types
    assert 'OBLIGATION_EVIDENCE_PRESENTED' in types
    assert 'OBLIGATION_REVIEW_APPLIED' in types
    assert 'SEMANTIC_STATE_COMMIT' in types
    ready_i=types.index('EVIDENCE_OBLIGATIONS_READY')
    present_i=types.index('OBLIGATION_EVIDENCE_PRESENTED')
    review_i=types.index('OBLIGATION_REVIEW_APPLIED')
    assert ready_i < present_i < review_i


def test_information_need_trace_is_retrieval_only_without_semantic_status(monkeypatch,tmp_path):
    repo=tmp_path/'repo'; repo.mkdir(); (repo/'a.py').write_text('def foo():\n    return 0\n')
    fake=_LifecycleLLM(); monkeypatch.setattr('debug_assistant.harness.runtime.build_llm',lambda cfg: fake)
    cfg=AppConfig(model=ModelConfig(provider='mock',planner_model='mock',critic_model='mock'),
                  harness=HarnessConfig(build_task_index=True,trace_dir=str(tmp_path/'traces'),reflect_every=99,max_steps=1,
                                        planner_start_guard_seconds=0,reflection_start_guard_seconds=0))
    result=AgentHarness(cfg).run(TaskSpec('need-only','foo issue',str(repo)))
    events=_trace_events(result['trace']['trace_path'])
    need=next(e['payload']['need'] for e in events if e['type']=='INFORMATION_NEED')
    assert 'status' not in need
    assert 'exhausted' in need


def test_optional_to_required_same_symbol_reuses_one_obligation_id(tmp_path):
    tr=EvidenceObligationTracker(repo_root=tmp_path,symbol_lookup=_symbol_lookup)
    optional={'target':'a.foo behavior','location':'a.py','goal_type':'behavior','reason':'nice to verify'}
    required={'target':'foo behavior for boundary input','location':'a.py','goal_type':'behavior','reason':'now causal and required'}
    tr.sync([], [optional]); oid=next(iter(tr.items)); assert tr.items[oid].status is ObligationStatus.OPTIONAL
    tr.sync([required], [])
    assert len(tr.items)==1
    assert next(iter(tr.items))==oid
    assert tr.items[oid].active_required is True
    assert tr.items[oid].status is ObligationStatus.OPEN


def test_successful_still_open_review_blocks_same_evidence_presentation_storm(tmp_path):
    tr=EvidenceObligationTracker(repo_root=tmp_path,symbol_lookup=_symbol_lookup)
    tr.sync([{'target':'foo behavior','location':'a.py','goal_type':'behavior','reason':'confirm'}])
    obj=next(iter(tr.items.values())); tr.note_evidence(_source('ev-1'),'def foo(): pass')
    fp=tr.evidence_fingerprint(obj); tr.mark_presented(obj.obligation_id,reflection_id='R1',projection_id='P1',evidence_fingerprint=fp)
    ok,_=tr.apply_explicit_review({'obligation_id':obj.obligation_id,'decision':'still_open','reason':'need downstream caller'},reflection_id='R1')
    assert ok is True
    assert tr.presentation_candidates()==[]
    # New source changes the evidence fingerprint and makes another review eligible.
    tr.note_evidence(_source('ev-2'),'def foo(): pass')
    assert [x.obligation_id for x in tr.presentation_candidates()]==[obj.obligation_id]


def test_semantic_transaction_rolls_back_tracker_and_hypothesis_together(monkeypatch,tmp_path):
    repo=tmp_path/'repo'; repo.mkdir(); (repo/'a.py').write_text('def foo():\n    return 0\n\nx=1\n'+'x=2\n'*10)
    fake=_LifecycleLLM(); monkeypatch.setattr('debug_assistant.harness.runtime.build_llm',lambda cfg: fake)
    from debug_assistant.memory.hypothesis import HypothesisManager
    original=HypothesisManager.update
    def fail_review_commit(self,review,step):
        if review.get('obligation_reviews'):
            raise RuntimeError('synthetic semantic commit failure')
        return original(self,review,step)
    monkeypatch.setattr(HypothesisManager,'update',fail_review_commit)
    cfg=AppConfig(model=ModelConfig(provider='mock',planner_model='mock',critic_model='mock'),
                  harness=HarnessConfig(build_task_index=True,trace_dir=str(tmp_path/'traces'),reflect_every=2,max_steps=3,
                                        obligation_review_min_seconds=0,planner_start_guard_seconds=0,reflection_start_guard_seconds=0))
    result=AgentHarness(cfg).run(TaskSpec('v145-rollback','foo boundary issue',str(repo)))
    events=_trace_events(result['trace']['trace_path']); types=[e['type'] for e in events]
    assert 'OBLIGATION_EVIDENCE_PRESENTED' in types
    assert 'SEMANTIC_STATE_ROLLBACK' in types
    # Candidate review events are buffered until commit, so rollback cannot leave fake applied history.
    assert 'OBLIGATION_REVIEW_APPLIED' not in types
    required=(result['state'].get('current_hypothesis') or {}).get('required_missing_evidence') or []
    assert any('foo behavior' in x.get('target','') for x in required)


class _PlannerDeadlineLLM(_LifecycleLLM):
    def complete_json(self,system,user,model=None,logical_timeout_seconds=None,on_attempt_started=None):
        if 'REFLECTION_SCHEMA' not in user and 'FINAL_REPORT_SCHEMA' not in user:
            self.call_n+=1; cid=f'c{self.call_n}'
            self.events.append({'type':'LLM_LOGICAL_CALL_STARTED','payload':{'logical_call_id':cid,'logical_timeout_seconds':1.0}})
            meta={'logical_call_id':cid,'attempt_index':1,'attempt_timeout_seconds':1.0,'logical_remaining_seconds':1.0}
            self.events.append({'type':'LLM_ATTEMPT_STARTED','payload':meta})
            self.events.append({'type':'LLM_ATTEMPT_FAILED','payload':{'logical_call_id':cid,'attempt_index':1,'error_type':'logical_deadline','retryable':False}})
            self.events.append({'type':'LLM_LOGICAL_CALL_FINISHED','payload':{'logical_call_id':cid,'provider_attempts':1,'logical_elapsed_ms':1.0,'success':False,'provider_success':False,'provider_failure':True,'error_type':'logical_deadline','logical_timeout_seconds':1.0}})
            raise LLMDeadlineExceeded('planner deadline')
        return super().complete_json(system,user,model=model,logical_timeout_seconds=logical_timeout_seconds,on_attempt_started=on_attempt_started)


def test_planner_deadline_uses_stage_failure_taxonomy_not_action_validation(monkeypatch,tmp_path):
    repo=tmp_path/'repo'; repo.mkdir(); (repo/'a.py').write_text('x=1\n')
    fake=_PlannerDeadlineLLM(); monkeypatch.setattr('debug_assistant.harness.runtime.build_llm',lambda cfg: fake)
    cfg=AppConfig(model=ModelConfig(provider='mock',planner_model='mock',critic_model='mock'),
                  harness=HarnessConfig(build_task_index=False,trace_dir=str(tmp_path/'traces'),max_steps=1,
                                        planner_start_guard_seconds=0,reflection_start_guard_seconds=0))
    result=AgentHarness(cfg).run(TaskSpec('planner-taxonomy','issue',str(repo)))
    events=_trace_events(result['trace']['trace_path']); types=[e['type'] for e in events]
    assert 'PLANNER_FAILED' in types
    assert 'ACTION_VALIDATION_ERROR' not in types


def test_trace_llm_lifecycle_precedes_usage_for_successful_call(monkeypatch,tmp_path):
    repo=tmp_path/'repo'; repo.mkdir(); (repo/'a.py').write_text('def foo():\n    return 0\n')
    fake=_LifecycleLLM(); monkeypatch.setattr('debug_assistant.harness.runtime.build_llm',lambda cfg: fake)
    cfg=AppConfig(model=ModelConfig(provider='mock',planner_model='mock',critic_model='mock'),
                  harness=HarnessConfig(build_task_index=False,trace_dir=str(tmp_path/'traces'),max_steps=1,
                                        planner_start_guard_seconds=0,reflection_start_guard_seconds=0))
    result=AgentHarness(cfg).run(TaskSpec('chronology','issue',str(repo)))
    events=_trace_events(result['trace']['trace_path']); types=[e['type'] for e in events]
    a=types.index('LLM_LOGICAL_CALL_STARTED'); b=types.index('LLM_ATTEMPT_STARTED'); c=types.index('LLM_LOGICAL_CALL_FINISHED'); d=types.index('LLM_CALL_USAGE')
    assert a < b < c < d
