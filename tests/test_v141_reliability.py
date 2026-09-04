import json
import time
from dataclasses import asdict

import httpx
import pytest

from debug_assistant.agent.reflection import Reflector
from debug_assistant.config import AppConfig, ModelConfig, HarnessConfig
from debug_assistant.contracts import ReflectionContract
from debug_assistant.harness.action_policy import ActionPolicy
from debug_assistant.harness.budget import BudgetController
from debug_assistant.harness.information_need import InformationNeedTracker
from debug_assistant.harness.runtime import AgentHarness
from debug_assistant.harness.trace import TraceRecorder
from debug_assistant.models import ActionProposal, ActionKind, Evidence, TaskSpec
from debug_assistant.repository.chunks import build_chunk_manifest, estimate_embedding_tokens
from debug_assistant.repository.embeddings import SiliconFlowEmbeddingProvider, EmbeddingInputError
from debug_assistant.repository.safe_fs import SafeRepositoryFS
from debug_assistant.agent.reporter import build_finalization_context


def _source_evidence(path='src/a.py', eid='ev-a', obs='obs-a', step=3, need='N1'):
    return Evidence(eid,'read_file','read_file','source',file=path,raw_observation_id=obs,tags=[f'step:{step}',f'need:{need}','source_class:source'])


def test_trace_redacts_secret_recursively(tmp_path):
    key='sk-'+'supersecretvalue1234567890'
    t=TraceRecorder(str(tmp_path),'x')
    t.record('RUN_START',{'semantic_search':{'api_key':key},'header':f'Bearer {key}','message':f'failed with {key}'})
    raw=t.path.read_text()
    assert key not in raw
    row=json.loads(raw)
    assert row['payload']['semantic_search']['api_key']=='***'
    assert 'Bearer ***' in row['payload']['header']


def test_reflection_nullable_root_fields_are_valid():
    obj=ReflectionContract.model_validate({
        'decision':'continue','reason':'not enough evidence','current_diagnosis':'package loading is suspect',
        'root_cause_target':None,'root_cause_location':None,'root_cause_mechanism':None,
        'evidence_sufficient':False,'supporting_evidence_ids':[],'contradicting_evidence_ids':[],
        'required_missing_evidence':[],'optional_validation':[],'recommended_next_goal':'inspect resolver','confidence':.3,
    })
    assert obj.root_cause_target is None and obj.root_cause_mechanism is None


def test_reflector_performs_one_bounded_schema_repair():
    class LLM:
        def __init__(self): self.n=0; self.calls=[]
        def complete_json(self,system,user,model=None):
            self.n+=1; self.calls.append({})
            if self.n==1:
                return {'decision':'continue','reason':'partial','current_diagnosis':'x','root_cause_target':{'file':'a.py'},
                        'root_cause_location':'a.py','root_cause_mechanism':None,'evidence_sufficient':False,
                        'supporting_evidence_ids':[],'contradicting_evidence_ids':[],'required_missing_evidence':[],
                        'optional_validation':[],'recommended_next_goal':'read source','confidence':.3}
            return {'decision':'continue','reason':'partial','current_diagnosis':'x','root_cause_target':None,
                    'root_cause_location':'a.py','root_cause_mechanism':None,'evidence_sufficient':False,
                    'supporting_evidence_ids':[],'contradicting_evidence_ids':[],'required_missing_evidence':[],
                    'optional_validation':[],'recommended_next_goal':'read source','confidence':.3}
    r=Reflector(LLM())
    out=r.review('ctx')
    assert r.last_repair_attempted is True
    assert out['root_cause_target'] is None


def test_structured_information_need_merges_paraphrases_without_embedding():
    t=InformationNeedTracker(max_no_gain_attempts=3)
    a=t.get_or_create('Where is the missing __init__.py path created?',1,{
        'target':'package path resolution','question_type':'location','evidence_goal':'find generation of missing __init__.py path'})
    b=t.get_or_create('Which function turns a package directory into a nonexistent __init__.py file?',2,{
        'target':'module path resolution','question_type':'location','evidence_goal':'locate generation of missing __init__.py path'})
    assert a.need_id==b.need_id
    assert b.match_quality.startswith(('lexical_semantic','exact_'))


def test_lexical_no_gain_survives_intermediate_read_and_triggers_semantic_advisory():
    t=InformationNeedTracker(max_no_gain_attempts=4)
    n=t.get_or_create('find missing init path generation',1,{'target':'package path resolution','question_type':'location','evidence_goal':'missing init generation'})
    t.note_attempt(n,'lexical'); t.note_result(n,[],False,'lexical')
    t.note_attempt(n,'read_file'); t.note_result(n,['ev-1'],True,'read_file')
    t.note_attempt(n,'lexical'); t.note_result(n,[],False,'lexical')
    assert n.lexical_no_gain_attempts==2
    assert 'semantic or hybrid' in t.advisory(n)


def test_action_policy_converge_allows_only_scoped_repo_tree():
    policy=ActionPolicy(); evidence=[_source_evidence('astroid/modutils.py')]
    broad=ActionProposal(ActionKind.TOOL,'repository_exploration','tree',tool='repo_tree',arguments={'path':'.','depth':2,'max_entries':100})
    scoped=ActionProposal(ActionKind.TOOL,'repository_exploration','tree',tool='repo_tree',arguments={'path':'astroid','depth':2,'max_entries':100})
    assert policy.evaluate(broad,budget_phase='converge',convergence_mode='convergence_required',evidence=evidence).allowed is False
    assert policy.evaluate(scoped,budget_phase='converge',convergence_mode='convergence_required',evidence=evidence).allowed is True


def test_action_policy_budget_critical_blocks_new_tool():
    policy=ActionPolicy(); action=ActionProposal(ActionKind.TOOL,'x','read',tool='read_file',arguments={'path':'a.py','start_line':1,'end_line':2})
    d=policy.evaluate(action,budget_phase='verify_only',convergence_mode='budget_critical',evidence=[_source_evidence()])
    assert d.allowed is False and d.hard_block is True


def test_budget_reserve_enters_finalize_before_hard_wall_time():
    started=time.time()-8.5
    b=BudgetController(max_steps=20,max_tool_calls=20,max_llm_calls=20,max_total_tokens=10000,max_wall_time_seconds=10,finalization_reserve_seconds=2,started_at=started)
    s=b.snapshot(steps=1,tool_calls=1,llm_calls=1,tokens=100)
    assert s.phase=='finalize'
    assert b.exhausted(s) is None


def test_large_python_symbol_is_split_below_embedding_budget(tmp_path):
    repo=tmp_path/'repo'; repo.mkdir()
    body='\n'.join(f'    x_{i} = "' + ('a'*120) + '"' for i in range(300))
    (repo/'big.py').write_text('def huge():\n'+body+'\n')
    manifest=build_chunk_manifest(SafeRepositoryFS(repo),max_embedding_tokens=400)
    chunks=[c for c in manifest.chunks if c.path=='big.py' and c.qualified_name=='huge']
    assert len(chunks)>1
    assert all(c.part_count==len(chunks) for c in chunks)
    # A pathological single source line may exceed the estimate, otherwise parts are bounded.
    assert sum(estimate_embedding_tokens(c.embedding_text())<=450 for c in chunks) >= len(chunks)-1


class _Resp:
    def __init__(self,status,data=None,text='bad'): self.status_code=status; self._data=data or {}; self.text=text
    def json(self): return self._data
    def raise_for_status(self):
        if self.status_code>=400: raise httpx.HTTPStatusError('bad',request=httpx.Request('POST','https://x'),response=httpx.Response(self.status_code))


class _Client:
    queue=[]
    def __init__(self,*a,**k): pass
    def __enter__(self): return self
    def __exit__(self,*a): return False
    def post(self,*a,**k): return self.queue.pop(0)


def test_embedding_400_batch_is_bounded_and_isolated(monkeypatch):
    _Client.queue=[_Resp(400),_Resp(200,{'data':[{'index':0,'embedding':[1,0,0]}]}),_Resp(200,{'data':[{'index':0,'embedding':[0,1,0]}]})]
    monkeypatch.setattr(httpx,'Client',_Client)
    p=SiliconFlowEmbeddingProvider(api_key='x',dimension=3,batch_size=2,max_retries=0,max_isolation_depth=2)
    rows=p.embed_documents(['a','b'])
    assert rows==[[1.0,0.0,0.0],[0.0,1.0,0.0]]
    assert p.stats.isolated_batches==1


def test_embedding_single_bad_input_reports_index(monkeypatch):
    _Client.queue=[_Resp(400),_Resp(200,{'data':[{'index':0,'embedding':[1,0,0]}]}),_Resp(400)]
    monkeypatch.setattr(httpx,'Client',_Client)
    p=SiliconFlowEmbeddingProvider(api_key='x',dimension=3,batch_size=2,max_retries=0,max_isolation_depth=2)
    with pytest.raises(EmbeddingInputError) as exc:
        p.embed_documents(['good','bad'])
    assert exc.value.input_index==1


def test_reporter_context_uses_source_evidence_when_hypothesis_has_no_support():
    ev1=_source_evidence('astroid/manager.py','ev-1','obs-1',7,'N1')
    ev2=_source_evidence('astroid/modutils.py','ev-2','obs-2',9,'N2')
    text,meta=build_finalization_context(task_id='t',issue='issue',state_summary={},hypothesis={'status':'partial'},evidence=[ev1,ev2])
    assert meta['evidence_fallback_used'] is True
    assert meta['reporter_source_fallback_evidence_count']==2
    assert 'SOURCE_EVIDENCE_FALLBACK_SUMMARIES' in text
    assert set(meta['fallback_candidate_files'])=={'astroid/manager.py','astroid/modutils.py'}


def test_runtime_reserve_finishes_after_current_tool_and_starts_no_new_tool(monkeypatch,tmp_path):
    class LLM:
        def __init__(self): self.calls=[]; self.plans=0; self.last_raw_content=None
        def _usage(self,system,user):
            self.calls.append({'model':'fake','prompt_tokens':10,'completion_tokens':2,'total_tokens':12,
                               'input_tokens':10,'output_tokens':2,'prompt_chars':len(system)+len(user),
                               'completion_chars':20,'latency_ms':1.0,'cached_tokens':None,'reasoning_tokens':None})
        def complete_json(self,system,user,model=None):
            self._usage(system,user)
            if 'FINAL_REPORT_SCHEMA' in user:
                ids=[]
                import re
                ids=list(dict.fromkeys(re.findall(r'ev-[0-9a-f]+',user)))
                return {'summary':'partial','root_cause':'not confirmed','likely_files':[],'likely_symbols':[],
                        'impact_scope':[],'recommended_change_points':[],'uncertainties':['budget reserve reached'],
                        'next_checks':[],'evidence_ids':ids,'confidence':.2}
            self.plans+=1
            if self.plans==1:
                return {'kind':'tool','skill':'repository_exploration','reason':'read source','confidence':.6,
                        'tool':'read_file','arguments':{'path':'a.py','start_line':1,'end_line':5},
                        'expected_evidence':'source','information_need':'inspect source'}
            time.sleep(1.1)
            return {'kind':'tool','skill':'repository_exploration','reason':'would start another tool','confidence':.5,
                    'tool':'repo_tree','arguments':{'path':'.','depth':2,'max_entries':100},
                    'expected_evidence':'more','information_need':'explore more'}

    repo=tmp_path/'repo'; repo.mkdir(); (repo/'a.py').write_text('x=1\n'*5)
    fake=LLM(); monkeypatch.setattr('debug_assistant.harness.runtime.build_llm',lambda cfg: fake)
    cfg=AppConfig(model=ModelConfig(provider='mock',planner_model='mock',critic_model='mock'),
                  harness=HarnessConfig(build_task_index=False,trace_dir=str(tmp_path/'traces'),reflect_every=99,
                                        max_steps=10,max_wall_time_seconds=2,finalization_reserve_seconds=1))
    result=AgentHarness(cfg).run(TaskSpec('reserve','issue',str(repo)))
    assert result['state']['tool_calls']==1
    assert result['state']['status']=='partial_success'
    assert result['report']['likely_files']==['a.py']
    events=[json.loads(x) for x in open(result['trace']['trace_path'],encoding='utf-8') if x.strip()]
    assert any(e['type']=='LLM_STAGE_SKIPPED' and e['payload']['stage']=='planner' for e in events)
    assert any(e['type']=='FINALIZATION_TRIGGERED' for e in events)
