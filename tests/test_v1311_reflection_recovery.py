import json,re
from pathlib import Path
from debug_assistant.config import AppConfig
from debug_assistant.harness.runtime import AgentHarness
from debug_assistant.models import TaskSpec
from debug_assistant.llm.base import LLMError

class RecoveryLLM:
    def __init__(self): self.calls=[]; self.planner_n=0
    def _u(self,s,u):
        self.calls.append({'model':'fake','prompt_tokens':10,'completion_tokens':2,'total_tokens':12,'input_tokens':10,'output_tokens':2,
                           'prompt_chars':len(s)+len(u),'completion_chars':20,'latency_ms':1,'cached_tokens':None,'reasoning_tokens':None})
    def complete_json(self,system,user,model=None):
        self._u(system,user)
        if 'FINAL_REPORT_SCHEMA' in user:
            ids=re.findall(r'ev-[A-Za-z0-9-]+',user)
            return {'summary':'boundary defect','root_cause':'a.py f boundary condition','likely_files':['a.py'],'likely_symbols':['f'],
                    'impact_scope':[],'recommended_change_points':[],'uncertainties':[],'next_checks':[],
                    'evidence_ids':list(dict.fromkeys(ids))[:1],'confidence':.9}
        self.planner_n+=1
        if self.planner_n==1:
            return {'kind':'tool','skill':'repository_exploration','reason':'read target','tool':'read_file',
                    'arguments':{'path':'a.py','start_line':1,'end_line':20},'expected_evidence':'source',
                    'information_need':'inspect boundary source','confidence':.9}
        return {'kind':'finish','skill':'hypothesis_validation','reason':'enough grounded evidence','tool':None,'arguments':{},
                'expected_evidence':'','information_need':'','confidence':.9}

class FailOnceReflector:
    failures=0
    def __init__(self,llm,model=''): self.llm=llm; self.model=model; self.last_prompt_breakdown={}
    def review(self,context):
        if FailOnceReflector.failures==0:
            FailOnceReflector.failures+=1
            raise LLMError('LLM request failed after retries: The read operation timed out')
        ids=re.findall(r'ev-[A-Za-z0-9-]+',context)
        return {'decision':'finish','reason':'grounded','current_diagnosis':'a.py f has boundary defect',
                'root_cause_target':'f','root_cause_location':'a.py:1','root_cause_mechanism':'boundary condition fails',
                'evidence_sufficient':True,'supporting_evidence_ids':list(dict.fromkeys(ids))[:1],
                'contradicting_evidence_ids':[],'required_missing_evidence':[],'optional_validation':[],
                'recommended_next_goal':'finish','confidence':.9,'hypothesis_changed':False}

def test_retryable_reflection_timeout_preserves_run(monkeypatch,tmp_path):
    repo=tmp_path/'repo'; repo.mkdir(); (repo/'a.py').write_text('\n'.join(f'value={i}' for i in range(30)))
    cfg=AppConfig(); cfg.model.provider='mock'; cfg.harness.build_task_index=False; cfg.harness.trace_dir=str(tmp_path/'traces')
    cfg.harness.reflect_every=2; cfg.harness.max_consecutive_reflection_failures=2
    fake=RecoveryLLM(); FailOnceReflector.failures=0
    monkeypatch.setattr('debug_assistant.harness.runtime.build_llm',lambda c: fake)
    monkeypatch.setattr('debug_assistant.harness.runtime.Reflector',FailOnceReflector)
    r=AgentHarness(cfg).run(TaskSpec('recover','boundary bug',str(repo)))
    assert r['state']['status']=='success'
    assert r['state']['reflection_failure_count']==1
    assert r['state']['max_consecutive_reflection_failures_observed']==1
    events=[json.loads(x) for x in Path(r['trace']['trace_path']).read_text().splitlines()]
    assert any(e['type']=='REFLECTION_FAILED' for e in events)
    assert any(e['type']=='REFLECTION_FAILURE_RECOVERED' for e in events)

class AlwaysFailReflector(FailOnceReflector):
    def review(self,context):
        raise LLMError('LLM request failed after retries: The read operation timed out')

class NeverFinishLLM(RecoveryLLM):
    def complete_json(self,system,user,model=None):
        self._u(system,user)
        if 'FINAL_REPORT_SCHEMA' in user:
            return super().complete_json(system,user,model)
        self.planner_n+=1
        return {'kind':'tool','skill':'repository_exploration','reason':'keep reading','tool':'read_file',
                'arguments':{'path':'a.py','start_line':1,'end_line':20},'expected_evidence':'source',
                'information_need':'inspect boundary source','confidence':.7}

def test_repeated_reflection_timeouts_do_not_loop_forever(monkeypatch,tmp_path):
    repo=tmp_path/'repo'; repo.mkdir(); (repo/'a.py').write_text('\n'.join(f'value={i}' for i in range(30)))
    cfg=AppConfig(); cfg.model.provider='mock'; cfg.harness.build_task_index=False; cfg.harness.trace_dir=str(tmp_path/'traces')
    cfg.harness.reflect_every=2; cfg.harness.max_consecutive_reflection_failures=2; cfg.harness.max_steps=6
    fake=NeverFinishLLM()
    monkeypatch.setattr('debug_assistant.harness.runtime.build_llm',lambda c: fake)
    monkeypatch.setattr('debug_assistant.harness.runtime.Reflector',AlwaysFailReflector)
    r=AgentHarness(cfg).run(TaskSpec('fail-limit','boundary bug',str(repo)))
    assert r['state']['reflection_failure_count'] >= 2
    assert r['state']['budget_critical_entered'] is True or r['state']['status'] in {'budget_exhausted','failed'}
    assert r['state']['step'] <= cfg.harness.max_steps
