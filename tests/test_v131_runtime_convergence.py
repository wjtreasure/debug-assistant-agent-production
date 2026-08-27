import json,re
from pathlib import Path
from debug_assistant.config import AppConfig
from debug_assistant.harness.runtime import AgentHarness
from debug_assistant.models import TaskSpec

class ConvergenceLLM:
    def __init__(self): self.calls=[]; self.planner_n=0
    def _u(self,s,u): self.calls.append({'model':'fake','prompt_tokens':10,'completion_tokens':2,'total_tokens':12,'input_tokens':10,'output_tokens':2,'prompt_chars':len(s)+len(u),'completion_chars':20,'latency_ms':1,'cached_tokens':None,'reasoning_tokens':None})
    def complete_json(self,system,user,model=None):
        self._u(system,user)
        if 'REFLECTION_SCHEMA' in user:
            ids=re.findall(r'\[(ev-[^\]]+)\]',user)
            support=[ids[-1]] if ids else []
            return {'decision':'continue','reason':'causal mechanism already grounded','current_diagnosis':'a.py boundary condition causes the failure',
                    'evidence_sufficient':True,'supporting_evidence_ids':support,'contradicting_evidence_ids':[],
                    'required_missing_evidence':[],'optional_validation':[{'target':'optional caller confirmation','location':'a.py','reason':'additional confidence only'}],
                    'recommended_next_goal':'finish unless required evidence appears','confidence':.92,'hypothesis_changed':False}
        if 'FINAL_REPORT_SCHEMA' in user:
            ids=re.findall(r'ev-[A-Za-z0-9-]+',user)
            return {'summary':'boundary defect','root_cause':'a.py boundary condition','likely_files':['a.py'],'likely_symbols':[],
                    'impact_scope':[],'recommended_change_points':[],'uncertainties':[],'next_checks':[],
                    'evidence_ids':list(dict.fromkeys(ids))[:1],'confidence':.9}
        self.planner_n+=1
        if self.planner_n==1:
            return {'kind':'tool','skill':'repository_exploration','reason':'read target','tool':'read_file','arguments':{'path':'a.py','start_line':1,'end_line':20},'expected_evidence':'source','information_need':'inspect boundary source','confidence':.9}
        # Keep asking for already-covered source with changing nonessential intents. Rehydration + reflection should converge before the mock can run forever.
        return {'kind':'tool','skill':'hypothesis_validation','reason':'optional confirmation','tool':'read_file','arguments':{'path':'a.py','start_line':5,'end_line':10},
                'expected_evidence':'same source','information_need':f'optional check {self.planner_n}','confidence':.8}

def test_runtime_force_finalizes_after_stable_no_progress(monkeypatch,tmp_path):
    repo=tmp_path/'repo'; repo.mkdir(); (repo/'a.py').write_text('\n'.join(f'value={i}' for i in range(30)))
    cfg=AppConfig(); cfg.model.provider='mock'; cfg.harness.build_task_index=False; cfg.harness.trace_dir=str(tmp_path/'traces'); cfg.harness.reflect_every=2
    fake=ConvergenceLLM(); monkeypatch.setattr('debug_assistant.harness.runtime.build_llm',lambda c: fake)
    r=AgentHarness(cfg).run(TaskSpec('conv','boundary bug',str(repo)))
    assert r['state']['status']=='success'
    assert r['state']['forced_finalization'] is True
    assert r['state']['convergence_mode']=='force_finalization'
    events=[json.loads(x) for x in Path(r['trace']['trace_path']).read_text().splitlines()]
    assert any(e['type']=='CONVERGENCE_MODE_CHANGED' for e in events)
    assert any(e['type']=='FORCE_FINALIZATION' for e in events)
    assert any(e['type']=='OBSERVATION_REHYDRATED' for e in events)
    assert r['state']['step'] < cfg.harness.max_steps
