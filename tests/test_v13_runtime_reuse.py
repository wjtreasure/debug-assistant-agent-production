import json
from pathlib import Path
from debug_assistant.config import AppConfig
from debug_assistant.harness.runtime import AgentHarness
from debug_assistant.models import TaskSpec

class ReuseLLM:
    def __init__(self): self.n=0; self.planner_n=0; self.calls=[]
    def _u(self,s,u): self.calls.append({'model':'fake','prompt_tokens':10,'completion_tokens':2,'total_tokens':12,'input_tokens':10,'output_tokens':2,'prompt_chars':len(s)+len(u),'completion_chars':20,'latency_ms':1,'cached_tokens':None,'reasoning_tokens':None})
    def complete_json(self,system,user,model=None):
        self.n+=1; self._u(system,user)
        if 'REFLECTION_SCHEMA' in user:
            return {'decision':'continue','reason':'reuse acknowledged','current_diagnosis':'needle source','evidence_sufficient':False,'supporting_evidence_ids':[],'contradicting_evidence_ids':[],'missing':['finish report'],'contradictions':[],'recommended_next_goal':'finish','confidence':.7}
        if 'FINAL_REPORT_SCHEMA' in user:
            return {'summary':'done','root_cause':'needle path','likely_files':['a.py'],'likely_symbols':[], 'impact_scope':[],
                    'recommended_change_points':[],'uncertainties':[],'next_checks':[],'evidence_ids':[],'confidence':.8}
        self.planner_n+=1
        if self.planner_n==1:
            return {'kind':'tool','skill':'repository_exploration','reason':'find','tool':'grep','arguments':{'query':'needle','glob':'*.py','max_results':20},'expected_evidence':'candidate','confidence':.8}
        if self.planner_n==2:
            return {'kind':'tool','skill':'hypothesis_validation','reason':'read full','tool':'read_file','arguments':{'path':'a.py','start_line':1,'end_line':20},'expected_evidence':'source','information_need':'inspect source','confidence':.9}
        if self.planner_n==3:
            return {'kind':'tool','skill':'hypothesis_validation','reason':'recheck subset','tool':'read_file','arguments':{'path':'./a.py','start_line':5,'end_line':10},'expected_evidence':'subset','information_need':'inspect source','confidence':.9}
        return {'kind':'finish','skill':'report_synthesis','reason':'done','tool':None,'arguments':{},'confidence':.9}


def test_runtime_reuses_covered_read_and_rehydrates_context(monkeypatch,tmp_path):
    repo=tmp_path/'repo'; repo.mkdir(); (repo/'a.py').write_text('\n'.join(f'needle = {i}' for i in range(30)))
    cfg=AppConfig(); cfg.model.provider='mock'; cfg.harness.build_task_index=False; cfg.harness.trace_dir=str(tmp_path/'traces')
    fake=ReuseLLM(); monkeypatch.setattr('debug_assistant.harness.runtime.build_llm',lambda c: fake)
    r=AgentHarness(cfg).run(TaskSpec('reuse','needle bug',str(repo)))
    assert r['state']['status']=='success'
    assert r['state']['tool_calls']==2  # grep + first read; covered subrange is not re-executed
    assert r['state']['observation_reuse_count']==1
    events=[json.loads(x) for x in Path(r['trace']['trace_path']).read_text().splitlines()]
    assert any(e['type']=='OBSERVATION_REUSED' for e in events)
    assert any(e['type']=='OBSERVATION_SUBRANGE_REHYDRATED' for e in events)
    assert any(e['type']=='PATH_RESOLVED' and e['payload'].get('strategy')=='normalized_relative' for e in events)
    assert any(e['type']=='CONTEXT_BUILT' and any(x.get('reason')=='observation_reused' for x in e['payload'].get('selected',[])) for e in events)
