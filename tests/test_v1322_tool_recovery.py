import json, re
from pathlib import Path

from debug_assistant.config import AppConfig
from debug_assistant.harness.runtime import AgentHarness
from debug_assistant.models import TaskSpec
from debug_assistant.harness.convergence import ConvergenceController, ConvergenceMode


class RepairLLM:
    def __init__(self):
        self.calls=[]; self.planner_n=0; self.last_raw_content=None
    def _usage(self,system,user):
        self.calls.append({'model':'fake','prompt_tokens':10,'completion_tokens':2,'total_tokens':12,
                           'input_tokens':10,'output_tokens':2,'prompt_chars':len(system)+len(user),
                           'completion_chars':20,'latency_ms':1,'cached_tokens':None,'reasoning_tokens':None})
    def complete_json(self,system,user,model=None):
        self._usage(system,user)
        low=user.lower()
        if 'reflection_schema' in low:
            ids=list(dict.fromkeys(re.findall(r'ev-[0-9a-f]+',user)))
            obj={'decision':'finish','reason':'range source confirms behavior','current_diagnosis':'target line is in repaired read',
                 'root_cause_target':'f','root_cause_location':'a.py','root_cause_mechanism':'bad boundary',
                 'evidence_sufficient':True,'supporting_evidence_ids':ids[-1:] or [],'contradicting_evidence_ids':[],
                 'required_missing_evidence':[],'optional_validation':[],'recommended_next_goal':'','confidence':.9,'hypothesis_changed':True}
            self.last_raw_content=json.dumps(obj); return obj
        if 'final_report_schema' in low:
            ids=list(dict.fromkeys(re.findall(r'ev-[0-9a-f]+',user)))
            obj={'summary':'s','root_cause':'bad boundary','likely_files':['a.py'],'likely_symbols':['f'],'impact_scope':[],
                 'recommended_change_points':[],'uncertainties':[],'next_checks':[],'evidence_ids':ids[-1:] or [],'confidence':.9}
            self.last_raw_content=json.dumps(obj); return obj
        self.planner_n += 1
        obj={'kind':'tool','skill':'hypothesis_validation','reason':'inspect exact region','confidence':.9,'tool':'read_file',
             'arguments':{'path':'a.py','start_line':200,'end_line':400},'expected_evidence':'target implementation',
             'information_need':'inspect implementation around line 300'}
        self.last_raw_content=json.dumps(obj); return obj


def test_runtime_repairs_201_line_read_without_invalid_route(monkeypatch,tmp_path):
    repo=tmp_path/'repo'; repo.mkdir(); (repo/'a.py').write_text('\n'.join(f'line_{i}' for i in range(1,500)),encoding='utf-8')
    cfg=AppConfig(); cfg.model.provider='mock'; cfg.harness.build_task_index=False; cfg.harness.trace_dir=str(tmp_path/'traces'); cfg.harness.reflect_every=2; cfg.harness.max_steps=6
    fake=RepairLLM(); monkeypatch.setattr('debug_assistant.harness.runtime.build_llm',lambda c: fake)
    r=AgentHarness(cfg).run(TaskSpec('repair','boundary failure',str(repo)))
    assert r['state']['status']=='success'
    assert r['state']['invalid_routes']==0
    assert r['state']['tool_calls']==1
    events=[json.loads(x) for x in Path(r['trace']['trace_path']).read_text().splitlines()]
    repair=[e for e in events if e['type']=='ACTION_ARGUMENT_REPAIRED']
    assert repair and repair[0]['payload']['original_arguments']['end_line']==400
    assert repair[0]['payload']['repaired_arguments']['end_line']==399
    obs=[e for e in events if e['type']=='TOOL_OBSERVATION'][0]['payload']
    assert obs['metadata']['start_line']==200 and obs['metadata']['end_line']==399


def test_budget_critical_does_not_recover_from_reflection_after_rejection_only():
    c=ConvergenceController(no_progress_limit=2)
    c.state.mode=ConvergenceMode.BUDGET_CRITICAL
    hyp={'status':'supported','supporting_evidence_ids':['ev-1'],'evidence_sufficient':False,
         'required_missing_evidence':[{'target':'x'}],'contradicting_evidence_ids':[],
         'diagnosis_fingerprint':'a','required_gap_fingerprint':'a','stable_diagnosis_transitions':0,'updated_step':1}
    c.assess_reflection(hyp,allow_budget_recovery=False)
    hyp2=dict(hyp); hyp2['diagnosis_fingerprint']='b'; hyp2['updated_step']=2
    assessment=c.assess_reflection(hyp2,allow_budget_recovery=False)
    assert assessment.kind.value=='progress'
    assert c.state.mode is ConvergenceMode.BUDGET_CRITICAL
