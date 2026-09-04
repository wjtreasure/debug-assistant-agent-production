import json
from debug_assistant.config import AppConfig, ModelConfig, HarnessConfig
from debug_assistant.models import TaskSpec
from debug_assistant.harness.runtime import AgentHarness
from debug_assistant.llm.base import LLMError


def _cfg(tmp_path):
    return AppConfig(
        model=ModelConfig(provider='mock',planner_model='mock',critic_model='mock',temperature=0),
        harness=HarnessConfig(build_task_index=False,trace_dir=str(tmp_path/'traces'),reflect_every=99,max_steps=10)
    )


def _events(trace_path):
    return [json.loads(x) for x in open(trace_path,encoding='utf-8') if x.strip()]


class PlannerFailLLM:
    def __init__(self): self.calls=[]
    def complete_json(self,system,user,model=None):
        raise LLMError('planner request timeout')


class ReporterFailLLM:
    def __init__(self): self.calls=[]; self.n=0
    def _usage(self,system,user):
        self.calls.append({'model':'fake','prompt_tokens':10,'completion_tokens':2,'total_tokens':12,
                           'input_tokens':10,'output_tokens':2,'prompt_chars':len(system)+len(user),
                           'completion_chars':20,'latency_ms':1.0,'cached_tokens':None,'reasoning_tokens':None})
    def complete_json(self,system,user,model=None):
        self.n+=1
        low=user.lower()
        if 'final_report_schema' in low:
            raise LLMError('reporter request timeout')
        self._usage(system,user)
        if self.n == 1:
            return {'kind':'tool','skill':'repository_exploration','reason':'grep','confidence':.8,
                    'tool':'grep','arguments':{'query':'needle','glob':'*.py','max_results':20},'expected_evidence':'match'}
        if self.n == 2:
            return {'kind':'tool','skill':'hypothesis_validation','reason':'read','confidence':.9,
                    'tool':'read_file','arguments':{'path':'a.py','start_line':1,'end_line':20},'expected_evidence':'source'}
        return {'kind':'finish','skill':'report_synthesis','reason':'done','confidence':.9,'tool':None,'arguments':{},'expected_evidence':''}


def test_planner_failure_is_traced_with_stage(monkeypatch,tmp_path):
    repo=tmp_path/'repo'; repo.mkdir()
    monkeypatch.setattr('debug_assistant.harness.runtime.build_llm',lambda cfg: PlannerFailLLM())
    r=AgentHarness(_cfg(tmp_path)).run(TaskSpec('planner-fail','issue',str(repo)))
    assert r['state']['status']=='failed'
    assert r['failure']['stage']=='planner'
    evs=_events(r['trace']['trace_path'])
    assert any(e['type']=='RUN_FAILED' and e['payload']['stage']=='planner' for e in evs)
    assert any(e['type']=='LLM_USAGE' for e in evs)


def test_reporter_failure_keeps_trace_and_prior_usage(monkeypatch,tmp_path):
    repo=tmp_path/'repo'; repo.mkdir(); (repo/'a.py').write_text('needle = 1\n' * 20)
    fake=ReporterFailLLM()
    monkeypatch.setattr('debug_assistant.harness.runtime.build_llm',lambda cfg: fake)
    r=AgentHarness(_cfg(tmp_path)).run(TaskSpec('reporter-fail','issue',str(repo)))
    assert r['state']['status']=='partial_success'
    assert r['failure'] is None
    assert r['report_source']=='fallback'
    evs=_events(r['trace']['trace_path'])
    assert any(e['type']=='REPORTER_FAILED' and e['payload']['stage']=='reporter' for e in evs)
    assert any(e['type']=='FALLBACK_REPORT_BUILT' for e in evs)
    usage=[e for e in evs if e['type']=='LLM_USAGE'][-1]['payload']
    assert usage['totals']['calls'] >= 3
    assert usage['totals']['prompt_tokens'] >= 30
    per_call=[e for e in evs if e['type']=='LLM_CALL_USAGE']
    assert per_call and all('prompt_tokens' in e['payload'] for e in per_call)
    assert all('completion_tokens' in e['payload'] for e in per_call)


def test_finalization_usage_failure_does_not_hide_primary_run_failed(monkeypatch,tmp_path):
    repo=tmp_path/'repo'; repo.mkdir()
    monkeypatch.setattr('debug_assistant.harness.runtime.build_llm',lambda cfg: PlannerFailLLM())
    monkeypatch.setattr('debug_assistant.harness.runtime._usage_snapshot',lambda llm: (_ for _ in ()).throw(RuntimeError('usage boom')))
    r=AgentHarness(_cfg(tmp_path)).run(TaskSpec('finalizer-fail','issue',str(repo)))
    evs=_events(r['trace']['trace_path'])
    assert any(e['type']=='FINALIZATION_ERROR' and e['payload']['component']=='usage_snapshot' for e in evs)
    assert any(e['type']=='RUN_FAILED' and e['payload']['stage']=='planner' for e in evs)
