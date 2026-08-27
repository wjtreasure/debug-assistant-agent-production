import json
import re

from debug_assistant.agent.reporter import (
    Reporter,
    ReporterContractViolation,
    build_finalization_context,
    detect_tool_action,
)
from debug_assistant.config import AppConfig, HarnessConfig, ModelConfig
from debug_assistant.harness.runtime import AgentHarness
from debug_assistant.memory.evidence_memory import EvidenceMemory
from debug_assistant.memory.observation_store import ObservationStore
from debug_assistant.models import Evidence, TaskSpec, ToolObservation


def _ev(eid, path, obs_id, start=1, end=20, summary="source"):
    return Evidence(
        evidence_id=eid,
        kind="read_file",
        source="read_file",
        summary=summary,
        excerpt=f"{start:5d} | excerpt",
        file=path,
        line_start=start,
        line_end=end,
        raw_observation_id=obs_id,
        source_start_line=start,
        source_end_line=end,
        excerpt_start_line=start,
        excerpt_end_line=start,
        excerpt_truncated=True,
    )


def test_detect_tool_action_is_conservative():
    assert detect_tool_action('<read_file path="a.py" start_line="1" end_line="20" />') == "xml_tool_action"
    assert detect_tool_action('tool_call: {"name":"read_file"}') == "tool_call_marker"
    assert detect_tool_action({"kind":"tool","skill":"x","tool":"grep","arguments":{}}) == "agent_action_object"
    # Natural prose mentioning tool names must not be rejected.
    assert detect_tool_action('The report notes that read_file evidence was incomplete.') is None
    assert detect_tool_action('{"summary":"grep results were inspected"}') is None


def test_finalization_context_contains_full_hypothesis_contradictions_and_no_known_index():
    store=ObservationStore()
    obs1=ToolObservation('read_file', True, '\n'.join(f'{i:5d} | root_{i}' for i in range(10,31)), {'path':'src/a.py','start_line':10,'end_line':30}, observation_id='obs-1')
    obs2=ToolObservation('read_file', True, '\n'.join(f'{i:5d} | contra_{i}' for i in range(1,11)), {'path':'src/b.py','start_line':1,'end_line':10}, observation_id='obs-2')
    store.add(obs1); store.add(obs2)
    evidence=[_ev('ev-1','src/a.py','obs-1',10,30,'supports root cause'), _ev('ev-2','src/b.py','obs-2',1,10,'contradicts edge case')]
    hyp={
        'description':'root cause', 'root_cause_target':'f', 'root_cause_location':'src/a.py',
        'root_cause_mechanism':'bad branch', 'status':'supported', 'confidence':.7,
        'supporting_evidence_ids':['ev-1'], 'contradicting_evidence_ids':['ev-2'],
        'required_missing_evidence':[{'target':'caller','location':'src/c.py','reason':'not inspected'}],
        'optional_validation':[{'target':'test','location':'tests/test_a.py','reason':'nice to validate'}],
    }
    text,meta=build_finalization_context(task_id='t',issue='issue',state_summary={'status':'running'},hypothesis=hyp,evidence=evidence,observation_store=store)
    assert 'COMPLETE_HYPOTHESIS' in text
    assert 'contradicting_evidence_ids' in text and 'ev-2' in text
    assert 'REQUIRED_MISSING_EVIDENCE' in text and 'caller' in text
    assert 'OPTIONAL_VALIDATION' in text and 'test' in text
    assert 'KNOWN_CONTEXT_INDEX' not in text
    assert meta['known_context_included'] is False
    assert meta['reporter_projection_count'] == 1  # contradictions are summary-only by default


def test_reporter_valid_json_with_required_missing_evidence_is_still_llm_report():
    class LLM:
        def __init__(self): self.calls=[]; self.last_raw_content=None
        def complete_json(self,system,user,model=None):
            self.calls.append({})
            self.last_raw_content='{"summary":"s","root_cause":"r","likely_files":["a.py"],"likely_symbols":["f"],"impact_scope":[],"recommended_change_points":[],"uncertainties":["caller not inspected"],"next_checks":["inspect caller"],"evidence_ids":["ev-1"],"confidence":0.7}'
            return json.loads(self.last_raw_content)
    ev=_ev('ev-1','a.py','obs-1')
    ctx='COMPLETE_HYPOTHESIS: {"required_missing_evidence":[{"target":"caller"}]}'
    report=Reporter(LLM(),compact_prompt=True).build('t',ctx,[ev])
    assert report.report_source == 'llm'
    assert report.uncertainties == ['caller not inspected']


def test_reporter_tool_action_is_explicit_contract_violation():
    class ToolLikeLLM:
        def __init__(self): self.calls=[]; self.last_raw_content=None
        def complete_json(self,system,user,model=None):
            self.last_raw_content='<read_file path="a.py" start_line="1" end_line="20" />'
            raise RuntimeError('not json')
    reporter=Reporter(ToolLikeLLM(),compact_prompt=True)
    try:
        reporter.build('t','ctx',[_ev('ev-1','a.py','obs-1')])
        assert False, 'expected ReporterContractViolation'
    except ReporterContractViolation:
        assert reporter.last_contract_violation['reason'] == 'xml_tool_action'


class _ColdReporterE2ELLm:
    """1343-shaped flow: search -> read schema -> read marshalling -> reflection -> reporter."""
    def __init__(self): self.calls=[]; self.last_raw_content=None; self.planner_n=0
    def _usage(self,system,user):
        self.calls.append({'model':'fake','prompt_tokens':10,'completion_tokens':2,'total_tokens':12,
                           'input_tokens':10,'output_tokens':2,'prompt_chars':len(system)+len(user),
                           'completion_chars':100,'latency_ms':1.0,'cached_tokens':None,'reasoning_tokens':None})
    def complete_json(self,system,user,model=None):
        self._usage(system,user)
        low=user.lower()
        if 'final_report_schema' in low:
            assert 'KNOWN_CONTEXT_INDEX' not in user
            assert 'SOURCE_PROJECTION' in user
            assert '_do_load' in user
            self.last_raw_content=json.dumps({
                'summary':'Nested invalid input exposes None validator path',
                'root_cause':'Schema._do_load invokes field validators after failed unmarshal leaves result None.',
                'likely_files':['src/marshmallow/schema.py'],
                'likely_symbols':['Schema._do_load','Schema._invoke_field_validators'],
                'impact_scope':['Nested invalid input'],
                'recommended_change_points':[{'file':'src/marshmallow/schema.py','symbol':'Schema._do_load','reason':'guard validator invocation when result is None'}],
                'uncertainties':[], 'next_checks':[], 'evidence_ids':re.findall(r'ev-[0-9a-f]+',user)[:2], 'confidence':.88,
            })
            return json.loads(self.last_raw_content)
        if 'reflection_schema' in low:
            ids=list(dict.fromkeys(re.findall(r'ev-[0-9a-f]+',user)))
            self.last_raw_content=json.dumps({
                'decision':'finish','reason':'causal chain confirmed','current_diagnosis':'_do_load validates None result',
                'root_cause_target':'Schema._do_load','root_cause_location':'src/marshmallow/schema.py',
                'root_cause_mechanism':'failed unmarshal leaves result None and validators index it',
                'evidence_sufficient':True,'supporting_evidence_ids':ids[:2],'contradicting_evidence_ids':[],
                'required_missing_evidence':[],'optional_validation':[], 'recommended_next_goal':'',
                'confidence':.88,'hypothesis_changed':True,
            })
            return json.loads(self.last_raw_content)
        self.planner_n += 1
        if self.planner_n == 1:
            obj={'kind':'tool','skill':'repository_exploration','reason':'find load path','confidence':.8,'tool':'grep',
                 'arguments':{'query':'_do_load|_invoke_field_validators','glob':'*.py','max_results':20},'expected_evidence':'locations','information_need':'find load path'}
        elif self.planner_n == 2:
            obj={'kind':'tool','skill':'hypothesis_validation','reason':'read schema','confidence':.9,'tool':'read_file',
                 'arguments':{'path':'src/marshmallow/schema.py','start_line':1,'end_line':80},'expected_evidence':'_do_load source','information_need':'inspect _do_load'}
        else:
            obj={'kind':'tool','skill':'hypothesis_validation','reason':'read unmarshal','confidence':.9,'tool':'read_file',
                 'arguments':{'path':'src/marshmallow/marshalling.py','start_line':1,'end_line':40},'expected_evidence':'unmarshal error path','information_need':'inspect unmarshal'}
        self.last_raw_content=json.dumps(obj)
        return obj


def test_mock_e2e_cold_support_is_restored_for_reporter_without_tool_call(monkeypatch,tmp_path):
    repo=tmp_path/'repo'
    (repo/'src/marshmallow').mkdir(parents=True)
    (repo/'src/marshmallow/schema.py').write_text('\n'.join([
        'def _do_load(self, data):',
        '    try:',
        '        result = self._unmarshal(data)',
        '    except ValidationError as error:',
        '        result = error.data',
        '    self._invoke_field_validators(data=result)',
        'def _invoke_field_validators(self, data):',
        '    return data["x"]',
    ] + ['# pad']*72),encoding='utf-8')
    (repo/'src/marshmallow/marshalling.py').write_text('class ValidationError(Exception):\n    data = None\n' + '# pad\n'*38,encoding='utf-8')
    fake=_ColdReporterE2ELLm()
    monkeypatch.setattr('debug_assistant.harness.runtime.build_llm',lambda cfg: fake)
    cfg=AppConfig(
        model=ModelConfig(provider='mock',planner_model='mock',critic_model='mock',temperature=0),
        harness=HarnessConfig(build_task_index=False,trace_dir=str(tmp_path/'traces'),reflect_every=4,max_steps=8)
    )
    cfg.harness.context.fallback_recent_count=1
    result=AgentHarness(cfg).run(TaskSpec('mock-1343','nested invalid input',str(repo)))
    assert result['state']['status'] == 'success'
    assert result['report_source'] == 'llm'
    assert result['report']['likely_symbols'][0] == 'Schema._do_load'
    events=[json.loads(x) for x in open(result['trace']['trace_path'],encoding='utf-8') if x.strip()]
    reporter_ctx=[e for e in events if e['type']=='REPORTER_CONTEXT_BUILT'][-1]['payload']
    assert reporter_ctx['known_context_included'] is False
    assert reporter_ctx['reporter_projection_count'] >= 1
    assert not any(e['type']=='REPORTER_CONTRACT_VIOLATION' for e in events)
    # The first schema source is old enough to be cold by reflection time, yet reporter succeeds from store-backed projection.
    reflection_context=[e for e in events if e['type']=='CONTEXT_BUILT' and e['payload']['stage']=='reflection'][-1]['payload']
    assert reflection_context['cold_item_count'] >= 1
