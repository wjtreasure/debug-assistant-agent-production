import copy
import json
import re

import pytest

from debug_assistant.agent.planner import Planner, PlannerContractError
from debug_assistant.agent.reporter import Reporter
from debug_assistant.config import AppConfig, HarnessConfig, ModelConfig
from debug_assistant.harness.guards import RouterGuard
from debug_assistant.harness.runtime import AgentHarness
from debug_assistant.models import ActionKind, ActionProposal, AgentState, Evidence, TaskSpec
from debug_assistant.tools.registry import PARALLEL_ALLOWED_TOOLS, ToolRegistry


def _planner_output(**changes):
    value={
        'kind':'tool', 'skill':'repository_exploration', 'reason':'inspect source',
        'tool':'read_file', 'arguments':{'path':'a.py'}, 'actions':[],
    }
    value.update(changes)
    return value


class _SequenceLLM:
    def __init__(self, *responses):
        self.responses=list(responses); self.calls=[]

    def complete_json(self, system, user, model=None, logical_timeout_seconds=None):
        self.calls.append({'system':system,'user':user})
        return copy.deepcopy(self.responses.pop(0))


def _registry(tmp_path):
    (tmp_path/'a.py').write_text('value = 1\n', encoding='utf-8')
    return ToolRegistry(tmp_path)


def test_planner_prompt_catalog_is_live_and_parallel_policy_is_shared(tmp_path):
    registry=_registry(tmp_path)
    llm=_SequenceLLM(_planner_output())
    Planner(llm,registry).propose(AgentState(TaskSpec('t','issue',str(tmp_path))), 'ctx')
    prompt=llm.calls[0]['user']
    assert 'VALID_SKILLS:' in prompt and 'VALID_TOOLS:' in prompt
    assert 'PARALLEL_ALLOWED_TOOLS:' in prompt
    assert 'git_log' in prompt  # registered, but not bounded-local parallel
    parallel_line=next(x for x in prompt.splitlines() if x.startswith('PARALLEL_ALLOWED_TOOLS:'))
    assert 'git_log' not in parallel_line
    assert set(PARALLEL_ALLOWED_TOOLS)=={'read_file','grep','symbol_search','code_search','inspect_symbol_context'}
    assert llm.calls and llm.calls[0]['user'].count('VALID_QUESTION_TYPES:') == 1


@pytest.mark.parametrize('bad', [
    {'skill':'general'},
    {'tool':'noop'},
])
def test_unknown_planner_objects_are_rejected_before_execution(tmp_path, bad):
    registry=_registry(tmp_path)
    raw=_planner_output(**bad)
    action=Planner(_SequenceLLM(raw),registry).propose(AgentState(TaskSpec('t','issue',str(tmp_path))), 'ctx')
    decision=RouterGuard(registry).validate(action, AgentState(TaskSpec('t','issue',str(tmp_path))))
    assert decision.ok is False
    assert decision.error['error_type'] in {'unknown_skill','unknown_tool'}


def test_planner_repair_cannot_change_skill(tmp_path):
    primary=_planner_output(arguments=[])
    repair=_planner_output(skill='general', arguments={'path':'a.py'})
    with pytest.raises(PlannerContractError) as raised:
        Planner(_SequenceLLM(primary,repair),_registry(tmp_path)).propose(AgentState(TaskSpec('t','i',str(tmp_path))), 'ctx')
    assert raised.value.metadata['repair_rejection_reason']=='immutable_field_modified'


def test_planner_repair_cannot_change_existing_path(tmp_path):
    primary=_planner_output(actions=[{'tool':'read_file','arguments':{}}])
    repair=_planner_output(arguments={'path':'B.py'}, actions=[])
    with pytest.raises(PlannerContractError) as raised:
        Planner(_SequenceLLM(primary,repair),_registry(tmp_path)).propose(AgentState(TaskSpec('t','i',str(tmp_path))), 'ctx')
    assert raised.value.metadata['repair_rejection_reason']=='semantic_change_attempted'


def test_planner_repair_rejects_placeholder_children(tmp_path):
    primary=_planner_output(kind='parallel', tool=None, arguments={}, actions=[])
    repair={**primary, 'actions':[
        {'action_id':'noop1','tool':'noop','arguments':{}},
        {'action_id':'noop2','tool':'noop','arguments':{}},
    ]}
    with pytest.raises(PlannerContractError) as raised:
        Planner(_SequenceLLM(primary,repair),_registry(tmp_path)).propose(AgentState(TaskSpec('t','i',str(tmp_path))), 'ctx')
    assert raised.value.metadata['repair_rejection_reason']=='semantic_change_attempted'


def test_planner_repair_only_canonicalizes_question_type(tmp_path):
    primary=_planner_output(information_need_structured={
        'target':'file_build', 'question_type':'not-a-question-type', 'evidence_goal':'inspect source',
    })
    repair=_planner_output(information_need_structured={
        'target':'file_build', 'question_type':'behavior', 'evidence_goal':'inspect source',
    })
    action=Planner(_SequenceLLM(primary,repair),_registry(tmp_path)).propose(AgentState(TaskSpec('t','i',str(tmp_path))), 'ctx')
    assert action.information_need_structured['question_type']=='behavior'
    assert action.information_need_structured['target']=='file_build'
    assert action.arguments=={'path':'a.py'}


def test_parallel_git_log_remains_fail_closed_even_when_registered(tmp_path):
    registry=_registry(tmp_path)
    action=ActionProposal(ActionKind.PARALLEL,'repository_exploration','inspect',tool=None,arguments={},actions=[
        {'action_id':'a1','tool':'git_log','arguments':{}},
        {'action_id':'a2','tool':'read_file','arguments':{'path':'a.py'}},
    ])
    decision=RouterGuard(registry).validate(action,AgentState(TaskSpec('t','i',str(tmp_path))))
    assert decision.ok is False
    assert decision.error['error_type']=='parallel_tool_not_allowed'


def _report(claim_type='source_fact', status='observed', evidence_ids=None, claims=None):
    ids=list(evidence_ids or ['ev-1'])
    return {
        'summary':'partial', 'root_cause':'uncertain', 'likely_files':['a.py'], 'likely_symbols':[],
        'impact_scope':[], 'recommended_change_points':[], 'uncertainties':[], 'next_checks':[],
        'evidence_ids':ids, 'confidence':0.3,
        'claims':claims if claims is not None else [{
            'text':'source was inspected','claim_type':claim_type,'status':status,'evidence_ids':ids,
        }],
    }


def _evidence():
    return Evidence('ev-1','read_file','read_file','source',file='a.py',raw_observation_id='obs-1',source_start_line=1,source_end_line=1)


def test_reporter_repair_drops_added_claim_and_keeps_grounded_original():
    primary=_report(claim_type='bad-type',status='bad-status')
    repaired=_report(claims=[
        {'text':'source was inspected','claim_type':'source_fact','status':'observed','evidence_ids':['ev-1']},
        {'text':'new unsupported fact','claim_type':'diagnosis','status':'supported','evidence_ids':['ev-1']},
    ])
    reporter=Reporter(_SequenceLLM(primary,repaired),compact_prompt=True)
    report=reporter.build('t','ctx',[_evidence()])
    assert len(report.claims)==1
    assert any(event=='REPORTER_REPAIR_DROPPED_CLAIM' and payload['reason']=='new_claim_forbidden'
               for event,payload in reporter.last_events)


def test_reporter_repair_cannot_upgrade_acquired_unreviewed():
    primary=_report(claim_type='bad-type',status='acquired_unreviewed')
    repaired=_report(claims=[{
        'text':'source was inspected','claim_type':'source_fact','status':'supported','evidence_ids':['ev-1'],
    }])
    report=Reporter(_SequenceLLM(primary,repaired),compact_prompt=True).build('t','ctx',[_evidence()])
    assert report.claims[0]['status']=='acquired_unreviewed'


class _InvalidReviewLLM:
    def __init__(self):
        self.reflections=0; self.planner=0; self.calls=[]

    def complete_json(self, system, user, model=None, logical_timeout_seconds=None):
        self.calls.append({'system':system,'user':user})
        low=user.lower()
        if 'final_report_schema' in low:
            return _report()
        if 'reflection_schema' in low:
            self.reflections+=1
            if self.reflections == 1:
                return {
                    'decision':'continue','reason':'source needs semantic review','current_diagnosis':'foo source is relevant',
                    'root_cause_target':'foo','root_cause_location':'a.py','root_cause_mechanism':'source behavior',
                    'evidence_sufficient':False,'supporting_evidence_ids':[],'contradicting_evidence_ids':[],
                    'required_missing_evidence':[{'target':'foo behavior','location':'a.py:1-1','goal_type':'behavior','reason':'review foo'}],
                    'optional_validation':[],'obligation_reviews':[],'confidence':.3,
                }
            match=re.search(r'\bO[0-9a-f]{8}\b',user)
            oid=match.group(0) if match else 'Omissing'
            return {
                'decision':'continue','reason':'malformed review should be ignored','current_diagnosis':'foo source is relevant',
                'root_cause_target':'foo','root_cause_location':'a.py','root_cause_mechanism':'source behavior',
                'evidence_sufficient':False,'supporting_evidence_ids':[],'contradicting_evidence_ids':[],
                'required_missing_evidence':[],'optional_validation':[],
                'obligation_reviews':[{'obligation_id':oid,'decision':'resolved','reason':'bad',
                                      'refined_requirement':{'target':'foo behavior','line_start':1}}],
                'confidence':.3,
            }
        self.planner+=1
        if self.planner == 1:
            return {'kind':'tool','skill':'repository_exploration','reason':'read source','tool':'read_file','arguments':{'path':'a.py'}}
        return {'kind':'finish','skill':'report_synthesis','reason':'finish','tool':None,'arguments':{}}


def test_invalid_obligation_review_is_ignored_without_implicit_resolution(monkeypatch,tmp_path):
    (tmp_path/'a.py').write_text('foo = 1\n',encoding='utf-8')
    fake=_InvalidReviewLLM()
    monkeypatch.setattr('debug_assistant.harness.runtime.build_llm',lambda cfg:fake)
    cfg=AppConfig(model=ModelConfig(provider='mock',planner_model='mock',critic_model='mock'),harness=HarnessConfig(
        build_task_index=False,trace_dir=str(tmp_path/'traces'),max_steps=5,reflect_every=1,
        planner_start_guard_seconds=0,reflection_start_guard_seconds=0,obligation_review_min_seconds=0,
        finalization_reserve_seconds=0,
    ))
    result=AgentHarness(cfg).run(TaskSpec('invalid-review','issue',str(tmp_path)))
    events=[json.loads(line) for line in open(result['trace']['trace_path'],encoding='utf-8')]
    types=[event['type'] for event in events]
    assert 'INVALID_OBLIGATION_REVIEW' in types
    assert 'OBLIGATION_REVIEW_IGNORED' in types
    assert 'OBLIGATION_REVIEW_APPLIED' not in types
    assert 'EVIDENCE_OBLIGATIONS_SATISFIED' not in types
    ignored=[event for event in events if event['type']=='OBLIGATION_REVIEW_IGNORED'][-1]['payload']
    assert ignored['reason']=='schema_invalid'
    commits=[event['payload'] for event in events if event['type']=='SEMANTIC_STATE_COMMIT']
    assert commits and commits[-1]['to_revision']==commits[-1]['from_revision']
    reconciliation=[event['payload'] for event in events if event['type']=='FINAL_STATE_RECONCILIATION'][-1]
    assert reconciliation['evidence_ready_unreviewed']
    assert result['state']['status'] in {'partial_success','success'}
