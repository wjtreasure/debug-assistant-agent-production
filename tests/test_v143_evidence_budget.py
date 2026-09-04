import json

from debug_assistant.config import AppConfig, HarnessConfig, ModelConfig
from debug_assistant.harness.obligations import EvidenceObligationTracker, ObligationStatus
from debug_assistant.harness.runtime import AgentHarness
from debug_assistant.models import Evidence, TaskSpec


def _repo(tmp_path):
    repo=tmp_path/'repo'
    (repo/'astroid').mkdir(parents=True)
    (repo/'astroid/builder.py').write_text('\n'.join(f'line_{i}' for i in range(1,241)),encoding='utf-8')
    (repo/'astroid/modutils.py').write_text('def get_source_file():\n    pass\n',encoding='utf-8')
    return repo


def _read_ev(eid='ev-1', path='astroid/builder.py', start=40, end=80):
    return Evidence(eid,'read_file','read_file','source',file=path,
                    source_start_line=start,source_end_line=end,
                    line_start=start,line_end=end)


def test_ranged_obligation_requires_full_same_file_source_coverage(tmp_path):
    repo=_repo(tmp_path)
    tr=EvidenceObligationTracker(repo_root=repo)
    item={'target':'open_source_file body at astroid/builder.py:40-80',
          'location':'astroid/builder.py','reason':'confirm directory-path behavior'}
    tr.sync([item])
    oid=next(iter(tr.items))
    assert tr.items[oid].canonical_files==('astroid/builder.py',)
    assert tr.items[oid].line_hint==(40,80)

    # Same file but wrong range: 40-80 is not contained in 80-200.
    assert tr.note_evidence(_read_ev(path='astroid/builder.py',start=80,end=200),'open_source_file')==[]
    assert tr.items[oid].status is ObligationStatus.OPEN

    # Full containment makes semantic source READY, but does not close behavior until a Critic review.
    assert tr.note_evidence(_read_ev('ev-cover',path='astroid/builder.py',start=35,end=90),'open_source_file')==[]
    obj=tr.items[oid]
    assert obj.status is ObligationStatus.ATTEMPTED
    assert obj.evidence_ready is True
    assert obj.evidence_ids==['ev-cover']
    fp=tr.evidence_fingerprint(obj)
    tr.mark_presented(oid,reflection_id='R1',projection_id='P1',evidence_fingerprint=fp)
    ok,_=tr.apply_explicit_review({'obligation_id':oid,'decision':'resolved','reason':'source proves behavior'},reflection_id='R1')
    assert ok is True
    assert obj.status is ObligationStatus.SATISFIED


def test_ranged_obligation_rejects_null_range_and_discovery_evidence(tmp_path):
    repo=_repo(tmp_path)
    tr=EvidenceObligationTracker(repo_root=repo)
    item={'target':'open_source_file body','location':'astroid/builder.py:40-80','reason':'confirm behavior'}
    tr.sync([item]); oid=next(iter(tr.items))

    no_range=Evidence('ev-source','read_file','read_file','source',file='astroid/builder.py')
    assert tr.note_evidence(no_range,'def open_source_file(): pass')==[]
    grep=Evidence('ev-grep','grep','grep','astroid/builder.py:52: def open_source_file',file='astroid/builder.py')
    assert tr.note_evidence(grep,'astroid/builder.py:52: def open_source_file')==[]
    symbol=Evidence('ev-symbol','symbol_search','symbol_search','FunctionDef open_source_file',file='astroid/builder.py')
    assert tr.note_evidence(symbol,'astroid/builder.py:52-58 FunctionDef open_source_file')==[]
    assert tr.items[oid].status is ObligationStatus.OPEN
    assert tr.items[oid].evidence_ids==[]


def test_wrong_file_never_satisfies_ranged_obligation(tmp_path):
    repo=_repo(tmp_path)
    tr=EvidenceObligationTracker(repo_root=repo)
    item={'target':'open_source_file body','location':'astroid/builder.py:40-80','reason':'confirm behavior'}
    tr.sync([item]); oid=next(iter(tr.items))
    ev=_read_ev(path='astroid/modutils.py',start=1,end=100)
    assert tr.note_evidence(ev,'open_source_file')==[]
    assert tr.items[oid].status is ObligationStatus.OPEN


def test_omitted_required_gap_is_not_falsely_satisfied(tmp_path):
    repo=_repo(tmp_path)
    tr=EvidenceObligationTracker(repo_root=repo)
    required={'target':'2.9.0 -> 2.9.1 change','location':'git history/diff','reason':'confirm regression history'}
    tr.sync([required])
    oid=next(iter(tr.items))
    tr.sync([])
    assert tr.items[oid].status is ObligationStatus.OPEN
    assert tr.items[oid].evidence_ids==[]
    assert tr.items[oid].active_required is False
    assert tr.open_critical()==[]


def test_required_gap_explicitly_downgrades_to_optional(tmp_path):
    repo=_repo(tmp_path)
    tr=EvidenceObligationTracker(repo_root=repo)
    gap={'target':'2.9.0 -> 2.9.1 diff','location':'git history/diff','reason':'confirm regression history'}
    tr.sync([gap],[])
    oid=next(iter(tr.items))
    tr.sync([], [gap])
    obj=tr.items[oid]
    assert obj.status is ObligationStatus.OPTIONAL
    assert obj.critical is False
    assert obj.active_required is False
    assert obj.evidence_ids==[]


def test_satisfied_status_always_has_evidence_ids(tmp_path):
    repo=_repo(tmp_path)
    tr=EvidenceObligationTracker(repo_root=repo)
    item={'target':'builder source','location':'astroid/builder.py:40-80','reason':'read source'}
    tr.sync([item]); oid=next(iter(tr.items))
    tr.mark_satisfied(oid,[])
    assert tr.items[oid].status is ObligationStatus.OPEN
    tr.mark_satisfied(oid,['ev-real'])
    assert tr.items[oid].status is ObligationStatus.SATISFIED
    assert tr.items[oid].evidence_ids==['ev-real']


class _StageGuardLLM:
    def __init__(self):
        self.calls=[]; self.plan_calls=0; self.reflection_calls=0
    def _usage(self,system,user):
        self.calls.append({'model':'fake','prompt_tokens':10,'completion_tokens':2,'total_tokens':12,
                           'input_tokens':10,'output_tokens':2,'prompt_chars':len(system)+len(user),
                           'completion_chars':20,'latency_ms':1.0,'cached_tokens':None,'reasoning_tokens':None})
    def complete_json(self,system,user,model=None):
        self._usage(system,user)
        low=user.lower()
        if 'final_report_schema' in low:
            return {'summary':'guarded','root_cause':'not fully confirmed','likely_files':['a.py'],'likely_symbols':[],
                    'impact_scope':[],'recommended_change_points':[],'uncertainties':['stage guard'],
                    'next_checks':[],'evidence_ids':[],'confidence':.2}
        if 'reflection_schema' in low:
            self.reflection_calls+=1
            return {'decision':'continue','reason':'need more','current_diagnosis':'partial',
                    'evidence_sufficient':False,'supporting_evidence_ids':[], 'contradicting_evidence_ids':[],
                    'required_missing_evidence':[], 'optional_validation':[], 'recommended_next_goal':'more',
                    'confidence':.2,'hypothesis_changed':False}
        self.plan_calls+=1
        return {'kind':'tool','skill':'repository_exploration','reason':'read source','confidence':.6,
                'tool':'read_file','arguments':{'path':'a.py','start_line':1,'end_line':5},
                'expected_evidence':'source','information_need':'inspect source'}


def _events(path):
    return [json.loads(x) for x in open(path,encoding='utf-8') if x.strip()]


def test_planner_stage_guard_skips_new_planner_when_finalize_window_is_too_small(monkeypatch,tmp_path):
    repo=tmp_path/'repo'; repo.mkdir(); (repo/'a.py').write_text('x=1\n'*5)
    fake=_StageGuardLLM(); monkeypatch.setattr('debug_assistant.harness.runtime.build_llm',lambda cfg: fake)
    cfg=AppConfig(model=ModelConfig(provider='mock',planner_model='mock',critic_model='mock'),
                  harness=HarnessConfig(build_task_index=False,trace_dir=str(tmp_path/'traces'),reflect_every=99,
                                        max_steps=10,max_wall_time_seconds=120,finalization_reserve_seconds=90,
                                        planner_start_guard_seconds=60,reflection_start_guard_seconds=60))
    result=AgentHarness(cfg).run(TaskSpec('planner-guard','issue',str(repo)))
    # First planner is necessary because there is no evidence yet. After the first read,
    # only ~30 seconds remain before the 90-second finalization reserve, so no second planner starts.
    assert fake.plan_calls==1
    assert result['state']['tool_calls']==1
    assert result['state']['status']=='partial_success'
    events=_events(result['trace']['trace_path'])
    assert any(e['type']=='LLM_STAGE_SKIPPED' and e['payload']['stage']=='planner' for e in events)


def test_reflection_stage_guard_skips_reflection_and_finalizes_from_existing_evidence(monkeypatch,tmp_path):
    repo=tmp_path/'repo'; repo.mkdir(); (repo/'a.py').write_text('x=1\n'*5)
    fake=_StageGuardLLM(); monkeypatch.setattr('debug_assistant.harness.runtime.build_llm',lambda cfg: fake)
    cfg=AppConfig(model=ModelConfig(provider='mock',planner_model='mock',critic_model='mock'),
                  harness=HarnessConfig(build_task_index=False,trace_dir=str(tmp_path/'traces'),reflect_every=2,
                                        max_steps=10,max_wall_time_seconds=120,finalization_reserve_seconds=90,
                                        planner_start_guard_seconds=0,reflection_start_guard_seconds=60))
    result=AgentHarness(cfg).run(TaskSpec('reflection-guard','issue',str(repo)))
    assert fake.reflection_calls==0
    assert result['state']['tool_calls']==1
    assert result['state']['status']=='partial_success'
    events=_events(result['trace']['trace_path'])
    assert any(e['type']=='LLM_STAGE_SKIPPED' and e['payload']['stage']=='reflection' for e in events)


def test_superseded_transition_is_explicit_not_inferred_from_omission(tmp_path):
    repo=_repo(tmp_path)
    tr=EvidenceObligationTracker(repo_root=repo)
    old={'target':'broad package path cause','location':'astroid/modutils.py','reason':'identify cause'}
    tr.sync([old]); oid=next(iter(tr.items))
    tr.sync([])
    assert tr.items[oid].status is ObligationStatus.OPEN
    assert tr.mark_superseded(oid,'Oreplacement') is True
    assert tr.items[oid].status is ObligationStatus.SUPERSEDED
    assert 'superseded_by:Oreplacement' in tr.items[oid].aliases
