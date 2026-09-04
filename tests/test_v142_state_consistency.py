from debug_assistant.config import ContextConfig
from debug_assistant.context.manager import ContextManager
from debug_assistant.contracts import InformationNeedContract
from debug_assistant.harness.obligations import EvidenceObligationTracker, ObligationStatus
from debug_assistant.memory.evidence_memory import EvidenceMemory
from debug_assistant.memory.hypothesis import HypothesisManager
from debug_assistant.memory.observation_store import ObservationStore
from debug_assistant.models import AgentState, TaskSpec, ToolObservation


def _read(path,start,end,prefix='x'):
    content='\n'.join(f'{i:5d} | {prefix}_{i}_' + ('z'*40) for i in range(start,end+1))
    return ToolObservation('read_file',True,content,{'path':path,'start_line':start,'end_line':end})


def test_evidence_aware_projection_restores_truncated_support_tail(tmp_path):
    state=AgentState(TaskSpec('t','issue',str(tmp_path))); store=ObservationStore(); memory=EvidenceMemory()
    obs=_read('builder.py',80,160,'B'); store.add(obs); state.observations.append(obs); ev=memory.add_observation(obs); state.evidence.append(ev)
    assert ev.excerpt_truncated is True and ev.excerpt_end_line < 151
    state.current_hypothesis={'status':'partial','supporting_evidence_ids':[ev.evidence_id]}
    mgr=ContextManager(ContextConfig(max_item_chars=12000,safety_margin_chars=500,fallback_recent_count=0,target_active_items=6,hard_active_items=8))
    result=mgr.build(state,memory,store,max_context_chars=30000)
    assert '151 | B_151_' in result.text
    assert mgr.is_visible_range('builder.py',115,151)


def test_obligation_line_refinement_deduplicates_same_target(tmp_path):
    repo=tmp_path/'repo'; (repo/'astroid').mkdir(parents=True); (repo/'astroid/builder.py').write_text('def file_build():\n    pass\n')
    tr=EvidenceObligationTracker(max_attempts=2,repo_root=repo)
    tr.sync([{'target':'file_build implementation','location':'astroid/builder.py','reason':'inspect implementation'}])
    first=list(tr.items)[0]
    tr.sync([{'target':'file_build implementation','location':'astroid/builder.py:115-151','reason':'confirm failure condition'}])
    assert list(tr.items)==[first]
    obj=tr.items[first]
    assert obj.line_hint==(115,151)


def test_satisfied_obligation_does_not_reopen_when_critic_reemits(tmp_path):
    repo=tmp_path/'repo'; (repo/'astroid').mkdir(parents=True); (repo/'astroid/builder.py').write_text('def file_build():\n    pass\n')
    tr=EvidenceObligationTracker(max_attempts=2,repo_root=repo)
    item={'target':'file_build implementation','location':'astroid/builder.py:1-2','reason':'confirm implementation'}
    tr.sync([item]); oid=list(tr.items)[0]
    from debug_assistant.models import Evidence
    ev=Evidence('ev-1','read_file','read_file','source',file='astroid/builder.py',source_start_line=1,source_end_line=2)
    closed=tr.note_evidence(ev,'def file_build():\n    pass')
    assert closed==[oid] and tr.items[oid].status is ObligationStatus.SATISFIED
    tr.sync([item])
    assert tr.items[oid].status is ObligationStatus.SATISFIED


def test_hypothesis_without_explicit_causal_fields_is_partial():
    h=HypothesisManager()
    s=h.update({'current_diagnosis':'builder probably fails','supporting_evidence_ids':['ev-1'],
                'contradicting_evidence_ids':[],'required_missing_evidence':[],'optional_validation':[],
                'evidence_sufficient':False,'confidence':.3},1)
    assert s.status=='partial'


def test_hypothesis_supported_and_confirmed_require_causal_structure():
    h=HypothesisManager()
    base={'current_diagnosis':'x','root_cause_target':'file_build','root_cause_mechanism':'missing path reaches open_source_file',
          'supporting_evidence_ids':['ev-1'],'contradicting_evidence_ids':[],'optional_validation':[],'confidence':.8}
    s=h.update({**base,'required_missing_evidence':[{'target':'caller','location':'a.py','reason':'need caller'}],'evidence_sufficient':False},1)
    assert s.status=='supported'
    c=h.update({**base,'required_missing_evidence':[],'evidence_sufficient':True},2)
    assert c.status=='confirmed'


def test_question_type_common_aliases_are_canonicalized():
    assert InformationNeedContract.model_validate({'question_type':'implementation'}).question_type=='behavior'
    assert InformationNeedContract.model_validate({'question_type':'mechanism'}).question_type=='causality'
    assert InformationNeedContract.model_validate({'question_type':'callsite'}).question_type=='caller'

def test_satisfied_obligation_is_removed_from_required_gap_view(tmp_path):
    repo=tmp_path/'repo'; (repo/'astroid').mkdir(parents=True); (repo/'astroid/builder.py').write_text('def file_build():\n    pass\n')
    tr=EvidenceObligationTracker(max_attempts=2,repo_root=repo)
    item={'target':'file_build implementation','location':'astroid/builder.py:1-2','reason':'confirm implementation'}
    tr.sync([item])
    from debug_assistant.models import Evidence
    ev=Evidence('ev-1','read_file','read_file','source',file='astroid/builder.py',source_start_line=1,source_end_line=2)
    tr.note_evidence(ev,'def file_build():\n    pass')
    assert tr.remaining_items([item])==[]
