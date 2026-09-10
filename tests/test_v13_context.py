from debug_assistant.config import ContextConfig
from debug_assistant.context.manager import ContextManager
from debug_assistant.memory.observation_store import ObservationStore
from debug_assistant.memory.evidence_memory import EvidenceMemory
from debug_assistant.models import AgentState, TaskSpec, ToolObservation


def obs(path,start,end,marker):
    content='\n'.join(f'{i:5d} | {marker}_{i}' for i in range(start,end+1))
    return ToolObservation('read_file',True,content,{'path':path,'start_line':start,'end_line':end,'truncated':False})


def add(state,store,memory,o):
    state.observations.append(o); store.add(o); ev=memory.add_observation(o)
    if ev: state.evidence.append(ev)
    return ev


def test_cross_file_rehydration_keeps_old_observation_without_fixed_n(tmp_path):
    state=AgentState(TaskSpec('t','issue',str(tmp_path))); store=ObservationStore(); memory=EvidenceMemory()
    a=obs('tools.py',1,20,'A'); b=obs('single.py',1,20,'B'); c=obs('single.py',21,40,'C')
    for o in (a,b,c): add(state,store,memory,o)
    mgr=ContextManager(ContextConfig(max_item_chars=12000,safety_margin_chars=500,fallback_recent_count=2),enable_catalog=True,enable_model_selection=False)
    mgr.rehydrate(a.observation_id)
    r=mgr.build(state,memory,store,max_context_chars=30000,max_steps=20,max_tool_calls=45)
    ids={x['id'] for x in r.selected}
    evidence_ids={item.evidence_id for item in state.evidence}
    assert evidence_ids.issubset(ids)
    assert any(x['reason']=='observation_reused' for x in r.selected if x.get('provenance_observation_id')==a.observation_id)
    assert all(observation.observation_id not in r.text for observation in (a,b,c))


def test_model_context_ids_are_optional_priority_hints_and_invalid_id_is_nonfatal(tmp_path):
    state=AgentState(TaskSpec('t','issue',str(tmp_path))); store=ObservationStore(); memory=EvidenceMemory()
    a=obs('a.py',1,10,'A'); ev=add(state,store,memory,a)
    mgr=ContextManager(ContextConfig(max_item_chars=12000,safety_margin_chars=500,fallback_recent_count=1),enable_catalog=True,enable_model_selection=True)
    r=mgr.build(state,memory,store,max_context_chars=15000,requested_ids=[ev.evidence_id,'ev-missing'])
    assert 'ev-missing' in r.invalid_requested_ids
    assert any(x['id']==ev.evidence_id and x['reason']=='model_requested' for x in r.selected)


def test_context_budget_and_raw_evidence_dedup(tmp_path):
    state=AgentState(TaskSpec('t','issue',str(tmp_path))); store=ObservationStore(); memory=EvidenceMemory()
    a=obs('a.py',1,100,'UNIQUE'); ev=add(state,store,memory,a)
    mgr=ContextManager(ContextConfig(max_item_chars=3500,safety_margin_chars=500,fallback_recent_count=1),enable_catalog=True,enable_model_selection=False)
    r=mgr.build(state,memory,store,max_context_chars=12000)
    assert r.used_chars <= 12000
    # If raw is selected, evidence body should be compact, not duplicate the full excerpt.
    assert r.text.count('    1 | UNIQUE_1') == 1
    assert ev.evidence_id in r.text


def test_item_truncation_is_explicit_and_line_safe(tmp_path):
    state=AgentState(TaskSpec('t','issue',str(tmp_path))); store=ObservationStore(); memory=EvidenceMemory()
    a=obs('a.py',1,200,'LONG'); add(state,store,memory,a)
    mgr=ContextManager(ContextConfig(max_item_chars=700,safety_margin_chars=300,fallback_recent_count=1),enable_catalog=True)
    items=mgr.catalog(state,memory,store); x=next(i for i in items if i.raw_observation_id==a.observation_id)
    assert x.metadata['context_truncated'] is True
    assert x.metadata['display_chars'] <= 700
    # No partial final source line is introduced.
    assert all(' | ' in line for line in x.full_content.splitlines())


def test_rehydrate_uses_immutable_store_after_runtime_copy_is_bounded(tmp_path):
    state=AgentState(TaskSpec('t','issue',str(tmp_path))); store=ObservationStore(); memory=EvidenceMemory()
    original=obs('a.py',1,80,'ORIGINAL')
    store.add(original)
    original.content='\n'.join(original.content.splitlines()[:5]) + '\n...[bounded in runtime state]'
    original.metadata['end_line']=5
    state.observations.append(original)
    ev=memory.add_observation(original); state.evidence.append(ev)
    manager=ContextManager(ContextConfig(max_item_chars=12000,safety_margin_chars=300,fallback_recent_count=1))

    manager.rehydrate(original.observation_id,path='a.py',start_line=70,end_line=80,information_need='tail')
    result=manager.build(state,memory,store,max_context_chars=15000)

    assert '70 | ORIGINAL_70' in result.text
    assert '80 | ORIGINAL_80' in result.text
