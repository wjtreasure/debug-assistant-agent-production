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
    assert a.observation_id in ids and b.observation_id in ids and c.observation_id in ids
    assert any(x['reason']=='observation_reused' for x in r.selected if x['id']==a.observation_id)


def test_model_context_ids_are_optional_priority_hints_and_invalid_id_is_nonfatal(tmp_path):
    state=AgentState(TaskSpec('t','issue',str(tmp_path))); store=ObservationStore(); memory=EvidenceMemory()
    a=obs('a.py',1,10,'A'); add(state,store,memory,a)
    mgr=ContextManager(ContextConfig(max_item_chars=12000,safety_margin_chars=500,fallback_recent_count=1),enable_catalog=True,enable_model_selection=True)
    r=mgr.build(state,memory,store,max_context_chars=15000,requested_ids=[a.observation_id,'obs-missing'])
    assert 'obs-missing' in r.invalid_requested_ids
    assert any(x['id']==a.observation_id and x['reason']=='model_requested' for x in r.selected)


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
    items=mgr.catalog(state,memory,store); x=next(i for i in items if i.context_id==a.observation_id)
    assert x.metadata['context_truncated'] is True
    assert x.metadata['display_chars'] <= 700
    # No partial final source line is introduced.
    assert all(' | ' in line for line in x.full_content.splitlines())
