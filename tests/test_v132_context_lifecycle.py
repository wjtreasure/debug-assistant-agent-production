from pathlib import Path
from debug_assistant.config import ContextConfig
from debug_assistant.context.manager import ContextManager
from debug_assistant.memory.evidence_memory import EvidenceMemory
from debug_assistant.memory.observation_store import ObservationStore
from debug_assistant.models import AgentState, TaskSpec, ToolObservation


def read_obs(path,start,end,prefix='L'):
    content='\n'.join(f'{i:5d} | {prefix}_{i}' for i in range(start,end+1))
    return ToolObservation('read_file',True,content,{'path':path,'start_line':start,'end_line':end})


def add(state,store,memory,obs):
    state.observations.append(obs); store.add(obs)
    ev=memory.add_observation(obs)
    if ev: state.evidence.append(ev)
    return ev


def test_known_context_survives_cold_eviction_and_exact_rehydrate(tmp_path):
    state=AgentState(TaskSpec('t','issue',str(tmp_path))); store=ObservationStore(); memory=EvidenceMemory()
    old=read_obs('a.py',100,180,'A'); newest=read_obs('b.py',1,20,'B')
    add(state,store,memory,old); add(state,store,memory,newest)
    mgr=ContextManager(ContextConfig(max_item_chars=12000,safety_margin_chars=300,fallback_recent_count=1,target_active_items=4,hard_active_items=6))
    first=mgr.build(state,memory,store,max_context_chars=12000)
    assert '- a.py: read 100-180' in first.text  # existence remains model-visible
    assert not mgr.is_visible_range('a.py',130,160)

    mgr.rehydrate(old.observation_id,path='a.py',start_line=130,end_line=160,information_need='revisit A')
    second=mgr.build(state,memory,store,max_context_chars=12000)
    assert '130 | A_130' in second.text and '160 | A_160' in second.text
    assert '129 | A_129' not in second.text and '161 | A_161' not in second.text
    assert mgr.is_visible_range('a.py',130,160)


def test_display_coverage_tracks_only_final_visible_lines(tmp_path):
    state=AgentState(TaskSpec('t','issue',str(tmp_path))); store=ObservationStore(); memory=EvidenceMemory()
    obs=read_obs('wide.py',1,100,'W'); add(state,store,memory,obs)
    mgr=ContextManager(ContextConfig(max_item_chars=260,safety_margin_chars=300,fallback_recent_count=1))
    r=mgr.build(state,memory,store,max_context_chars=8000)
    assert 'wide.py: read 1-100' in r.text  # raw/known coverage
    assert mgr.is_visible_range('wide.py',1,5)
    assert not mgr.is_visible_range('wide.py',80,90)  # display coverage is truthful, not raw coverage


def test_range_level_supersession_preserves_uncovered_search_hits(tmp_path):
    state=AgentState(TaskSpec('t','issue',str(tmp_path))); store=ObservationStore(); memory=EvidenceMemory()
    grep=ToolObservation('grep',True,'a.py:10: foo\nb.py:20: foo\na.py:90: foo',{'pattern':'foo'})
    add(state,store,memory,grep)
    read=read_obs('a.py',1,20,'A'); add(state,store,memory,read)
    mgr=ContextManager(ContextConfig(fallback_recent_count=1))
    known=mgr.known_context_text(store)
    assert 'a.py: read 1-20; search hits 90' in known
    assert 'search hits 10' not in known
    assert 'b.py: search hits 20' in known


def test_overlapping_rehydrate_requests_are_coalesced(tmp_path):
    state=AgentState(TaskSpec('t','issue',str(tmp_path))); store=ObservationStore(); memory=EvidenceMemory()
    old=read_obs('a.py',300,400,'A'); newest=read_obs('b.py',1,5,'B')
    add(state,store,memory,old); add(state,store,memory,newest)
    mgr=ContextManager(ContextConfig(fallback_recent_count=1,max_item_chars=12000,safety_margin_chars=300))
    mgr.build(state,memory,store,max_context_chars=15000)
    mgr.rehydrate(old.observation_id,path='a.py',start_line=320,end_line=360,information_need='x')
    mgr.rehydrate(old.observation_id,path='a.py',start_line=350,end_line=380,information_need='y')
    r=mgr.build(state,memory,store,max_context_chars=15000)
    assert r.text.count('350 | A_350') == 1
    assert mgr.is_visible_range('a.py',320,380)
