from debug_assistant.memory.evidence_memory import EvidenceMemory
from debug_assistant.memory.observation_store import ObservationStore
from debug_assistant.models import AgentState, TaskSpec, ToolObservation
from debug_assistant.config import ContextConfig
from debug_assistant.context.manager import ContextManager


def _read_obs(name: str, start: int, end: int, marker: str = ''):
    content='\n'.join(f"{i:5d} | value_{i} {marker}" for i in range(start,end+1))
    return ToolObservation(
        'read_file',True,content,
        {'path':name,'start_line':start,'end_line':end,'requested_line_count':end-start+1,'truncated':False}
    )


def _build_context(state, memory, *, recent=2):
    store=ObservationStore()
    for observation in state.observations:
        store.add(observation)
    manager=ContextManager(ContextConfig(
        max_item_chars=16000, fallback_recent_count=recent,
        target_active_items=recent * 2, hard_active_items=recent * 3,
    ))
    return manager.build(state,memory,store,max_context_chars=50000).text


def test_recent_raw_preserves_tail_when_evidence_excerpt_is_compressed(tmp_path):
    state=AgentState(TaskSpec('t','issue',str(tmp_path)))
    memory=EvidenceMemory()
    obs=_read_obs('a.py',1,150,'TAILMARK')
    state.observations.append(obs)
    ev=memory.add_observation(obs); state.evidence.append(ev)
    assert ev.excerpt_truncated is True
    assert ev.excerpt_end_line is not None and ev.excerpt_end_line < 150
    assert ev.source_end_line == 150
    ctx=_build_context(state,memory)
    assert f"EVIDENCE {ev.evidence_id}" in ctx
    assert obs.observation_id not in ctx
    assert '  150 | value_150 TAILMARK' in ctx


def test_evidence_provenance_and_truthful_coverage(tmp_path):
    memory=EvidenceMemory(); obs=_read_obs('a.py',10,120)
    ev=memory.add_observation(obs)
    assert ev.raw_observation_id == obs.observation_id
    assert ev.source_start_line == 10 and ev.source_end_line == 120
    assert ev.line_start == 10 and ev.line_end == 120
    assert ev.excerpt_truncated is True
    assert ev.excerpt_start_line == 10
    assert ev.excerpt_end_line is not None and ev.excerpt_end_line < 120


def test_recent_window_keeps_two_and_evicts_oldest_full_content(tmp_path):
    state=AgentState(TaskSpec('t','issue',str(tmp_path))); memory=EvidenceMemory()
    observations=[]
    for idx,name in enumerate(('A.py','B.py','C.py'),1):
        obs=_read_obs(name,1,5,f'FULL_{name}')
        observations.append(obs); state.observations.append(obs)
        ev=memory.add_observation(obs); state.evidence.append(ev)
    ctx=_build_context(state,memory)
    a,b,c=observations
    assert memory.pinned[0].evidence_id in ctx  # compact known source/evidence pointer
    assert f"EVIDENCE {memory.pinned[0].evidence_id}" not in ctx
    assert f"EVIDENCE {memory.pinned[1].evidence_id}" in ctx
    assert f"EVIDENCE {memory.pinned[2].evidence_id}" in ctx
    # Old A remains traceable through historical evidence metadata/summary.
    assert memory.pinned[0].evidence_id in ctx
    # Each selected fact owns one model-visible body.
    assert ctx.count('FULL_B.py') == 5
    assert ctx.count('FULL_C.py') == 5


def test_failed_observation_is_visible_in_recent_raw_window(tmp_path):
    state=AgentState(TaskSpec('t','issue',str(tmp_path))); memory=EvidenceMemory()
    obs=ToolObservation('git_log',False,'fatal: not a git repository',{'retryable':False},'git_error')
    state.observations.append(obs)
    ctx=_build_context(state,memory)
    assert obs.observation_id not in ctx
    assert 'not Evidence; do not cite' in ctx
    assert 'ok=False' in ctx
    assert 'git_error' in ctx
    assert 'fatal: not a git repository' in ctx
