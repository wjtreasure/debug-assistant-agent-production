import time
from debug_assistant.harness.budget import BudgetController
from debug_assistant.harness.information_need import InformationNeedTracker
from debug_assistant.memory.evidence_memory import EvidenceMemory
from debug_assistant.models import ToolObservation

def test_budget_phases_are_cost_aware():
    b=BudgetController(max_steps=20,max_tool_calls=40,max_llm_calls=40,max_total_tokens=1000,max_wall_time_seconds=100,started_at=time.time())
    assert b.snapshot(steps=1,tool_calls=1,llm_calls=1,tokens=100).phase=='explore'
    assert b.snapshot(steps=12,tool_calls=1,llm_calls=1,tokens=100).phase=='converge'
    assert b.snapshot(steps=17,tool_calls=1,llm_calls=1,tokens=100).phase=='verify_only'
    assert b.snapshot(steps=19,tool_calls=1,llm_calls=1,tokens=100).phase=='finalize'

def test_repeated_lexical_no_gain_emits_semantic_advisory():
    t=InformationNeedTracker(max_no_gain_attempts=3); n=t.get_or_create('find compatibility behavior',1)
    for _ in range(2):
        t.note_attempt(n,'lexical'); t.note_result(n,[],False)
    assert 'semantic or hybrid' in t.advisory(n)

def test_code_search_candidate_does_not_become_evidence():
    m=EvidenceMemory(); obs=ToolObservation('code_search',True,'x.py: candidate',{'information_source':'candidate_retrieval'})
    assert m.add_observation(obs) is None
