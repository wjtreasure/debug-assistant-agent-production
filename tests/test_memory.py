from debug_assistant.memory.evidence_memory import EvidenceMemory
from debug_assistant.models import ToolObservation

def test_dedup_memory():
    m=EvidenceMemory(); o=ToolObservation('grep',True,'a.py:1: boom')
    assert m.add_observation(o) is not None
    assert m.add_observation(o) is None
