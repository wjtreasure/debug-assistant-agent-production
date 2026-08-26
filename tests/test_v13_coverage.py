from debug_assistant.memory.coverage import ReadCoverageIndex
from debug_assistant.memory.observation_store import ObservationStore
from debug_assistant.models import ToolObservation


def test_read_coverage_exact_and_subrange_reuse_only():
    c=ReadCoverageIndex(); c.add(path='a.py',start_line=300,end_line=400,observation_id='obs-a')
    assert c.find_covering(path='a.py',start_line=300,end_line=400).observation_id=='obs-a'
    assert c.find_covering(path='a.py',start_line=320,end_line=380).observation_id=='obs-a'
    assert c.find_covering(path='a.py',start_line=350,end_line=450) is None


def test_observation_store_rehydrates_original_object():
    s=ObservationStore(); o=ToolObservation('read_file',True,'x',{'path':'a.py','start_line':1,'end_line':2})
    s.add(o)
    assert s.get(o.observation_id) is o
