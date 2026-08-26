from debug_assistant.reporting.fallback import FallbackReportBuilder
from debug_assistant.memory.hypothesis import HypothesisState
from debug_assistant.models import Evidence


def test_fallback_report_uses_only_structured_hypothesis_and_evidence():
    h=HypothesisState(description='division by zero',status='confirmed',confidence=.9,supporting_evidence_ids=['ev-1'])
    e=Evidence('ev-1','read_file','read_file','golden section source',file='pvlib/tools.py',tags=['symbol:_golden_sect_DataFrame'])
    r=FallbackReportBuilder().build('t',h,[e])
    assert r.report_source=='fallback'
    assert r.likely_files==['pvlib/tools.py']
    assert r.likely_symbols==['_golden_sect_DataFrame']
    assert r.evidence_ids==['ev-1']
