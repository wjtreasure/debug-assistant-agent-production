import pytest
from pydantic import ValidationError
from debug_assistant.contracts import ReflectionContract

def test_reflection_v2_bounds_optional_and_required():
    base={'decision':'continue','reason':'x','current_diagnosis':'x','evidence_sufficient':False,
          'supporting_evidence_ids':[],'contradicting_evidence_ids':[],'confidence':.5}
    ok=ReflectionContract.model_validate({**base,'required_missing_evidence':[{'target':'a','reason':'r'}],
                                          'optional_validation':[{'target':'b','reason':'r'},{'target':'c','reason':'r'}]})
    assert len(ok.optional_validation)==2
    with pytest.raises(ValidationError):
        ReflectionContract.model_validate({**base,'required_missing_evidence':[],
            'optional_validation':[{'target':str(i),'reason':'r'} for i in range(3)]})
