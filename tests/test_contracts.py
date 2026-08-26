import pytest
from pydantic import ValidationError
from debug_assistant.contracts import AgentActionContract, ReflectionContract


def test_action_contract_rejects_extra_fields():
    with pytest.raises(ValidationError):
        AgentActionContract.model_validate({
            'kind':'tool','skill':'repository_exploration','reason':'x','confidence':0.5,
            'tool':'grep','arguments':{'query':'x'},'expected_evidence':'x','params':{}
        })


def test_action_contract_requires_tool_for_tool_action():
    with pytest.raises(ValidationError):
        AgentActionContract.model_validate({'kind':'tool','skill':'issue_triage','reason':'x'})


def test_reflection_contract_is_typed():
    r=ReflectionContract.model_validate({
        'decision':'finish','reason':'enough','current_diagnosis':'x','evidence_sufficient':True,
        'supporting_evidence_ids':['ev-1'],'missing':[],'contradictions':[],'recommended_next_goal':''
    })
    assert r.evidence_sufficient is True
