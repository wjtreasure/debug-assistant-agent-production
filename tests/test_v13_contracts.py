import json
import pytest
from pydantic import ValidationError
from debug_assistant.contracts import DiagnosisReportContract, AgentActionContract, render_contract
from debug_assistant.harness.feature_flags import FeatureFlags


def test_report_contract_ssot_contains_evidence_ids_and_forbids_unknown():
    rendered=render_contract(DiagnosisReportContract,'FINAL_REPORT_SCHEMA')
    schema=json.loads(rendered.split('\n',1)[1])
    assert 'evidence_ids' in schema['properties']
    with pytest.raises(ValidationError):
        DiagnosisReportContract.model_validate({'some_new_field':123})


def test_action_contract_optional_context_ids_and_information_need():
    x=AgentActionContract.model_validate({'kind':'tool','skill':'repository_exploration','reason':'inspect','tool':'read_file',
                                          'arguments':{},'retain_context_ids':['obs-1'],'information_need':'compare implementation'})
    assert x.retain_context_ids==['obs-1']
    assert x.information_need=='compare implementation'


def test_feature_flag_dependencies_reject_invalid_combinations():
    with pytest.raises(ValueError): FeatureFlags(context_catalog=False,model_context_selection=True).validate()
    with pytest.raises(ValueError): FeatureFlags(hypothesis_state=False,fallback_reporter=True).validate()
    FeatureFlags().validate()


def test_each_non_dependent_feature_can_be_disabled_for_ablation():
    names=['observation_reuse','context_catalog','context_budget_packing','model_context_selection','termination_advisory','fallback_reporter']
    for name in names:
        kwargs={name:False}
        # Disabling context_catalog requires model selection to remain disabled (the default).
        flags=FeatureFlags(**kwargs)
        flags.validate()
    # hypothesis_state can be disabled when its dependent fallback is also disabled.
    FeatureFlags(hypothesis_state=False,fallback_reporter=False).validate()
