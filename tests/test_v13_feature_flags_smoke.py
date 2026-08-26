import pytest
from debug_assistant.config import AppConfig
from debug_assistant.harness.runtime import AgentHarness
from debug_assistant.models import TaskSpec

@pytest.mark.parametrize('flag_name',[
    'observation_reuse','context_catalog','context_budget_packing','termination_advisory','fallback_reporter'
])
def test_single_feature_off_smoke(flag_name,tmp_path):
    repo=tmp_path/'repo'; repo.mkdir(); (repo/'src').mkdir()
    (repo/'src'/'parser.py').write_text("def parse_value(values, idx):\n    if idx > len(values):\n        raise ValueError('invalid boundary')\n    return values[idx]\n")
    cfg=AppConfig(); cfg.model.provider='mock'; cfg.harness.build_task_index=False; cfg.harness.trace_dir=str(tmp_path/'traces')
    setattr(cfg.harness.features,flag_name,False)
    cfg.harness.features.validate()
    r=AgentHarness(cfg).run(TaskSpec(f'off-{flag_name}','parse_value boundary invalid',str(repo)))
    assert r['state']['status'] in ('success','partial_success')


def test_hypothesis_off_requires_fallback_off_and_still_smokes(tmp_path):
    repo=tmp_path/'repo'; repo.mkdir(); (repo/'src').mkdir()
    (repo/'src'/'parser.py').write_text("def parse_value(values, idx):\n    if idx > len(values):\n        raise ValueError('invalid boundary')\n    return values[idx]\n")
    cfg=AppConfig(); cfg.model.provider='mock'; cfg.harness.build_task_index=False; cfg.harness.trace_dir=str(tmp_path/'traces')
    cfg.harness.features.hypothesis_state=False; cfg.harness.features.fallback_reporter=False
    cfg.harness.features.validate()
    r=AgentHarness(cfg).run(TaskSpec('off-hypothesis','parse_value boundary invalid',str(repo)))
    assert r['state']['status']=='success'
