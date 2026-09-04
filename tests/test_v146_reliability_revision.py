import asyncio
import httpx
import pytest

from debug_assistant.agent.planner import Planner, PlannerContractError, normalize_planner_action
from debug_assistant.contracts import AgentActionContract
from debug_assistant.harness.deadline import RunDeadline
from debug_assistant.models import AgentState, TaskSpec
from debug_assistant.tools.repository import ReadFileTool
from debug_assistant.repository.index import RepositoryIndex, IndexDeadlineExceeded
from debug_assistant.repository.safe_fs import SafeRepositoryFS
from debug_assistant.llm.openai_compatible import OpenAICompatibleClient


def test_run_deadline_is_monotonic_and_clamps_timeout():
    d = RunDeadline(10)
    assert d.remaining() <= 10
    assert d.effective_timeout(100) <= 10
    assert d.can_start(0)


def test_read_file_rejects_start_beyond_eof(tmp_path):
    path = tmp_path / 'one.py'
    path.write_text('x = 1\n')
    obs = ReadFileTool(tmp_path).execute('one.py', 99, 100)
    assert obs.ok is False
    assert obs.error_type == 'range_out_of_bounds'
    assert obs.metadata['actual_line_count'] == 1


def test_index_deadline_rolls_back_partial_build(tmp_path):
    (tmp_path / 'a.py').write_text('def a():\n    return 1\n')
    idx = RepositoryIndex(tmp_path, tmp_path / 'idx.sqlite', fs=SafeRepositoryFS(tmp_path))
    with pytest.raises(IndexDeadlineExceeded):
        idx.build(deadline=RunDeadline(0))
    assert idx.search('return') == []
    idx.close()


def test_planner_contract_has_one_bounded_repair():
    class LLM:
        def __init__(self): self.calls = 0
        def complete_json(self, system, user, model=None, logical_timeout_seconds=None):
            self.calls += 1
            if self.calls == 1:
                return {'kind': 'tool', 'skill': 'repository_exploration', 'reason': 'inspect',
                        'tool': 'read_file', 'arguments': ['secret-not-in-repair-prompt']}
            assert 'secret-not-in-repair-prompt' not in user
            return {'kind': 'tool', 'skill': 'repository_exploration', 'reason': 'inspect',
                    'tool': 'read_file', 'arguments': {'path': 'a.py'}}
    llm = LLM()
    state = AgentState(TaskSpec('t', 'issue', str('.')))
    action = Planner(llm, type('Tools', (), {'render': lambda self, compact=False: ''})()).propose(state, 'context')
    assert action.arguments == {'path': 'a.py'}
    assert llm.calls == 2


def _planner_tools():
    return type('Tools', (), {'render': lambda self, compact=False: ''})()


def test_single_child_parallel_is_normalized_without_repair():
    child = {'tool': 'read_file', 'arguments': {'path': 'a.py'}}
    class LLM:
        calls = 0
        def complete_json(self, *args, **kwargs):
            self.calls += 1
            return {'kind': 'parallel', 'tool': None, 'arguments': {}, 'actions': [child],
                    'skill': 'repository_exploration', 'reason': 'one child'}
    llm = LLM()
    action = Planner(llm, _planner_tools()).propose(AgentState(TaskSpec('t','i','.')), 'c')
    assert action.kind.value == 'tool'
    assert action.tool == child['tool'] and action.arguments == child['arguments']
    assert llm.calls == 1


def test_repaired_single_child_parallel_is_normalized_without_third_call():
    child = {'tool': 'read_file', 'arguments': {'path': 'a.py'}}
    class LLM:
        def __init__(self): self.calls = 0
        def complete_json(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {'kind': 'tool', 'skill': 'repository_exploration', 'reason': 'bad', 'tool': 'read_file', 'arguments': []}
            return {'kind': 'parallel', 'tool': None, 'arguments': {}, 'actions': [child],
                    'skill': 'repository_exploration', 'reason': 'repair'}
    llm = LLM()
    action = Planner(llm, _planner_tools()).propose(AgentState(TaskSpec('t','i','.')), 'c')
    assert action.kind.value == 'tool' and action.arguments == child['arguments']
    assert llm.calls == 2


@pytest.mark.parametrize('count,valid', [(0, False), (2, True), (5, False)])
def test_parallel_cardinality_is_not_silently_changed(count, valid):
    children = [{'tool': 'read_file', 'arguments': {'path': f'{i}.py'}} for i in range(count)]
    normalized, meta = normalize_planner_action({'kind': 'parallel', 'skill': 'repository_exploration', 'reason': 'parallel', 'actions': children})
    assert meta is None
    result = AgentActionContract.model_validate(normalized) if valid else None
    if valid:
        assert result.kind == 'parallel' and len(result.actions) == count
    else:
        with pytest.raises(Exception):
            AgentActionContract.model_validate(normalized)


def test_normalization_preserves_child_tool_and_arguments_exactly():
    child = {'tool': 'read_file', 'arguments': {'path': 'a.py', 'start_line': 2, 'end_line': 4}}
    normalized, _ = normalize_planner_action({'kind': 'parallel', 'tool': None, 'arguments': {}, 'actions': [child]})
    assert normalized['tool'] == child['tool']
    assert normalized['arguments'] == child['arguments']


def test_runtime_trace_records_normalization_before_any_contract_failure(monkeypatch, tmp_path):
    from debug_assistant.config import AppConfig
    from debug_assistant.harness.runtime import AgentHarness
    repo = tmp_path / 'repo'; repo.mkdir(); (repo / 'a.py').write_text('value = 1\n')
    class LLM:
        def __init__(self): self.calls = []; self.n = 0
        def complete_json(self, system, user, model=None, logical_timeout_seconds=None):
            self.n += 1
            self._usage(system, user)
            if self.n == 1:
                return {'kind': 'tool', 'skill': 'repository_exploration', 'reason': 'bad', 'tool': 'read_file', 'arguments': []}
            if self.n == 2:
                return {'kind': 'parallel', 'tool': None, 'arguments': {}, 'actions': [
                    {'tool': 'read_file', 'arguments': {'path': 'a.py'}}
                ], 'skill': 'repository_exploration', 'reason': 'repair'}
            return {'kind': 'finish', 'skill': 'report_synthesis', 'reason': 'stop', 'tool': None, 'arguments': {}}
        def _usage(self, system, user):
            self.calls.append({'model': 'fake', 'prompt_tokens': 1, 'completion_tokens': 1,
                               'total_tokens': 2, 'input_tokens': 1, 'output_tokens': 1,
                               'prompt_chars': len(system) + len(user), 'completion_chars': 1})
    fake = LLM(); monkeypatch.setattr('debug_assistant.harness.runtime.build_llm', lambda cfg: fake)
    cfg = AppConfig(); cfg.model.provider = 'mock'; cfg.harness.build_task_index = False
    cfg.harness.max_steps = 10; cfg.harness.reflect_every = 99; cfg.harness.trace_dir = str(tmp_path / 'traces')
    result = AgentHarness(cfg).run(TaskSpec('normalization-trace', 'issue', str(repo)))
    events = [__import__('json').loads(x) for x in open(result['trace']['trace_path'], encoding='utf-8')]
    types = [e['type'] for e in events]
    assert 'PLANNER_ACTION_NORMALIZED' in types
    assert 'PLANNER_CONTRACT_INVALID' not in types
    assert types.index('PLANNER_ACTION_NORMALIZED') < types.index('ACTION_PROPOSED')


def test_runtime_recovers_after_one_final_planner_contract_rejection(monkeypatch, tmp_path):
    from debug_assistant.config import AppConfig
    from debug_assistant.harness.runtime import AgentHarness
    repo = tmp_path / 'repo'; repo.mkdir()
    (repo / 'a.py').write_text('a = 1\n'); (repo / 'b.py').write_text('b = 2\n')
    class LLM:
        def __init__(self): self.n = 0; self.calls = []
        def complete_json(self, system, user, model=None, logical_timeout_seconds=None):
            self.n += 1; self.calls.append({'model':'fake','prompt_tokens':1,'completion_tokens':1,'total_tokens':2})
            if 'FINAL_REPORT_SCHEMA' in user:
                return {'summary':'done','root_cause':'uncertain','likely_files':['a.py'],'likely_symbols':[],
                        'impact_scope':[],'recommended_change_points':[],'uncertainties':[],
                        'next_checks':[],'evidence_ids':[],'confidence':.5}
            if self.n == 1:
                return {'kind':'tool','skill':'repository_exploration','reason':'read a','tool':'read_file','arguments':{'path':'a.py'}}
            if self.n == 2:
                return {'kind':'tool','skill':'repository_exploration','reason':'bad','tool':'read_file','arguments':[]}
            if self.n == 3:
                return {'kind':'parallel','skill':'repository_exploration','reason':'still bad','tool':None,'arguments':{},'actions':[]}
            if self.n == 4:
                return {'kind':'tool','skill':'repository_exploration','reason':'read b','tool':'read_file','arguments':{'path':'b.py'}}
            return {'kind':'finish','skill':'report_synthesis','reason':'done','tool':None,'arguments':{}}
    fake = LLM(); monkeypatch.setattr('debug_assistant.harness.runtime.build_llm', lambda cfg: fake)
    cfg = AppConfig(); cfg.model.provider='mock'; cfg.harness.build_task_index=False
    cfg.harness.max_steps=10; cfg.harness.reflect_every=99; cfg.harness.trace_dir=str(tmp_path/'traces')
    result = AgentHarness(cfg).run(TaskSpec('planner-recovery','issue',str(repo)))
    events = [__import__('json').loads(x) for x in open(result['trace']['trace_path'], encoding='utf-8')]
    rejected = [e for e in events if e['type']=='PLANNER_CONTRACT_REJECTED']
    assert len(rejected) == 1
    assert result['state']['tool_calls'] == 2
    assert result['state']['evidence'] == 2
    assert result['state']['planner_contract_failure_count'] == 0
    assert not any(e['type']=='PROVIDER_DEGRADED' for e in events)
    assert 'raw' not in __import__('json').dumps(rejected)


def test_repeated_planner_contract_rejection_finalizes_with_existing_evidence(monkeypatch, tmp_path):
    from debug_assistant.config import AppConfig
    from debug_assistant.harness.runtime import AgentHarness
    repo = tmp_path / 'repo'; repo.mkdir(); (repo/'a.py').write_text('a = 1\n')
    class LLM:
        def __init__(self): self.n=0; self.calls=[]
        def complete_json(self, system, user, model=None, logical_timeout_seconds=None):
            self.n+=1; self.calls.append({'model':'fake','prompt_tokens':1,'completion_tokens':1,'total_tokens':2})
            if 'FINAL_REPORT_SCHEMA' in user:
                return {'summary':'partial','root_cause':'uncertain','likely_files':['a.py'],'likely_symbols':[],
                        'impact_scope':[],'recommended_change_points':[],'uncertainties':['planner contract repeatedly invalid'],
                        'next_checks':[],'evidence_ids':[],'confidence':.3}
            if self.n == 1:
                return {'kind':'tool','skill':'repository_exploration','reason':'read','tool':'read_file','arguments':{'path':'a.py'}}
            return {'kind':'parallel','skill':'repository_exploration','reason':'invalid','tool':None,'arguments':{},'actions':[]}
    fake=LLM(); monkeypatch.setattr('debug_assistant.harness.runtime.build_llm',lambda cfg:fake)
    cfg=AppConfig(); cfg.model.provider='mock'; cfg.harness.build_task_index=False
    cfg.harness.max_steps=10; cfg.harness.reflect_every=99; cfg.harness.trace_dir=str(tmp_path/'traces')
    result=AgentHarness(cfg).run(TaskSpec('planner-repeated','issue',str(repo)))
    events=[__import__('json').loads(x) for x in open(result['trace']['trace_path'],encoding='utf-8')]
    rejected=[e for e in events if e['type']=='PLANNER_CONTRACT_REJECTED']
    assert len(rejected)==2
    assert result['state']['status']=='partial_success'
    assert result['state']['evidence']==1
    assert not any(e['type']=='PROVIDER_DEGRADED' for e in events)


@pytest.mark.asyncio
async def test_invalid_json_closes_logical_lifecycle_as_provider_success():
    class Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(200, request=request, json={
                'choices': [{'message': {'content': 'not json'}}],
                'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3},
            })
    client = OpenAICompatibleClient('https://example.invalid/v1', 'key', 'm', async_transport=Transport())
    with pytest.raises(Exception):
        await client.acomplete_json('s', 'u', logical_timeout_seconds=1)
    finished = [e for e in client.events if e['type'] == 'LLM_LOGICAL_CALL_FINISHED']
    assert len(finished) == 1
    assert finished[0]['payload']['success'] is False
    assert finished[0]['payload']['error_type'] == 'invalid_json'
    assert finished[0]['payload']['provider_success'] is True
    assert finished[0]['payload']['provider_failure'] is False
    assert client.calls[-1]['total_tokens'] == 3
