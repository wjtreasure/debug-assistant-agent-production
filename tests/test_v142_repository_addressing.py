from pathlib import Path

from debug_assistant.repository.paths import (
    RepositoryPathResolver, RepositoryPathMatcher, ResolutionMode,
    AmbiguousPathError, PathRejectedError,
)
from debug_assistant.repository.safe_fs import SafeRepositoryFS
from debug_assistant.tools.repository import GrepTool, ReadFileTool


def _repo(tmp_path):
    repo=tmp_path/'repo'; (repo/'astroid/interpreter').mkdir(parents=True); (repo/'tests').mkdir()
    (repo/'astroid/modutils.py').write_text('def get_source_file(path):\n    return path\n')
    (repo/'astroid/manager.py').write_text('class Manager:\n    pass\n')
    (repo/'astroid/interpreter/spec.py').write_text('VALUE=1\n')
    return repo


def test_resolver_normalizes_relative_and_windows_separators(tmp_path):
    repo=_repo(tmp_path); r=RepositoryPathResolver(SafeRepositoryFS(repo))
    a=r.resolve_file('./astroid/modutils.py')
    b=r.resolve_file(r'astroid\modutils.py')
    assert a.relative_path==b.relative_path=='astroid/modutils.py'
    assert a.strategy=='normalized_relative' and b.strategy=='normalized_relative'


def test_resolver_accepts_absolute_inside_repo_but_rejects_outside(tmp_path):
    repo=_repo(tmp_path); r=RepositoryPathResolver(SafeRepositoryFS(repo))
    target=(repo/'astroid/modutils.py').resolve()
    got=r.resolve_file(str(target))
    assert got.relative_path=='astroid/modutils.py' and got.strategy=='absolute_inside_repo'
    outside=tmp_path/'outside.py'; outside.write_text('x=1')
    try:
        r.resolve_file(str(outside))
    except PathRejectedError:
        pass
    else:
        raise AssertionError('outside absolute path must be rejected')


def test_resolver_unique_suffix_and_basename_are_read_tolerant_only(tmp_path):
    repo=_repo(tmp_path); r=RepositoryPathResolver(SafeRepositoryFS(repo))
    assert r.resolve_file('interpreter/spec.py').strategy=='unique_suffix'
    assert r.resolve_file('modutils.py').strategy=='unique_basename'
    try:
        r.resolve_file('modutils.py',mode=ResolutionMode.EXACT)
    except Exception as exc:
        assert getattr(exc,'error_type',None)=='path_not_found'
    else:
        raise AssertionError('EXACT must not basename-recover')


def test_ambiguous_basename_is_not_silently_guessed(tmp_path):
    repo=_repo(tmp_path); (repo/'src').mkdir(); (repo/'src/parser.py').write_text('x=1'); (repo/'tests/parser.py').write_text('x=2')
    r=RepositoryPathResolver(SafeRepositoryFS(repo))
    try:
        r.resolve_file('parser.py')
    except AmbiguousPathError as exc:
        assert set(exc.candidates)=={'src/parser.py','tests/parser.py'}
    else:
        raise AssertionError('ambiguous basename must not resolve')


def test_ambiguous_read_is_structured_tool_observation_not_exception(tmp_path):
    repo=_repo(tmp_path); (repo/'src').mkdir(); (repo/'src/parser.py').write_text('x=1'); (repo/'tests/parser.py').write_text('x=2')
    obs=ReadFileTool(repo).execute('parser.py',1,5)
    assert obs.ok is False and obs.error_type=='ambiguous_path'
    assert obs.metadata['planner_retryable'] is True
    assert obs.metadata['retryable'] is False  # do not retry the same deterministic call
    assert set(obs.metadata['candidates'])=={'src/parser.py','tests/parser.py'}


def test_grep_path_matcher_real_regression_exact_repo_relative(tmp_path):
    repo=_repo(tmp_path)
    obs=GrepTool(repo).execute('def get_source_file',glob='astroid/modutils.py')
    assert obs.ok and obs.metadata['matches']==1
    assert 'astroid/modutils.py:1:' in obs.content
    assert obs.metadata['path_pattern']['strategy']=='repo_relative_exact'


def test_grep_glob_semantics_are_explicit(tmp_path):
    repo=_repo(tmp_path); (repo/'astroid/interpreter/nested.py').write_text('MARK=1\n')
    g=GrepTool(repo)
    assert 'astroid/interpreter/nested.py' in g.execute('MARK',glob='*.py').content
    assert g.execute('MARK',glob='astroid/*.py').metadata['matches']==0
    assert 'astroid/interpreter/nested.py' in g.execute('MARK',glob='astroid/**/*.py').content
    # ** spans zero or more directories, so direct astroid/*.py files also match.
    assert 'astroid/modutils.py' in g.execute('def get_source_file',glob='astroid/**/*.py').content


def test_glob_does_not_suffix_recover_missing_repo_relative_prefix(tmp_path):
    repo=tmp_path/'repo'; (repo/'src/astroid').mkdir(parents=True); (repo/'src/astroid/modutils.py').write_text('NEEDLE=1\n')
    obs=GrepTool(repo).execute('NEEDLE',glob='astroid/modutils.py')
    assert obs.ok and obs.metadata['matches']==0


def test_path_matcher_rejects_parent_traversal_pattern():
    m=RepositoryPathMatcher()
    try:
        m.normalize_pattern('../*.py')
    except PathRejectedError:
        pass
    else:
        raise AssertionError('parent traversal glob must be rejected')

def test_glob_preserves_leading_dot_directories(tmp_path):
    repo=tmp_path/'repo'; (repo/'.github').mkdir(parents=True); (repo/'.github/workflow.py').write_text('DOTMARK=1\n')
    obs=GrepTool(repo).execute('DOTMARK',glob='.github/*.py')
    assert obs.ok and '.github/workflow.py:1:' in obs.content

def test_read_resolver_rejects_symlink_escape(tmp_path):
    repo=tmp_path/'repo'; repo.mkdir(); outside=tmp_path/'secret.py'; outside.write_text('SECRET=1\n')
    link=repo/'secret.py'
    try:
        link.symlink_to(outside)
    except OSError:
        return
    obs=ReadFileTool(repo).execute('secret.py',1,5)
    assert obs.ok is False and obs.error_type=='path_rejected'
    assert 'SECRET=1' not in obs.content

def test_runtime_ambiguous_path_returns_to_planner_without_reflection(monkeypatch,tmp_path):
    import json
    from debug_assistant.config import AppConfig
    from debug_assistant.harness.runtime import AgentHarness
    from debug_assistant.models import TaskSpec

    repo=tmp_path/'repo'; (repo/'src').mkdir(parents=True); (repo/'tests').mkdir()
    (repo/'src/parser.py').write_text('ROOT_CAUSE=1\n'); (repo/'tests/parser.py').write_text('TEST_ONLY=1\n')

    class LLM:
        def __init__(self): self.calls=[]; self.n=0
        def _u(self,s,u): self.calls.append({'model':'fake','prompt_tokens':10,'completion_tokens':2,'total_tokens':12,'input_tokens':10,'output_tokens':2,'prompt_chars':len(s)+len(u),'completion_chars':10,'latency_ms':1,'cached_tokens':None,'reasoning_tokens':None})
        def complete_json(self,system,user,model=None):
            self._u(system,user)
            if 'FINAL_REPORT_SCHEMA' in user:
                return {'summary':'done','root_cause':'src parser','likely_files':['src/parser.py'],'likely_symbols':[],
                        'impact_scope':[],'recommended_change_points':[],'uncertainties':[],'next_checks':[],
                        'evidence_ids':[],'confidence':.5}
            if 'REFLECTION_SCHEMA' in user:
                raise AssertionError('path ambiguity should replan without a reflection call')
            self.n+=1
            if self.n==1:
                return {'kind':'tool','skill':'repository_exploration','reason':'read parser','tool':'read_file',
                        'arguments':{'path':'parser.py','start_line':1,'end_line':5},'expected_evidence':'source',
                        'information_need':'inspect parser','confidence':.5}
            if self.n==2:
                return {'kind':'tool','skill':'repository_exploration','reason':'choose source parser','tool':'read_file',
                        'arguments':{'path':'src/parser.py','start_line':1,'end_line':5},'expected_evidence':'source',
                        'information_need':'inspect source parser','confidence':.7}
            return {'kind':'finish','skill':'report_synthesis','reason':'enough','tool':None,'arguments':{},'confidence':.7}

    fake=LLM(); monkeypatch.setattr('debug_assistant.harness.runtime.build_llm',lambda cfg: fake)
    cfg=AppConfig(); cfg.model.provider='mock'; cfg.harness.build_task_index=False; cfg.harness.reflect_every=99; cfg.harness.trace_dir=str(tmp_path/'traces')
    result=AgentHarness(cfg).run(TaskSpec('ambiguous','parser bug',str(repo)))
    assert result['state']['tool_calls']==2
    assert result['state']['reflection_count']==0
    events=[json.loads(x) for x in open(result['trace']['trace_path'],encoding='utf-8') if x.strip()]
    assert any(e['type']=='PATH_AMBIGUOUS' for e in events)
    assert any(e['type']=='PATH_REPLAN_REQUESTED' for e in events)
