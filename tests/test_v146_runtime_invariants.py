from __future__ import annotations

import random
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from debug_assistant.agent.reflection import Reflector
from debug_assistant.contracts import ReflectionContract
from debug_assistant.harness.budget_gate import RemainingBudgetGate
from debug_assistant.harness.evidence_bundle import build_evidence_bundle, select_ready_obligations
from debug_assistant.harness.guards import RouterGuard
from debug_assistant.harness.obligations import EvidenceObligationTracker, ObligationStatus
from debug_assistant.harness.parallel import execute_parallel_group
from debug_assistant.harness.retry import RetryPolicy
from debug_assistant.harness.semantic_invariants import SemanticInvariantError, validate_semantic_candidate
from debug_assistant.memory.observation_store import ObservationStore
from debug_assistant.models import ActionKind, ActionProposal, AgentState, Evidence, ToolObservation, TaskSpec
from debug_assistant.reporting.rules import apply_reporting_rules, derive_claim_status
from debug_assistant.repository.index import RepositoryIndex
from debug_assistant.tools.base import Tool, ToolArgs, ToolSpec
from debug_assistant.tools.registry import ToolRegistry


def _symbol_lookup(rows):
    def lookup(query, limit=100):
        q=str(query)
        return [r for r in rows if q in {r['name'],r['qualified_name']} or r['qualified_name'].endswith('.'+q)][:limit]
    return lookup


def _source(eid: str, file: str, a: int, b: int, obs_id: str | None = None):
    return Evidence(
        evidence_id=eid, kind='read_file', source='read_file', summary='source', file=file,
        line_start=a, line_end=b, raw_observation_id=obs_id,
        source_start_line=a, source_end_line=b, excerpt_start_line=a, excerpt_end_line=b,
    )


def test_structured_symbol_scope_corrects_wrong_file_and_range(tmp_path):
    rows=[{'path':'pkg/mod.py','name':'target','qualified_name':'C.target','kind':'FunctionDef','start_line':20,'end_line':30}]
    tr=EvidenceObligationTracker(repo_root=tmp_path,symbol_lookup=_symbol_lookup(rows))
    tr.sync([{
        'target':'target implementation','file':'wrong.py','symbol':'target','line_start':1,'line_end':2,
        'goal_type':'behavior','reason':'inspect implementation',
    }])
    obj=next(iter(tr.items.values()))
    assert obj.scope_valid is True
    assert obj.canonical_files == ('pkg/mod.py',)
    assert obj.line_hint == (20,30)
    assert obj.canonical_symbols == ('target',)
    events=tr.pop_events()
    reasons=[e[1].get('reason') for e in events if e[0]=='OBLIGATION_SCOPE_CORRECTED']
    assert 'unique_symbol_file_authoritative' in reasons
    assert 'symbol_range_authoritative' in reasons


def test_structured_symbol_scope_fails_closed_when_ambiguous(tmp_path):
    rows=[
        {'path':'a.py','name':'target','qualified_name':'A.target','kind':'FunctionDef','start_line':1,'end_line':2},
        {'path':'b.py','name':'target','qualified_name':'B.target','kind':'FunctionDef','start_line':5,'end_line':6},
    ]
    tr=EvidenceObligationTracker(repo_root=tmp_path,symbol_lookup=_symbol_lookup(rows))
    tr.sync([{'target':'target body','symbol':'target','goal_type':'behavior','reason':'inspect'}])
    obj=next(iter(tr.items.values()))
    assert obj.scope_valid is False
    assert obj.scope_error == 'ambiguous_or_missing_symbol'
    assert any(t=='AMBIGUOUS_SCOPE' for t,_ in tr.pop_events())


def test_composite_legacy_scope_never_becomes_ready(tmp_path):
    rows=[
        {'path':'m.py','name':'first_function','qualified_name':'first_function','kind':'FunctionDef','start_line':1,'end_line':3},
        {'path':'m.py','name':'second_function','qualified_name':'second_function','kind':'FunctionDef','start_line':5,'end_line':7},
    ]
    tr=EvidenceObligationTracker(repo_root=tmp_path,symbol_lookup=_symbol_lookup(rows))
    tr.sync([{'target':'first_function / second_function behavior','location':'m.py','goal_type':'causality','reason':'need both'}])
    obj=next(iter(tr.items.values()))
    assert obj.scope_valid is False
    tr.note_evidence(_source('ev','m.py',1,7),'all source')
    assert obj.evidence_ready is False


class _Args(ToolArgs):
    value: int = 0


class _DelayTool(Tool):
    spec=ToolSpec('delay','delay',_Args,side_effect='none')
    def __init__(self, delay: float, *, ok=True):
        self.delay=delay; self._ok=ok
    def execute(self,value=0):
        time.sleep(self.delay)
        return ToolObservation('delay',self._ok,f'value={value}',{'retryable':False},None if self._ok else 'PermanentError')


class _Registry:
    def __init__(self, tools): self.tools=tools
    def get(self,name): return self.tools[name]


def test_parallel_completion_order_cannot_change_ingestion_order():
    reg=_Registry({'slow':_DelayTool(.03),'fast':_DelayTool(.001)})
    # ToolObservation.tool names are irrelevant here; child rows must follow Planner action_index.
    actions=[{'action_id':'first','tool':'slow','arguments':{'value':1}}, {'action_id':'second','tool':'fast','arguments':{'value':2}}]
    result=execute_parallel_group(reg,actions,group_id='G',max_workers=2,group_timeout_seconds=1,retry_policy=RetryPolicy(max_attempts=1))
    assert [x.action_id for x in result.children] == ['first','second']
    assert [x.action_index for x in result.children] == [0,1]
    assert [x.observation.content for x in result.children] == ['value=1','value=2']


def test_parallel_partial_failure_keeps_successful_child_result():
    reg=_Registry({'good':_DelayTool(.001,ok=True),'bad':_DelayTool(.001,ok=False)})
    actions=[{'tool':'good','arguments':{'value':1}}, {'tool':'bad','arguments':{'value':2}}]
    result=execute_parallel_group(reg,actions,group_id='G',max_workers=2,group_timeout_seconds=1,retry_policy=RetryPolicy(max_attempts=1))
    assert result.status == 'partial'
    assert result.children[0].observation.ok is True
    assert result.children[1].observation.ok is False


def test_parallel_dependency_is_rejected_not_silently_rewritten(tmp_path):
    repo=tmp_path/'repo'; repo.mkdir(); (repo/'a.py').write_text('x=1\n')
    registry=ToolRegistry(repo)
    guard=RouterGuard(registry)
    action=ActionProposal(
        kind=ActionKind.PARALLEL, skill='repository_exploration', reason='independent reads',
        actions=[
            {'action_id':'a0','tool':'read_file','arguments':{'path':'a.py','start_line':1,'end_line':1}},
            {'action_id':'a1','tool':'grep','arguments':{'query':'{{result_of_action_0}}','glob':'*.py'}},
        ],
    )
    gd=guard.validate(action,AgentState(TaskSpec('t','issue',str(repo))))
    assert gd.ok is False
    assert gd.error['error_type']=='parallel_dependency'


def test_inspect_symbol_context_returns_bounded_source_and_trust_markers(tmp_path):
    repo=tmp_path/'repo'; repo.mkdir()
    (repo/'mod.py').write_text(
        'def leaf():\n    return 1\n\n'
        'def target():\n    return leaf()\n\n'
        'def caller():\n    return target()\n'
    )
    idx=RepositoryIndex(repo,tmp_path/'idx.sqlite'); idx.build()
    try:
        data=idx.inspect_symbol_context('target','mod.py',include_source=True,max_source_chars=5000)
        assert data['ok'] is True
        assert data['definition']['source_code']
        assert any(x['symbol']=='caller' and x['resolution_kind']=='exact' and x.get('source_code') for x in data['callers'])
        assert any(x['symbol']=='leaf' and x['resolution_kind']=='exact' and x.get('source_code') for x in data['callees'])
        assert data['source_chars_used'] <= 5000
    finally:
        idx.close()


def test_evidence_bundle_batches_only_same_root_and_deduplicates_source(tmp_path):
    rows=[{'path':'m.py','name':'f','qualified_name':'f','kind':'FunctionDef','start_line':1,'end_line':3}]
    tr=EvidenceObligationTracker(repo_root=tmp_path,symbol_lookup=_symbol_lookup(rows))
    # Two distinct atomic obligations deliberately share the same source but same root.
    tr.sync([
        {'target':'f behavior','file':'m.py','symbol':'f','goal_type':'behavior','reason':'behavior','information_need_root_id':'N1'},
    ])
    first=next(iter(tr.items.values()))
    # Create a second obligation with a different semantic goal but same exact source/root.
    tr.sync([
        {'target':'f behavior','file':'m.py','symbol':'f','goal_type':'behavior','reason':'behavior','information_need_root_id':'N1'},
        {'target':'f causal role','file':'m.py','symbol':'f','goal_type':'causality','reason':'causal','information_need_root_id':'N1'},
        {'target':'other root','file':'m.py','symbol':'f','goal_type':'caller','reason':'other','information_need_root_id':'N2'},
    ])
    store=ObservationStore(); obs=ToolObservation('read_file',True,'    1 | def f():\n    2 |     x=1\n    3 |     return x',{'path':'m.py','start_line':1,'end_line':3}); store.add(obs)
    ev=_source('ev','m.py',1,3,obs.observation_id)
    tr.note_evidence(ev,obs.content)
    selected=select_ready_obligations(tr,max_items=3)
    assert selected and all(x.information_need_root_id=='N1' for x in selected)
    bundle=build_evidence_bundle(tr,[ev],store,bundle_id='B',max_items=3,max_chars=4000)
    assert bundle is not None
    assert bundle.root_id=='N1'
    assert len(bundle.items)==2
    assert bundle.text.count('def f():') == 1
    assert len({x.obligation_id for x in bundle.items})==2


def test_semantic_invariants_reject_presented_without_submission_and_supported_gap(tmp_path):
    tr=EvidenceObligationTracker(repo_root=tmp_path,symbol_lookup=lambda q,limit=100: [])
    tr.sync([{'target':'body','file':'m.py','line_start':1,'line_end':2,'goal_type':'behavior','reason':'body'}])
    obj=next(iter(tr.items.values()))
    obj.last_presented_reflection_id='R1'
    hs=SimpleNamespace(status='partial',supporting_evidence_ids=[],required_missing_evidence=[{'target':'body'}],evidence_sufficient=False)
    with pytest.raises(SemanticInvariantError,match='PRESENTED'):
        validate_semantic_candidate(tr,hs,submitted_reflection_ids={'R2'})

    obj.last_presented_reflection_id=None
    hs=SimpleNamespace(status='supported',supporting_evidence_ids=['ev'],required_missing_evidence=[{'target':'body'}],evidence_sufficient=True)
    with pytest.raises(SemanticInvariantError,match='required gaps'):
        validate_semantic_candidate(tr,hs)


def test_report_status_is_deterministically_derived_and_change_point_grounded(tmp_path):
    repo=tmp_path/'repo'; repo.mkdir(); (repo/'m.py').write_text('def f():\n    return 1\n')
    idx=RepositoryIndex(repo,tmp_path/'idx.sqlite'); idx.build()
    try:
        ev=_source('ev','m.py',1,2)
        claim={'text':'f is defined here','claim_type':'source_fact','status':'hypothesis','evidence_ids':['ev'],'file':'m.py','line_start':1,'line_end':2}
        assert derive_claim_status(claim,evidence_by_id={'ev':ev})=='observed'
        report=SimpleNamespace(
            claims=[claim],
            recommended_change_points=[{'file':'wrong.py','symbol':'f','reason':'candidate'}],
        )
        report,corrections=apply_reporting_rules(report,evidence=[ev],repository_index=idx,obligations=None,hypothesis=SimpleNamespace(status='partial'))
        assert report.claims[0]['status']=='observed'
        assert report.recommended_change_points[0]['file']=='m.py'
        assert any(x['kind']=='change_point_file' for x in corrections)
    finally:
        idx.close()


def test_reflector_keeps_valid_review_when_sibling_review_schema_is_invalid():
    raw={
        'decision':'continue','reason':'review','current_diagnosis':'partial','evidence_sufficient':False,
        'supporting_evidence_ids':[],'contradicting_evidence_ids':[],
        'required_missing_evidence':[],'optional_validation':[],
        'obligation_reviews':[
            {'obligation_id':'O1','decision':'still_open','reason':'needs more'},
            {'obligation_id':'O2','decision':'refine','reason':'bad because requirement missing'},
        ],
        'recommended_next_goal':'continue','confidence':.4,
    }
    sanitized,invalid=Reflector._sanitize_individual_reviews(raw)
    parsed=ReflectionContract.model_validate(sanitized)
    assert [x.obligation_id for x in parsed.obligation_reviews]==['O1']
    assert invalid[0]['obligation_id']=='O2'


def test_budget_gate_parallel_estimate_is_wave_based_and_full_jitter_is_bounded():
    gate=RemainingBudgetGate(time.time()+1000)
    assert gate.estimate('parallel',child_costs=[8,8,15,8],max_workers=4)==20
    assert gate.estimate('parallel',child_costs=[8,8,15,8,8],max_workers=4)==35
    policy=RetryPolicy(max_attempts=3,base_delay_seconds=.4,max_delay_seconds=2)
    rng=random.Random(1)
    d=policy.delay(2,rng=rng)
    assert 0 <= d <= .8


def test_real_indexed_parallel_reads_are_thread_safe(tmp_path):
    repo=tmp_path/'repo'; repo.mkdir()
    (repo/'m.py').write_text('def leaf():\n    return 1\n\ndef target():\n    return leaf()\n')
    idx=RepositoryIndex(repo,tmp_path/'idx.sqlite'); idx.build()
    registry=ToolRegistry(repo,index=idx)
    try:
        actions=[
            {'action_id':'s','tool':'symbol_search','arguments':{'query':'target','max_results':10}},
            {'action_id':'i','tool':'inspect_symbol_context','arguments':{'symbol':'target','file':'m.py','include_source':True}},
        ]
        result=execute_parallel_group(registry,actions,group_id='real-index',max_workers=2,group_timeout_seconds=2,retry_policy=RetryPolicy(max_attempts=1))
        assert result.status=='success'
        assert all(x.observation.ok for x in result.children)
    finally:
        idx.close()


def test_parallel_semantic_code_search_is_rejected_as_not_bounded_local(tmp_path):
    repo=tmp_path/'repo'; repo.mkdir(); (repo/'m.py').write_text('x=1\n')
    idx=RepositoryIndex(repo,tmp_path/'idx.sqlite'); idx.build()
    registry=ToolRegistry(repo,index=idx)
    try:
        action=ActionProposal(kind=ActionKind.PARALLEL,skill='repository_exploration',reason='bad parallel network search',actions=[
            {'tool':'code_search','arguments':{'query':'concept','mode':'semantic','max_results':5}},
            {'tool':'read_file','arguments':{'path':'m.py','start_line':1,'end_line':1}},
        ])
        gd=RouterGuard(registry).validate(action,AgentState(TaskSpec('t','issue',str(repo))))
        assert gd.ok is False
        assert gd.error['error_type']=='parallel_tool_not_bounded_local'
    finally:
        idx.close()
