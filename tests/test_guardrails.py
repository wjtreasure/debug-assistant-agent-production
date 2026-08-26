from debug_assistant.harness.guards import RouterGuard,LoopGuard
from debug_assistant.tools.registry import ToolRegistry
from debug_assistant.models import *


def state(tmp_path):
    return AgentState(TaskSpec('x','issue',str(tmp_path)))


def test_cross_skill_tool_is_advisory_not_hard_rejected(tmp_path):
    g=RouterGuard(ToolRegistry(tmp_path)); s=state(tmp_path)
    a=ActionProposal(ActionKind.TOOL,'issue_triage','unusual but read-only route',0.99,'read_file',{'path':'x'})
    d=g.validate(a,s)
    assert d.ok and 'unusual' in d.advisory


def test_tool_schema_rejects_unknown_argument(tmp_path):
    g=RouterGuard(ToolRegistry(tmp_path)); s=state(tmp_path)
    a=ActionProposal(ActionKind.TOOL,'repository_exploration','bad args',0.7,'read_file',{'path':'x','range':'1-10'})
    d=g.validate(a,s)
    assert not d.ok
    assert d.error['error_type']=='schema_validation'
    assert 'range' in str(d.error['details'])


def test_tool_schema_canonicalizes_defaults(tmp_path):
    g=RouterGuard(ToolRegistry(tmp_path)); s=state(tmp_path)
    a=ActionProposal(ActionKind.TOOL,'repository_exploration','read',0.7,'read_file',{'path':'x'})
    d=g.validate(a,s)
    assert d.ok
    assert d.canonical_arguments=={'path':'x','start_line':1,'end_line':200}


def test_read_file_rejects_more_than_200_lines(tmp_path):
    g=RouterGuard(ToolRegistry(tmp_path)); s=state(tmp_path)
    a=ActionProposal(ActionKind.TOOL,'repository_exploration','read too much',0.7,'read_file',{'path':'x','start_line':1,'end_line':500})
    d=g.validate(a,s)
    assert not d.ok and d.error['error_type']=='schema_validation'


def test_premature_finish_rejected(tmp_path):
    g=RouterGuard(ToolRegistry(tmp_path)); s=state(tmp_path)
    a=ActionProposal(ActionKind.FINISH,'report_synthesis','done',0.99)
    d=g.validate(a,s); assert not d.ok and d.force_reflection


def test_loop_guard_blocks_repeat(tmp_path):
    s=state(tmp_path); l=LoopGuard(max_repeat=2); a=ActionProposal(ActionKind.TOOL,'repository_exploration','x',.5,'grep',{'query':'x'})
    assert l.observe_action(a,s)[0]; assert l.observe_action(a,s)[0]; assert not l.observe_action(a,s)[0]
