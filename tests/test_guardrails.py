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


def test_read_file_repairs_more_than_200_lines_deterministically(tmp_path):
    g=RouterGuard(ToolRegistry(tmp_path)); s=state(tmp_path)
    a=ActionProposal(ActionKind.TOOL,'repository_exploration','read too much',0.7,'read_file',{'path':'x','start_line':200,'end_line':400})
    d=g.validate(a,s)
    assert d.ok
    assert d.canonical_arguments == {'path':'x','start_line':200,'end_line':399}
    assert d.repair['requested_line_count'] == 201
    assert d.repair['repaired_line_count'] == 200


def test_read_file_does_not_repair_ambiguous_invalid_arguments(tmp_path):
    g=RouterGuard(ToolRegistry(tmp_path)); s=state(tmp_path)
    a=ActionProposal(ActionKind.TOOL,'repository_exploration','bad args',0.7,'read_file',{'path':'x','range':'1-500'})
    d=g.validate(a,s)
    assert not d.ok and d.error['error_type']=='schema_validation'
    assert d.repair is None


def test_premature_finish_rejected(tmp_path):
    g=RouterGuard(ToolRegistry(tmp_path)); s=state(tmp_path)
    a=ActionProposal(ActionKind.FINISH,'report_synthesis','done',0.99)
    d=g.validate(a,s); assert not d.ok and d.force_reflection


def test_loop_guard_blocks_repeat(tmp_path):
    s=state(tmp_path); l=LoopGuard(max_repeat=2); a=ActionProposal(ActionKind.TOOL,'repository_exploration','x',.5,'grep',{'query':'x'})
    assert l.observe_action(a,s)[0]; assert l.observe_action(a,s)[0]; assert not l.observe_action(a,s)[0]


def test_rejected_action_repeat_guard_counts_rejections(tmp_path):
    s=state(tmp_path); l=LoopGuard(max_repeat=2)
    a=ActionProposal(ActionKind.TOOL,'repository_exploration','bad',.5,'read_file',{'path':'x','range':'1-10'})
    c1,ex1,sig1=l.observe_rejected_action(a,s,'schema_validation')
    c2,ex2,sig2=l.observe_rejected_action(a,s,'schema_validation')
    c3,ex3,sig3=l.observe_rejected_action(a,s,'schema_validation')
    assert (c1,ex1)==(1,False)
    assert (c2,ex2)==(2,False)
    assert (c3,ex3)==(3,True)
    assert sig1==sig2==sig3
    assert s.repeated_actions==2
