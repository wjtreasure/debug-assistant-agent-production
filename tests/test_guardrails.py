from debug_assistant.harness.guards import RouterGuard,LoopGuard
from debug_assistant.tools.registry import ToolRegistry
from debug_assistant.models import *

def state(tmp_path):
    return AgentState(TaskSpec('x','issue',str(tmp_path)))

def test_cross_skill_tool_rejected(tmp_path):
    g=RouterGuard(ToolRegistry(tmp_path)); s=state(tmp_path)
    a=ActionProposal(ActionKind.TOOL,'issue_triage','bad route',0.99,'read_file',{'path':'x'})
    assert not g.validate(a,s).ok

def test_premature_finish_rejected(tmp_path):
    g=RouterGuard(ToolRegistry(tmp_path)); s=state(tmp_path)
    a=ActionProposal(ActionKind.FINISH,'report_synthesis','done',0.99)
    d=g.validate(a,s); assert not d.ok and d.force_reflection

def test_loop_guard_blocks_repeat(tmp_path):
    s=state(tmp_path); l=LoopGuard(max_repeat=2); a=ActionProposal(ActionKind.TOOL,'repository_exploration','x',.5,'grep',{'query':'x'})
    assert l.observe_action(a,s)[0]; assert l.observe_action(a,s)[0]; assert not l.observe_action(a,s)[0]
