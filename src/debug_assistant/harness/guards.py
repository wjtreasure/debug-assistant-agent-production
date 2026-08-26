from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
from debug_assistant.models import ActionKind, ActionProposal, AgentState
from debug_assistant.skills.catalog import SKILLS

@dataclass(slots=True)
class GuardDecision:
    ok: bool
    reason: str = ""
    force_reflection: bool = False

class RouterGuard:
    def __init__(self, tool_registry): self.tools=tool_registry
    def validate(self, action: ActionProposal, state: AgentState) -> GuardDecision:
        if action.skill not in SKILLS: return GuardDecision(False,f"unknown skill: {action.skill}")
        skill=SKILLS[action.skill]
        if action.kind == ActionKind.TOOL:
            if not action.tool: return GuardDecision(False,'tool action missing tool name')
            tool=self.tools.get(action.tool)
            if not tool: return GuardDecision(False,f"unknown tool: {action.tool}")
            if action.tool not in skill.allowed_tools: return GuardDecision(False,f"tool {action.tool} not allowed by skill {action.skill}")
            for arg in tool.spec.required_args:
                if arg not in action.arguments or action.arguments[arg] in ('',None): return GuardDecision(False,f"missing required tool arg: {arg}")
        # High confidence never bypasses guardrails. Suspicious early finish is escalated to reflection.
        if action.kind == ActionKind.FINISH and len(state.evidence)<2:
            return GuardDecision(False,'finish rejected: insufficient grounded evidence',True)
        if skill.prerequisites and 'evidence' in skill.prerequisites and not state.evidence:
            return GuardDecision(False,f"skill {action.skill} requires evidence")
        return GuardDecision(True)

class LoopGuard:
    def __init__(self,max_repeat=2,max_no_progress=4):
        self.max_repeat=max_repeat; self.max_no_progress=max_no_progress; self.counts=Counter(); self.last_evidence=0; self.no_progress=0
    def observe_action(self,action,state):
        fp=action.fingerprint(); self.counts[fp]+=1
        if self.counts[fp]>self.max_repeat:
            state.repeated_actions+=1; return False,f"repeated action exceeded limit: {fp}"
        return True,''
    def observe_progress(self,state):
        now=len(state.evidence)
        if now<=self.last_evidence: self.no_progress+=1
        else: self.no_progress=0; self.last_evidence=now
        return self.no_progress < self.max_no_progress
