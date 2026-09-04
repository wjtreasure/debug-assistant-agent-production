from __future__ import annotations
from dataclasses import dataclass, field
from collections import Counter
from typing import Any
from debug_assistant.models import ActionKind, ActionProposal, AgentState
from debug_assistant.skills.catalog import SKILLS
from debug_assistant.tools.registry import PARALLEL_ALLOWED_TOOLS

@dataclass(slots=True)
class GuardDecision:
    ok: bool
    reason: str = ""
    force_reflection: bool = False
    canonical_arguments: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    advisory: str = ""
    repair: dict[str, Any] | None = None
    canonical_actions: list[dict[str,Any]] | None = None

class RouterGuard:
    """Validate capability + action/tool contracts. Skill/tool affinity is advisory, not a security permission."""
    def __init__(self, tool_registry): self.tools=tool_registry

    def validate(self, action: ActionProposal, state: AgentState) -> GuardDecision:
        if action.skill not in SKILLS:
            return GuardDecision(False,f"unknown skill: {action.skill}",error={"error_type":"unknown_skill","retryable":True})
        skill=SKILLS[action.skill]

        if action.kind == ActionKind.PARALLEL:
            if not (2 <= len(action.actions) <= 4):
                return GuardDecision(False,'parallel action must contain 2-4 children',error={"error_type":"schema_validation","retryable":True})
            canonical_actions=[]
            import json,re
            for idx,child in enumerate(action.actions):
                tool_name=str(child.get('tool') or '')
                if tool_name not in PARALLEL_ALLOWED_TOOLS:
                    return GuardDecision(False,f'parallel tool is not in bounded local allowlist: {tool_name}',error={'error_type':'parallel_tool_not_allowed','retryable':True,'action_index':idx})
                args=dict(child.get('arguments') or {})
                # Parallel workers are deliberately local/bounded. Semantic/hybrid
                # code_search can invoke an external embedding provider and therefore
                # must remain serial under the normal provider deadline/circuit breaker.
                if tool_name=='code_search' and str(args.get('mode','lexical')).lower()!='lexical':
                    return GuardDecision(False,'parallel code_search must use lexical mode',error={'error_type':'parallel_tool_not_bounded_local','retryable':True,'action_index':idx})
                blob=json.dumps(args,ensure_ascii=False)
                if re.search(r'\{\{\s*(?:result_of_)?action[_-]?\d+|\{\{[^}]*action_id',blob,re.I):
                    return GuardDecision(False,'parallel child arguments depend on sibling results',error={"error_type":"parallel_dependency","retryable":True,"action_index":idx})
                tool=self.tools.get(tool_name)
                if not tool:
                    return GuardDecision(False,f'unknown tool: {tool_name}',error={"error_type":"unknown_tool","retryable":True})
                if tool.spec.side_effect!='none':
                    return GuardDecision(False,f'capability denied: tool {tool_name} side_effect={tool.spec.side_effect}',error={"error_type":"capability_denied","retryable":False})
                canonical,error=self.tools.validate_arguments(tool_name,args)
                if error:return GuardDecision(False,error['message'],error={**error,'action_index':idx})
                canonical_actions.append({**child,'action_id':child.get('action_id') or f'a{idx}','arguments':canonical})
            return GuardDecision(True,canonical_actions=canonical_actions)

        if action.kind == ActionKind.TOOL:
            if not action.tool:
                return GuardDecision(False,'tool action missing tool name',error={"error_type":"schema_validation","retryable":True})
            tool=self.tools.get(action.tool)
            if not tool:
                return GuardDecision(False,f"unknown tool: {action.tool}",error={"error_type":"unknown_tool","retryable":True})
            # Read-only Debug Assistant capability boundary. New mutable tools must be explicitly reviewed here.
            if tool.spec.side_effect != 'none':
                return GuardDecision(False,f"capability denied: tool {action.tool} side_effect={tool.spec.side_effect}",error={"error_type":"capability_denied","retryable":False})
            canonical,error=self.tools.validate_arguments(action.tool,action.arguments)
            repair=None
            if error:
                repaired,repair_meta=self.tools.repair_arguments(action.tool,action.arguments,error)
                if repaired is not None:
                    canonical,error=self.tools.validate_arguments(action.tool,repaired)
                    if error is None:
                        repair=repair_meta
                if error:
                    return GuardDecision(False,error['message'],error=error)
            advisory=''
            if action.tool not in skill.suggested_tools:
                advisory=f"tool {action.tool} is unusual for skill {action.skill}, but allowed by read-only capability policy"
            return GuardDecision(True,canonical_arguments=canonical,advisory=advisory,repair=repair)

        if action.kind == ActionKind.FINISH and len(state.evidence)<2:
            return GuardDecision(False,'finish rejected: insufficient grounded evidence',True,error={"error_type":"premature_finish","retryable":True})
        if skill.prerequisites and 'evidence' in skill.prerequisites and not state.evidence:
            return GuardDecision(False,f"skill {action.skill} requires evidence",error={"error_type":"missing_prerequisite","retryable":True})
        return GuardDecision(True)

class LoopGuard:
    def __init__(self,max_repeat=2,max_no_progress=4):
        self.max_repeat=max_repeat; self.max_no_progress=max_no_progress; self.counts=Counter(); self.rejected_counts=Counter(); self.last_evidence=0; self.no_progress=0
    def observe_action(self,action,state):
        fp=action.fingerprint(); self.counts[fp]+=1
        if self.counts[fp]>self.max_repeat:
            state.repeated_actions+=1; return False,f"repeated action exceeded limit: {fp}"
        return True,''
    def observe_rejected_action(self,action,state,error_type="action_rejected"):
        sig=f"{action.fingerprint()}|{error_type}"
        self.rejected_counts[sig]+=1
        count=self.rejected_counts[sig]
        repeated=count>1
        if repeated:
            state.repeated_actions+=1
        return count, count>self.max_repeat, sig

    def observe_progress(self,state):
        now=len(state.evidence)
        if now<=self.last_evidence: self.no_progress+=1
        else: self.no_progress=0; self.last_evidence=now
        return self.no_progress < self.max_no_progress
