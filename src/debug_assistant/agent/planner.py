from __future__ import annotations
from debug_assistant.models import ActionProposal, ActionKind, AgentState
from debug_assistant.skills.catalog import render_skill_catalog

SYSTEM="""You are the planner inside a read-only software debugging agent. Diagnose the issue; never propose edits, patches, write commands, package installation, network side effects, or repository mutation. Every conclusion must be grounded in repository evidence. Choose one next action, not a workflow plan. Prefer falsification over confirmation. Do not repeat equivalent calls. High confidence does not grant permission."""

SCHEMA="Return exactly one JSON object:\\n{\"kind\":\"tool|reflect|finish\",\"skill\":\"skill_name\",\"reason\":\"why this is the best next step\",\"confidence\":0.0,\"tool\":\"tool_name or null\",\"arguments\":{},\"expected_evidence\":\"what would change the diagnosis\"}"

class Planner:
    def __init__(self,llm,tools,model=''): self.llm=llm; self.tools=tools; self.model=model
    def propose(self,state:AgentState,context:str) -> ActionProposal:
        user=f"""{context}\n\nSKILLS:\n{render_skill_catalog()}\n\nTOOLS:\n{self.tools.render()}\n\n{SCHEMA}"""
        data=self.llm.complete_json(SYSTEM,user,model=self.model or None)
        kind=ActionKind(data.get('kind','tool'))
        return ActionProposal(kind=kind,skill=str(data.get('skill','repository_exploration')),reason=str(data.get('reason','')),
            confidence=max(0.0,min(1.0,float(data.get('confidence',0.5)))),tool=data.get('tool'),arguments=data.get('arguments') or {},expected_evidence=str(data.get('expected_evidence','')))
