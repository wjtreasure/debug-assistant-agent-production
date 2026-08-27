from __future__ import annotations
from pydantic import ValidationError
from debug_assistant.models import ActionProposal, ActionKind, AgentState
from debug_assistant.contracts import AgentActionContract, compact_validation_error, render_contract, render_contract_compact
from debug_assistant.skills.catalog import render_skill_catalog

SYSTEM="""You are the planner inside a read-only software debugging agent. Diagnose the issue; never propose edits, patches, write commands, package installation, network side effects, or repository mutation. Every conclusion must be grounded in repository evidence. Choose one next action, not a workflow plan. Prefer falsification over confirmation. Do not repeat equivalent calls. High confidence does not grant permission. Tool argument names and constraints are strict: use only fields shown in the tool catalog. For read_file, start_line/end_line are inclusive and a request may contain at most 200 lines, so end_line - start_line + 1 <= 200. Context IDs are optional hints: only reference IDs that appear in CONTEXT_CATALOG."""

class Planner:
    def __init__(self,llm,tools,model='',compact_prompt=False): self.llm=llm; self.tools=tools; self.model=model; self.compact_prompt=compact_prompt; self.last_prompt_breakdown={}
    def propose(self,state:AgentState,context:str) -> ActionProposal:
        contract=(render_contract_compact(AgentActionContract,"AGENT_ACTION_SCHEMA") if self.compact_prompt else render_contract(AgentActionContract,"AGENT_ACTION_SCHEMA"))
        skills=render_skill_catalog(compact=self.compact_prompt); tools_text=self.tools.render(compact=self.compact_prompt)
        instruction="retain_context_ids is optional. If present, use only IDs from CONTEXT_CATALOG. information_need should state the precise unresolved fact that justifies another tool call."
        user=f"{context}\n\nSKILLS:\n{skills}\n\nTOOLS (strict schemas; suggested skill/tool affinity is guidance, not permission):\n{tools_text}\n\n{contract}\n{instruction}"
        self.last_prompt_breakdown={'system_chars':len(SYSTEM),'context_chars':len(context),'skill_catalog_chars':len(skills),'tool_catalog_chars':len(tools_text),'contract_chars':len(contract),'instruction_chars':len(instruction)}
        data=self.llm.complete_json(SYSTEM,user,model=self.model or None)
        try:
            c=AgentActionContract.model_validate(data)
        except ValidationError as exc:
            details=compact_validation_error(exc)
            raise ValueError(f"agent action schema validation failed: {details}") from exc
        return ActionProposal(
            kind=ActionKind(c.kind), skill=c.skill, reason=c.reason, confidence=c.confidence,
            tool=c.tool, arguments=c.arguments, expected_evidence=c.expected_evidence,
            information_need=c.information_need, retain_context_ids=c.retain_context_ids,
        )
