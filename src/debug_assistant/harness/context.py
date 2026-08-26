from __future__ import annotations
from debug_assistant.models import AgentState

def build_context(state: AgentState, memory, max_chars: int) -> str:
    issue=state.task.issue[:max_chars//2]
    evidence=memory.context(max_chars=max_chars//2)
    recent='\n'.join(f"- {a.skill}/{a.tool or a.kind.value}: {a.reason}" for a in state.actions[-6:]) or '(none)'
    return f"""TASK_ID: {state.task.task_id}\nISSUE:\n{issue}\n\nEVIDENCE_LEDGER:\n{evidence}\n\nRECENT_ACTIONS:\n{recent}\n\nSTATE: {state.to_summary()}"""[:max_chars]
