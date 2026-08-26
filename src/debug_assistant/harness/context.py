from __future__ import annotations
from dataclasses import asdict
from debug_assistant.models import AgentState, ToolObservation


def _select_recent_observations(state: AgentState, max_count: int, max_chars: int) -> list[ToolObservation]:
    """Select newest observations without truncating an individual observation.

    Latest observation is always preferred. The window is bounded by both count and chars.
    """
    if max_count <= 0 or max_chars <= 0:
        return []
    selected=[]; used=0
    for obs in reversed(state.observations):
        if len(selected) >= max_count:
            break
        size=len(obs.content)
        if selected and used + size > max_chars:
            continue
        if not selected and size > max_chars:
            # Harness Tool output limit should normally make this impossible. Preserve the latest
            # observation rather than silently losing fidelity; the global context gate remains.
            selected.append(obs)
            break
        selected.append(obs); used += size
    return list(reversed(selected))


def _render_recent_observations(observations: list[ToolObservation]) -> str:
    if not observations:
        return '(none)'
    chunks=[]
    for obs in observations:
        meta=obs.metadata or {}
        head=(f"OBSERVATION {obs.observation_id}\n"
              f"tool={obs.tool} ok={str(obs.ok).lower()} error_type={obs.error_type or 'none'}\n"
              f"metadata={meta}\n")
        chunks.append(head + obs.content + f"\nEND OBSERVATION {obs.observation_id}")
    return '\n\n'.join(chunks)


def build_context(
    state: AgentState,
    memory,
    max_chars: int,
    *,
    max_steps:int|None=None,
    max_tool_calls:int|None=None,
    recent_observation_count:int=2,
    recent_observation_chars:int=16000,
) -> str:
    # Reserve explicit space for short-term raw fidelity. Do not rely on a final blind [:max_chars]
    # cut that could silently remove the newest observation tail.
    recent_obs=_select_recent_observations(state,recent_observation_count,recent_observation_chars)
    recent_ids={o.observation_id for o in recent_obs}
    recent_raw=_render_recent_observations(recent_obs)

    issue_budget=max(2000,min(max_chars//3,18000))
    issue=state.task.issue[:issue_budget]
    recent_actions='\n'.join(f"- {a.skill}/{a.tool or a.kind.value}: {a.reason}" for a in state.actions[-6:]) or '(none)'
    budget=[]
    if max_steps is not None:
        budget.append(f"step={state.step}/{max_steps}")
        budget.append(f"remaining_steps={max(0,max_steps-state.step)}")
    if max_tool_calls is not None:
        budget.append(f"tool_calls={state.tool_calls}/{max_tool_calls}")
        budget.append(f"remaining_tool_calls={max(0,max_tool_calls-state.tool_calls)}")
    budget_text=', '.join(budget) or 'not configured'
    warning=''
    if max_steps is not None and max_steps-state.step <= max(2,max_steps//5):
        warning='\nBUDGET_NOTICE: You are nearing the step budget. Prioritize validating the strongest diagnosis and finish when evidence is sufficient.'

    fixed=(f"TASK_ID: {state.task.task_id}\nISSUE:\n{issue}\n\n"
           f"RECENT_ACTIONS:\n{recent_actions}\n\n"
           f"RUNTIME_BUDGET: {budget_text}{warning}\n\nSTATE: {state.to_summary()}\n\n")
    raw_section=("RECENT_RAW_OBSERVATIONS:\n"
                 "These are authoritative complete bounded results of the most recent tool executions. "
                 "Historical evidence may be compressed; do not infer missing content from a compressed summary when the raw observation is present here.\n"
                 f"{recent_raw}\n\n")

    remaining=max(0,max_chars-len(fixed)-len(raw_section)-64)
    evidence=memory.context(max_chars=remaining,recent_observation_ids=recent_ids)
    text=f"{fixed}HISTORICAL_EVIDENCE_LEDGER:\n{evidence}\n\n{raw_section}"
    if len(text) <= max_chars:
        return text
    # Defensive fallback: keep fixed state + latest raw facts, shrink historical evidence first.
    overflow=len(text)-max_chars
    if overflow > 0 and evidence not in ('','(no evidence yet)'):
        keep=max(0,len(evidence)-overflow)
        evidence=evidence[:keep]
        text=f"{fixed}HISTORICAL_EVIDENCE_LEDGER:\n{evidence}\n\n{raw_section}"
    return text[:max_chars]
