"""DAG-aware finalization predicate without a second convergence loop."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from .verification_dag import VerificationDAGState


class DAGFinalizationDecision(BaseModel):
    """A routing hint; the Harness still owns the terminal status and budget."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    finalize_allowed: bool
    terminal_status: Literal["READY_FOR_REVIEW", "INCONCLUSIVE"]
    reasons: tuple[str, ...] = ()


def evaluate_dag_finalization(
    dag_state: VerificationDAGState,
    *,
    candidate_complete: bool,
    evidence_sufficient: bool,
    blocking_contradiction_open: bool,
    source_requirement_satisfied: bool,
    hypothesis_stable: bool,
    runtime_gate_allowed: bool = True,
) -> DAGFinalizationDecision:
    """Combine semantic/runtime signals for one fail-closed route decision.

    ``runtime_gate_allowed`` represents the existing Harness budget, timeout,
    no-progress and deadline gates.  LangGraph may route on this decision but
    cannot set it to true or bypass it.
    """

    reasons: list[str] = []
    if not candidate_complete:
        reasons.append("candidate_incomplete")
    if not evidence_sufficient:
        reasons.append("evidence_insufficient")
    if dag_state.critical_open_task_ids:
        reasons.append("critical_verification_tasks_open")
    if blocking_contradiction_open:
        reasons.append("blocking_contradiction_open")
    if not source_requirement_satisfied:
        reasons.append("source_requirement_unsatisfied")
    if not hypothesis_stable:
        reasons.append("hypothesis_not_stable")
    if not runtime_gate_allowed:
        reasons.append("runtime_gate_blocked")
    allowed = not reasons
    return DAGFinalizationDecision(
        finalize_allowed=allowed,
        terminal_status="READY_FOR_REVIEW" if allowed else "INCONCLUSIVE",
        reasons=tuple(reasons),
    )

