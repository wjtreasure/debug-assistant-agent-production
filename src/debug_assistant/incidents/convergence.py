"""Immutable convergence telemetry emitted by ProgressPolicy."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


ConvergenceStatus = Literal[
    "CONVERGED", "CONTINUE", "REPLAN", "INCONCLUSIVE_BUDGET", "INCONCLUSIVE_CAPABILITY"
]


class ConvergenceDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ConvergenceStatus
    evidence_sufficient: bool
    hypothesis_stable: bool
    open_critical_obligations: tuple[str, ...] = ()
    blocking_contradictions: tuple[str, ...] = ()
    no_progress: bool = False
    hard_budget: bool = False
    capability_blocked: bool = False
    reasons: tuple[str, ...] = ()
