"""Post-commit action admission; no diagnostic routing or semantic writes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from .verification_plan import (
    PlanAdmissionDecision, VerificationPlanError, EnrichedVerificationPlan,
    admit_verification_plan, validate_plan_acquisitions,
)

from .rca_state import RCAState


@dataclass(frozen=True, slots=True)
class ContinuationDecision:
    action: Literal["execute", "replan", "reject", "finalize"]
    reason: str
    target_obligation_id: str | None = None
    committed_state_version: int = 0

class ContinuationPolicy:
    @staticmethod
    def admit_plan(
        state: RCAState, plan: EnrichedVerificationPlan, *,
        state_snapshot_hash: str,
        generated_state_version: int,
        generated_snapshot_hash: str,
        schema_sha256_by_operation: dict[str, str],
        registry: Any, acquisition_projection: Mapping[str, Any],
        compilation_failure: VerificationPlanError | None = None,
    ) -> PlanAdmissionDecision:
        """Bind exact approved tasks after the single canonical semantic commit."""
        version = state.event_count
        if generated_state_version != version or generated_snapshot_hash != state_snapshot_hash:
            return PlanAdmissionDecision("reject", "plan_snapshot_stale", committed_state_version=version)
        if compilation_failure is not None:
            if plan.tasks:
                raise ValueError('a rejected compilation cannot contain executable tasks')
            return PlanAdmissionDecision('replan', compilation_failure.reason_code,
                detail=compilation_failure.detail, committed_state_version=version)
        if not plan.tasks:
            return PlanAdmissionDecision("replan", "plan_complete",
                                         committed_state_version=version)
        if any(schema_sha256_by_operation.get(task.operation) != task.schema_sha256
               for task in plan.tasks):
            return PlanAdmissionDecision("replan", "plan_schema_stale",
                                         committed_state_version=version)
        try:
            admitted = admit_verification_plan(
                plan, state=state, state_snapshot_hash=state_snapshot_hash,
            )
            validate_plan_acquisitions(plan, registry, projection=acquisition_projection)
        except VerificationPlanError as exc:
            return PlanAdmissionDecision("replan", exc.reason_code, detail=exc.detail,
                                         committed_state_version=version)
        return PlanAdmissionDecision("execute", "plan_admitted", admitted, version)

    @staticmethod
    def execution_barrier(
        state: RCAState, admission: PlanAdmissionDecision, *,
        target_obligation_id: str | None = None,
        state_snapshot_hash: str | None = None,
        schema_valid: bool = True, binding_reason: str = "",
        ready: bool = True, complete: bool = False, failed: bool = False,
    ) -> ContinuationDecision:
        """Single barrier owner; raw Observation meaning is never an input."""
        plan = admission.admitted_plan
        reason = (
            "plan_snapshot_stale" if plan is None or (state_snapshot_hash is not None
                and plan.state_snapshot_hash != state_snapshot_hash) else
            "plan_state_stale" if plan.committed_state_version != admission.committed_state_version else
            "plan_state_stale" if state.event_count != admission.committed_state_version else
            "plan_schema_stale" if not schema_valid else
            binding_reason if binding_reason else
            "plan_task_failed" if failed else
            "plan_complete" if complete else
            "plan_dag_blocked" if not ready else ""
        )
        if reason:
            return ContinuationDecision("replan", reason, target_obligation_id, state.event_count)
        if plan is None or target_obligation_id not in {
            task.target_obligation_id for task in plan.tasks
        }:
            return ContinuationDecision("replan", "plan_target_unknown",
                                        target_obligation_id, state.event_count)
        target = next((item for item in state.obligations if item.id == target_obligation_id), None)
        if target is None or target.status != "OPEN":
            return ContinuationDecision("replan", "plan_target_terminal" if target else "plan_target_unknown",
                                        target_obligation_id, state.event_count)
        return ContinuationDecision("execute", "plan_task_ready", target.id, state.event_count)
