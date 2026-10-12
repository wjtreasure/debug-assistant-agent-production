from __future__ import annotations
from dataclasses import dataclass

@dataclass(slots=True)
class FeatureFlags:
    # Incident benchmark bundle switches.  These are intentionally separate
    # from the lower-level V1 feature switches so a paired evaluation can
    # disable one experimental bundle without copying the Agent runtime.
    evidence_lifecycle: bool = True
    context_manager: bool = True
    no_progress_detection: bool = True
    finalization_gate: bool = True
    context_projection_v2: bool = True
    lightweight_skills: bool = True
    native_tool_calling: bool = True
    structured_reflection: bool = True
    # Canonical split semantic/action contract with one immutable turn input.
    planner_contract_version: str = "planner_turn_v1"
    # Phase 2 typed Observation-to-Obligation gate.  Kept opt-in for the
    # historical DAG fixture until its scripted tool outputs carry semantic
    # qualification metadata; A1/A2 bundles enable it explicitly.
    evidence_qualification: bool = False
    # Shared public Cloud-OpsBench 57-type output contract. Disabled for old
    # generic incident fixtures; all refactor experiment arms enable it.
    public_fault_taxonomy: bool = False
    evidence_rehydration: bool = True
    semantic_convergence_v2: bool = True

    def validate(self) -> None:
        if self.context_projection_v2 and not self.context_manager:
            raise ValueError("context_projection_v2 requires context_manager")
