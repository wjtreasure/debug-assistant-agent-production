from __future__ import annotations
from dataclasses import dataclass

@dataclass(slots=True)
class FeatureFlags:
    observation_reuse: bool = True
    context_catalog: bool = True
    context_budget_packing: bool = True
    model_context_selection: bool = False
    hypothesis_state: bool = True
    termination_advisory: bool = True
    fallback_reporter: bool = True
    convergence_control: bool = True
    context_lifecycle_v2: bool = True
    context_projection_v2: bool = True
    compact_prompt_rendering: bool = True
    semantic_code_search: bool = True
    lightweight_skills: bool = True
    information_need_tracking: bool = True
    evidence_obligations: bool = True
    cost_aware_convergence: bool = True
    trace_v2: bool = True
    native_tool_calling: bool = True
    structured_reflection: bool = True

    def validate(self) -> None:
        if self.model_context_selection and not self.context_catalog:
            raise ValueError("model_context_selection requires context_catalog")
        if self.fallback_reporter and not self.hypothesis_state:
            raise ValueError("fallback_reporter requires hypothesis_state")
        if self.context_projection_v2 and not self.context_lifecycle_v2:
            raise ValueError("context_projection_v2 requires context_lifecycle_v2")
