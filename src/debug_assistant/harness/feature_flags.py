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

    def validate(self) -> None:
        if self.model_context_selection and not self.context_catalog:
            raise ValueError("model_context_selection requires context_catalog")
        if self.fallback_reporter and not self.hypothesis_state:
            raise ValueError("fallback_reporter requires hypothesis_state")
