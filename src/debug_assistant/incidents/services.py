"""Pure RCA orchestration services kept independent of Tool execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from debug_assistant.memory.evidence_projector import EvidenceProjector
from .progress_policy import ProgressDecision, ProgressFacts, ProgressPolicy


@dataclass(slots=True)
class RCAOrchestrationServices:
    """Dependency-injected control plane; no Registry, Provider, or Gold access."""

    projector: EvidenceProjector = field(default_factory=EvidenceProjector)

    def project(self, observation: Any, **kwargs: Any):
        return self.projector.project(observation, **kwargs)

    def converge(self, facts: ProgressFacts) -> ProgressDecision:
        return ProgressPolicy.evaluate(facts)
