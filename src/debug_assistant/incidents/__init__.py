"""CloudOps diagnosis runtime public API."""

from .contracts import IncidentCase, IncidentRunResult, RootCauseCandidate, TerminalOutcome
from .verification_dag import (
    LangGraphUnavailable,
    LangGraphVerificationAdapter,
    VerificationDAGError,
    VerificationDAGHarnessAdapter,
    VerificationDAGState,
    VerificationTask,
    VerificationTaskExecution,
)
from .rca_state import RCAEvent, RCAState, RCAStateError
from .evidence_pipeline import EvidencePipeline, EvidencePipelineError, EvidenceSet
from .finalization import (
    FinalizationConditions,
    FinalizationDecision,
    FinalizationFacts,
    FinalizationPolicy,
    FinalizationReason,
)
from .convergence import ConvergenceDecision
from .progress_policy import ProgressDecision, ProgressFacts, ProgressPolicy
from .services import RCAOrchestrationServices


def __getattr__(name):
    """Load the Runtime lazily so contract/agent imports do not cycle.

    ``debug_assistant.agent.planner`` consumes incident contracts. Eagerly
    importing Runtime here would
    make either direct agent import re-enter the partially initialized module.
    The public ``from debug_assistant.incidents import DiagnosisHarness`` API is
    preserved through this standard module-level lazy export.
    """
    if name in {"DiagnosisHarness", "DiagnosisHarnessConfig"}:
        from .runtime import (
            DiagnosisHarness, DiagnosisHarnessConfig,
        )
        return {
            "DiagnosisHarness": DiagnosisHarness,
            "DiagnosisHarnessConfig": DiagnosisHarnessConfig,
        }[name]
    raise AttributeError(name)

__all__ = [
    "IncidentCase", "DiagnosisHarness", "DiagnosisHarnessConfig",
    "IncidentRunResult", "RootCauseCandidate", "TerminalOutcome",
    "VerificationTask", "VerificationTaskExecution", "VerificationDAGState",
    "RCAEvent", "RCAState", "RCAStateError",
    "EvidencePipeline", "EvidencePipelineError", "EvidenceSet",
    "FinalizationConditions", "FinalizationDecision", "FinalizationFacts",
    "FinalizationPolicy", "FinalizationReason",
    "ConvergenceDecision", "ProgressDecision", "ProgressFacts", "ProgressPolicy",
    "RCAOrchestrationServices",
    "VerificationDAGHarnessAdapter", "LangGraphVerificationAdapter",
    "VerificationDAGError", "LangGraphUnavailable",
]
