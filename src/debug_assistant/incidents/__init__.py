"""CloudOps diagnosis runtime and compatibility exports."""

from .contracts import IncidentCase, IncidentRunResult, RootCauseCandidate
from .dag_convergence import DAGFinalizationDecision, evaluate_dag_finalization
from .verification_dag import (
    LangGraphUnavailable,
    LangGraphVerificationAdapter,
    VerificationDAGError,
    VerificationDAGHarnessAdapter,
    VerificationDAGState,
    VerificationTask,
    VerificationTaskExecution,
)


def __getattr__(name):
    """Load the Runtime lazily so contract/agent imports do not cycle.

    ``debug_assistant.agent.planner`` and ``debug_assistant.agent.reflection``
    both consume incident contracts.  Eagerly importing Runtime here would
    make either direct agent import re-enter the partially initialized module.
    The public ``from debug_assistant.incidents import DiagnosisHarness`` API is
    preserved through this standard module-level lazy export.
    """
    if name in {
        "DiagnosisHarness", "DiagnosisHarnessConfig",
        "IncidentHarness", "IncidentHarnessConfig",
    }:
        from .runtime import (
            DiagnosisHarness, DiagnosisHarnessConfig,
            IncidentHarness, IncidentHarnessConfig,
        )
        return {
            "DiagnosisHarness": DiagnosisHarness,
            "DiagnosisHarnessConfig": DiagnosisHarnessConfig,
            "IncidentHarness": IncidentHarness,
            "IncidentHarnessConfig": IncidentHarnessConfig,
        }[name]
    raise AttributeError(name)

__all__ = [
    "IncidentCase", "DiagnosisHarness", "DiagnosisHarnessConfig",
    "IncidentHarness", "IncidentHarnessConfig",
    "IncidentRunResult", "RootCauseCandidate",
    "VerificationTask", "VerificationTaskExecution", "VerificationDAGState",
    "VerificationDAGHarnessAdapter", "LangGraphVerificationAdapter",
    "VerificationDAGError", "LangGraphUnavailable",
    "DAGFinalizationDecision", "evaluate_dag_finalization",
]
