from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any

class ProgressKind(str, Enum):
    PROGRESS="progress"
    NO_PROGRESS="no_progress"

class ConvergenceMode(str, Enum):
    NORMAL="normal"
    CONVERGENCE_REQUIRED="convergence_required"
    BUDGET_CRITICAL="budget_critical"
    FORCE_FINALIZATION="force_finalization"

@dataclass(slots=True)
class ProgressAssessment:
    kind: ProgressKind
    reasons: list[str]
    diagnosis_changed: bool=False
    required_gap_changed: bool=False
    contradiction_changed: bool=False
    support_changed: bool=False

@dataclass(slots=True)
class ConvergenceState:
    mode: ConvergenceMode=ConvergenceMode.NORMAL
    no_progress_streak: int=0
    redundant_request_streak: int=0
    critical_attempt_used: bool=False
    first_supported_hypothesis_step: int | None=None
    first_stable_diagnosis_step: int | None=None
    prompt_tokens_at_first_stable_diagnosis: int | None=None
    completion_tokens_at_first_stable_diagnosis: int | None=None
    tokens_at_first_stable_diagnosis: int | None=None
    forced_finalization: bool=False
    budget_critical_entered: bool=False

class ConvergenceController:
    """Deterministic guard over exploration cost; never decides the root cause."""
    def __init__(self, *, no_progress_limit: int=2):
        self.no_progress_limit=max(1,int(no_progress_limit))
        self.state=ConvergenceState()
        self._previous_hypothesis: dict[str,Any] | None=None

    @staticmethod
    def can_finalize(hyp: dict[str,Any]) -> bool:
        return bool(
            hyp
            and hyp.get('status') in {'supported','confirmed'}
            and hyp.get('supporting_evidence_ids')
            and bool(hyp.get('evidence_sufficient'))
            and not hyp.get('contradicting_evidence_ids')
            and not hyp.get('required_missing_evidence')
        )

    def note_redundant(self) -> int:
        self.state.redundant_request_streak += 1
        return self.state.redundant_request_streak

    def note_nonredundant_action(self) -> None:
        self.state.redundant_request_streak=0

    def assess_reflection(self, hyp: dict[str,Any], *, usage_totals: dict[str,Any] | None=None, allow_budget_recovery: bool=True) -> ProgressAssessment:
        prev=self._previous_hypothesis
        if not prev:
            assessment=ProgressAssessment(ProgressKind.PROGRESS,['reflection_baseline'])
        else:
            diagnosis_changed=hyp.get('diagnosis_fingerprint') != prev.get('diagnosis_fingerprint')
            gap_changed=hyp.get('required_gap_fingerprint') != prev.get('required_gap_fingerprint')
            contradiction_changed=set(hyp.get('contradicting_evidence_ids') or []) != set(prev.get('contradicting_evidence_ids') or [])
            # New supporting evidence is useful progress, but repeated observations that
            # never become support do not count merely because bytes were new.
            support_changed=set(hyp.get('supporting_evidence_ids') or []) != set(prev.get('supporting_evidence_ids') or [])
            reasons=[]
            if diagnosis_changed: reasons.append('diagnosis_changed')
            if gap_changed: reasons.append('required_gap_changed')
            if contradiction_changed: reasons.append('contradiction_changed')
            if support_changed: reasons.append('support_changed')
            kind=ProgressKind.PROGRESS if reasons else ProgressKind.NO_PROGRESS
            if not reasons: reasons=['diagnostic_state_unchanged']
            assessment=ProgressAssessment(kind,reasons,diagnosis_changed,gap_changed,contradiction_changed,support_changed)
        self._previous_hypothesis=dict(hyp)

        if hyp.get('status') in {'supported','confirmed'} and self.state.first_supported_hypothesis_step is None:
            self.state.first_supported_hypothesis_step=hyp.get('updated_step')
        if int(hyp.get('stable_diagnosis_transitions',0) or 0) >= 1 and self.state.first_stable_diagnosis_step is None:
            self.state.first_stable_diagnosis_step=hyp.get('updated_step')
            usage_totals=usage_totals or {}
            self.state.prompt_tokens_at_first_stable_diagnosis=int(usage_totals.get('prompt_tokens',0) or 0)
            self.state.completion_tokens_at_first_stable_diagnosis=int(usage_totals.get('completion_tokens',0) or 0)
            self.state.tokens_at_first_stable_diagnosis=int(usage_totals.get('tokens',0) or 0)

        if assessment.kind is ProgressKind.PROGRESS:
            self.state.no_progress_streak=0
            if self.state.mode is ConvergenceMode.BUDGET_CRITICAL and allow_budget_recovery:
                self.state.critical_attempt_used=False
                self.state.mode=(ConvergenceMode.CONVERGENCE_REQUIRED
                                 if self._should_converge(hyp) else ConvergenceMode.NORMAL)
        else:
            self.state.no_progress_streak += 1

        # Enter convergence when diagnosis has been unchanged across two consecutive
        # reflections and no required causal gap / direct contradiction remains.
        if self.state.mode is ConvergenceMode.NORMAL and self._should_converge(hyp):
            self.state.mode=ConvergenceMode.CONVERGENCE_REQUIRED

        if self.state.no_progress_streak >= self.no_progress_limit and self.state.mode in {ConvergenceMode.NORMAL,ConvergenceMode.CONVERGENCE_REQUIRED}:
            if self.can_finalize(hyp):
                self.state.mode=ConvergenceMode.FORCE_FINALIZATION
                self.state.forced_finalization=True
            else:
                # V1.4.6: semantic stagnation is not budget exhaustion. Keep the
                # investigation in convergence mode; Runtime's semantic_no_progress
                # policy decides whether to conservatively finalize. BUDGET_CRITICAL
                # is reserved for actual cost/wall-time pressure from apply_budget().
                self.state.mode=ConvergenceMode.CONVERGENCE_REQUIRED
        elif self.state.mode is ConvergenceMode.BUDGET_CRITICAL and self.state.critical_attempt_used:
            if assessment.kind is ProgressKind.NO_PROGRESS:
                # Caller converts this terminal condition to BUDGET_EXHAUSTED.
                pass
        return assessment

    @staticmethod
    def _should_converge(hyp: dict[str,Any]) -> bool:
        return bool(
            int(hyp.get('stable_diagnosis_transitions',0) or 0) >= 1
            and not hyp.get('required_missing_evidence')
            and not hyp.get('contradicting_evidence_ids')
            and bool(hyp.get('evidence_sufficient'))
            and hyp.get('status') in {'supported','confirmed'}
        )


    def apply_budget(self, *, remaining_ratio: float, hyp: dict[str,Any]) -> ConvergenceMode:
        """Cost-aware mode pressure. Budget never decides diagnosis; it only narrows exploration."""
        r=max(0.0,min(1.0,float(remaining_ratio)))
        if r <= 0.10:
            if self.can_finalize(hyp):
                self.state.mode=ConvergenceMode.FORCE_FINALIZATION; self.state.forced_finalization=True
            else:
                self.state.mode=ConvergenceMode.BUDGET_CRITICAL; self.state.budget_critical_entered=True
        elif r <= 0.50 and self.state.mode is ConvergenceMode.NORMAL:
            # The 20%-10% band is governed by VERIFY_ONLY ActionPolicy. Do not
            # prematurely turn it into BUDGET_CRITICAL and skip the final exact check.
            self.state.mode=ConvergenceMode.CONVERGENCE_REQUIRED
        return self.state.mode
    def allow_critical_tool_attempt(self, information_need: str, hyp: dict[str,Any]) -> bool:
        # V1.4.1 compatibility shim: once BUDGET_CRITICAL is entered the Runtime
        # finalizes at the next state boundary and never starts a new tool.
        return self.state.mode is not ConvergenceMode.BUDGET_CRITICAL

    def critical_failed_after_reflection(self, assessment: ProgressAssessment) -> bool:
        return bool(self.state.mode is ConvergenceMode.BUDGET_CRITICAL
                    and self.state.critical_attempt_used
                    and assessment.kind is ProgressKind.NO_PROGRESS)
