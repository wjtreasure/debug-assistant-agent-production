from __future__ import annotations

"""Pure-ish semantic transaction reducer for typed Reflection decisions."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable

from pydantic import ValidationError

from debug_assistant.contracts import ReflectionDecision, ObligationReview, EvidenceRequirement
from debug_assistant.memory.hypothesis import HypothesisState, diagnosis_fingerprint, evidence_fingerprint, required_gap_fingerprint
from .obligations import ObligationStatus
from .semantic_invariants import validate_semantic_candidate


@dataclass(frozen=True, slots=True)
class CandidateSemanticState:
    hypothesis: HypothesisState
    obligations: dict[str, Any]
    ignored_review_ids: tuple[str, ...] = ()
    accepted_review_ids: tuple[str, ...] = ()
    dropped_evidence_ids: tuple[str, ...] = ()
    revision: int = 0


class SemanticReducer:
    """Derive and atomically commit semantic state from evidence-grounded input."""

    def __init__(self, obligations, hypothesis_manager, evidence: Iterable[Any] = (), *, revision: int = 0):
        self.obligations = obligations
        self.hypothesis_manager = hypothesis_manager
        self.evidence = list(evidence)
        self.revision = int(revision)

    @staticmethod
    def _decision(value) -> tuple[ReflectionDecision, list[dict[str, Any]]]:
        if isinstance(value, ReflectionDecision):
            return value, []
        if not isinstance(value, dict):
            raise ValueError("reflection decision must be an object")
        raw = dict(value)
        invalid = []
        rows = raw.get("obligation_reviews") or []
        valid = []
        for index, row in enumerate(rows):
            try:
                valid.append(ObligationReview.model_validate(row))
            except (ValidationError, TypeError, ValueError):
                oid = row.get("obligation_id") if isinstance(row, dict) else ""
                invalid.append({"obligation_id": str(oid or ""), "index": index})
        raw["obligation_reviews"] = valid
        # A raw ReflectionContract response can be migrated at this boundary only
        # for semantic fields; derived fields are intentionally ignored.
        if "diagnosis" not in raw:
            raw["diagnosis"] = raw.get("current_diagnosis", "")
        return ReflectionDecision.model_validate(raw), invalid

    def reduce(self, decision, *, reflection_id: str = "", presented_evidence_ids: set[str] | None = None) -> CandidateSemanticState:
        typed, invalid_rows = self._decision(decision)
        available = {str(x.evidence_id) for x in self.evidence}
        support = tuple(x for x in typed.supporting_evidence_ids if x in available)
        contra = tuple(x for x in typed.contradicting_evidence_ids if x in available)
        dropped = tuple(x for x in typed.supporting_evidence_ids + typed.contradicting_evidence_ids if x not in available)
        old_items = deepcopy(self.obligations.items)
        old_hypothesis = deepcopy(self.hypothesis_manager.state)
        try:
            # Requirements are additive in the typed path. Existing open obligations
            # remain active until an explicit valid review resolves/refines them.
            existing = self.obligations.active_required_items()
            requirements = existing + [x.model_dump() for x in typed.new_requirements]
            self.obligations.sync(requirements, [x.model_dump() for x in typed.optional_validation])
            accepted = []
            for review in typed.obligation_reviews:
                if review.obligation_id not in self.obligations.items:
                    invalid_rows.append({"obligation_id": review.obligation_id, "reason": "unknown_obligation"})
                    continue
                ok, _ = self.obligations.apply_explicit_review(review.model_dump(), reflection_id=reflection_id)
                if ok:
                    accepted.append(review.obligation_id)
                else:
                    invalid_rows.append({"obligation_id": review.obligation_id, "reason": "not_presented_or_not_applicable"})

            gaps = self.obligations.active_required_items()
            has_required = bool(self.obligations.open_critical())
            source_support = [x for x in support if next((e for e in self.evidence if str(e.evidence_id) == x and e.source in {"read_file", "git_show"}), None)]
            sufficient = bool(typed.diagnosis and support and source_support and not has_required)
            if contra:
                status = "contradicted"
            elif sufficient and typed.root_cause_target and typed.root_cause_mechanism:
                status = "supported"
            elif typed.diagnosis or typed.root_cause_target or support:
                status = "partial"
            else:
                status = "none"
            root_location = typed.root_cause_location or ""
            hyp = HypothesisState(
                description=typed.diagnosis,
                root_cause_target=typed.root_cause_target or "",
                root_cause_location=root_location,
                root_cause_mechanism=typed.root_cause_mechanism or "",
                status=status,
                confidence=typed.confidence,
                supporting_evidence_ids=list(support),
                contradicting_evidence_ids=list(contra),
                required_missing_evidence=gaps,
                optional_validation=self.obligations.optional_items(),
                updated_step=0,
                stable_diagnosis_transitions=0,
                diagnosis_fingerprint=diagnosis_fingerprint(typed.diagnosis, status, list(contra), root_cause_target=typed.root_cause_target or "", root_cause_location=root_location),
                evidence_fingerprint=evidence_fingerprint(list(support), list(contra)),
                required_gap_fingerprint=required_gap_fingerprint(gaps, self.hypothesis_manager.repo_root),
                model_claimed_changed=None,
                evidence_sufficient=sufficient,
            )
            validate_semantic_candidate(self.obligations, hyp)
            semantic_input = bool(typed.diagnosis or typed.root_cause_target or typed.root_cause_location or
                                  typed.root_cause_mechanism or support or contra)
            new_revision = self.revision + (1 if accepted or typed.new_requirements or
                                            (semantic_input and hyp.diagnosis_fingerprint != old_hypothesis.diagnosis_fingerprint) else 0)
            candidate = CandidateSemanticState(hyp, deepcopy(self.obligations.items),
                                          tuple(x.get("obligation_id", "") for x in invalid_rows),
                                          tuple(accepted), tuple(sorted(set(dropped))), new_revision)
            # ``reduce`` is a dry-run. Only ``commit`` mutates the live semantic state.
            self.obligations.items = old_items
            self.hypothesis_manager.state = old_hypothesis
            return candidate
        except Exception:
            self.obligations.items = old_items
            self.hypothesis_manager.state = old_hypothesis
            raise

    def commit(self, candidate: CandidateSemanticState) -> HypothesisState:
        self.obligations.items = deepcopy(candidate.obligations)
        self.hypothesis_manager.state = candidate.hypothesis
        self.revision = candidate.revision
        return candidate.hypothesis

    def reduce_and_commit(self, decision, *, reflection_id: str = "", presented_evidence_ids: set[str] | None = None) -> CandidateSemanticState:
        candidate = self.reduce(decision, reflection_id=reflection_id, presented_evidence_ids=presented_evidence_ids)
        self.commit(candidate)
        return candidate
