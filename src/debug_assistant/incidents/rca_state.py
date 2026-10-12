"""Canonical RCA state and fail-closed reducer.

The runtime once carried the same semantic gap in several fields. This module
defines the single state boundary: an Obligation is
the semantic question, while a Task is only its executable scheduling state.
The reducer is deliberately pure and does not execute tools or inspect Gold.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import (
    Contradiction,
    EvidenceId,
    IncidentHypothesis,
    SemanticDecisionRecord,
    VerificationObligation,
)


RCA_STATE_SCHEMA_VERSION = "rca-state-v2"


class RCAStateError(ValueError):
    """Raised when a state event would violate the canonical lifecycle."""


class RCAEvent(BaseModel):
    """A small serializable reducer event.

    ``payload`` is intentionally opaque to the persistence layer but each
    event kind is validated by ``RCAState.reduce`` before it can mutate state.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal[
        "ADD_OBLIGATION", "LINK_EVIDENCE", "SET_CONTRADICTIONS",
        "CONTRADICTION", "INVALIDATE_BRANCH", "REOPEN_OBLIGATION",
        "UPDATE_OBLIGATION", "SUPERSEDE_OBLIGATION", "SET_HYPOTHESIS",
        "ADD_SEMANTIC_RECORD",
    ]
    payload: dict[str, Any] = Field(default_factory=dict)


class RCAState(BaseModel):
    """Incident-scoped canonical state.

    ``VerificationObligation`` owns the semantic claim and evidence
    requirement. Execution tasks are owned by ``VerificationDAGState`` and
    never enter this semantic state.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["rca-state-v2"] = RCA_STATE_SCHEMA_VERSION
    incident_id: str = Field(min_length=1)
    hypotheses: tuple[IncidentHypothesis, ...] = ()
    obligations: tuple[VerificationObligation, ...] = ()
    evidence_ids: tuple[EvidenceId, ...] = ()
    contradictions: tuple[Contradiction, ...] = ()
    semantic_records: tuple[SemanticDecisionRecord, ...] = ()
    event_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_invariants(self) -> "RCAState":
        obligation_ids = [item.id for item in self.obligations]
        if len(obligation_ids) != len(set(obligation_ids)):
            raise RCAStateError("obligation IDs must be unique")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise RCAStateError("evidence IDs must be unique")
        if any(item not in set(self.evidence_ids) for item in self._referenced_evidence()):
            raise RCAStateError("state references evidence that is not admitted")
        contradiction_ids = [item.evidence_id for item in self.contradictions]
        if len(contradiction_ids) != len(set(contradiction_ids)):
            raise RCAStateError("contradiction evidence IDs must be unique")
        if any(
            hypothesis.verification_obligations
            or hypothesis.contradictions
            or hypothesis.required_gaps
            or hypothesis.evidence_gap
            or hypothesis.contradicting_evidence_ids
            or hypothesis.evidence_sufficient
            for hypothesis in self.hypotheses
        ):
            raise RCAStateError(
                "canonical hypothesis history cannot embed lifecycle projections"
            )
        record_ids = [item.record_id for item in self.semantic_records]
        if len(record_ids) != len(set(record_ids)):
            raise RCAStateError("semantic record IDs must be unique")
        admitted = set(self.evidence_ids)
        if any(
            evidence.evidence_id not in admitted
            for record in self.semantic_records
            for evidence in record.evidence
        ):
            raise RCAStateError("semantic records reference evidence that is not admitted")
        return self

    def _referenced_evidence(self) -> tuple[EvidenceId, ...]:
        refs: list[EvidenceId] = []
        for obligation in self.obligations:
            refs.extend(obligation.supporting_evidence_ids)
        for item in self.contradictions:
            refs.append(item.evidence_id)
        return tuple(refs)

    def obligation(self, obligation_id: str) -> VerificationObligation:
        for item in self.obligations:
            if item.id == obligation_id:
                return item
        raise RCAStateError(f"unknown obligation: {obligation_id}")

    @property
    def open_critical_obligation_ids(self) -> tuple[str, ...]:
        return tuple(
            item.id for item in self.obligations if item.blocks_finalization
        )

    def active_hypothesis_projection(self) -> IncidentHypothesis | None:
        """Build a detached read-only view from canonical semantic state."""
        if not self.hypotheses:
            return None
        hypothesis = self.hypotheses[-1]
        blocking = tuple(
            item for item in self.contradictions if item.blocks_finalization
        )
        required_gaps = tuple(
            item.claim for item in self.obligations if item.blocks_finalization
        )
        supporting_evidence_ids = tuple(dict.fromkeys((
            *hypothesis.supporting_evidence_ids,
            *(
                evidence_id
                for item in self.obligations
                if item.status == "SATISFIED"
                for evidence_id in item.supporting_evidence_ids
            ),
        )))
        evidence_sufficient = bool(supporting_evidence_ids) and not (
            required_gaps or blocking
        )
        return hypothesis.model_copy(update={
            "verification_obligations": self.obligations,
            "required_gaps": required_gaps,
            "evidence_gap": required_gaps[0] if required_gaps else "",
            "contradictions": self.contradictions,
            "contradicting_evidence_ids": tuple(
                item.evidence_id for item in blocking
            ),
            "supporting_evidence_ids": supporting_evidence_ids,
            "evidence_sufficient": evidence_sufficient,
            "status": (
                "supported" if evidence_sufficient
                else "rejected" if hypothesis.status == "rejected"
                else "open"
            ),
        })

    def hypothesis_projections(self) -> tuple[IncidentHypothesis, ...]:
        """Return history with only the active item enriched from canonical state."""
        active = self.active_hypothesis_projection()
        if active is None:
            return ()
        return (*self.hypotheses[:-1], active)

    def context_projection(self) -> dict[str, Any]:
        """Return the semantic slice that ContextManager may expose to a model.

        Observation bodies, action history and budget remain owned by their
        existing Runtime stores.  This projection is deliberately read-only:
        it carries the current hypothesis, canonical Obligation routing
        metadata, admitted Evidence pointers and contradictions without
        creating a second mutable state representation.
        """
        active = self.active_hypothesis_projection()
        hypothesis = active.model_dump(mode="json") if active else None
        obligations = [{**item.model_dump(mode="json"),
                        "required_fact_slots": list(item.required_fact_slots)}
                       for item in self.obligations]
        semantic_history: list[dict[str, Any]] = []
        semantic_budget = 6000
        used = 2
        for record in reversed(self.semantic_records):
            row = record.model_dump(mode="json")
            row_chars = len(str(row))
            if semantic_history and used + row_chars > semantic_budget:
                break
            semantic_history.append(row)
            used += row_chars
            if len(semantic_history) >= 12:
                break
        semantic_history.reverse()
        return {
            "incident_id": self.incident_id,
            "active_hypothesis": hypothesis,
            "obligations": obligations,
            "ready_obligations": [
                item for item in obligations
                if item.get("status") in {"OPEN", "BLOCKED_BY_CAPABILITY"}
            ],
            "evidence_ids": list(self.evidence_ids),
            "contradictions": [item.model_dump(mode="json") for item in self.contradictions],
            "open_critical_obligation_ids": list(self.open_critical_obligation_ids),
            # A bounded, immutable semantic journal. Evidence bodies remain in
            # EvidenceMemory and are recovered through these canonical refs.
            "semantic_history": semantic_history,
            "semantic_history_omitted": max(0, len(self.semantic_records) - len(semantic_history)),
        }

    def reduce(self, event: RCAEvent | dict[str, Any]) -> "RCAState":
        """Apply one validated event and return a new state.

        Invalid IDs, unknown transitions, duplicate Evidence and unsupported
        reopen operations fail closed instead of being silently coerced.
        """

        evt = event if isinstance(event, RCAEvent) else RCAEvent.model_validate(event)
        payload = evt.payload
        if evt.kind == "ADD_OBLIGATION":
            obligation = VerificationObligation.model_validate(payload)
            if any(item.id == obligation.id for item in self.obligations):
                raise RCAStateError(f"duplicate obligation: {obligation.id}")
            return self._next(obligations=self.obligations + (obligation,))
        if evt.kind == "LINK_EVIDENCE":
            evidence_id = str(payload.get("evidence_id", ""))
            if not evidence_id.startswith("ev-"):
                raise RCAStateError("linked evidence must use an ev- ID")
            if evidence_id in self.evidence_ids:
                raise RCAStateError(f"duplicate evidence: {evidence_id}")
            return self._next(evidence_ids=self.evidence_ids + (evidence_id,))
        if evt.kind == "UPDATE_OBLIGATION":
            return self._update_obligation(payload)
        if evt.kind == "SUPERSEDE_OBLIGATION":
            return self._supersede_obligation(str(payload.get("obligation_id", "")))
        if evt.kind == "SET_HYPOTHESIS":
            return self._set_hypothesis(payload)
        if evt.kind == "ADD_SEMANTIC_RECORD":
            record = SemanticDecisionRecord.model_validate(
                payload.get("record", payload)
            )
            if any(item.record_id == record.record_id for item in self.semantic_records):
                raise RCAStateError(f"duplicate semantic record: {record.record_id}")
            if any(
                evidence.evidence_id not in self.evidence_ids
                for evidence in record.evidence
            ):
                raise RCAStateError("semantic record must reference admitted Evidence")
            return self._next(semantic_records=self.semantic_records + (record,))
        if evt.kind == "SET_CONTRADICTIONS":
            return self._set_contradictions(payload)
        if evt.kind == "CONTRADICTION":
            contradiction = Contradiction.model_validate(payload)
            if contradiction.evidence_id not in self.evidence_ids:
                raise RCAStateError("contradiction must reference admitted Evidence")
            if any(item.evidence_id == contradiction.evidence_id for item in self.contradictions):
                raise RCAStateError("duplicate contradiction Evidence")
            return self._next(contradictions=self.contradictions + (contradiction,))
        if evt.kind == "INVALIDATE_BRANCH":
            return self._invalidate_branch(str(payload.get("obligation_id", "")))
        if evt.kind == "REOPEN_OBLIGATION":
            return self._reopen_obligation(str(payload.get("obligation_id", "")), payload)
        raise RCAStateError(f"unsupported reducer event: {evt.kind}")

    def _update_obligation(self, payload: dict[str, Any]) -> "RCAState":
        """Apply a Runtime-owned semantic transition through the reducer."""
        obligation_id = str(payload.get("obligation_id", ""))
        current = self.obligation(obligation_id)
        target = str(payload.get("status", current.status))
        allowed = {
            "OPEN": {"OPEN", "SATISFIED", "BLOCKED_BY_CAPABILITY", "NOT_APPLICABLE_BY_CAPABILITY", "SUPERSEDED_BY_REPLAN"},
            "SATISFIED": {"SATISFIED"},
            "BLOCKED_BY_CAPABILITY": {"BLOCKED_BY_CAPABILITY", "NOT_APPLICABLE_BY_CAPABILITY", "SUPERSEDED_BY_REPLAN"},
            "NOT_APPLICABLE_BY_CAPABILITY": {"NOT_APPLICABLE_BY_CAPABILITY"},
            "SUPERSEDED_BY_REPLAN": {"SUPERSEDED_BY_REPLAN"},
        }
        if target not in allowed[current.status]:
            raise RCAStateError(f"invalid obligation transition {current.status}->{target}")
        support = tuple(payload.get("supporting_evidence_ids", current.supporting_evidence_ids))
        blocked = tuple(payload.get("blocked_capabilities", current.blocked_capabilities))
        if any(item not in self.evidence_ids for item in support):
            raise RCAStateError("obligation references unknown Evidence")
        if target == "SATISFIED" and not support:
            raise RCAStateError("a satisfied obligation must cite Evidence")
        updated = current.model_copy(update={
            "status": target,
            "supporting_evidence_ids": support,
            "blocked_capabilities": blocked,
        })
        return self._next(obligations=tuple(
            updated if item.id == obligation_id else item for item in self.obligations
        ))

    def _supersede_obligation(self, obligation_id: str) -> "RCAState":
        current = self.obligation(obligation_id)
        if current.status not in {"OPEN", "BLOCKED_BY_CAPABILITY"}:
            raise RCAStateError(
                f"only an active obligation can be superseded: {obligation_id}={current.status}"
            )
        updated = current.model_copy(update={"status": "SUPERSEDED_BY_REPLAN"})
        return self._next(obligations=tuple(
            updated if item.id == obligation_id else item for item in self.obligations
        ))

    def _set_hypothesis(self, payload: dict[str, Any]) -> "RCAState":
        """Accept a fully Runtime-normalized hypothesis snapshot.

        The Planner never calls this event directly.  Runtime first allocates
        IDs, links known Evidence and applies obligation transitions, then
        submits the resulting snapshot here.
        """
        raw = payload.get("hypothesis", payload)
        hypothesis = IncidentHypothesis.model_validate(raw)
        referenced = {item.id for item in hypothesis.verification_obligations}
        if referenced - {item.id for item in self.obligations}:
            raise RCAStateError("hypothesis references unknown canonical obligation")
        refs = set(hypothesis.supporting_evidence_ids)
        refs.update(hypothesis.contradicting_evidence_ids)
        refs.update(item.evidence_id for item in hypothesis.contradictions)
        if refs - set(self.evidence_ids):
            raise RCAStateError("hypothesis references unknown Evidence")
        if any(item.evidence_id not in self.evidence_ids for item in hypothesis.contradictions):
            raise RCAStateError("contradiction references unknown Evidence")
        canonical_hypothesis = hypothesis.model_copy(update={
            "verification_obligations": (),
            "required_gaps": (),
            "evidence_gap": "",
            "contradictions": (),
            "contradicting_evidence_ids": (),
            "evidence_sufficient": False,
        })
        return self._next(hypotheses=self.hypotheses + (canonical_hypothesis,))

    def _set_contradictions(self, payload: dict[str, Any]) -> "RCAState":
        raw = payload.get("contradictions", ())
        contradictions = tuple(Contradiction.model_validate(item) for item in raw)
        if len({item.evidence_id for item in contradictions}) != len(contradictions):
            raise RCAStateError("contradiction evidence IDs must be unique")
        if any(item.evidence_id not in self.evidence_ids for item in contradictions):
            raise RCAStateError("contradiction must reference admitted Evidence")
        prior = {item.evidence_id: item for item in self.contradictions}
        allowed = {
            "OPEN": {"OPEN", "RESOLVED", "EXPLAINED"},
            "RESOLVED": {"RESOLVED"},
            "EXPLAINED": {"EXPLAINED"},
        }
        for item in contradictions:
            previous = prior.get(item.evidence_id)
            if previous is not None and item.status not in allowed[previous.status]:
                raise RCAStateError(
                    f"invalid contradiction transition {previous.status}->{item.status}"
                )
        return self._next(contradictions=contradictions)

    def _invalidate_branch(self, obligation_id: str) -> "RCAState":
        obligation = self.obligation(obligation_id)
        if obligation.status not in {"SATISFIED", "OPEN"}:
            raise RCAStateError("only an active/satisfied branch can be invalidated")
        updated = obligation.model_copy(update={"status": "OPEN", "supporting_evidence_ids": ()})
        return self._next(obligations=tuple(
            updated if item.id == obligation_id else item for item in self.obligations
        ))

    def _reopen_obligation(self, obligation_id: str, payload: dict[str, Any]) -> "RCAState":
        reason = str(payload.get("reason", "")).strip()
        if not reason:
            raise RCAStateError("reopening an obligation requires a reason")
        obligation = self.obligation(obligation_id)
        if obligation.status != "SATISFIED":
            raise RCAStateError("only a satisfied obligation can be reopened")
        updated = obligation.model_copy(update={"status": "OPEN", "supporting_evidence_ids": ()})
        return self._next(obligations=tuple(
            updated if item.id == obligation_id else item for item in self.obligations
        ))

    def _next(self, **updates: Any) -> "RCAState":
        updates["event_count"] = self.event_count + 1
        payload = self.model_dump(mode="python")
        payload.update(updates)
        return type(self).model_validate(payload)
