"""Pure projection of terminal RCA state into a reviewable report."""

from __future__ import annotations

from typing import Any

from .contracts import DiagnosticReport


def build_diagnostic_report(
    *,
    case: Any,
    run_status: str,
    termination_reason: str,
    candidate: Any | None,
    rca_state: Any,
    evidence_memory: Any,
    projected_facts: Any = (),
    review_status: str = "NOT_STARTED",
    review_reason: str = "",
    review_confidence_score: float | None = None,
    review_confidence_reason: str = "",
) -> DiagnosticReport:
    """Build a detached report without interpreting or ranking hypotheses."""

    hypothesis = rca_state.active_hypothesis_projection()
    records = tuple(rca_state.semantic_records)
    support = tuple(dict.fromkeys(
        evidence_id
        for item in (hypothesis.supporting_evidence_ids if hypothesis else ())
        for evidence_id in (str(item),)
    ))
    contradict = tuple(dict.fromkeys(
        item.evidence_id for item in (rca_state.contradictions or ())
        if item.blocks_finalization
    ))
    evidence_links: list[dict[str, Any]] = []
    facts = tuple(projected_facts or ())
    for item in evidence_memory.pinned:
        relations = [fact for fact in facts if fact.evidence_id == item.evidence_id]
        evidence_links.append({
            "evidence_id": item.evidence_id,
            "observation_id": item.raw_observation_id or "",
            "source": item.source,
            "file": item.file,
            "start_line": item.source_start_line,
            "end_line": item.source_end_line,
            "relations": [
                {
                    "obligation_id": fact.obligation_id,
                    "hypothesis_id": fact.hypothesis_id,
                    "claim": fact.fact,
                    "hypothesis_relation": fact.hypothesis_relation,
                    "obligation_relation": fact.relation,
                    "covered_fact_slots": list(fact.covered_fact_slots),
                    "reason": fact.reason,
                    "rationale": fact.rationale,
                    "source_spans": list(fact.source_spans),
                }
                for fact in relations[-8:]
            ],
        })
    required = tuple(
        item.claim for item in rca_state.obligations if item.blocks_finalization
    )
    exploratory = tuple(
        item.claim for item in rca_state.obligations if not item.blocks_finalization
    )
    unresolved = tuple(dict.fromkeys((
        *(required if hypothesis is not None else ()),
        *(
            (hypothesis.claim,)
            if hypothesis is not None and hypothesis.status != "supported"
            else ()
        ),
    )))
    next_steps = tuple(dict.fromkeys((*required, *exploratory)))
    status = (
        "SUPPORTED_ROOT_CAUSE" if run_status == "PASS" and candidate is not None
        else "FAILED" if run_status == "FAILED"
        else "INCONCLUSIVE"
    )
    confidence = review_confidence_score
    basis = review_confidence_reason.strip()
    if confidence is None and not basis:
        basis = "Independent Review has not supplied a confidence score."
    return DiagnosticReport(
        case_id=str(case.case_id),
        status=status,
        incident_summary=str(getattr(case, "summary", "") or ""),
        termination_reason=str(termination_reason or ""),
        hypothesis_id=str(getattr(hypothesis, "hypothesis_id", "") or ""),
        hypothesis_revision=int(getattr(hypothesis, "revision", 0) or 0),
        hypothesis_claim=str(getattr(hypothesis, "claim", "") or ""),
        hypothesis_status=(getattr(hypothesis, "status", "none") if hypothesis else "none"),
        root_cause_component=str(getattr(candidate, "component", "") or ""),
        root_cause_fault=str(
            getattr(candidate, "fault_explanation", "")
            or getattr(candidate, "fault", "")
            or ""
        ),
        root_cause_fault_code=str(getattr(candidate, "fault_code", "") or ""),
        root_cause_mechanism=str(getattr(candidate, "mechanism", "") or ""),
        root_cause_evidence_ids=tuple(
            getattr(candidate, "evidence_ids", ()) or ()
        ),
        confidence=confidence,
        confidence_basis=basis,
        semantic_record_ids=tuple(item.record_id for item in records),
        supporting_evidence_ids=support,
        contradicting_evidence_ids=contradict,
        evidence_links=tuple(evidence_links),
        open_required_questions=required,
        open_exploratory_questions=exploratory,
        unverified_claims=unresolved,
        next_steps=next_steps,
        review_status=review_status,
        review_reason=review_reason,
    )
