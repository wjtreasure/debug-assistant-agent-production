from __future__ import annotations
from debug_assistant.models import DiagnosisReport

class FallbackReportBuilder:
    """Deterministic report from persisted hypothesis/evidence metadata only."""
    def build(self, task_id: str, hypothesis, evidence) -> DiagnosisReport:
        ids=set(hypothesis.supporting_evidence_ids or [])
        selected=[e for e in evidence if not ids or e.evidence_id in ids]
        files=[]; symbols=[]
        for e in selected:
            if e.file and e.file not in files: files.append(e.file)
            for tag in e.tags:
                if tag.startswith('symbol:'):
                    sym=tag.split(':',1)[1]
                    if sym and sym not in symbols: symbols.append(sym)
        desc=hypothesis.description or "Diagnosis could not be fully formatted by the LLM reporter."
        ev=[{"evidence_id":e.evidence_id,"source":e.source,"file":e.file,"line_start":e.line_start,"line_end":e.line_end,
             "summary":e.summary,"raw_observation_id":e.raw_observation_id,"source_start_line":e.source_start_line,
             "source_end_line":e.source_end_line,"excerpt_start_line":e.excerpt_start_line,"excerpt_end_line":e.excerpt_end_line,
             "excerpt_truncated":e.excerpt_truncated} for e in selected]
        return DiagnosisReport(
            task_id=task_id,summary=desc,root_cause=desc,likely_files=files,likely_symbols=symbols,
            impact_scope=[],evidence=ev,recommended_change_points=[],uncertainties=list(hypothesis.missing_evidence or []),
            next_checks=list(hypothesis.missing_evidence or []),confidence=float(hypothesis.confidence or 0.0),
            policy_note="Read-only diagnosis. Report was deterministically reconstructed after LLM reporter failure.",
            report_source="fallback",evidence_ids=[e.evidence_id for e in selected],
        )
