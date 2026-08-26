from __future__ import annotations
from dataclasses import dataclass, field
import hashlib, re

@dataclass(slots=True)
class HypothesisState:
    description: str = ""
    status: str = "none"
    confidence: float = 0.0
    supporting_evidence_ids: list[str] = field(default_factory=list)
    contradicting_evidence_ids: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    updated_step: int = 0
    stable_reflections: int = 0
    fingerprint: str = ""
    model_claimed_changed: bool | None = None


def _norm(text: str) -> str:
    return re.sub(r"\s+"," ",text.strip().lower())

def fingerprint(description: str, supporting: list[str], contradicting: list[str], status: str) -> str:
    raw="|".join([_norm(description),status,",".join(sorted(supporting)),",".join(sorted(contradicting))])
    return hashlib.sha1(raw.encode("utf-8","ignore")).hexdigest()[:16]

class HypothesisManager:
    def __init__(self): self.state=HypothesisState()
    def update(self, review: dict, step: int) -> HypothesisState:
        desc=review.get("current_diagnosis","") or ""
        support=list(review.get("supporting_evidence_ids") or [])
        contradict=list(review.get("contradicting_evidence_ids") or [])
        missing=list(review.get("missing") or review.get("missing_evidence") or [])
        sufficient=bool(review.get("evidence_sufficient"))
        status="confirmed" if sufficient and not contradict else ("contradicted" if contradict else ("supported" if support else ("active" if desc else "none")))
        fp=fingerprint(desc,support,contradict,status)
        stable=self.state.stable_reflections+1 if fp and fp==self.state.fingerprint else 0
        self.state=HypothesisState(
            description=desc,status=status,confidence=float(review.get("confidence",0.0) or 0.0),
            supporting_evidence_ids=support,contradicting_evidence_ids=contradict,missing_evidence=missing,
            updated_step=step,stable_reflections=stable,fingerprint=fp,
            model_claimed_changed=review.get("hypothesis_changed"),
        )
        return self.state
