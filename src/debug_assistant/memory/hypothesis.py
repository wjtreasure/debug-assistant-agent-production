from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
import hashlib, os, re
from typing import Iterable
from debug_assistant.repository.safe_fs import SafeRepositoryFS
from debug_assistant.repository.paths import RepositoryPathResolver, ResolutionMode, RepositoryPathError

@dataclass(slots=True)
class MissingEvidenceState:
    target: str
    location: str | None = None
    reason: str = ""

@dataclass(slots=True)
class HypothesisState:
    description: str = ""
    root_cause_target: str = ""
    root_cause_location: str = ""
    root_cause_mechanism: str = ""
    status: str = "none"
    confidence: float = 0.0
    supporting_evidence_ids: list[str] = field(default_factory=list)
    contradicting_evidence_ids: list[str] = field(default_factory=list)
    required_missing_evidence: list[dict] = field(default_factory=list)
    optional_validation: list[dict] = field(default_factory=list)
    updated_step: int = 0
    stable_diagnosis_transitions: int = 0
    diagnosis_fingerprint: str = ""
    evidence_fingerprint: str = ""
    required_gap_fingerprint: str = ""
    model_claimed_changed: bool | None = None
    evidence_sufficient: bool = False

    @property
    def stable_reflections(self) -> int:
        """Backward-compatible alias for V1.3 callers/tests."""
        return self.stable_diagnosis_transitions

    @property
    def missing_evidence(self) -> list[str]:
        """Backward-compatible human-readable view."""
        return [x.get("target","") for x in self.required_missing_evidence]


def _sha(parts: Iterable[str]) -> str:
    raw="|".join(parts)
    return hashlib.sha1(raw.encode("utf-8","ignore")).hexdigest()[:16]


def normalize_target(text: str) -> str:
    text=(text or "").strip().lower()
    text=re.sub(r"[^\w\s/]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def normalize_location(location: str | None, repo_root: str | Path | None = None) -> str:
    """Normalize model-emitted source locations to a stable repo-relative key.

    The normalization is deliberately deterministic: separators are POSIX, trailing
    line decorations are removed, ``..`` segments are collapsed, absolute paths
    under the repository become relative, and a bare/suffix path is expanded to the
    unique matching repository file when that can be done unambiguously.
    """
    if not location:
        return ""
    s=str(location).strip().replace("\\","/")
    s=re.sub(r"^file://", "", s, flags=re.I)
    # Strip only explicit trailing line decorations. Digits inside file names stay intact.
    s=re.sub(r"(?i)(?:#L\d+(?:-L?\d+)?|:\d+(?:-\d+)?|\s*\(\s*lines?\s+\d+(?:-\d+)?\s*\))$", "", s).strip()
    root=Path(repo_root).resolve() if repo_root else None
    win_abs=bool(re.match(r"^[A-Za-z]:/",s))

    def _clean_posix(value: str) -> str:
        value=value.replace("\\","/")
        parts=[]
        for part in PurePosixPath(value).parts:
            if part in ('/', '.', ''):
                continue
            if part == '..':
                if parts: parts.pop()
                continue
            # Drop Windows drive segment after conversion, e.g. C:
            if re.fullmatch(r"[A-Za-z]:", part):
                continue
            parts.append(part)
        return PurePosixPath(*parts).as_posix() if parts else ""

    try:
        if root and not win_abs:
            p=Path(s)
            candidate=(p if p.is_absolute() else root/p).resolve(strict=False)
            try:
                normalized=candidate.relative_to(root).as_posix()
            except ValueError:
                # Absolute model output may contain the repo directory name.
                parts=PurePosixPath(candidate.as_posix()).parts
                if root.name in parts:
                    idx=len(parts)-1-list(reversed(parts)).index(root.name)
                    normalized=PurePosixPath(*parts[idx+1:]).as_posix()
                else:
                    normalized=_clean_posix(s)
        elif root and win_abs:
            raw=_clean_posix(re.sub(r"^[A-Za-z]:/", "", s))
            parts=PurePosixPath(raw).parts
            if root.name in parts:
                idx=len(parts)-1-list(reversed(parts)).index(root.name)
                normalized=PurePosixPath(*parts[idx+1:]).as_posix()
            else:
                normalized=raw
        else:
            normalized=_clean_posix(s)
    except Exception:
        normalized=_clean_posix(s)

    while normalized.startswith("./"):
        normalized=normalized[2:]

    # If the model emitted only a basename or shortened suffix, use the same
    # repository-safe resolver as Tool addressing. Never guess when ambiguous.
    if root and normalized and root.exists():
        try:
            resolver=RepositoryPathResolver(SafeRepositoryFS(root))
            normalized=resolver.resolve_file(normalized,mode=ResolutionMode.READ_TOLERANT).relative_path
        except RepositoryPathError:
            pass
    return normalized


def _gap_pairs(items: list[dict], repo_root=None) -> list[str]:
    out=[]
    for item in items or []:
        if hasattr(item,'model_dump'): item=item.model_dump()
        target=normalize_target((item or {}).get('target',''))
        location=normalize_location((item or {}).get('location'),repo_root)
        out.append(f"{target}@{location}")
    return sorted(set(out))


def diagnosis_fingerprint(
    description: str,
    status: str,
    contradicting: list[str],
    *,
    root_cause_target: str = "",
    root_cause_location: str = "",
) -> str:
    """Fingerprint diagnosis identity without coupling stability to prose rewrites.

    Structured target/location are authoritative when present. The natural-language
    diagnosis is retained only as a legacy fallback for older snapshots/tests.
    Evidence accumulation never participates; direct contradiction membership does.
    """
    target=normalize_target(root_cause_target)
    location=(root_cause_location or "").strip()
    identity=(f"{target}@{location}" if target or location else normalize_target(description))
    return _sha([identity,status,",".join(sorted(set(contradicting)))])


def evidence_fingerprint(supporting: list[str], contradicting: list[str]) -> str:
    return _sha([",".join(sorted(set(supporting))),",".join(sorted(set(contradicting)))])


def required_gap_fingerprint(items: list[dict], repo_root=None) -> str:
    return _sha(_gap_pairs(items,repo_root))


class HypothesisManager:
    def __init__(self, repo_root: str | Path | None = None):
        self.repo_root=Path(repo_root).resolve() if repo_root else None
        self.state=HypothesisState()

    def update(self, review: dict, step: int) -> HypothesisState:
        desc=review.get("current_diagnosis","") or ""
        root_target=review.get("root_cause_target","") or ""
        root_location=normalize_location(review.get("root_cause_location"),self.repo_root)
        root_mechanism=review.get("root_cause_mechanism","") or ""
        support=list(review.get("supporting_evidence_ids") or [])
        contradict=list(review.get("contradicting_evidence_ids") or [])
        required=list(review.get("required_missing_evidence") or [])
        optional=list(review.get("optional_validation") or [])
        # Backward-compatible ingestion for tests / old snapshots only. The V1.3.1
        # ReflectionContract no longer emits the ambiguous `missing` field.
        if not required:
            legacy=review.get("missing") or review.get("missing_evidence") or []
            required=[{"target":str(x),"location":None,"reason":"legacy reflection state"} for x in legacy]
        sufficient=bool(review.get("evidence_sufficient"))
        # Harness owns hypothesis status deterministically. Supporting evidence alone
        # is not enough to call a diagnosis supported: the causal target and mechanism
        # must both be explicit. CONFIRMED additionally requires sufficient evidence and
        # no remaining critical gap.
        if contradict:
            status="contradicted"
        elif sufficient and support and root_target and root_mechanism and not required:
            status="confirmed"
        elif support and root_target and root_mechanism:
            status="supported"
        elif desc or root_target or root_location or root_mechanism or support:
            status="partial"
        else:
            status="none"

        dfp=diagnosis_fingerprint(
            desc,status,contradict,root_cause_target=root_target,root_cause_location=root_location
        )
        efp=evidence_fingerprint(support,contradict)
        gfp=required_gap_fingerprint(required,self.repo_root)
        stable=(self.state.stable_diagnosis_transitions+1
                if dfp and dfp==self.state.diagnosis_fingerprint else 0)
        self.state=HypothesisState(
            description=desc,root_cause_target=root_target,root_cause_location=root_location,root_cause_mechanism=root_mechanism,
            status=status,confidence=float(review.get("confidence",0.0) or 0.0),
            supporting_evidence_ids=support,contradicting_evidence_ids=contradict,
            required_missing_evidence=[x.model_dump() if hasattr(x,'model_dump') else dict(x) for x in required],
            optional_validation=[x.model_dump() if hasattr(x,'model_dump') else dict(x) for x in optional],
            updated_step=step,stable_diagnosis_transitions=stable,
            diagnosis_fingerprint=dfp,evidence_fingerprint=efp,required_gap_fingerprint=gfp,
            model_claimed_changed=review.get("hypothesis_changed"),evidence_sufficient=sufficient,
        )
        return self.state
