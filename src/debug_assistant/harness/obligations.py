from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import re
from typing import Callable

from debug_assistant.memory.hypothesis import normalize_location, normalize_target


class ObligationStatus(str, Enum):
    OPEN = "open"
    ATTEMPTED = "attempted"
    SATISFIED = "satisfied"
    UNRESOLVED = "unresolved"
    OPTIONAL = "optional"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"  # Reserved; V1.4.5 still does not auto-expire obligations.


_LINE_SUFFIX_RE = re.compile(r"(?::(?P<a>\d+)(?:-(?P<b>\d+))?|#L(?P<c>\d+)(?:-L?(?P<d>\d+))?)$", re.I)
_PATH_IN_TEXT_RE = re.compile(
    r"(?P<path>(?:[A-Za-z0-9_.-]+[\\/])+[A-Za-z0-9_.-]+\.[A-Za-z0-9]+)"
    r"(?::(?P<a>\d+)(?:-(?P<b>\d+))?)?"
)
_PATH_LINES_TEXT_RE = re.compile(
    r"(?P<path>(?:[A-Za-z0-9_.-]+[\\/])+[A-Za-z0-9_.-]+\.[A-Za-z0-9]+)\s+(?:lines?|L)\s*(?P<a>\d+)(?:\s*[-–]\s*(?P<b>\d+))?", re.I
)
_GENERIC_TARGET_WORDS = {
    "implementation", "definition", "exact", "source", "code", "inspect", "read", "locate", "location",
    "function", "method", "symbol", "the", "a", "an", "of", "for", "in", "at", "lines", "line", "body",
    "behavior", "behaviour", "handling", "confirm", "verify", "determine", "whether", "path", "file",
    "directory", "without", "with", "from", "module", "package",
}

# Discovery evidence can guide the Planner but cannot prove behavior/causality.
_SOURCE_CAPABILITIES = {
    "read_file": {"location", "behavior", "causality", "caller", "test", "contradiction", "evidence"},
    "git_show": {"history", "location", "behavior", "causality", "contradiction", "evidence"},
}


SymbolLookup = Callable[[str, int], list[dict]]


def _looks_like_repo_file(value: str) -> bool:
    value = (value or "").strip().replace("\\", "/")
    if not value or any(x in value.lower() for x in ("git history", "git diff")):
        return False
    tail = value.rsplit("/", 1)[-1]
    return bool("." in tail and not tail.startswith("."))


def _location_parts(location: str | None, repo_root=None) -> tuple[tuple[str, ...], tuple[int, int] | None]:
    if not location:
        return (), None
    raw = str(location).strip().replace("\\", "/")
    line_hint = None
    if not re.search(r"\s+or\s+|\s*[,|]\s*", raw, re.I):
        m = _LINE_SUFFIX_RE.search(raw)
        if m:
            a = int(m.group("a") or m.group("c"))
            b = int(m.group("b") or m.group("d") or a)
            line_hint = (min(a, b), max(a, b))
            raw = raw[:m.start()]
    candidates = []
    for part in re.split(r"\s+or\s+|\s*[,|]\s*", raw, flags=re.I):
        part = part.strip()
        if not part or not _looks_like_repo_file(part):
            continue
        norm = normalize_location(part, repo_root)
        if norm:
            candidates.append(norm)
    return tuple(sorted(set(candidates))), line_hint


def _target_location_parts(target: str, repo_root=None) -> tuple[tuple[str, ...], tuple[int, int] | None]:
    files = []
    line_hint = None
    text=str(target or "").replace("\\", "/")
    for regex in (_PATH_IN_TEXT_RE,_PATH_LINES_TEXT_RE):
        for m in regex.finditer(text):
            norm = normalize_location(m.group("path"), repo_root)
            if norm:
                files.append(norm)
            if line_hint is None and m.group("a"):
                a = int(m.group("a")); b = int(m.group("b") or a)
                line_hint = (min(a, b), max(a, b))
    return tuple(sorted(set(files))), line_hint


def _legacy_goal_type(target: str, reason: str = "") -> str:
    """Conservative compatibility inference for old Reflection outputs.

    V1.4.5 treats an explicit structured ``goal_type`` as authoritative. This helper
    exists only for older model outputs and intentionally prioritizes concrete behavior /
    causality language over generic words such as "regression".
    """
    s = " ".join(x for x in (normalize_target(target), normalize_target(reason)) if x)
    if any(x in s for x in ("contradict", "falsif", "counterexample")):
        return "contradiction"
    if "test" in s:
        return "test"
    if any(x in s for x in ("caller", "callsite", "call site", "call path", "callee")):
        return "caller"
    if any(x in s for x in ("mechanism", "causal", "root cause", "cause", "why", "produced", "constructs")):
        return "causality"
    if any(x in s for x in ("existence check", "condition", "whether", "behavior", "behaviour", "handling", "handle", "opens", "returns", "raises", "converts", "resolution")):
        return "behavior"
    # History requires an actual history/diff intent, not merely the word "regression".
    if any(x in s for x in ("git history", "git diff", "commit", "compare version", "version diff", "introduced in", "history")):
        return "history"
    if any(x in s for x in ("implementation", "definition", "location", "path resolution", "resolve", "generation", "body")):
        return "location"
    return "evidence"


def _target_core(target: str) -> str:
    norm = normalize_target(target)
    words = [w for w in norm.split() if w not in _GENERIC_TARGET_WORDS and not w.isdigit()]
    return " ".join(words) or norm


def _meaningful_probes(target: str) -> tuple[list[str], int]:
    cleaned = _PATH_IN_TEXT_RE.sub(" ", str(target or ""))
    identifiers = [
        x for x in re.findall(r"\b[A-Za-z_]\w+\b", cleaned)
        if x.lower() not in _GENERIC_TARGET_WORDS and len(x) > 3
    ]
    codeish = [x for x in identifiers if "_" in x or any(c.isupper() for c in x)]
    if codeish:
        return list(dict.fromkeys(codeish[:4])), 1
    probes = list(dict.fromkeys(identifiers[:3]))
    return probes, min(2, len(probes))


def _symbol_probes(target: str) -> list[str]:
    """Extract code-symbol candidates without turning class qualifiers into extra duties."""
    cleaned=_PATH_IN_TEXT_RE.sub(' ',str(target or ''))
    dotted_refs=re.findall(r'\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b',cleaned)
    dotted_parts={part for ref in dotted_refs for part in ref.split('.')}
    dotted=[]
    for ref in dotted_refs:
        parts=ref.split('.')
        # RepositoryIndex qualified names do not include module names. Preserve a
        # Class.method pair, otherwise use the leaf symbol for module.symbol text.
        if len(parts)>=2 and parts[-2][:1].isupper():
            dotted.append('.'.join(parts[-2:]))
        else:
            dotted.append(parts[-1])
    identifiers=[x for x in re.findall(r'\b[A-Za-z_]\w+\b',cleaned)
                 if x.lower() not in _GENERIC_TARGET_WORDS and len(x)>2 and x not in dotted_parts]
    codeish=[x for x in identifiers if '_' in x or any(c.isupper() for c in x)]
    fallback=[x for x in identifiers if x not in codeish]
    return list(dict.fromkeys(dotted+codeish+fallback))[:6]


def _canonical_goal_type(raw: str | None, target: str, reason: str) -> tuple[str, str]:
    if raw:
        value = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "implementation": "behavior", "definition": "location", "where": "location", "symbol": "location",
            "mechanism": "causality", "cause": "causality", "root_cause": "causality", "why": "causality",
            "verification": "behavior", "validation": "behavior", "check": "behavior",
            "callsite": "caller", "call_site": "caller", "call_path": "caller", "callee": "caller",
            "tests": "test", "testing": "test", "history_diff": "history", "git_history": "history",
            "falsification": "contradiction", "conflict": "contradiction",
        }
        canonical=aliases.get(value, value)
        semantic_text=normalize_target(f'{target} {reason}')
        if canonical=='location' and any(token in semantic_text for token in (' implementation',' body',' behavior',' behaviour',' handling',' how ')):
            return 'behavior', 'structured_corrected'
        return canonical, "structured"
    return _legacy_goal_type(target, reason), "legacy_inference"


@dataclass(slots=True)
class EvidenceObligation:
    obligation_id: str
    target: str
    reason: str
    location: str | None = None
    canonical_target: str = ""
    canonical_files: tuple[str, ...] = ()
    goal_type: str = "evidence"
    goal_type_source: str = "legacy_inference"
    line_hint: tuple[int, int] | None = None
    canonical_symbols: tuple[str, ...] = ()
    # (path, symbol_name, start_line, end_line)
    symbol_ranges: tuple[tuple[str, str, int, int], ...] = ()
    critical: bool = True
    active_required: bool = True
    status: ObligationStatus = ObligationStatus.OPEN
    attempts: int = 0
    evidence_ids: list[str] = field(default_factory=list)
    evidence_ready: bool = False
    last_presented_reflection_id: str | None = None
    last_presented_projection_id: str | None = None
    last_presented_evidence_fingerprint: str | None = None
    last_reviewed_reflection_id: str | None = None
    last_reviewed_evidence_fingerprint: str | None = None
    last_review_decision: str | None = None
    review_decision_source: str | None = None
    superseded_by: str | None = None
    refined_from: str | None = None
    aliases: list[str] = field(default_factory=list)
    information_need_root_id: str | None = None
    scope_valid: bool = True
    scope_error: str | None = None


_SEMANTIC_REVIEW_GOALS = {"behavior", "causality", "caller", "contradiction"}
_TERMINAL_STATUSES = {ObligationStatus.SATISFIED, ObligationStatus.SUPERSEDED}
_SUPPORTING_ACTION_TERMS = {
    "caller", "callsite", "call site", "callee", "test", "tests", "import",
    "dependency", "definition", "symbol", "related", "context", "upstream", "downstream",
}


class EvidenceObligationTracker:
    """Task-scoped evidence obligations with separate evidence and semantic planes.

    V1.4.6 invariants:
    * source/range/symbol matching only makes semantic evidence READY;
    * behavior/causality/caller/contradiction never become SATISFIED from coverage alone;
    * semantic SATISFIED requires a successful review of evidence that was physically
      presented in that Reflection request;
    * presentation facts are execution history, while semantic status is transactional.
    """

    def __init__(self, max_attempts: int = 2, repo_root=None, symbol_lookup: SymbolLookup | None = None):
        self.max_attempts = max(1, int(max_attempts))
        self.items: dict[str, EvidenceObligation] = {}
        self.repo_root = repo_root
        self.symbol_lookup = symbol_lookup
        self._events: list[tuple[str,dict]] = []

    def pop_events(self) -> list[tuple[str,dict]]:
        rows=list(self._events); self._events.clear(); return rows

    def _event(self, event_type: str, payload: dict) -> None:
        self._events.append((event_type,payload))

    def set_symbol_lookup(self, symbol_lookup: SymbolLookup | None) -> None:
        self.symbol_lookup = symbol_lookup
        if symbol_lookup:
            for obj in self.items.values():
                symbols, ranges = self._resolve_symbol_scopes(obj.target, obj.canonical_files)
                if symbols:
                    obj.canonical_symbols = symbols
                    obj.symbol_ranges = ranges

    def _resolve_symbol_scopes(self, target: str, files: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[tuple[str, str, int, int], ...]]:
        if not self.symbol_lookup:
            return (), ()
        probes = _symbol_probes(target)
        found = []
        names = []
        for probe in probes:
            try:
                rows = self.symbol_lookup(probe, 40) or []
            except TypeError:
                rows = self.symbol_lookup(probe, limit=40) or []
            exact = []
            for row in rows:
                name = str(row.get("name") or "")
                qname = str(row.get("qualified_name") or "")
                path = normalize_location(row.get("path"), self.repo_root) if row.get("path") else ""
                if files and path not in files:
                    continue
                if probe not in {name, qname} and not qname.endswith("." + probe):
                    continue
                start = row.get("start_line"); end = row.get("end_line")
                if not path or start is None or end is None:
                    continue
                exact.append((path, name or probe, int(start), int(end)))
            unique = list(dict.fromkeys(exact))
            if len(unique) == 1:
                found.append(unique[0]); names.append(unique[0][1])
        return tuple(dict.fromkeys(names)), tuple(dict.fromkeys(found))

    def _structured_scope(self, raw: dict):
        file_raw=(raw.get("file") or "").strip() if isinstance(raw.get("file"),str) else raw.get("file")
        symbol=(raw.get("symbol") or "").strip() if isinstance(raw.get("symbol"),str) else raw.get("symbol")
        a=raw.get("line_start"); b=raw.get("line_end")
        file_norm=normalize_location(file_raw,self.repo_root) if file_raw else None
        if symbol and not self.symbol_lookup:
            self._event('AMBIGUOUS_SCOPE',{'symbol':symbol,'file':file_norm,'reason':'repository_symbol_index_unavailable'})
            return ((file_norm,) if file_norm else ()), ((int(a),int(b)) if a is not None and b is not None else None), (), (), False, 'repository_symbol_index_unavailable'
        if symbol and self.symbol_lookup:
            try: rows=self.symbol_lookup(str(symbol),100) or []
            except TypeError: rows=self.symbol_lookup(str(symbol),limit=100) or []
            exact=[]
            for row in rows:
                name=str(row.get("name") or ""); q=str(row.get("qualified_name") or "")
                if str(symbol) not in {name,q} and not q.endswith("."+str(symbol)): continue
                path=normalize_location(row.get("path"),self.repo_root) if row.get("path") else ""
                if not path or row.get("start_line") is None or row.get("end_line") is None: continue
                exact.append((path,name or str(symbol),int(row["start_line"]),int(row["end_line"])))
            exact=list(dict.fromkeys(exact))
            preferred=[x for x in exact if not file_norm or x[0]==file_norm]
            chosen=preferred if len(preferred)==1 else exact if len(exact)==1 else []
            if len(chosen)==1:
                path,name,start,end=chosen[0]
                if file_norm and path!=file_norm:
                    self._event('OBLIGATION_SCOPE_CORRECTED',{'reason':'unique_symbol_file_authoritative','symbol':symbol,'from_file':file_norm,'to_file':path})
                if a is not None and b is not None and (int(a),int(b))!=(start,end):
                    self._event('OBLIGATION_SCOPE_CORRECTED',{'reason':'symbol_range_authoritative','symbol':symbol,'from_range':[int(a),int(b)],'to_range':[start,end],'file':path})
                return (path,), (start,end), (name,), ((path,name,start,end),), True, None
            self._event('AMBIGUOUS_SCOPE',{'symbol':symbol,'file':file_norm,'candidate_count':len(exact),'candidates':[{'file':x[0],'symbol':x[1],'start_line':x[2],'end_line':x[3]} for x in exact[:10]]})
            return ((file_norm,) if file_norm else ()), ((int(a),int(b)) if a is not None and b is not None else None), (), (), False, 'ambiguous_or_missing_symbol'
        if file_norm and a is not None and b is not None:
            try:
                path=(self.repo_root / file_norm) if self.repo_root else None
                valid=(path is None or (path.is_file() and int(a)>=1 and int(b)>=int(a)))
            except Exception: valid=False
            if not valid:
                self._event('AMBIGUOUS_SCOPE',{'file':file_norm,'range':[a,b],'reason':'invalid_repository_range'})
            return (file_norm,), (int(a),int(b)), (), (), valid, None if valid else 'invalid_repository_range'
        if file_norm:
            return (file_norm,), None, (), (), True, None
        return None

    def _identity(self, target: str, reason: str, location: str | None, goal_type: str | None = None, raw: dict | None = None):
        core = _target_core(target)
        structured=self._structured_scope(raw or {}) if raw else None
        if structured is not None:
            files,line_hint,symbols,symbol_ranges,scope_valid,scope_error=structured
        else:
            loc_files, loc_line = _location_parts(location, self.repo_root)
            target_files, target_line = _target_location_parts(target, self.repo_root)
            files = tuple(sorted(set(loc_files + target_files)))
            line_hint = loc_line or target_line
            symbols, symbol_ranges = self._resolve_symbol_scopes(target, files)
            scope_valid=True; scope_error=None
            # Legacy free-text requirements that resolve to more than one independent
            # source body are composite semantic duties.  V1.4.6 fails them closed so
            # Reflection must emit/refine atomic requirements instead of relying on
            # accidental AND/OR matching.
            independent={(r[0],r[1]) for r in symbol_ranges}
            if len(independent)>1:
                scope_valid=False; scope_error='composite_scope_requires_atomization'
                self._event('AMBIGUOUS_SCOPE',{'reason':scope_error,'target':target,'symbols':list(symbols),'ranges':[list(r) for r in symbol_ranges]})
        goal, goal_source = _canonical_goal_type(goal_type, target, reason)
        key = f'{core}|{";".join(files)}|{";".join(symbols)}'
        oid = "O" + hashlib.sha1(key.encode("utf-8", "ignore")).hexdigest()[:8]
        return oid, core, files, line_hint, goal, goal_source, symbols, symbol_ranges, scope_valid, scope_error

    @staticmethod
    def _same_scope(a: EvidenceObligation, *, files: tuple[str, ...], goal: str, symbols: tuple[str, ...]) -> bool:
        if a.goal_type != goal:
            return False
        if a.canonical_files and files and a.canonical_files != files:
            return False
        if a.canonical_symbols and symbols:
            return bool(set(a.canonical_symbols) & set(symbols))
        return False

    def _lookup_equivalent(self, oid: str, core: str, files: tuple[str, ...], *, goal: str = "", symbols: tuple[str, ...] = ()) -> EvidenceObligation | None:
        obj = self.items.get(oid)
        if obj is not None and not (obj.status in _TERMINAL_STATUSES and obj.goal_type != goal):
            return obj
        candidates = []
        for existing in self.items.values():
            if existing.status is ObligationStatus.SUPERSEDED:
                continue
            # A terminal obligation's evidence semantics are immutable. A later
            # requirement over the same source identity but with a different goal
            # must become a new obligation instead of mutating e.g. a deterministically
            # satisfied location check into an unreviewed semantic obligation.
            if existing.status in _TERMINAL_STATUSES and existing.goal_type != goal:
                continue
            if existing.canonical_target == core and (existing.canonical_files == files or not existing.canonical_files or not files):
                candidates.append(existing)
                continue
            # Structured symbol identity is stronger than prose. This merges e.g.
            # "spec.find_spec behavior" and "find_spec behavior ..." across optional/required turns.
            if goal and symbols and self._same_scope(existing, files=files, goal=goal, symbols=symbols):
                candidates.append(existing)
        unique={x.obligation_id:x for x in candidates}
        return next(iter(unique.values())) if len(unique) == 1 else None

    def _upsert(self, raw: dict, *, required: bool, refined_from: str | None = None) -> tuple[EvidenceObligation | None, dict | None]:
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump()
        raw = raw or {}
        target = str(raw.get("target", "")).strip()
        reason = str(raw.get("reason", "")).strip()
        loc = raw.get("location")
        explicit_goal = raw.get("goal_type")
        if not target:
            return None, None
        oid, core, files, line_hint, goal, goal_source, symbols, symbol_ranges, scope_valid, scope_error = self._identity(target, reason, loc, explicit_goal, raw)
        obj = self._lookup_equivalent(oid, core, files, goal=goal, symbols=symbols)
        transition = None
        if obj is not None:
            oid = obj.obligation_id
        if obj is None:
            if oid in self.items:
                # ``_identity`` intentionally keeps the stable source identity
                # used by equivalent prose requirements. Once that identity is
                # terminal, however, a different goal needs its own lifecycle.
                base_oid = oid
                oid = "O" + hashlib.sha1(f"{base_oid}|{goal}".encode("utf-8", "ignore")).hexdigest()[:8]
                while oid in self.items:
                    oid = "O" + hashlib.sha1(f"{oid}|{goal}".encode("utf-8", "ignore")).hexdigest()[:8]
                self._event("OBLIGATION_SCOPE_SPLIT", {
                    "from_obligation_id": base_oid,
                    "to_obligation_id": oid,
                    "reason": "terminal_goal_type_immutable",
                    "from_goal_type": self.items[base_oid].goal_type,
                    "to_goal_type": goal,
                })
            status = ObligationStatus.OPEN if required else ObligationStatus.OPTIONAL
            obj = EvidenceObligation(
                obligation_id=oid, target=target, reason=reason, location=loc,
                canonical_target=core, canonical_files=files, goal_type=goal,
                goal_type_source=goal_source, line_hint=line_hint,
                canonical_symbols=symbols, symbol_ranges=symbol_ranges,
                critical=required, active_required=required, status=status,
                refined_from=refined_from, information_need_root_id=raw.get("information_need_root_id"), scope_valid=scope_valid, scope_error=scope_error,
            )
            self.items[oid] = obj
            return obj, {
                "obligation_id": oid, "from": None, "to": status.value,
                "reason": "required_created" if required else "optional_created",
            }
        if target != obj.target and target not in obj.aliases:
            obj.aliases.append(target)
        if reason:
            obj.reason = reason
        if line_hint and (obj.line_hint is None or (line_hint[1] - line_hint[0]) < (obj.line_hint[1] - obj.line_hint[0])):
            obj.line_hint = line_hint; obj.location = loc
        elif not obj.location and loc:
            obj.location = loc
        if files:
            obj.canonical_files = files
        obj.goal_type = goal
        obj.goal_type_source = goal_source
        if symbols:
            obj.canonical_symbols = symbols
            obj.symbol_ranges = symbol_ranges
        obj.scope_valid=scope_valid; obj.scope_error=scope_error
        if raw.get("information_need_root_id") and not obj.information_need_root_id: obj.information_need_root_id=raw.get("information_need_root_id")
        if required:
            # A repeated requirement is not an invalidation signal.  Terminal
            # obligations remain terminal until an explicit invalidation/reopen
            # operation exists; ordinary Reflection synchronization must not
            # create the illegal SATISFIED -> ACTIVE transition.
            if obj.status in _TERMINAL_STATUSES:
                obj.active_required = False
                obj.critical = False
                return obj, transition
            obj.active_required = True; obj.critical = True
            if obj.status is ObligationStatus.OPTIONAL:
                old = obj.status; obj.status = ObligationStatus.OPEN
                transition = {"obligation_id": oid, "from": old.value, "to": obj.status.value, "reason": "required_reactivated"}
        else:
            obj.active_required = False; obj.critical = False
            if obj.status not in {ObligationStatus.SATISFIED, ObligationStatus.SUPERSEDED}:
                old = obj.status; obj.status = ObligationStatus.OPTIONAL
                if old is not obj.status:
                    transition = {"obligation_id": oid, "from": old.value, "to": obj.status.value, "reason": "required_downgraded_to_optional"}
        return obj, transition

    def sync(self, missing: list[dict], optional: list[dict] | None = None):
        transitions = []
        for obj in self.items.values():
            obj.active_required = False
        for raw in missing or []:
            _, tr = self._upsert(raw, required=True)
            if tr:
                transitions.append(tr)
        for raw in optional or []:
            _, tr = self._upsert(raw, required=False)
            if tr:
                transitions.append(tr)
        return transitions

    def mark_superseded(self, obligation_id: str, replacement_id: str | None = None) -> bool:
        obj = self.items.get(obligation_id)
        if obj is None or obj.status is ObligationStatus.SATISFIED:
            return False
        obj.status = ObligationStatus.SUPERSEDED; obj.active_required = False; obj.critical = False
        obj.superseded_by = replacement_id
        if replacement_id and f"superseded_by:{replacement_id}" not in obj.aliases:
            obj.aliases.append(f"superseded_by:{replacement_id}")
        return True

    def create_refinement(self, obligation_id: str, raw: dict) -> tuple[EvidenceObligation | None, list[dict]]:
        parent=self.items.get(obligation_id)
        if parent is None:
            return None, []
        old_status=parent.status.value
        if hasattr(raw,"model_dump"): raw=raw.model_dump()
        raw=dict(raw or {})
        raw.setdefault("information_need_root_id",parent.information_need_root_id)
        child, tr=self._upsert(raw, required=True, refined_from=obligation_id)
        if child is None:
            return None, []
        transitions=[]
        if tr: transitions.append(tr)
        if child.obligation_id != obligation_id and self.mark_superseded(obligation_id, child.obligation_id):
            transitions.append({"obligation_id":obligation_id,"from":old_status,"to":"superseded","reason":"explicit_refinement","superseded_by":child.obligation_id})
        return child, transitions

    def mark_satisfied(self, obligation_id: str, evidence_ids: list[str], *, reflection_id: str | None = None, decision_source: str = "deterministic") -> bool:
        obj = self.items.get(obligation_id)
        ids = [str(x) for x in (evidence_ids or []) if str(x).strip()]
        if obj is None or not ids:
            return False
        if obj.goal_type in _SEMANTIC_REVIEW_GOALS:
            # Semantic closure requires the evidence to have been physically presented
            # in the same successful Reflection that makes the decision.
            if not reflection_id or obj.last_presented_reflection_id != reflection_id:
                return False
        for eid in ids:
            if eid not in obj.evidence_ids:
                obj.evidence_ids.append(eid)
        obj.evidence_ready = True
        obj.status = ObligationStatus.SATISFIED; obj.active_required = False; obj.critical = False
        if reflection_id:
            obj.last_reviewed_reflection_id=reflection_id
            obj.last_reviewed_evidence_fingerprint=self.evidence_fingerprint(obj)
            obj.last_review_decision="resolved"
            obj.review_decision_source=decision_source
        return True

    def note_attempt_for_need(self, need_text: str):
        norm = normalize_target(need_text or "")
        for obj in self.items.values():
            if not obj.active_required or obj.status not in {ObligationStatus.OPEN, ObligationStatus.ATTEMPTED}:
                continue
            terms = [x for x in obj.canonical_target.split() if len(x) > 3]
            if obj.canonical_target and (obj.canonical_target in norm or (terms and sum(x in norm for x in terms) >= max(1, min(2, len(terms))))):
                obj.attempts += 1; obj.status = ObligationStatus.ATTEMPTED
                if obj.attempts >= self.max_attempts:
                    obj.status = ObligationStatus.UNRESOLVED

    def action_scope(self, tool: str, arguments: dict | None = None, intent: str = "") -> dict:
        """Check whether an investigation action honors the current required scope.

        This is deliberately an execution advisory/guard, not a satisfaction rule:
        supporting actions may inspect callers, tests, symbols, or dependencies, but
        only compatible source evidence can make an obligation READY/SATISFIED.
        """
        args = dict(arguments or {})
        active = self.open_critical()
        if not active:
            return {"allowed": True, "relation": "no_active_required"}
        text = " ".join(str(x or "").lower() for x in (
            intent, args.get("query"), args.get("symbol"), args.get("target"),
        ))
        supporting = any(term in text for term in _SUPPORTING_ACTION_TERMS)
        path = normalize_location(str(args.get("path") or args.get("file") or ""), self.repo_root)
        try:
            start = int(args.get("start_line", 1))
            end = int(args.get("end_line", start))
        except (TypeError, ValueError):
            start = end = None

        scope_probe = None
        first_drift = None
        for obj in active:
            if not obj.canonical_files and not obj.line_hint and not obj.canonical_symbols:
                continue
            same_file = not obj.canonical_files or path in obj.canonical_files
            symbol_match = bool(
                same_file and obj.canonical_symbols and any(
                    candidate in set(obj.canonical_symbols)
                    for candidate in (str(args.get("symbol") or ""), str(args.get("query") or ""))
                )
            )
            range_match = bool(
                same_file and obj.line_hint and start is not None and end is not None
                and start <= obj.line_hint[0] and end >= obj.line_hint[1]
            )
            exact_candidate = range_match or symbol_match
            if exact_candidate:
                return {"allowed": True, "relation": "satisfying_candidate", "obligation_id": obj.obligation_id}
            if same_file and tool == "read_file" and not obj.line_hint:
                # A file-level read can be useful supporting context for a
                # symbol-scoped obligation; eligibility still requires the
                # symbol/range evidence before semantic closure.
                scope_probe = scope_probe or {"allowed": True, "relation": "scope_probe", "obligation_id": obj.obligation_id}
            # Explicitly motivated supporting exploration remains possible. It is
            # never treated as satisfying this obligation by note_evidence().
            if supporting:
                continue
            if tool in {"read_file", "inspect_symbol_context"} and (path or obj.canonical_files):
                first_drift = first_drift or {
                    "allowed": False,
                    "relation": "scope_drift",
                    "obligation_id": obj.obligation_id,
                    "reason": "action does not cover the active required file/symbol/range; provide an explicit supporting-action reason",
                    "required": {
                        "file": obj.canonical_files[0] if len(obj.canonical_files) == 1 else None,
                        "symbol": obj.canonical_symbols[0] if len(obj.canonical_symbols) == 1 else None,
                        "line_start": obj.line_hint[0] if obj.line_hint else None,
                        "line_end": obj.line_hint[1] if obj.line_hint else None,
                    },
                }
        if scope_probe is not None:
            return scope_probe
        if first_drift is not None:
            return first_drift
        return {"allowed": True, "relation": "supporting_action" if supporting else "exploratory_action"}

    def _same_file(self, obj: EvidenceObligation, evidence) -> tuple[bool, str]:
        ev_file = normalize_location(getattr(evidence, "file", None), self.repo_root) if getattr(evidence, "file", None) else ""
        if obj.canonical_files and (not ev_file or ev_file not in obj.canonical_files):
            return False, ev_file
        return True, ev_file

    @staticmethod
    def _coverage(evidence) -> tuple[int, int] | None:
        start = getattr(evidence, "source_start_line", None)
        end = getattr(evidence, "source_end_line", None)
        if start is None or end is None:
            return None
        return int(start), int(end)

    def _eligible(self, obj: EvidenceObligation, evidence, raw_content: str = "") -> bool:
        if not obj.scope_valid:
            return False
        source = str(getattr(evidence, "source", None) or "")
        capabilities = _SOURCE_CAPABILITIES.get(source, set())
        if obj.goal_type not in capabilities:
            return False
        eid = str(getattr(evidence, "evidence_id", "") or "").strip()
        if not eid:
            return False
        same_file, _ = self._same_file(obj, evidence)
        if not same_file:
            return False
        coverage = self._coverage(evidence)
        if obj.line_hint:
            if coverage is None:
                return False
            a, b = obj.line_hint
            return coverage[0] <= a and coverage[1] >= b
        if obj.goal_type in {"behavior", "causality", "caller", "contradiction"}:
            if obj.symbol_ranges:
                if coverage is None:
                    return False
                relevant = [r for r in obj.symbol_ranges if not obj.canonical_files or r[0] in obj.canonical_files]
                if not relevant:
                    return False
                return all(coverage[0] <= start and coverage[1] >= end for _, _, start, end in relevant)
            return False
        if source == "git_show" and obj.goal_type == "history":
            content = (raw_content or "").lower(); probes, need = _meaningful_probes(obj.target)
            return bool(content and probes and sum(p.lower() in content for p in probes) >= max(1, need))
        content = (raw_content or "").lower(); probes, need = _meaningful_probes(obj.target)
        if not probes or not content:
            return False
        return sum(p.lower() in content for p in probes) >= need

    def _relevant_but_insufficient(self, obj: EvidenceObligation, evidence) -> bool:
        if obj.line_hint:
            return False
        if obj.goal_type not in _SEMANTIC_REVIEW_GOALS:
            return False
        if str(getattr(evidence, "source", None) or "") != "read_file":
            return False
        same_file, _ = self._same_file(obj, evidence)
        return same_file and bool(obj.canonical_files)

    def note_evidence(self, evidence, raw_content: str = "") -> list[str]:
        """Attach eligible source evidence.

        Return only deterministically SATISFIED obligation IDs. Semantic obligations become
        READY/ATTEMPTED and require a later Reflection review.
        """
        if evidence is None:
            return []
        closed = []
        for obj in self.items.values():
            if obj.status in {ObligationStatus.SATISFIED, ObligationStatus.SUPERSEDED, ObligationStatus.OPTIONAL}:
                continue
            if self._eligible(obj, evidence, raw_content):
                eid=str(evidence.evidence_id)
                if eid not in obj.evidence_ids:
                    obj.evidence_ids.append(eid)
                obj.evidence_ready=True
                if obj.goal_type in _SEMANTIC_REVIEW_GOALS:
                    if obj.status in {ObligationStatus.OPEN, ObligationStatus.UNRESOLVED}:
                        obj.status=ObligationStatus.ATTEMPTED
                elif self.mark_satisfied(obj.obligation_id,[eid]):
                    closed.append(obj.obligation_id)
                continue
            if obj.status is ObligationStatus.OPEN and self._relevant_but_insufficient(obj, evidence):
                obj.status = ObligationStatus.ATTEMPTED
        return closed

    def evidence_fingerprint(self, obj: EvidenceObligation) -> str:
        raw="|".join(sorted(obj.evidence_ids))+f"|{obj.line_hint}|{obj.symbol_ranges}"
        return hashlib.sha1(raw.encode("utf-8","ignore")).hexdigest()[:16]

    def presentation_candidates(self) -> list[EvidenceObligation]:
        out=[]
        for obj in self.items.values():
            if not obj.active_required or not obj.critical:
                continue
            if obj.status not in {ObligationStatus.OPEN, ObligationStatus.ATTEMPTED, ObligationStatus.UNRESOLVED}:
                continue
            if obj.goal_type not in _SEMANTIC_REVIEW_GOALS or not obj.evidence_ready or not obj.evidence_ids:
                continue
            fp=self.evidence_fingerprint(obj)
            # A successful review already said the exact same evidence is insufficient;
            # repeating it automatically would create a presentation storm.
            if obj.last_review_decision == "still_open" and obj.last_reviewed_evidence_fingerprint == fp:
                continue
            out.append(obj)
        def rank(x: EvidenceObligation):
            precision=0 if x.line_hint else (1 if x.symbol_ranges else 2)
            return (precision, -len(x.canonical_symbols), x.obligation_id)
        return sorted(out,key=rank)

    def presentation_scope(self, obj: EvidenceObligation, evidence) -> tuple[str, int, int] | None:
        ev_file=normalize_location(getattr(evidence,"file",None),self.repo_root)
        coverage=self._coverage(evidence)
        if not ev_file or coverage is None:
            return None
        if obj.line_hint:
            a,b=obj.line_hint
            return (ev_file,a,b) if coverage[0] <= a and coverage[1] >= b else None
        ranges=[r for r in obj.symbol_ranges if r[0]==ev_file]
        if ranges:
            a=min(r[2] for r in ranges); b=max(r[3] for r in ranges)
            return (ev_file,a,b) if coverage[0] <= a and coverage[1] >= b else None
        return ev_file,coverage[0],coverage[1]

    def mark_presented(self, obligation_id: str, *, reflection_id: str, projection_id: str, evidence_fingerprint: str) -> bool:
        obj=self.items.get(obligation_id)
        if obj is None:
            return False
        obj.last_presented_reflection_id=reflection_id
        obj.last_presented_projection_id=projection_id
        obj.last_presented_evidence_fingerprint=evidence_fingerprint
        return True

    def apply_explicit_review(self, review: dict, *, reflection_id: str) -> tuple[bool, list[dict]]:
        """Apply one validated review to the candidate semantic state."""
        oid=str(review.get("obligation_id") or "")
        obj=self.items.get(oid)
        if obj is None:
            return False, []
        decision=str(review.get("decision") or "")
        transitions=[]
        if decision == "resolved":
            if obj.goal_type in _SEMANTIC_REVIEW_GOALS and obj.last_presented_reflection_id != reflection_id:
                return False, []
            if not self.mark_satisfied(oid,obj.evidence_ids,reflection_id=reflection_id,decision_source="explicit"):
                return False, []
            transitions.append({"obligation_id":oid,"to":"satisfied","reason":"explicit_review_resolved"})
            return True, transitions
        if decision == "still_open":
            if obj.status in _TERMINAL_STATUSES:
                # A normal review is not an invalidation/reopen operation. Keep
                # terminal obligations closed until such an operation exists.
                return False, []
            obj.active_required=True; obj.critical=True
            if obj.status is ObligationStatus.OPTIONAL:
                obj.status=ObligationStatus.OPEN
            obj.last_reviewed_reflection_id=reflection_id
            obj.last_reviewed_evidence_fingerprint=self.evidence_fingerprint(obj) if obj.evidence_ids else None
            obj.last_review_decision="still_open"; obj.review_decision_source="explicit"
            if obj.status is ObligationStatus.UNRESOLVED:
                obj.status=ObligationStatus.ATTEMPTED
            return True, transitions
        if decision == "refine":
            raw=review.get("refined_requirement")
            if hasattr(raw,"model_dump"): raw=raw.model_dump()
            if not raw:
                return False, []
            # Refinement is semantic: the parent evidence must have been presented when relevant.
            if obj.goal_type in _SEMANTIC_REVIEW_GOALS and obj.last_presented_reflection_id != reflection_id:
                return False, []
            child,trs=self.create_refinement(oid,raw)
            if child is None:
                return False, []
            obj.last_reviewed_reflection_id=reflection_id
            obj.last_reviewed_evidence_fingerprint=self.evidence_fingerprint(obj) if obj.evidence_ids else None
            obj.last_review_decision="refine"; obj.review_decision_source="explicit"
            return True, trs
        return False, []

    def apply_implicit_resolution(self, obligation_id: str, *, reflection_id: str) -> bool:
        obj=self.items.get(obligation_id)
        if obj is None or not obj.evidence_ids:
            return False
        if obj.goal_type in _SEMANTIC_REVIEW_GOALS and obj.last_presented_reflection_id != reflection_id:
            return False
        return self.mark_satisfied(obligation_id,obj.evidence_ids,reflection_id=reflection_id,decision_source="implicit")

    def active_required_items(self) -> list[dict]:
        rows=[]
        for obj in self.items.values():
            if obj.active_required and obj.status in {ObligationStatus.OPEN,ObligationStatus.ATTEMPTED,ObligationStatus.UNRESOLVED}:
                rows.append({"target":obj.target,"location":obj.location,"file":(obj.canonical_files[0] if len(obj.canonical_files)==1 else None),"symbol":(obj.canonical_symbols[0] if len(obj.canonical_symbols)==1 else None),"line_start":(obj.line_hint[0] if obj.line_hint else None),"line_end":(obj.line_hint[1] if obj.line_hint else None),"goal_type":obj.goal_type if obj.goal_type != "evidence" else None,"reason":obj.reason,"information_need_root_id":obj.information_need_root_id})
        return rows

    def optional_items(self) -> list[dict]:
        return [{"target":x.target,"location":x.location,"file":(x.canonical_files[0] if len(x.canonical_files)==1 else None),"symbol":(x.canonical_symbols[0] if len(x.canonical_symbols)==1 else None),"line_start":(x.line_hint[0] if x.line_hint else None),"line_end":(x.line_hint[1] if x.line_hint else None),"goal_type":x.goal_type if x.goal_type != "evidence" else None,"reason":x.reason,"information_need_root_id":x.information_need_root_id}
                for x in self.items.values() if x.status is ObligationStatus.OPTIONAL]

    def remaining_items(self, missing: list[dict]) -> list[dict]:
        remaining = []
        for raw in missing or []:
            item = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)
            oid, core, files, _, goal, _, symbols, _, _, _ = self._identity(
                str(item.get("target", "")), str(item.get("reason", "")), item.get("location"), item.get("goal_type"), item
            )
            obj = self._lookup_equivalent(oid, core, files, goal=goal, symbols=symbols)
            if obj is not None and obj.status is ObligationStatus.SATISFIED and obj.evidence_ids:
                continue
            remaining.append(item)
        return remaining

    def open_critical(self):
        return [x for x in self.items.values() if x.critical and x.active_required and x.status in {ObligationStatus.OPEN, ObligationStatus.ATTEMPTED, ObligationStatus.UNRESOLVED}]

    def summary(self):
        return [asdict(x) for x in self.items.values()]
