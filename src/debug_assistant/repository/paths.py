from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
import posixpath
import re

from .safe_fs import SafeRepositoryFS


class ResolutionMode(str, Enum):
    """How much deterministic recovery a path lookup may perform.

    EXACT is appropriate for operations whose target must never be guessed.
    READ_TOLERANT additionally permits a unique suffix or basename recovery.
    """

    EXACT = "exact"
    READ_TOLERANT = "read_tolerant"


class RepositoryPathError(ValueError):
    error_type = "path_error"
    retryable = False  # tool-executor retry; path errors need replanning, not same-call retry
    planner_retryable = False

    def __init__(self, raw_path: str, message: str, *, candidates: list[str] | None = None):
        super().__init__(message)
        self.raw_path = raw_path
        self.candidates = list(candidates or [])

    def metadata(self) -> dict:
        return {
            "input_path": self.raw_path,
            "candidates": self.candidates,
            "retryable": self.retryable,
            "planner_retryable": self.planner_retryable,
        }


class PathNotFoundError(RepositoryPathError):
    error_type = "path_not_found"
    planner_retryable = True


class AmbiguousPathError(RepositoryPathError):
    error_type = "ambiguous_path"
    planner_retryable = True


class PathRejectedError(RepositoryPathError):
    error_type = "path_rejected"
    retryable = False


@dataclass(slots=True, frozen=True)
class ResolvedRepoPath:
    relative_path: str
    absolute_path: Path
    strategy: str
    normalized_input: str

    def metadata(self, raw_path: str) -> dict:
        return {
            "input": raw_path,
            "canonical": self.relative_path,
            "strategy": self.strategy,
            "normalized_input": self.normalized_input,
        }


def _strip_wrapping_quotes(value: str) -> str:
    s=value.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {'"', "'", '`'}:
        s=s[1:-1].strip()
    return s


def normalize_path_syntax(raw_path: str) -> str:
    """Normalize path spelling without weakening repository safety.

    Parent traversal is deliberately preserved for SafeRepositoryFS to reject when it
    escapes the repository. This function only normalizes separators / harmless syntax.
    """
    if raw_path is None:
        return ""
    s=_strip_wrapping_quotes(str(raw_path))
    if "\x00" in s:
        raise PathRejectedError(str(raw_path), "path contains NUL byte")
    s=s.replace("\\", "/")
    s=re.sub(r"/{2,}", "/", s)
    # Keep Unix absolute paths absolute. posixpath.normpath also preserves leading '..'.
    s=posixpath.normpath(s) if s else "."
    while s.startswith("./"):
        s=s[2:]
    return s or "."


def _looks_windows_absolute(path: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:/", path))


class RepositoryPathResolver:
    """Resolve model-emitted paths against the task's safe repository snapshot.

    Fallback resolution is deterministic and only succeeds when the match is unique.
    It never scans outside SafeRepositoryFS and never silently picks the first candidate.
    """

    def __init__(self, fs: SafeRepositoryFS | str | Path):
        self.fs=fs if isinstance(fs,SafeRepositoryFS) else SafeRepositoryFS(fs)
        self._file_paths: tuple[str,...] | None=None
        self._dir_paths: tuple[str,...] | None=None

    def _files(self) -> tuple[str,...]:
        if self._file_paths is None:
            self._file_paths=tuple(sorted({sf.rel for sf in self.fs.iter_files()}))
        return self._file_paths

    def _dirs(self) -> tuple[str,...]:
        if self._dir_paths is None:
            self._dir_paths=tuple(sorted(set(self.fs.iter_directories())))
        return self._dir_paths

    def _exact(self, raw_path: str, normalized: str, *, expect: str) -> ResolvedRepoPath | None:
        if _looks_windows_absolute(normalized):
            # A Windows absolute path cannot be proven to be inside a Linux/WSL repo.
            raise PathRejectedError(raw_path, f"absolute path is not addressable inside this repository: {raw_path}")
        try:
            p=self.fs.resolve(normalized)
        except ValueError as exc:
            raise PathRejectedError(raw_path, str(exc)) from exc
        if expect == "file" and not p.is_file():
            return None
        if expect == "directory" and not p.is_dir():
            return None
        if expect == "any" and not p.exists():
            return None
        is_abs=Path(normalized).is_absolute()
        if p == self.fs.root:
            rel="."
        elif is_abs:
            rel=self.fs.relative(p)
        else:
            # Preserve the repository's logical relative name for in-repo symlinks;
            # SafeRepositoryFS still resolves the target for boundary enforcement.
            rel=PurePosixPath(normalized).as_posix()
        original=_strip_wrapping_quotes(str(raw_path))
        if is_abs:
            strategy="absolute_inside_repo"
        elif original != normalized:
            strategy="normalized_relative"
        else:
            strategy="exact_relative"
        return ResolvedRepoPath(rel,p,strategy,normalized)

    @staticmethod
    def _suffix_matches(normalized: str, candidates: tuple[str,...]) -> list[str]:
        suffix=normalized.strip("/")
        if not suffix or suffix == ".":
            return []
        needle="/"+suffix
        return [p for p in candidates if p == suffix or p.endswith(needle)]

    def _fallback(self, raw_path: str, normalized: str, candidates: tuple[str,...], *, allow_basename: bool) -> ResolvedRepoPath:
        suffix=self._suffix_matches(normalized,candidates) if "/" in normalized.strip("/") else []
        if len(suffix)==1:
            rel=suffix[0]
            return ResolvedRepoPath(rel,self.fs.resolve(rel),"unique_suffix",normalized)
        if len(suffix)>1:
            raise AmbiguousPathError(raw_path,f"path '{raw_path}' matches multiple repository entries",candidates=suffix[:20])
        if allow_basename:
            basename=PurePosixPath(normalized).name
            matches=[p for p in candidates if PurePosixPath(p).name == basename]
            if len(matches)==1:
                rel=matches[0]
                return ResolvedRepoPath(rel,self.fs.resolve(rel),"unique_basename",normalized)
            if len(matches)>1:
                raise AmbiguousPathError(raw_path,f"path '{raw_path}' matches multiple repository entries",candidates=matches[:20])
        raise PathNotFoundError(raw_path,f"path not found in repository: {raw_path}")

    def resolve_path(self, raw_path: str, *, mode: ResolutionMode=ResolutionMode.EXACT) -> ResolvedRepoPath:
        """Resolve an existing file or directory. Fuzzy recovery is intentionally unsupported."""
        normalized=normalize_path_syntax(raw_path)
        exact=self._exact(raw_path,normalized,expect="any")
        if exact is not None:
            return exact
        raise PathNotFoundError(raw_path,f"path not found in repository: {raw_path}")

    def resolve_file(self, raw_path: str, *, mode: ResolutionMode=ResolutionMode.READ_TOLERANT) -> ResolvedRepoPath:
        normalized=normalize_path_syntax(raw_path)
        exact=self._exact(raw_path,normalized,expect="file")
        if exact is not None:
            return exact
        if mode is ResolutionMode.EXACT:
            raise PathNotFoundError(raw_path,f"file not found in repository: {raw_path}")
        return self._fallback(raw_path,normalized,self._files(),allow_basename=True)

    def resolve_directory(self, raw_path: str='.', *, mode: ResolutionMode=ResolutionMode.READ_TOLERANT) -> ResolvedRepoPath:
        normalized=normalize_path_syntax(raw_path or '.')
        exact=self._exact(raw_path,normalized,expect="directory")
        if exact is not None:
            return exact
        if mode is ResolutionMode.EXACT:
            raise PathNotFoundError(raw_path,f"directory not found in repository: {raw_path}")
        # Directory basename recovery is intentionally disabled: names such as src/tests
        # are commonly duplicated and too easy to mis-address silently.
        return self._fallback(raw_path,normalized,self._dirs(),allow_basename=False)


_GLOB_MAGIC=re.compile(r"[*?[]")


@dataclass(slots=True, frozen=True)
class NormalizedPathPattern:
    pattern: str
    strategy: str

    def metadata(self, raw_pattern: str) -> dict:
        return {"input":raw_pattern,"normalized":self.pattern,"strategy":self.strategy}


class RepositoryPathMatcher:
    """Deterministic glob matching over canonical repository-relative paths.

    - patterns without '/' match basenames recursively (e.g. *.py)
    - patterns with '/' are anchored to repository root
    - '**' spans zero or more path segments
    - exact path patterns do not perform suffix/basename fallback
    """

    def normalize_pattern(self, raw_pattern: str) -> NormalizedPathPattern:
        raw="*" if raw_pattern is None else str(raw_pattern)
        p=_strip_wrapping_quotes(raw).replace("\\", "/").strip() or "*"
        p=re.sub(r"/{2,}", "/", p)
        if p.startswith('/') or _looks_windows_absolute(p):
            raise PathRejectedError(raw,"glob patterns must be repository-relative")
        raw_parts=p.split('/')
        if '..' in raw_parts:
            raise PathRejectedError(raw,"glob pattern may not traverse parent directories")
        # Remove harmless '.' path segments without changing ** semantics.
        p='/'.join(part for part in raw_parts if part not in ('','.')) or '*'
        strategy="basename_glob" if '/' not in p else ("repo_relative_glob" if _GLOB_MAGIC.search(p) else "repo_relative_exact")
        return NormalizedPathPattern(p,strategy)

    @staticmethod
    def _match_parts(path_parts: tuple[str,...], pattern_parts: tuple[str,...]) -> bool:
        if not pattern_parts:
            return not path_parts
        head=pattern_parts[0]
        if head == '**':
            # zero segments OR consume one path segment and keep ** active
            return RepositoryPathMatcher._match_parts(path_parts,pattern_parts[1:]) or (
                bool(path_parts) and RepositoryPathMatcher._match_parts(path_parts[1:],pattern_parts)
            )
        if not path_parts:
            return False
        return fnmatchcase(path_parts[0],head) and RepositoryPathMatcher._match_parts(path_parts[1:],pattern_parts[1:])

    def matches(self, relative_path: str, raw_pattern: str | NormalizedPathPattern='*') -> bool:
        rel=str(relative_path).replace("\\", "/")
        while rel.startswith("./"):
            rel=rel[2:]
        norm=raw_pattern if isinstance(raw_pattern,NormalizedPathPattern) else self.normalize_pattern(raw_pattern)
        p=norm.pattern
        if '/' not in p:
            return fnmatchcase(PurePosixPath(rel).name,p)
        if not _GLOB_MAGIC.search(p):
            return rel == p
        return self._match_parts(PurePosixPath(rel).parts,PurePosixPath(p).parts)
