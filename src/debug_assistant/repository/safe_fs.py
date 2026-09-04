from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

IGNORED={'.git','.venv','venv','node_modules','dist','build','__pycache__','.tox','.mypy_cache','.pytest_cache'}
TEXT_SUFFIXES={'.py','.md','.rst','.txt','.toml','.yaml','.yml','.json','.ini','.cfg','.c','.h','.cc','.cpp','.hpp','.java','.js','.ts','.tsx','.go','.rs'}

@dataclass(slots=True, frozen=True)
class SafeFile:
    path: Path
    rel: str

class SafeRepositoryFS:
    """Single path-safety boundary for all repository reads.

    Symlinks resolving outside the repository are rejected. Repository tools should
    never call Path.rglob/read_text directly after construction.
    """
    def __init__(self, root: Path | str):
        self.root=Path(root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(self.root)

    def resolve(self, rel: str | Path='.') -> Path:
        raw=self.root/Path(rel)
        p=raw.resolve()
        if p != self.root and self.root not in p.parents:
            raise ValueError('path escapes repository workspace')
        return p

    def relative(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.root)).replace('\\','/')

    def read_text(self, rel: str | Path, *, max_bytes: int | None=None) -> str:
        p=self.resolve(rel)
        if not p.is_file():
            raise FileNotFoundError(p)
        if max_bytes is not None and p.stat().st_size > max_bytes:
            raise ValueError(f'file exceeds max_bytes={max_bytes}')
        return p.read_text(encoding='utf-8',errors='ignore')

    def iter_files(self, *, suffixes: set[str] | None=None, max_file_bytes: int | None=None) -> Iterator[SafeFile]:
        suffixes=suffixes or TEXT_SUFFIXES
        # os.walk does not follow directory symlinks by default. Individual file
        # symlinks are resolved and boundary-checked before yielding.
        import os
        for cur, dirs, files in os.walk(self.root, followlinks=False):
            dirs[:]=[d for d in dirs if d not in IGNORED]
            for name in files:
                raw=Path(cur)/name
                if any(x in IGNORED for x in raw.parts) or raw.suffix.lower() not in suffixes:
                    continue
                try:
                    resolved=raw.resolve()
                    if resolved != self.root and self.root not in resolved.parents:
                        continue
                    if not resolved.is_file():
                        continue
                    if max_file_bytes is not None and resolved.stat().st_size > max_file_bytes:
                        continue
                    yield SafeFile(resolved, str(raw.relative_to(self.root)).replace('\\','/'))
                except (OSError,ValueError):
                    continue
    def iter_directories(self) -> Iterator[str]:
        """Yield safe repository-relative directory paths from the same boundary.

        The repository root is represented as ``.``. Directory symlinks are not
        followed, matching iter_files() safety semantics.
        """
        import os
        yield "."
        for cur, dirs, _files in os.walk(self.root, followlinks=False):
            dirs[:]=[d for d in dirs if d not in IGNORED]
            for name in list(dirs):
                raw=Path(cur)/name
                try:
                    resolved=raw.resolve()
                    if resolved != self.root and self.root not in resolved.parents:
                        continue
                    if not resolved.is_dir():
                        continue
                    yield str(raw.relative_to(self.root)).replace('\\','/')
                except (OSError,ValueError):
                    continue

