from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import time


class RepositoryCache:
    """Local mirror/workspace cache for SWE-bench repositories.

    The cache is intentionally resilient to interrupted network clones:
    - validates an existing bare mirror before reusing it;
    - removes corrupt/incomplete mirrors left by failed clones;
    - retries transient clone/fetch failures with exponential backoff;
    - validates the requested commit before creating/reusing a workspace;
    - recreates a corrupt workspace automatically.
    """

    def __init__(
        self,
        root: str | Path = "cache/swe_repos",
        *,
        max_retries: int = 3,
        retry_base_seconds: float = 2.0,
    ) -> None:
        self.root = Path(root)
        self.bare = self.root / "bare"
        self.work = self.root / "workspaces"
        self.max_retries = max(1, max_retries)
        self.retry_base_seconds = max(0.0, retry_base_seconds)
        self.bare.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _run(cmd: list[str], *, check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
        kwargs = {"text": True}
        if quiet:
            kwargs.update({"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL})
        return subprocess.run(cmd, check=check, **kwargs)

    @classmethod
    def _is_valid_bare_repo(cls, path: Path) -> bool:
        if not path.exists():
            return False
        result = cls._run(
            ["git", "--git-dir", str(path), "rev-parse", "--is-bare-repository"],
            check=False,
            quiet=True,
        )
        return result.returncode == 0

    @classmethod
    def _is_valid_worktree(cls, path: Path) -> bool:
        if not path.exists():
            return False
        result = cls._run(
            ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
            check=False,
            quiet=True,
        )
        return result.returncode == 0

    @classmethod
    def _has_commit(cls, bare: Path, commit: str) -> bool:
        result = cls._run(
            ["git", "--git-dir", str(bare), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=False,
            quiet=True,
        )
        return result.returncode == 0

    def _retry(self, operation: str, fn) -> None:
        last_error: subprocess.CalledProcessError | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                fn()
                return
            except subprocess.CalledProcessError as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                delay = self.retry_base_seconds * (2 ** (attempt - 1))
                print(
                    f"[repo-cache] {operation} failed (attempt {attempt}/{self.max_retries}); "
                    f"retrying in {delay:.1f}s..."
                )
                if delay:
                    time.sleep(delay)
        assert last_error is not None
        raise last_error

    def _ensure_bare_repo(self, repo: str, bare: Path) -> None:
        # A failed `git clone --mirror` can leave a directory behind. Existence
        # alone is therefore not proof that the cache entry is reusable.
        if bare.exists() and not self._is_valid_bare_repo(bare):
            print(f"[repo-cache] removing corrupt bare repository: {bare}")
            shutil.rmtree(bare, ignore_errors=True)

        if not bare.exists():
            url = f"https://github.com/{repo}.git"

            def clone() -> None:
                # Remove any partial directory left by the previous failed try.
                if bare.exists():
                    shutil.rmtree(bare, ignore_errors=True)
                self._run(["git", "clone", "--mirror", url, str(bare)], check=True)
                if not self._is_valid_bare_repo(bare):
                    raise subprocess.CalledProcessError(128, ["git", "clone", "--mirror", url, str(bare)])

            self._retry(f"clone {repo}", clone)
            return

        # Existing healthy mirror: refresh when possible. A temporary network
        # failure here should not invalidate already-cached objects.
        fetch = self._run(
            ["git", "--git-dir", str(bare), "fetch", "--prune"],
            check=False,
            quiet=True,
        )
        if fetch.returncode != 0:
            print(f"[repo-cache] warning: fetch failed for {repo}; using local mirror")

    def _ensure_commit(self, repo: str, bare: Path, commit: str) -> None:
        if self._has_commit(bare, commit):
            return

        def fetch_commit() -> None:
            self._run(
                ["git", "--git-dir", str(bare), "fetch", "origin", commit],
                check=True,
            )

        self._retry(f"fetch commit {commit} for {repo}", fetch_commit)
        if not self._has_commit(bare, commit):
            raise RuntimeError(f"commit {commit} is unavailable in cached repository {repo}")

    def _ensure_workspace(self, bare: Path, dst: Path) -> None:
        if dst.exists() and not self._is_valid_worktree(dst):
            print(f"[repo-cache] removing corrupt workspace: {dst}")
            shutil.rmtree(dst, ignore_errors=True)

        if not dst.exists():
            self._run(["git", "clone", "--no-checkout", str(bare), str(dst)], check=True)

    def prepare(self, repo: str, commit: str, instance_id: str) -> Path:
        name = repo.replace("/", "__")
        bare = self.bare / f"{name}.git"
        dst = self.work / instance_id

        self._ensure_bare_repo(repo, bare)
        self._ensure_commit(repo, bare, commit)
        self._ensure_workspace(bare, dst)

        self._run(["git", "-C", str(dst), "reset", "--hard"], check=True, quiet=True)
        self._run(["git", "-C", str(dst), "clean", "-fdx"], check=True, quiet=True)
        self._run(
            ["git", "-C", str(dst), "checkout", "--detach", commit],
            check=True,
            quiet=True,
        )
        return dst
