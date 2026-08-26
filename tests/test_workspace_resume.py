from pathlib import Path
import subprocess

from debug_assistant.datasets.workspace import RepositoryCache


def test_invalid_bare_repo_is_detected(tmp_path: Path):
    cache = RepositoryCache(tmp_path, max_retries=1, retry_base_seconds=0)
    bare = cache.bare / 'broken.git'
    bare.mkdir()
    (bare / 'partial').write_text('incomplete')
    assert cache._is_valid_bare_repo(bare) is False


def test_valid_bare_repo_is_detected(tmp_path: Path):
    cache = RepositoryCache(tmp_path, max_retries=1, retry_base_seconds=0)
    bare = cache.bare / 'repo.git'
    subprocess.run(['git', 'init', '--bare', str(bare)], check=True, stdout=subprocess.DEVNULL)
    assert cache._is_valid_bare_repo(bare) is True
