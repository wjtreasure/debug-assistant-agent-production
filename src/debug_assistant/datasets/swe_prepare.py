from __future__ import annotations
from pathlib import Path
import json
from .patch_parser import parse_unified_patch
from .python_locator import locate_symbols
from .workspace import RepositoryCache


def _rows_from_parquet(path):
    import pandas as pd
    return pd.read_parquet(path).to_dict('records')


def _prepared_ok(task_dir: Path) -> bool:
    """Return True only for a fully-written prepared SWE task."""
    required = (task_dir / 'issue.md', task_dir / 'task.json', task_dir / 'ground_truth.json')
    if not all(p.is_file() for p in required):
        return False
    try:
        task = json.loads((task_dir / 'task.json').read_text(encoding='utf-8'))
        gold = json.loads((task_dir / 'ground_truth.json').read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(task.get('task_id')) and isinstance(gold.get('files'), list)


def prepare_parquet(parquet, output, limit=0, clone=True):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    cache = RepositoryCache()
    count = 0

    for row in _rows_from_parquet(parquet):
        if limit and count >= limit:
            break

        iid = row['instance_id']
        d = out / iid

        # Idempotent resume: a task is skipped only when all expected output
        # files exist and are parseable. Empty/partial directories from a prior
        # interrupted run are safely rebuilt.
        if _prepared_ok(d):
            print(f"[prepare-swe] skip completed task: {iid}")
            count += 1
            continue

        patch = parse_unified_patch(row.get('patch', ''))
        symbols = []
        workspace = ''

        # Prepare the repository before creating task output files so a failed
        # network clone cannot leave a task that looks complete.
        if clone:
            ws = cache.prepare(row['repo'], row['base_commit'], iid)
            workspace = str(ws)
            for f in patch['files']:
                p = ws / f['path']
                if p.exists() and p.suffix == '.py':
                    for s in locate_symbols(p, f['modified_ranges']):
                        s['file'] = f['path']
                        symbols.append(s)

        gold = {
            "files": [f['path'] for f in patch['files']],
            "symbols": symbols,
            "patch_ranges": patch['files'],
        }
        task = {
            "task_id": iid,
            "repo": row['repo'],
            "base_commit": row['base_commit'],
            "issue": row['problem_statement'],
            "workspace": workspace,
        }

        d.mkdir(parents=True, exist_ok=True)
        (d / 'issue.md').write_text(row['problem_statement'], encoding='utf-8')
        (d / 'task.json').write_text(json.dumps(task, indent=2), encoding='utf-8')
        (d / 'ground_truth.json').write_text(json.dumps(gold, indent=2), encoding='utf-8')
        count += 1

    return count