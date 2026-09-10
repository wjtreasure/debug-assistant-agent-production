from __future__ import annotations
from pathlib import Path
import json
from .ground_truth import GROUND_TRUTH_SCHEMA_VERSION, normalize_repo_path, validate_ground_truth
from .patch_parser import apply_file_patch, parse_unified_patch
from .python_locator import locate_symbols
from .workspace import RepositoryCache


def _rows_from_parquet(path):
    import pandas as pd
    return pd.read_parquet(path).to_dict('records')


def _prepared_ok(task_dir: Path) -> bool:
    """Return True only for a fully-written prepared SWE task."""
    required = (
        task_dir / 'issue.md', task_dir / 'task.json',
        task_dir / 'ground_truth.json', task_dir / 'evaluation_only.json',
    )
    if not all(p.is_file() for p in required):
        return False
    try:
        task = json.loads((task_dir / 'task.json').read_text(encoding='utf-8'))
        gold = json.loads((task_dir / 'ground_truth.json').read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return False
    try:
        validate_ground_truth(gold, instance_id=task.get('task_id'))
    except ValueError:
        return False
    return bool(task.get('task_id'))


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
        fix_locations = []
        workspace = ''

        # Prepare the repository before creating task output files so a failed
        # network clone cannot leave a task that looks complete.
        if clone:
            ws = cache.prepare(row['repo'], row['base_commit'], iid)
            workspace = str(ws)
        for f in patch['files']:
            old_path = normalize_repo_path(f.get('old_path'))
            new_path = normalize_repo_path(f.get('new_path'))
            old_symbols = []
            new_symbols = []
            old_text = ''
            old_file = ws / old_path if clone and old_path else None
            if old_file is not None and old_file.is_file() and old_file.suffix == '.py':
                old_text = old_file.read_text(encoding='utf-8', errors='ignore')
                old_symbols = locate_symbols(
                    old_file, f['old_edit_ranges'], f['insertion_anchors'],
                    source_text=old_text,
                )
            if clone and new_path and (new_path.endswith('.py') or old_text):
                new_text = apply_file_patch(old_text, f)
                new_file = ws / new_path
                new_symbols = locate_symbols(
                    new_file, f['new_edit_ranges'], source_text=new_text,
                )

            fix_locations.append({
                'old_path': old_path,
                'new_path': new_path,
                'status': f['status'],
                'old_edit_ranges': f['old_edit_ranges'],
                'new_edit_ranges': f['new_edit_ranges'],
                'insertion_anchors': f['insertion_anchors'],
                'hunk_ranges': f['hunk_ranges'],
                'symbols': old_symbols,
                'new_symbols': new_symbols,
            })

        gold = {
            'schema_version': GROUND_TRUTH_SCHEMA_VERSION,
            'instance_id': iid,
            'fix_locations': fix_locations,
        }
        validate_ground_truth(gold, instance_id=iid)
        task = {
            "task_id": iid,
            "repo": row['repo'],
            "base_commit": row['base_commit'],
            "issue": row['problem_statement'],
            "workspace": workspace,
        }
        evaluation_only = {
            'schema_version': 1,
            'instance_id': iid,
            'patch': row.get('patch', ''),
            'test_patch': row.get('test_patch', ''),
            'fail_to_pass': row.get('FAIL_TO_PASS', row.get('fail_to_pass', [])) or [],
            'pass_to_pass': row.get('PASS_TO_PASS', row.get('pass_to_pass', [])) or [],
            'version': row.get('version'),
            'environment_setup_commit': row.get('environment_setup_commit'),
            'hints_text': row.get('hints_text'),
        }

        d.mkdir(parents=True, exist_ok=True)
        (d / 'issue.md').write_text(row['problem_statement'], encoding='utf-8')
        (d / 'task.json').write_text(json.dumps(task, indent=2), encoding='utf-8')
        (d / 'ground_truth.json').write_text(json.dumps(gold, indent=2), encoding='utf-8')
        (d / 'evaluation_only.json').write_text(json.dumps(evaluation_only, indent=2), encoding='utf-8')
        count += 1

    return count
