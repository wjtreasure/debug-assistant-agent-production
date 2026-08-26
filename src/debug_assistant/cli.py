from __future__ import annotations
import argparse, json
from pathlib import Path
from debug_assistant.config import AppConfig
from debug_assistant.models import TaskSpec
from debug_assistant.harness.runtime import AgentHarness

def load_env(path='.env'):
    p=Path(path)
    if not p.exists(): return
    import os
    for line in p.read_text(encoding='utf-8').splitlines():
        line=line.strip()
        if not line or line.startswith('#') or '=' not in line: continue
        k,v=line.split('=',1); os.environ.setdefault(k.strip(),v.strip())

def main(argv=None):
    load_env(); ap=argparse.ArgumentParser(prog='debug-assistant'); sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('diagnose'); p.add_argument('--issue',required=True); p.add_argument('--repo',required=True); p.add_argument('--output',required=True); p.add_argument('--task-id',default='local-issue')
    p=sub.add_parser('diagnose-task'); p.add_argument('--task',required=True); p.add_argument('--output',required=True)
    p=sub.add_parser('prepare-swe'); p.add_argument('--parquet',required=True); p.add_argument('--output',required=True); p.add_argument('--limit',type=int,default=0); p.add_argument('--no-clone',action='store_true')
    p=sub.add_parser('run-swe'); p.add_argument('--tasks',required=True); p.add_argument('--output',required=True); p.add_argument('--limit',type=int,default=0); p.add_argument('--no-resume',action='store_true')
    p=sub.add_parser('eval-localization'); p.add_argument('--gold',required=True); p.add_argument('--predictions',required=True); p.add_argument('--output',required=True)
    a=ap.parse_args(argv)
    if a.cmd=='prepare-swe':
        from debug_assistant.datasets.swe_prepare import prepare_parquet
        n=prepare_parquet(a.parquet,a.output,a.limit,clone=not a.no_clone); print(json.dumps({'prepared':n,'output':a.output})); return
    if a.cmd=='run-swe':
        from debug_assistant.evaluation.batch import run_prepared
        cfg=AppConfig.from_env(); r=run_prepared(a.tasks,a.output,AgentHarness(cfg),a.limit,resume=not a.no_resume); print(json.dumps(r,indent=2)); return
    if a.cmd=='eval-localization':
        from debug_assistant.evaluation.localization import evaluate_dataset
        r=evaluate_dataset(a.gold,a.predictions); Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(r,indent=2),encoding='utf-8'); print(json.dumps(r['aggregate'],indent=2)); return
    cfg=AppConfig.from_env(); harness=AgentHarness(cfg)
    if a.cmd=='diagnose':
        issue=Path(a.issue).read_text(encoding='utf-8'); task=TaskSpec(a.task_id,issue,str(Path(a.repo).resolve()))
    else:
        d=Path(a.task); meta=json.loads((d/'task.json').read_text()); issue=(d/'issue.md').read_text(); repo=meta.get('workspace')
        if not repo: raise SystemExit('task has no workspace; prepare without --no-clone or set task.json workspace')
        task=TaskSpec(meta['task_id'],issue,repo,meta.get('repo',''),meta.get('base_commit',''),meta)
    r=harness.run(task,a.output); print(json.dumps({"status":r['state']['status'],"output":a.output,"trace":r['trace']},indent=2))

if __name__=='__main__': main()
