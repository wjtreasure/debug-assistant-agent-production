from __future__ import annotations
from pathlib import Path
import json,time
from debug_assistant.models import TaskSpec


def run_prepared(root,output,harness,limit=0,resume=True):
    root=Path(root); out=Path(output); out.mkdir(parents=True,exist_ok=True); pred=out/'predictions.jsonl'
    done=set()
    if resume and pred.exists():
        for line in pred.read_text(encoding='utf-8').splitlines():
            if line.strip(): done.add(json.loads(line)['task_id'])
    total=0; success=0
    for d in sorted(root.iterdir()):
        if limit and total>=limit: break
        if not (d/'task.json').exists(): continue
        meta=json.loads((d/'task.json').read_text()); iid=meta['task_id']
        if iid in done: continue
        workspace=meta.get('workspace')
        if not workspace:
            raise RuntimeError(f'{iid}: no workspace. Prepare SWE data with cloning enabled.')
        task=TaskSpec(iid,(d/'issue.md').read_text(),workspace,meta.get('repo',''),meta.get('base_commit',''),meta)
        case_out=out/'cases'/iid; started=time.time()
        try:
            r=harness.run(task,str(case_out)); report=r.get('report') or {}; status=r['state']['status']
        except Exception as exc:
            r={'state':{'status':'failed'},'report':None,'trace':{},'error':str(exc)}; report={}; status='failed'
        row={'task_id':iid,'status':status,'report':report,'trace':r.get('trace',{}),'latency_s':time.time()-started}
        with pred.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        total+=1; success+=int(status=='success')
    return {'processed':total,'success':success,'predictions':str(pred)}
