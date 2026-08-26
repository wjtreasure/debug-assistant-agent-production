from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json,time,random
from debug_assistant.models import TaskSpec
from debug_assistant.evaluation.trace_metrics import summarize_trace

RUNTIME_VERSION='1.3'

def _safe_experiment_config(harness, task_ids, *, sample=0, seed=42, limit=0):
    cfg=harness.config
    return {'runtime_version':RUNTIME_VERSION,'created_at':time.time(),'task_ids':task_ids,'selection':{'sample':sample,'seed':seed if sample else None,'limit':limit},
            'model':{'provider':cfg.model.provider,'planner_model':cfg.model.planner_model,'critic_model':cfg.model.critic_model or cfg.model.planner_model,'temperature':cfg.model.temperature,'base_url':cfg.model.base_url},
            'harness':asdict(cfg.harness)}

def run_prepared(root,output,harness,limit=0,resume=True,*,sample=0,seed=42):
    root=Path(root); out=Path(output); out.mkdir(parents=True,exist_ok=True); pred=out/'predictions.jsonl'; metrics_path=out/'trace_metrics.jsonl'
    candidates=[d for d in sorted(root.iterdir()) if (d/'task.json').exists()]
    if sample: candidates=random.Random(seed).sample(candidates,min(sample,len(candidates)))
    elif limit: candidates=candidates[:limit]
    task_ids=[json.loads((d/'task.json').read_text())['task_id'] for d in candidates]
    (out/'experiment_manifest.json').write_text(json.dumps(_safe_experiment_config(harness,task_ids,sample=sample,seed=seed,limit=limit),ensure_ascii=False,indent=2),encoding='utf-8')
    done=set()
    if resume and pred.exists():
        for line in pred.read_text(encoding='utf-8').splitlines():
            if line.strip(): done.add(json.loads(line)['task_id'])
    total=0; success=0; partial=0
    for d in candidates:
        meta=json.loads((d/'task.json').read_text()); iid=meta['task_id']
        if iid in done: continue
        workspace=meta.get('workspace'); case_out=out/'cases'/iid; started=time.time()
        try:
            if not workspace: raise RuntimeError(f'{iid}: no workspace. Prepare SWE data with cloning enabled.')
            task=TaskSpec(iid,(d/'issue.md').read_text(),workspace,meta.get('repo',''),meta.get('base_commit',''),meta)
            r=harness.run(task,str(case_out)); report=r.get('report') or {}; status=r['state']['status']
        except Exception as exc:
            r={'state':{'status':'failed'},'report':None,'trace':{},'failure':{'stage':'batch','error_type':'unexpected_error','exception_type':type(exc).__name__,'message':str(exc)},'report_source':None}; report={}; status='failed'
        row={'task_id':iid,'status':status,'report_source':r.get('report_source') or report.get('report_source'),'report':report,'trace':r.get('trace',{}),'state':r.get('state',{}),
             'failure':r.get('failure') or r.get('state',{}).get('failure'),'latency_s':time.time()-started}
        with pred.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        trace_path=(r.get('trace') or {}).get('trace_path')
        if trace_path and Path(trace_path).exists():
            tm=summarize_trace(trace_path); tm['task_id']=iid
            with metrics_path.open('a',encoding='utf-8') as f:f.write(json.dumps(tm,ensure_ascii=False)+'\n')
        total+=1; success+=int(status=='success'); partial+=int(status=='partial_success')
    return {'processed':total,'success':success,'partial_success':partial,'predictions':str(pred),'trace_metrics':str(metrics_path),'manifest':str(out/'experiment_manifest.json'),'task_ids':task_ids}
