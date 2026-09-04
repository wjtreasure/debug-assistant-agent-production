from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import hashlib,json,time,random,subprocess
from debug_assistant.models import TaskSpec
from debug_assistant.evaluation.trace_metrics import summarize_trace
from debug_assistant import __version__
from debug_assistant.agent.planner import SYSTEM as PLANNER_SYSTEM
from debug_assistant.security.redaction import redact_sensitive

RUNTIME_VERSION=__version__

def _skill_hash():
    root=Path(__file__).resolve().parents[1]/'skills'
    parts=[]
    for p in sorted(root.glob('*/SKILL.md')):
        parts.append(p.parent.name+'\n'+p.read_text(encoding='utf-8'))
    return hashlib.sha256('\n---\n'.join(parts).encode()).hexdigest()

def _dataset_hash(root:Path,candidates:list[Path]):
    h=hashlib.sha256()
    for d in candidates:
        for name in ('task.json','issue.md','ground_truth.json'):
            p=d/name
            if p.exists():
                h.update(str(d.name+'/'+name).encode()); h.update(p.read_bytes())
    return h.hexdigest()

def _git_commit():
    try:return subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,timeout=3,check=False).stdout.strip()
    except Exception:return ''

def _safe_harness_dict(cfg):
    return redact_sensitive(asdict(cfg.harness))

def _fingerprint_payload(harness,task_ids,*,sample=0,seed=42,limit=0,dataset_hash=''):
    cfg=harness.config
    return {
        'runtime_version':RUNTIME_VERSION,'git_commit':_git_commit(),'task_ids':task_ids,'dataset_hash':dataset_hash,'planner_prompt_hash':hashlib.sha256(PLANNER_SYSTEM.encode()).hexdigest(),'skill_hash':_skill_hash(),
        'selection':{'sample':sample,'seed':seed if sample else None,'limit':limit},
        'model':{'provider':cfg.model.provider,'planner_model':cfg.model.planner_model,'critic_model':cfg.model.critic_model or cfg.model.planner_model,'temperature':cfg.model.temperature,'base_url':cfg.model.base_url},
        'harness':_safe_harness_dict(cfg),
    }

def _safe_experiment_config(harness, task_ids, *, sample=0, seed=42, limit=0, dataset_hash=''):
    payload=_fingerprint_payload(harness,task_ids,sample=sample,seed=seed,limit=limit,dataset_hash=dataset_hash)
    canonical=json.dumps(payload,sort_keys=True,ensure_ascii=False,separators=(',',':'))
    return {**payload,'created_at':time.time(),'experiment_fingerprint':hashlib.sha256(canonical.encode()).hexdigest()}

def run_prepared(root,output,harness,limit=0,resume=True,*,sample=0,seed=42):
    root=Path(root); out=Path(output); out.mkdir(parents=True,exist_ok=True); pred=out/'predictions.jsonl'; metrics_path=out/'trace_metrics.jsonl'; manifest_path=out/'experiment_manifest.json'
    candidates=[d for d in sorted(root.iterdir()) if (d/'task.json').exists()]
    if sample: candidates=random.Random(seed).sample(candidates,min(sample,len(candidates)))
    elif limit: candidates=candidates[:limit]
    task_ids=[json.loads((d/'task.json').read_text())['task_id'] for d in candidates]
    manifest=_safe_experiment_config(harness,task_ids,sample=sample,seed=seed,limit=limit,dataset_hash=_dataset_hash(root,candidates))
    if resume and manifest_path.exists():
        old=json.loads(manifest_path.read_text(encoding='utf-8'))
        if old.get('experiment_fingerprint') != manifest['experiment_fingerprint']:
            raise RuntimeError('refusing resume: experiment fingerprint differs from existing output')
    if not resume:
        for path in (pred,metrics_path):
            if path.exists(): path.unlink()
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
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
        row=redact_sensitive({'task_id':iid,'status':status,'report_source':r.get('report_source') or report.get('report_source'),'report':report,'trace':r.get('trace',{}),'state':r.get('state',{}),
             'failure':r.get('failure') or r.get('state',{}).get('failure'),'latency_s':time.time()-started,'experiment_fingerprint':manifest['experiment_fingerprint']})
        with pred.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        trace_path=(r.get('trace') or {}).get('trace_path')
        if trace_path and Path(trace_path).exists():
            tm=summarize_trace(trace_path); tm['task_id']=iid
            with metrics_path.open('a',encoding='utf-8') as f:f.write(json.dumps(tm,ensure_ascii=False)+'\n')
        total+=1; success+=int(status=='success'); partial+=int(status=='partial_success')
    return {'processed':total,'success':success,'partial_success':partial,'predictions':str(pred),'trace_metrics':str(metrics_path),'manifest':str(manifest_path),'experiment_fingerprint':manifest['experiment_fingerprint'],'task_ids':task_ids}
