from __future__ import annotations
from pathlib import Path
import json, math

def _norm(p): return str(p).replace('\\','/').lstrip('./')
def evaluate_one(gold,pred):
    gf=[_norm(x) for x in gold.get('files',[])]
    pf=[_norm(x) for x in (pred.get('likely_files') or [])]
    gs={(x.get('file'),x.get('symbol')) for x in gold.get('symbols',[])}
    ps=set()
    for s in pred.get('likely_symbols') or []:
        ps.add((None,str(s)))
    for x in pred.get('recommended_change_points') or []:
        ps.add((_norm(x.get('file','')) if x.get('file') else None,str(x.get('symbol',''))))
    rank=next((i+1 for i,x in enumerate(pf) if x in gf),None)
    symbol_hit=any((f,s) in gs or any(gsym==s for _,gsym in gs) for f,s in ps if s)
    return {"file_hit1":int(bool(pf and pf[0] in gf)),"file_hit3":int(any(x in gf for x in pf[:3])),"file_mrr":0.0 if rank is None else 1.0/rank,"symbol_hit":int(symbol_hit),"gold_files":gf,"pred_files":pf[:5]}

def evaluate_dataset(gold_root,predictions_path):
    preds={}
    p=Path(predictions_path)
    if p.suffix=='.jsonl':
        for line in p.read_text(encoding='utf-8').splitlines():
            if line.strip():
                r=json.loads(line); preds[r['task_id']]=r.get('report',r)
    else:
        data=json.loads(p.read_text(encoding='utf-8')); rows=data if isinstance(data,list) else data.get('predictions',[])
        preds={r['task_id']:r.get('report',r) for r in rows}
    rows=[]
    for d in Path(gold_root).iterdir():
        gt=d/'ground_truth.json'
        if not gt.exists() or d.name not in preds: continue
        m=evaluate_one(json.loads(gt.read_text()),preds[d.name]); m['task_id']=d.name; rows.append(m)
    n=len(rows) or 1
    agg={"n":len(rows),"file_hit@1":sum(r['file_hit1'] for r in rows)/n,"file_hit@3":sum(r['file_hit3'] for r in rows)/n,"file_mrr":sum(r['file_mrr'] for r in rows)/n,"symbol_hit":sum(r['symbol_hit'] for r in rows)/n}
    return {"aggregate":agg,"cases":rows}
