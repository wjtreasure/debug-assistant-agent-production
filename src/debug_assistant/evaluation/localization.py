from __future__ import annotations
from pathlib import Path
import json

def _norm(p): return str(p).replace('\\','/').lstrip('./')


def _gold_ranges(gold):
    rows = list(gold.get('ranges') or gold.get('modified_ranges') or [])
    rows.extend(x for x in gold.get('symbols', []) if isinstance(x, dict) and
                x.get('line_start') is not None and x.get('line_end') is not None)
    out=[]
    for row in rows:
        if not isinstance(row, dict):
            continue
        path=row.get('file') or row.get('path')
        start=row.get('line_start', row.get('new_start', row.get('old_start')))
        end=row.get('line_end', row.get('new_end', row.get('old_end')))
        if path and start is not None and end is not None:
            out.append((_norm(path), int(start), int(end)))
    return list(dict.fromkeys(out))


def _pred_symbols(pred):
    rows=[]
    for point in pred.get('recommended_change_points') or []:
        if not isinstance(point, dict):
            continue
        symbol=str(point.get('symbol') or '').strip()
        if symbol:
            rows.append((_norm(point.get('file','')) if point.get('file') else None, symbol,
                         point.get('line_start'), point.get('line_end')))
    rows.extend((None, str(symbol), None, None) for symbol in pred.get('likely_symbols') or [] if str(symbol).strip())
    return rows


def _ranked_symbol_hit(gold_symbols, predicted):
    for index, (_, symbol, _, _) in enumerate(predicted, 1):
        if any(symbol == gold_symbol for _, gold_symbol in gold_symbols):
            return index
    return None


def _range_hit(gold_ranges, predicted):
    for gfile, gstart, gend in gold_ranges:
        for pfile, _, pstart, pend in predicted:
            if pfile != gfile or pstart is None or pend is None:
                continue
            if int(pstart) <= gend and int(pend) >= gstart:
                return 1
    return 0


def evaluate_one(gold,pred):
    gf=[_norm(x) for x in gold.get('files',[])]; pf=[_norm(x) for x in (pred.get('likely_files') or [])]
    gs={(_norm(x.get('file','')) if x.get('file') else None,str(x.get('symbol','')))
        for x in gold.get('symbols',[]) if isinstance(x,dict) and x.get('symbol')}
    predicted_symbols=_pred_symbols(pred)
    rank=next((i+1 for i,x in enumerate(pf) if x in gf),None)
    symbol_rank=_ranked_symbol_hit(gs, predicted_symbols)
    qualified_symbol_hit=any((f,s) in gs for f,s,_,_ in predicted_symbols if f and s)
    return {
        'file_hit1':int(bool(pf and pf[0] in gf)),
        'file_hit3':int(any(x in gf for x in pf[:3])),
        'file_hit5':int(any(x in gf for x in pf[:5])),
        'file_hit@1':int(bool(pf and pf[0] in gf)),
        'file_hit@3':int(any(x in gf for x in pf[:3])),
        'file_hit@5':int(any(x in gf for x in pf[:5])),
        'file_mrr':0.0 if rank is None else 1.0/rank,
        'mrr':0.0 if rank is None else 1.0/rank,
        'gold_file_rank':rank,
        'gold_symbol_rank':symbol_rank,
        'symbol_hit':int(symbol_rank is not None),
        'qualified_symbol_hit':int(qualified_symbol_hit),
        'range_hit':_range_hit(_gold_ranges(gold), predicted_symbols),
        'gold_files':gf,'pred_files':pf[:5],
    }

def _aggregate(rows):
    n=len(rows) or 1
    return {'n':len(rows),
            'file_hit@1':sum(r['file_hit1'] for r in rows)/n,
            'file_hit@3':sum(r['file_hit3'] for r in rows)/n,
            'file_hit@5':sum(r['file_hit5'] for r in rows)/n,
            'file_mrr':sum(r['file_mrr'] for r in rows)/n,
            'symbol_hit':sum(r['symbol_hit'] for r in rows)/n,
            'qualified_symbol_hit':sum(r.get('qualified_symbol_hit',0) for r in rows)/n,
            'range_hit':sum(r.get('range_hit',0) for r in rows)/n}

def _trace_gold_support(trace_path: str | None, gold_files: list[str]):
    if not trace_path or not Path(trace_path).exists(): return {}
    events=[json.loads(x) for x in Path(trace_path).read_text(encoding='utf-8').splitlines() if x.strip()]
    evidence_file={}
    cumulative_tokens=0
    first_step=None; tokens_at=None
    gold=set(gold_files)
    for e in events:
        if e['type']=='LLM_CALL_USAGE': cumulative_tokens += int(e['payload'].get('total_tokens',0) or 0)
        elif e['type']=='EVIDENCE_ADDED':
            payload=e['payload']; f=_norm(payload.get('file') or '') if payload.get('file') else None
            evidence_file[payload.get('evidence_id')]=f
        elif e['type']=='HYPOTHESIS_UPDATED' and first_step is None:
            payload=e['payload']; support=payload.get('supporting_evidence_ids') or []
            if any(evidence_file.get(i) in gold for i in support):
                first_step=payload.get('updated_step'); tokens_at=cumulative_tokens
    total=sum(int(e['payload'].get('total_tokens',0) or 0) for e in events if e['type']=='LLM_CALL_USAGE')
    post=max(0,total-int(tokens_at)) if tokens_at is not None else None
    return {'first_gold_support_step':first_step,'tokens_at_first_gold_support':tokens_at,
            'post_gold_support_tokens':post,'post_gold_support_ratio':(post/total if post is not None and total else None)}

def evaluate_dataset(gold_root,predictions_path):
    preds={}; meta={}; p=Path(predictions_path)
    if p.suffix=='.jsonl':
        for line in p.read_text(encoding='utf-8').splitlines():
            if line.strip():
                r=json.loads(line); preds[r['task_id']]=r.get('report') or {}; meta[r['task_id']]={'status':r.get('status'),'report_source':r.get('report_source') or (r.get('report') or {}).get('report_source'),'state':r.get('state') or {},'trace':r.get('trace') or {}}
    else:
        data=json.loads(p.read_text(encoding='utf-8')); rr=data if isinstance(data,list) else data.get('predictions',[])
        for r in rr: preds[r['task_id']]=r.get('report') or {}; meta[r['task_id']]={'status':r.get('status'),'report_source':r.get('report_source'),'state':r.get('state') or {},'trace':r.get('trace') or {}}
    rows=[]; expected=0; missing=[]
    for d in Path(gold_root).iterdir():
        gt=d/'ground_truth.json'
        if not gt.exists(): continue
        expected+=1
        gold=json.loads(gt.read_text())
        if d.name not in preds:
            missing.append(d.name); m=evaluate_one(gold,{}) ; md={}
        else:
            m=evaluate_one(gold,preds[d.name]); md=meta.get(d.name,{})
        m['task_id']=d.name; m['status']=md.get('status'); m['report_source']=md.get('report_source'); m['forced_finalization']=bool((md.get('state') or {}).get('forced_finalization'))
        m.update(_trace_gold_support((md.get('trace') or {}).get('trace_path'),m['gold_files']))
        rows.append(m)
    llm=[r for r in rows if r.get('report_source')=='llm']; fallback=[r for r in rows if r.get('report_source')=='fallback']; forced=[r for r in rows if r.get('forced_finalization')]
    return {'aggregate':_aggregate(rows),'by_report_source':{'llm':_aggregate(llm),'fallback':_aggregate(fallback)},
            'forced_finalization':_aggregate(forced),
            'coverage':{'expected_tasks':expected,'predicted_tasks':expected-len(missing),'missing_predictions':len(missing),'coverage':(expected-len(missing))/(expected or 1),'missing_task_ids':missing},
            'runtime':{'fallback_count':len(fallback),'fallback_rate':len(fallback)/(len(rows) or 1),'partial_success_count':sum(r.get('status')=='partial_success' for r in rows),'forced_finalization_count':len(forced)},'cases':rows}
