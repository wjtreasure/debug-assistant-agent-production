#!/usr/bin/env python3
"""Extract adjacent-reflection required-evidence candidate pairs for manual labeling.

This script never reads SWE-bench gold. It operates only on runtime traces and writes a
CSV-like JSONL calibration candidate file. Label `same_fact` manually before using it.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

def needs_from_event(ev):
    payload=ev.get('payload') or ev.get('data') or {}
    hyp=payload if ev.get('type')=='HYPOTHESIS_UPDATED' else None
    if not hyp: return []
    out=[]
    for n in hyp.get('required_missing_evidence') or []:
        if not isinstance(n,dict): continue
        out.append({'target':n.get('target',''),'question':n.get('question') or n.get('target',''),'location':n.get('location'),'reason':n.get('reason','')})
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('traces',nargs='+'); ap.add_argument('-o','--output',required=True); args=ap.parse_args()
    rows=[]
    for name in args.traces:
        p=Path(name); prev=[]
        for line in p.read_text(encoding='utf-8').splitlines():
            try: ev=json.loads(line)
            except Exception: continue
            cur=needs_from_event(ev)
            if ev.get('type')!='HYPOTHESIS_UPDATED': continue
            for a in prev:
                for b in cur:
                    rows.append({'trace':p.name,'a':a,'b':b,'same_fact':None})
            prev=cur
    Path(args.output).write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in rows)+'\n',encoding='utf-8')
    print(json.dumps({'pairs':len(rows),'output':args.output},ensure_ascii=False))
if __name__=='__main__': main()
