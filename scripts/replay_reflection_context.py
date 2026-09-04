#!/usr/bin/env python3
"""Replay captured V1.3.4 Reflection contexts against the configured critic model.

Capture is opt-in because snapshots contain repository excerpts:
  DEBUG_AGENT_TRACE_REFLECTION_CONTEXT=1 debug-assistant run ...

Then:
  python scripts/replay_reflection_context.py .debug_assistant/traces/<run>.jsonl --out replay.json

The script compares structured epistemic fields, not prose wording. It never executes Tools.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from debug_assistant.config import AppConfig
from debug_assistant.llm.factory import build_llm
from debug_assistant.agent.reflection import Reflector

FIELDS=("phase","decision","root_cause_target","root_cause_location","root_cause_mechanism","evidence_sufficient","supporting_evidence_ids","contradicting_evidence_ids","resolved_required_evidence","required_missing_evidence")

def load(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]

def canonical(v):
    if isinstance(v,list): return sorted((canonical(x) for x in v),key=lambda x:json.dumps(x,sort_keys=True,default=str))
    if isinstance(v,dict): return {k:canonical(v[k]) for k in sorted(v)}
    return v

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('trace'); ap.add_argument('--out',default='reflection_replay.json'); ap.add_argument('--limit',type=int,default=20)
    args=ap.parse_args(); rows=load(args.trace)
    originals={}
    for r in rows:
        if r.get('type') in {'REFLECTION','TERMINAL_RECONCILIATION_REFLECTION'}:
            originals.setdefault(r['payload'].get('step'),[]).append(r['payload'])
    snaps=[r['payload'] for r in rows if r.get('type')=='REFLECTION_CONTEXT_SNAPSHOT'][:args.limit]
    if not snaps: raise SystemExit('No REFLECTION_CONTEXT_SNAPSHOT events. Re-run with DEBUG_AGENT_TRACE_REFLECTION_CONTEXT=1.')
    cfg=AppConfig.from_env(); llm=build_llm(cfg.model); ref=Reflector(llm,cfg.model.critic_model or cfg.model.planner_model,compact_prompt=cfg.harness.features.compact_prompt_rendering)
    out=[]
    for s in snaps:
        mode=s.get('mode','normal'); ctx=s['context']
        got=ref.review_terminal(ctx) if mode=='terminal' else (ref.review_retry(ctx) if mode=='retry' else ref.review(ctx))
        candidates=originals.get(s.get('step')) or []; orig=candidates[-1] if candidates else None
        diffs={}
        if orig:
            for f in FIELDS:
                if canonical(orig.get(f)) != canonical(got.get(f)): diffs[f]={'original':orig.get(f),'replay':got.get(f)}
        out.append({'step':s.get('step'),'mode':mode,'context_chars':len(ctx),'same_structured_outcome':not bool(diffs) if orig else None,'diffs':diffs,'replay':got})
    Path(args.out).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'snapshots':len(out),'exact_structural_matches':sum(x['same_structured_outcome'] is True for x in out),'output':args.out},ensure_ascii=False))
if __name__=='__main__': main()
