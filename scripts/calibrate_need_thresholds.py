#!/usr/bin/env python3
"""Summarize manually labeled semantic-need scores and suggest conservative thresholds.
Input JSONL rows must contain: same_fact (bool), similarity (float).
This does not auto-update runtime configuration.
"""
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path

def q(vals,p):
    vals=sorted(vals)
    if not vals:return None
    return vals[min(len(vals)-1,max(0,round((len(vals)-1)*p)))]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input'); args=ap.parse_args()
    rows=[json.loads(x) for x in Path(args.input).read_text().splitlines() if x.strip()]
    pos=[float(x['similarity']) for x in rows if x.get('same_fact') is True and x.get('similarity') is not None]
    neg=[float(x['similarity']) for x in rows if x.get('same_fact') is False and x.get('similarity') is not None]
    # Conservative recommendation: high above 95th percentile of hard negatives and near lower positive tail;
    # low below lower positive tail / upper easy-negative region. Human review remains required.
    high=max(0.90,(q(neg,.95) or 0)+.01) if neg else .90
    high=min(.99,high)
    low=min(.70,(q(pos,.10) or .70)-.01) if pos else .70
    low=max(.30,low)
    print(json.dumps({'n_positive':len(pos),'n_negative':len(neg),'positive_min':min(pos) if pos else None,'positive_p10':q(pos,.1),'negative_max':max(neg) if neg else None,'negative_p95':q(neg,.95),'provisional_high_recommendation':round(high,3),'provisional_low_recommendation':round(low,3),'note':'manual review required; do not auto-write config'},indent=2))
if __name__=='__main__':main()
