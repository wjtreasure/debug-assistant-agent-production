#!/usr/bin/env python3
"""Score manually extracted Need pairs with the configured BGE-M3 embedding endpoint.
Does not change thresholds or runtime config. API failures are emitted per row as errors.
"""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from debug_assistant.memory.evidence_need import RequiredEvidenceLedger, NeedMatcherConfig, SiliconFlowBgeM3Matcher

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input'); ap.add_argument('-o','--output',required=True); args=ap.parse_args()
    cfg=NeedMatcherConfig(base_url=os.getenv('DEBUG_AGENT_EMBEDDING_BASE_URL','https://api.siliconflow.cn/v1'),api_key=os.getenv('DEBUG_AGENT_EMBEDDING_API_KEY') or os.getenv('DEBUG_AGENT_API_KEY',''),model=os.getenv('DEBUG_AGENT_EMBEDDING_MODEL','BAAI/bge-m3'),timeout=float(os.getenv('DEBUG_AGENT_EMBEDDING_TIMEOUT','15')))
    matcher=SiliconFlowBgeM3Matcher(cfg); helper=RequiredEvidenceLedger()
    out=[]
    for i,line in enumerate(Path(args.input).read_text(encoding='utf-8').splitlines()):
        if not line.strip(): continue
        row=json.loads(line)
        try:
            a=helper._make(row['a'],1); b=helper._make(row['b'],1); d=matcher.compare(a,b)
            row['similarity']=d.similarity; row['semantic_result']=d.result.value; row['semantic_reason']=d.reason
        except Exception as exc:
            row['score_error']=f'{type(exc).__name__}: {exc}'
        out.append(row)
    Path(args.output).write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in out)+'\n',encoding='utf-8')
    print(json.dumps({'pairs':len(out),'api_calls':matcher.calls,'api_failures':matcher.api_failure_count,'output':args.output},ensure_ascii=False))
if __name__=='__main__':main()
