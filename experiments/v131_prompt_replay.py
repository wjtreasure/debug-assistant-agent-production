"""V1.3.1 prompt semantic replay helper.

This intentionally does NOT reconstruct arbitrary historical runtime checkpoints.
It accepts a prepared JSON snapshot containing a single Reflection context and
runs the current ReflectionContract against it. Existing V1.3 traces did not
persist full prompt text, so exact historical prompt replay requires explicitly
prepared snapshots rather than pretending metadata-only traces are sufficient.

Snapshot format:
{
  "context": "...exact reflection context..."
}
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from debug_assistant.config import AppConfig
from debug_assistant.llm.factory import build_llm
from debug_assistant.agent.reflection import Reflector


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('snapshot')
    ap.add_argument('--output')
    args=ap.parse_args()
    data=json.loads(Path(args.snapshot).read_text(encoding='utf-8'))
    context=data.get('context')
    if not isinstance(context,str) or not context.strip():
        raise SystemExit('snapshot must contain non-empty string field: context')
    cfg=AppConfig.from_env(); llm=build_llm(cfg.model)
    model=cfg.model.critic_model or cfg.model.planner_model
    result=Reflector(llm,model).review(context)
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output: Path(args.output).write_text(text,encoding='utf-8')
    print(text)

if __name__=='__main__': main()
