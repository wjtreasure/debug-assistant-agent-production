from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from dataclasses import dataclass
import time
from debug_assistant.harness.tool_executor import execute_with_retry
from debug_assistant.harness.retry import RetryPolicy

@dataclass(slots=True)
class ParallelChildResult:
    action_index: int
    action_id: str
    tool: str
    arguments: dict
    observation: object

@dataclass(slots=True)
class ParallelGroupResult:
    group_id: str
    children: list[ParallelChildResult]
    status: str
    elapsed_ms: float

def execute_parallel_group(registry, actions, *, group_id:str, max_workers:int=4, group_timeout_seconds:float=20.0, retry_policy:RetryPolicy|None=None, on_retry=None):
    started=time.monotonic(); deadline=started+max(0.01,float(group_timeout_seconds)); retry_policy=retry_policy or RetryPolicy()
    def run(idx,child):
        tool=registry.get(child['tool'])
        obs=execute_with_retry(tool,child.get('arguments') or {},policy=retry_policy,absolute_deadline=deadline,on_retry=(lambda n,o:on_retry(idx,n,o) if on_retry else None))
        return ParallelChildResult(idx,str(child.get('action_id') or f'a{idx}'),child['tool'],dict(child.get('arguments') or {}),obs)
    pool=ThreadPoolExecutor(max_workers=max(1,min(int(max_workers),len(actions))))
    futures=[pool.submit(run,idx,child) for idx,child in enumerate(actions)]
    try:
        wait(futures,timeout=max(0.01,deadline-time.monotonic()),return_when=ALL_COMPLETED)
        rows=[]
        from debug_assistant.models import ToolObservation
        for idx,fut in enumerate(futures):
            if fut.done():
                try: rows.append(fut.result())
                except Exception as exc: rows.append(ParallelChildResult(idx,str(actions[idx].get('action_id') or f'a{idx}'),actions[idx]['tool'],dict(actions[idx].get('arguments') or {}),ToolObservation(actions[idx]['tool'],False,str(exc),{'retryable':False},type(exc).__name__)))
            else:
                fut.cancel(); rows.append(ParallelChildResult(idx,str(actions[idx].get('action_id') or f'a{idx}'),actions[idx]['tool'],dict(actions[idx].get('arguments') or {}),ToolObservation(actions[idx]['tool'],False,'parallel group deadline exceeded',{'retryable':True},'TimeoutError')))
    finally:
        # Never let completion order or a slow sibling block deterministic ingestion.
        # Parallel tools are restricted to bounded local read-only operations; running
        # worker threads are not allowed to mutate Harness state.
        pool.shutdown(wait=False,cancel_futures=True)
    rows.sort(key=lambda x:x.action_index)
    oks=sum(1 for x in rows if x.observation.ok); status='success' if oks==len(rows) else 'failed' if oks==0 else 'partial'
    return ParallelGroupResult(group_id,rows,status,(time.monotonic()-started)*1000)
