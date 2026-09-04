from __future__ import annotations
from dataclasses import dataclass
import time


@dataclass(slots=True)
class BudgetSnapshot:
    steps_used:int; tool_calls_used:int; llm_calls_used:int; tokens_used:int; wall_time_seconds:float
    step_ratio:float; tool_ratio:float; llm_ratio:float; token_ratio:float; time_ratio:float; remaining_ratio:float
    phase:str
    finalization_reserve_seconds:float=0.0
    seconds_until_forced_finalization:float=0.0


class BudgetController:
    def __init__(self,*,max_steps:int,max_tool_calls:int,max_llm_calls:int,max_total_tokens:int,max_wall_time_seconds:int,finalization_reserve_seconds:int=0,started_at:float|None=None):
        self.max_steps=max(1,int(max_steps)); self.max_tool_calls=max(1,int(max_tool_calls)); self.max_llm_calls=max(1,int(max_llm_calls)); self.max_total_tokens=max(1,int(max_total_tokens)); self.max_wall_time_seconds=max(1,int(max_wall_time_seconds))
        self.finalization_reserve_seconds=max(0,min(int(finalization_reserve_seconds),max(0,self.max_wall_time_seconds-1)))
        self.started_at=started_at or time.time()

    def snapshot(self,*,steps:int,tool_calls:int,llm_calls:int,tokens:int)->BudgetSnapshot:
        elapsed=max(0.0,time.time()-self.started_at)
        ratios=[steps/self.max_steps,tool_calls/self.max_tool_calls,llm_calls/self.max_llm_calls,tokens/self.max_total_tokens,elapsed/self.max_wall_time_seconds]
        remaining=max(0.0,1.0-max(ratios))
        forced_at=max(0.0,self.max_wall_time_seconds-self.finalization_reserve_seconds)
        until_finalize=max(0.0,forced_at-elapsed)
        if self.finalization_reserve_seconds and elapsed>=forced_at:
            phase='finalize'
        else:
            phase='explore' if remaining>0.50 else 'converge' if remaining>0.20 else 'verify_only' if remaining>0.10 else 'finalize'
        return BudgetSnapshot(steps,tool_calls,llm_calls,tokens,elapsed,*ratios,remaining,phase,float(self.finalization_reserve_seconds),until_finalize)

    def exhausted(self,s:BudgetSnapshot)->str|None:
        if s.steps_used>=self.max_steps:return 'max_steps'
        if s.tool_calls_used>=self.max_tool_calls:return 'max_tool_calls'
        if s.llm_calls_used>=self.max_llm_calls:return 'max_llm_calls'
        if s.tokens_used>=self.max_total_tokens:return 'max_total_tokens'
        if s.wall_time_seconds>=self.max_wall_time_seconds:return 'max_wall_time'
        return None
