from __future__ import annotations
from collections import deque
from dataclasses import dataclass, asdict


@dataclass(slots=True)
class ProviderHealthSample:
    logical_call_id: str
    stage: str
    provider_success: bool
    provider_failure: bool
    error_type: str = ""
    elapsed_seconds: float = 0.0
    allowed_seconds: float = 0.0

    @property
    def healthy_success(self) -> bool:
        if not self.provider_success:
            return False
        if self.allowed_seconds <= 0:
            return True
        return self.elapsed_seconds / self.allowed_seconds < 0.8


class ProviderCircuitBreaker:
    """Small logical-call provider-health circuit breaker with automatic recovery."""
    def __init__(self, *, window:int=5, failure_threshold:int=3, consecutive_failures:int=2,
                 recovery_successes:int=2, degraded_timeout_seconds:float=60.0):
        self.window=max(1,int(window)); self.samples=deque(maxlen=self.window)
        self.failure_threshold=max(1,int(failure_threshold))
        self.consecutive_failures=max(1,int(consecutive_failures))
        self.recovery_successes=max(1,int(recovery_successes))
        self.degraded_timeout_seconds=max(1.0,float(degraded_timeout_seconds))
        self.degraded=False
        self._healthy_success_streak=0

    def observe(self, sample:ProviderHealthSample) -> str|None:
        self.samples.append(sample)
        if self.degraded:
            if sample.healthy_success:
                self._healthy_success_streak += 1
                if self._healthy_success_streak >= self.recovery_successes:
                    self.degraded=False; self._healthy_success_streak=0
                    return 'recovered'
            else:
                self._healthy_success_streak=0
            return None
        recent=list(self.samples)
        failures=sum(1 for x in recent if x.provider_failure)
        suffix=0
        for x in reversed(recent):
            if x.provider_failure: suffix+=1
            else: break
        if len(recent)>=self.window and failures>=self.failure_threshold and suffix>=self.consecutive_failures:
            self.degraded=True; self._healthy_success_streak=0
            return 'degraded'
        return None

    def cap(self, configured:float, stage:str) -> float:
        if self.degraded and stage in {'planner','reflection'}:
            return min(float(configured),self.degraded_timeout_seconds)
        return float(configured)

    def summary(self):
        return {'degraded':self.degraded,'samples':[asdict(x) for x in self.samples]}
