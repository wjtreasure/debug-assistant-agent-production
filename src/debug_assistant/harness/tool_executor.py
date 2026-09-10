from __future__ import annotations
import time
from .retry import RetryPolicy

TRANSIENT={'TimeoutExpired','TimeoutError','ConnectionError','OSError','timeout','transient_error'}

def execute_with_retry(tool, arguments, *, attempts=2, policy:RetryPolicy|None=None, absolute_deadline:float|None=None, on_retry=None, sleep_fn=time.sleep, rng=None):
    from debug_assistant.models import ToolObservation
    policy=policy or RetryPolicy(max_attempts=attempts)
    obs=None
    for n in range(1,max(1,int(policy.max_attempts))+1):
        if absolute_deadline is not None and time.monotonic() >= absolute_deadline:
            from debug_assistant.models import ToolObservation
            return ToolObservation(getattr(getattr(tool,'spec',None),'name','tool'),False,'tool retry deadline exceeded',{'retryable':True},'TimeoutError')
        try:
            obs=tool.execute(**arguments)
        except Exception as exc:
            error_type=type(exc).__name__
            retryable=bool(getattr(exc, "retryable", False)) or error_type in TRANSIENT
            obs=ToolObservation(
                getattr(getattr(tool, "spec", None), "name", "tool"), False, str(exc),
                {
                    "retryable": retryable,
                    "execution_failure": True,
                    "failure_category": "execution_failure",
                },
                error_type,
            )
        retryable=bool((obs.metadata or {}).get('retryable')) or obs.error_type in TRANSIENT
        if obs.ok or not retryable or n>=policy.max_attempts:
            return obs
        delay=policy.delay(n,rng=rng)
        remaining=(absolute_deadline-time.monotonic()) if absolute_deadline is not None else None
        if not policy.can_retry(n,remaining_seconds=remaining,next_cost_seconds=delay):
            return obs
        if on_retry:on_retry(n,obs)
        sleep_fn(delay)
    return obs
