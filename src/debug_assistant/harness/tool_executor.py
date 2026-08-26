from __future__ import annotations
import time

TRANSIENT={'TimeoutExpired','TimeoutError','ConnectionError','OSError','timeout','transient_error'}

def execute_with_retry(tool, arguments, *, attempts=2, on_retry=None):
    obs=None
    for n in range(1,attempts+1):
        obs=tool.execute(**arguments)
        retryable=bool(obs.metadata.get('retryable')) or obs.error_type in TRANSIENT
        if obs.ok or not retryable or n==attempts:
            return obs
        if on_retry: on_retry(n,obs)
        time.sleep(0.1*n)
    return obs
