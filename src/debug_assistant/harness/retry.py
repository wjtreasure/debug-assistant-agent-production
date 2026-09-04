from __future__ import annotations
from dataclasses import dataclass
import random

@dataclass(slots=True)
class RetryPolicy:
    max_attempts: int = 2
    base_delay_seconds: float = 0.2
    max_delay_seconds: float = 2.0

    def delay(self, attempt_index: int, *, rng=None) -> float:
        """Full-jitter exponential backoff. attempt_index is 1-based failed attempt."""
        cap=min(float(self.max_delay_seconds), float(self.base_delay_seconds)*(2 ** max(0, int(attempt_index)-1)))
        r=rng or random
        return float(r.uniform(0.0,max(0.0,cap)))

    def can_retry(self, attempt_index: int, *, remaining_seconds: float | None=None, next_cost_seconds: float=0.0) -> bool:
        if int(attempt_index) >= max(1,int(self.max_attempts)):
            return False
        if remaining_seconds is None:
            return True
        return float(remaining_seconds) > max(0.0,float(next_cost_seconds))
