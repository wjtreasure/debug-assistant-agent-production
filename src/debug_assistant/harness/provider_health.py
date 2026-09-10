from __future__ import annotations
from collections import deque
from dataclasses import dataclass, asdict
from debug_assistant.models import ToolObservation


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


@dataclass(slots=True)
class ToolHealth:
    """Per-run health for one Tool; business-negative results are not failures."""

    state: str = "CLOSED"
    consecutive_execution_failures: int = 0
    total_execution_failures: int = 0
    last_failure_type: str = ""


class ToolCircuitBreaker:
    """Small per-run CLOSED/OPEN Tool circuit breaker.

    It deliberately counts only execution/infrastructure failures.  A valid empty
    result, snapshot miss, resource-not-found result, or argument error is useful
    diagnostic information and must not poison the Tool's health.
    """

    _NON_FAILURE_TYPES = {
        "snapshot_unavailable", "not_found", "resource_not_found", "path_not_found",
        "ambiguous_path", "ambiguous_symbol", "symbol_not_found", "schema_validation",
        "invalid_arguments", "path_rejected", "permission_denied",
    }

    def __init__(self, *, failure_threshold: int = 2):
        self.failure_threshold = max(1, int(failure_threshold))
        self._health: dict[str, ToolHealth] = {}
        self.open_count = 0

    def health(self, tool: str) -> ToolHealth:
        return self._health.setdefault(str(tool), ToolHealth())

    def is_open(self, tool: str) -> bool:
        return self.health(tool).state == "OPEN"

    def before_call(self, tool: str) -> bool:
        return not self.is_open(tool)

    @classmethod
    def is_execution_failure(cls, observation: ToolObservation | None = None,
                             *, error: Exception | None = None) -> bool:
        if error is not None:
            return True
        if observation is None or observation.ok:
            return False
        metadata = observation.metadata or {}
        if metadata.get("semantic_negative") is True:
            return False
        error_type = str(observation.error_type or metadata.get("failure_category") or "")
        if error_type in cls._NON_FAILURE_TYPES:
            return False
        if metadata.get("failure_category") in {"validation", "business_negative", "not_found"}:
            return False
        # Explicit execution/infrastructure classification wins. Any remaining
        # non-empty Tool error is conservatively treated as an execution failure:
        # the breaker must fail closed for an unknown backend error, while the
        # allowlist above prevents valid negative/validation observations from
        # poisoning tool health.
        return bool(
            metadata.get("execution_failure") is True
            or metadata.get("infrastructure_failure") is True
            or metadata.get("retryable") is True
            or error_type
            or not observation.ok
        )

    def observe(self, tool: str, observation: ToolObservation | None = None,
                *, error: Exception | None = None) -> str | None:
        health = self.health(tool)
        failed = self.is_execution_failure(observation, error=error)
        if not failed:
            health.consecutive_execution_failures = 0
            return None
        health.consecutive_execution_failures += 1
        health.total_execution_failures += 1
        health.last_failure_type = type(error).__name__ if error else str(
            (observation or ToolObservation(str(tool), False, "")).error_type or "execution_failure"
        )
        if health.state == "CLOSED" and health.consecutive_execution_failures >= self.failure_threshold:
            health.state = "OPEN"
            self.open_count += 1
            return "opened"
        return None

    def summary(self) -> dict[str, dict]:
        return {
            name: asdict(health) for name, health in sorted(self._health.items())
        }

    def blocked_observation(self, tool: str) -> ToolObservation:
        return ToolObservation(
            tool=str(tool), ok=False,
            content=f"Tool {tool} is temporarily unavailable after repeated execution failures.",
            metadata={
                "status": "OPEN",
                "failure_category": "environment_unavailable",
                "capability_failure": True,
                "retryable": False,
            },
            error_type="tool_circuit_open",
        )
