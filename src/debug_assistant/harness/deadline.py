from __future__ import annotations

import time


class RunDeadline:
    """Single monotonic deadline shared by every runtime stage."""

    def __init__(self, timeout_seconds: float, *, started_at: float | None = None):
        self.started_at = time.monotonic() if started_at is None else float(started_at)
        self.absolute_deadline = self.started_at + max(0.0, float(timeout_seconds))

    def remaining(self) -> float:
        return max(0.0, self.absolute_deadline - time.monotonic())

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def effective_timeout(self, configured_timeout: float) -> float:
        return max(0.0, min(float(configured_timeout), self.remaining()))

    def can_start(self, required_seconds: float = 0.0) -> bool:
        return self.remaining() >= max(0.0, float(required_seconds))

    def check(self) -> None:
        if self.expired():
            raise TimeoutError("global run deadline exceeded")
