from __future__ import annotations
from debug_assistant.models import ToolObservation

class ObservationStore:
    def __init__(self):
        self._items: dict[str, ToolObservation] = {}
        self._order: list[str] = []

    def add(self, obs: ToolObservation) -> None:
        if obs.observation_id not in self._items:
            self._order.append(obs.observation_id)
        self._items[obs.observation_id] = obs

    def get(self, observation_id: str) -> ToolObservation | None:
        return self._items.get(observation_id)

    def recent(self, n: int) -> list[ToolObservation]:
        if n <= 0: return []
        return [self._items[x] for x in self._order[-n:] if x in self._items]

    def all(self) -> list[ToolObservation]:
        return [self._items[x] for x in self._order if x in self._items]
