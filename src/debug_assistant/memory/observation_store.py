from __future__ import annotations
import copy
from debug_assistant.models import ToolObservation

class ObservationStore:
    def __init__(self):
        self._items: dict[str, ToolObservation] = {}
        self._order: list[str] = []

    def add(self, obs: ToolObservation) -> None:
        if obs.observation_id not in self._items:
            self._order.append(obs.observation_id)
        # The Store is the immutable rehydration source. Runtime may later bound a
        # state/trace copy, but that must not corrupt the authoritative Observation.
        self._items[obs.observation_id] = copy.deepcopy(obs)

    def get(self, observation_id: str) -> ToolObservation | None:
        item=self._items.get(observation_id)
        return copy.deepcopy(item) if item is not None else None

    def all(self) -> list[ToolObservation]:
        return [copy.deepcopy(self._items[x]) for x in self._order if x in self._items]
