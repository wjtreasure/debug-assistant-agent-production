from __future__ import annotations
from dataclasses import dataclass, field
import math, time

DEFAULT_COSTS={
    'read_file':8.0,
    'inspect_symbol_context':15.0,
    'inspect_symbol_context_metadata':8.0,
    'tool':10.0,
    'parallel_margin':5.0,
    'focused_reflection':20.0,
    'reflection':40.0,
    'planner':40.0,
    'reporter':30.0,
}

@dataclass(slots=True)
class BudgetReservation:
    reservation_id: str
    seconds: float

@dataclass
class RemainingBudgetGate:
    absolute_deadline: float
    finalization_reserve_seconds: float=90.0
    cleanup_margin_seconds: float=15.0
    costs: dict[str,float]=field(default_factory=lambda:dict(DEFAULT_COSTS))
    _reserved: float=0.0
    _counter: int=0

    def usable_remaining(self, *, now: float | None=None, include_finalization_reserve: bool=False) -> float:
        now=time.time() if now is None else float(now)
        reserve=float(self.cleanup_margin_seconds) + (0.0 if include_finalization_reserve else float(self.finalization_reserve_seconds))
        return max(0.0,float(self.absolute_deadline)-now-reserve-self._reserved)

    def estimate(self, action_type: str, *, child_costs: list[float] | None=None, max_workers: int=4) -> float:
        if action_type=='parallel':
            costs=[max(0.0,float(x)) for x in (child_costs or [])]
            if not costs: return self.costs['parallel_margin']
            waves=max(1,math.ceil(len(costs)/max(1,int(max_workers))))
            return waves*max(costs)+self.costs['parallel_margin']
        return float(self.costs.get(action_type,self.costs.get('tool',10.0)))

    def admit(self, action_type: str, *, estimated_cost: float | None=None, child_costs: list[float] | None=None, max_workers: int=4, now: float | None=None, include_finalization_reserve: bool=False):
        cost=float(estimated_cost if estimated_cost is not None else self.estimate(action_type,child_costs=child_costs,max_workers=max_workers))
        remaining=self.usable_remaining(now=now,include_finalization_reserve=include_finalization_reserve)
        if remaining < cost:
            return None, {'action_type':action_type,'estimated_cost_seconds':cost,'usable_remaining_seconds':remaining}
        self._counter+=1; self._reserved+=cost
        return BudgetReservation(f'br-{self._counter}',cost), None

    def release(self, reservation: BudgetReservation | None):
        if reservation is not None:
            self._reserved=max(0.0,self._reserved-float(reservation.seconds))
