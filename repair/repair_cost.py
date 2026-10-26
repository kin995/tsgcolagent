from __future__ import annotations

from dataclasses import asdict, dataclass


SCOPE_COSTS = {
    "assertion-level": 1.0,
    "line-level": 1.5,
    "test-method-level": 2.0,
    "test-class-level": 4.0,
    "file-level": 6.0,
}


@dataclass
class RepairCost:
    scope_cost: float
    loc_change: int = 0
    risk_penalty: float = 0.0
    llm_cost: float = 0.0

    @property
    def total(self) -> float:
        return round(self.scope_cost + 0.05 * self.loc_change + self.risk_penalty + self.llm_cost, 4)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["total"] = self.total
        return data


def estimate_repair_cost(scope_level: str, loc_change: int = 0, llm_calls: int = 0) -> RepairCost:
    risk = 1.5 if scope_level in {"test-class-level", "file-level"} else 0.2 if scope_level == "test-method-level" else 0.0
    return RepairCost(
        scope_cost=SCOPE_COSTS.get(scope_level, 3.0),
        loc_change=loc_change,
        risk_penalty=risk,
        llm_cost=0.5 * llm_calls,
    )

