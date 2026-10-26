from __future__ import annotations


class MutationScoreMetric:
    def calculate(self, killed: int, total: int) -> float | None:
        if total <= 0:
            return None
        return round(killed / total, 4)
