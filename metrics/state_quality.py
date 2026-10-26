from __future__ import annotations


class StateFlowQualityMetric:
    """Compatibility name for the paper-aligned Test State Quality metric."""

    def calculate(
        self,
        sfc: float,
        ae: float,
        mutation_score: float | None,
        pv: float = 1.0,
        nrr: float = 1.0,
    ) -> float:
        weighted = [(sfc, 0.30), (ae, 0.20), (pv, 0.15), (nrr, 0.15)]
        if mutation_score is not None:
            weighted.append((mutation_score, 0.20))
        total_weight = sum(weight for _, weight in weighted)
        if total_weight <= 0:
            return 0.0
        return round(sum(value * weight for value, weight in weighted) / total_weight, 4)


TestStateQualityMetric = StateFlowQualityMetric
