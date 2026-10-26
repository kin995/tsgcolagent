from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class CompleteStateGraphReport:
    csg: float
    sfc: float
    dvs: float
    ae: float
    ms: float
    tic: float
    rcs: float
    complete: bool
    unresolved_validation_errors: int
    required_target_states_covered: bool
    no_regression_after_repair: bool
    threshold: float
    tsq: float
    pv: float
    nrr: float
    max_gap: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class CompleteStateGraphMetric:
    def __init__(self, threshold: float = 0.80):
        self.threshold = threshold

    def calculate(self, graph: Any, validation_report: Any | None = None, intent_plan: dict[str, Any] | None = None) -> CompleteStateGraphReport:
        last_report = graph.metadata.get("last_report", {})
        sfc = float(last_report.get("sfc", 0.0) or 0.0)
        dvs = float(getattr(validation_report, "score", graph.metadata.get("last_validation", {}).get("score", 0.0)) or 0.0)
        ae = float(last_report.get("ae", 0.0) or 0.0)
        ms_raw = last_report.get("mutation_score")
        ms = float(ms_raw) if ms_raw is not None else 0.0
        pv = float(last_report.get("pv", 1.0 if last_report.get("pytest_passed") else 0.0) or 0.0)
        nrr = float(last_report.get("nrr", 1.0) or 0.0)
        tsq_raw = last_report.get("tsq", last_report.get("sfq"))
        if tsq_raw is None:
            from metrics.state_quality import TestStateQualityMetric

            tsq_raw = TestStateQualityMetric().calculate(sfc, ae, ms_raw, pv=pv, nrr=nrr)
        tsq = float(tsq_raw or 0.0)
        tic = self._test_intent_coverage(graph, intent_plan)
        rcs = self._repair_closure_score(graph)
        from graph.graph_query import GraphQuery

        gap_values = GraphQuery(graph).get_gaps().as_dict()
        max_gap = max(gap_values.values(), default=0.0)
        csg = round(tsq, 4)
        unresolved = self._unresolved_validation_errors(graph, validation_report)
        required_covered = tic >= 0.999
        repair = graph.metadata.get("state_positioning_repair") or graph.metadata.get("minimal_target_repair", {})
        no_regression = not bool(repair.get("regression_detected", False))
        return CompleteStateGraphReport(
            csg=csg,
            sfc=round(sfc, 4),
            dvs=round(dvs, 4),
            ae=round(ae, 4),
            ms=round(ms, 4),
            tic=round(tic, 4),
            rcs=round(rcs, 4),
            complete=bool(csg >= self.threshold and pv >= 0.999 and nrr >= 0.999 and max_gap <= 0.40 and unresolved == 0 and required_covered and no_regression),
            unresolved_validation_errors=unresolved,
            required_target_states_covered=required_covered,
            no_regression_after_repair=no_regression,
            threshold=self.threshold,
            tsq=round(tsq, 4),
            pv=round(pv, 4),
            nrr=round(nrr, 4),
            max_gap=round(max_gap, 4),
        )

    def _test_intent_coverage(self, graph: Any, intent_plan: dict[str, Any] | None) -> float:
        plan = intent_plan or graph.metadata.get("last_test_intent_plan", {}) or graph.metadata.get("last_planning_decision", {})
        targets = set()
        if isinstance(plan, dict):
            for key in ("target_states", "target_methods", "target_branches"):
                value = plan.get(key, [])
                if isinstance(value, list):
                    targets.update(str(item) for item in value if item)
        if not targets:
            program_nodes = [node for node in graph.nodes.values() if node.node_type not in {"Dependency", "Parameter", "ReturnType", "Annotation"}]
            if not program_nodes:
                return 0.0
            return min(1.0, sum(1 for node in program_nodes if node.visited) / max(1, len(program_nodes)))
        covered = 0
        for target in targets:
            for node in graph.nodes.values():
                if target in {node.node_id, node.name, str(node.metadata.get("qualified_name", ""))} and node.visited:
                    covered += 1
                    break
        return covered / max(1, len(targets))

    def _repair_closure_score(self, graph: Any) -> float:
        unresolved = graph.metadata.get("unresolved_validation_errors", [])
        if isinstance(unresolved, list) and unresolved:
            return 0.0
        suggestions = graph.metadata.get("repair_suggestions", [])
        if suggestions:
            repair = graph.metadata.get("state_positioning_repair") or graph.metadata.get("minimal_target_repair", {})
            handled = repair.get("scope")
            return 1.0 if handled else 0.7
        return 1.0

    def _unresolved_validation_errors(self, graph: Any, validation_report: Any | None) -> int:
        if validation_report is not None:
            return len(getattr(validation_report, "errors", []) or [])
        unresolved = graph.metadata.get("unresolved_validation_errors", [])
        return len(unresolved) if isinstance(unresolved, list) else 0
