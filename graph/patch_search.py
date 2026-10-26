from __future__ import annotations

import ast
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from executor.coverage_runner import CoverageRunner
from executor.mutation_runner import MutationRunner
from executor.pytest_runner import PytestRunner
from executor.runtime_observer import RuntimeObserver
from executor.test_quality import analyze_test_usability
from graph.graph_defect_propagation import GraphDefectPropagation
from graph.graph_updater import GraphUpdater
from graph.state_graph import StateFlowGraph
from metrics.assertion_metric import AssertionEffectivenessMetric
from metrics.coverage_metric import StateFlowCoverageMetric
from metrics.state_quality import StateFlowQualityMetric


@dataclass
class PatchCandidate:
    name: str
    patch: str
    replace_file: bool = False


@dataclass
class PatchEvaluation:
    candidate: PatchCandidate
    pytest_passed: bool
    coverage_percent: float
    mutation_score: float
    estimated_sfc: float
    estimated_ae: float
    estimated_sfq: float
    accepted: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.name,
            "pytest_passed": self.pytest_passed,
            "coverage_percent": self.coverage_percent,
            "mutation_score": self.mutation_score,
            "estimated_sfc": self.estimated_sfc,
            "estimated_ae": self.estimated_ae,
            "estimated_sfq": self.estimated_sfq,
            "accepted": self.accepted,
            "reason": self.reason,
        }


class PatchSearchRepair:
    def __init__(self, source_path: str | Path, baseline_report: dict[str, object] | None = None, graph: StateFlowGraph | None = None):
        self.source_path = Path(source_path).resolve()
        self.baseline_report = baseline_report or {}
        self.graph = graph
        self.pytest_runner = PytestRunner()
        self.coverage_runner = CoverageRunner()
        self.mutation_runner = MutationRunner(max_mutants=6)
        self.runtime_observer = RuntimeObserver()
        self.last_evaluations: list[PatchEvaluation] = []

    def choose(self, test_path: str | Path, candidates: list[PatchCandidate]) -> PatchEvaluation | None:
        self.last_evaluations = [self._evaluate_cached(test_path, candidate) for candidate in candidates if candidate.patch.strip()]
        accepted = [item for item in self.last_evaluations if item.accepted]
        if not accepted:
            return None
        return max(accepted, key=lambda item: (item.estimated_sfq, item.mutation_score, item.coverage_percent))

    def evaluate(self, test_path: str | Path, candidate: PatchCandidate) -> PatchEvaluation:
        test_path = Path(test_path).resolve()
        try:
            ast.parse(candidate.patch)
        except SyntaxError as exc:
            return PatchEvaluation(candidate, False, 0.0, 0.0, 0.0, 0.0, 0.0, False, f"syntax error: {exc}")

        with tempfile.TemporaryDirectory(prefix="stateflow_patch_") as temp:
            temp_test = Path(temp) / test_path.name
            original = test_path.read_text(encoding="utf-8")
            if candidate.replace_file:
                temp_test.write_text(candidate.patch.strip() + "\n", encoding="utf-8")
            else:
                temp_test.write_text(original.rstrip() + "\n\n" + candidate.patch.strip() + "\n", encoding="utf-8")

            pytest_result = self.pytest_runner.run(temp_test, cwd=temp_test.parent, timeout=30)
            if not pytest_result.passed:
                return PatchEvaluation(candidate, False, 0.0, 0.0, 0.0, 0.0, 0.0, False, "pytest failed")

            usability = analyze_test_usability(temp_test)
            if usability.normal_behavior_tests == 0 or usability.exception_only:
                return PatchEvaluation(
                    candidate, True, 0.0, 0.0, 0.0, 0.0, 0.0, False,
                    "semantic gate rejected exception-only tests without normal behavior",
                )
            if usability.weak_assertions_only:
                return PatchEvaluation(
                    candidate, True, 0.0, 0.0, 0.0, 0.0, 0.0, False,
                    "semantic gate rejected weak assertions only",
                )

            coverage_result = self.coverage_runner.run(self.source_path, temp_test, cwd=temp_test.parent, timeout=40)
            mutation_result = self.mutation_runner.run(
                self.source_path,
                temp_test,
                cwd=temp_test.parent,
                timeout=5,
                evaluation_profile="screening",
            )
            runtime_observations = self.runtime_observer.collect(self.source_path, temp_test)
            metrics = self._real_metrics(temp_test, pytest_result, coverage_result, mutation_result, runtime_observations)
            coverage = coverage_result.percent
            mutation = mutation_result.score
            accepted, reason = self._accept(
                metrics["sfc"],
                metrics["ae"],
                mutation,
                metrics["sfq"],
                coverage_result.line_coverage,
                coverage_result.branch_coverage,
            )
            return PatchEvaluation(
                candidate=candidate,
                pytest_passed=True,
                coverage_percent=coverage,
                mutation_score=mutation if mutation is not None else 0.0,
                estimated_sfc=metrics["sfc"],
                estimated_ae=metrics["ae"],
                estimated_sfq=metrics["sfq"],
                accepted=accepted,
                reason=reason,
            )

    def _evaluate_cached(self, test_path: str | Path, candidate: PatchCandidate) -> PatchEvaluation:
        cache = self.graph.metadata.setdefault("patch_evaluation_cache", {}) if self.graph is not None else {}
        key = self._candidate_key(test_path, candidate)
        cached = cache.get(key) if isinstance(cache, dict) else None
        if isinstance(cached, dict):
            return PatchEvaluation(
                candidate=candidate,
                pytest_passed=bool(cached.get("pytest_passed")),
                coverage_percent=float(cached.get("coverage_percent", 0.0) or 0.0),
                mutation_score=float(cached.get("mutation_score", 0.0) or 0.0),
                estimated_sfc=float(cached.get("estimated_sfc", 0.0) or 0.0),
                estimated_ae=float(cached.get("estimated_ae", 0.0) or 0.0),
                estimated_sfq=float(cached.get("estimated_sfq", 0.0) or 0.0),
                accepted=bool(cached.get("accepted")),
                reason=str(cached.get("reason") or "cached patch evaluation"),
            )
        result = self.evaluate(test_path, candidate)
        if isinstance(cache, dict):
            cache[key] = result.to_dict()
            while len(cache) > 24:
                cache.pop(next(iter(cache)))
        return result

    def _candidate_key(self, test_path: str | Path, candidate: PatchCandidate) -> str:
        test_file = Path(test_path)
        payload = {
            "source": hashlib.sha256(self.source_path.read_bytes()).hexdigest(),
            "test": hashlib.sha256(test_file.read_bytes()).hexdigest() if test_file.exists() else "missing",
            "candidate": hashlib.sha256(candidate.patch.encode("utf-8")).hexdigest(),
            "replace_file": candidate.replace_file,
            "baseline": {
                key: self.baseline_report.get(key)
                for key in ("pytest_passed", "line_coverage", "branch_coverage", "sfc", "ae", "mutation_score", "sfq")
            },
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def _real_metrics(self, temp_test: Path, pytest_result, coverage_result, mutation_result, runtime_observations: list[object] | None = None) -> dict[str, float]:
        if self.graph is None:
            effective_mutation = mutation_result.score
            coverage = coverage_result.percent
            assertion_count = self._assert_count(temp_test)
            ae = round(min(1.0, assertion_count / 3), 4)
            return {
                "sfc": coverage,
                "ae": ae,
                "sfq": StateFlowQualityMetric().calculate(coverage, ae, effective_mutation),
            }
        graph = self.graph.clone()
        updater = GraphUpdater(graph)
        updater.reset_current_run_state()
        updater.apply_coverage_lines(coverage_result.covered_lines)
        updater.apply_coverage_arcs(getattr(coverage_result, "covered_arcs", set()))
        total_asserts = updater.apply_assert_counts(temp_test)
        updater.record_execution_evidence(temp_test, pytest_result, coverage_result, mutation_result, runtime_observations or [])
        updater.apply_mutation_results(mutation_result.to_dict())
        updater.mark_probable_edges()
        graph.record_flow(Path(temp_test).name, [node.node_id for node in graph.nodes.values() if node.visited])
        GraphDefectPropagation(graph).run()
        sfc = StateFlowCoverageMetric().calculate(graph)
        if coverage_result.valid:
            coverage_backed = coverage_result.line_coverage
            if coverage_result.branch_coverage is not None:
                coverage_backed = (coverage_result.line_coverage * 0.55) + (coverage_result.branch_coverage * 0.45)
            sfc = round(max(sfc, coverage_backed), 4)
        effective_mutation = mutation_result.score
        ae = AssertionEffectivenessMetric().calculate(graph, total_asserts=total_asserts, mutation_score=effective_mutation or 0.0)
        sfq = StateFlowQualityMetric().calculate(sfc, ae, effective_mutation)
        return {"sfc": sfc, "ae": ae, "sfq": sfq}

    def _accept(
        self,
        sfc: float,
        ae: float,
        mutation: float | None,
        sfq: float,
        line_coverage: float,
        branch_coverage: float | None,
    ) -> tuple[bool, str]:
        base_passed = bool(self.baseline_report.get("pytest_passed", False))
        if not base_passed:
            if sfc > 0.0 or sfq > 0.0:
                return True, "baseline pytest failed; accepted passing patch with measurable evidence"
            return False, "baseline failed and patch produced no measurable evidence"
        base_line = float(self.baseline_report.get("line_coverage", self.baseline_report.get("coverage_percent", 0.0)) or 0.0)
        base_branch_raw = self.baseline_report.get("branch_coverage")
        base_branch = float(base_branch_raw) if base_branch_raw is not None else None
        base_sfc = float(self.baseline_report.get("sfc", 0.0))
        base_ae = float(self.baseline_report.get("ae", 0.0))
        base_mutation_raw = self.baseline_report.get("mutation_score")
        base_mutation = float(base_mutation_raw) if base_mutation_raw is not None else 0.0
        mutation_value = mutation if mutation is not None else 0.0
        base_sfq = float(self.baseline_report.get("sfq", 0.0))
        if line_coverage + 1e-9 < base_line:
            return False, "line coverage regressed"
        if base_branch is not None and branch_coverage is not None and branch_coverage + 1e-9 < base_branch:
            return False, "branch coverage regressed"
        if sfc + 1e-9 < base_sfc:
            return False, "state-flow coverage regressed"
        if ae + 1e-9 < base_ae:
            return False, "assertion effectiveness regressed"
        if mutation_value + 1e-9 < base_mutation:
            return False, "mutation score regressed"
        if sfq + 1e-9 < base_sfq:
            return False, "overall quality regressed"
        line_improved = line_coverage > base_line + 1e-9
        branch_improved = (
            base_branch is not None
            and branch_coverage is not None
            and branch_coverage > base_branch + 1e-9
        )
        if (
            sfq == base_sfq
            and mutation_value == base_mutation
            and sfc == base_sfc
            and not line_improved
            and not branch_improved
        ):
            return False, "no measurable quality improvement"
        return True, "accepted by patch search validation"

    def _assert_count(self, path: Path) -> int:
        text = path.read_text(encoding="utf-8")
        return text.count("assert ") + text.count("pytest.raises")
