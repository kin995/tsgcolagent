from __future__ import annotations

import re
import os
import shutil
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.analysis_agent import SourceAnalysisAgent
from agents.generation_agent import TestGenerationAgent
from agents.test_plan import PROMPT_VERSION, PYTHON_TARGET_LOADER_IMPORTS, SCHEMA_VERSION, python_target_loader_body
from executor.coverage_runner import CoverageResult, CoverageRunner
from executor.evaluation_cache import EvaluationCache, dependency_files, source_dependency_files
from executor.failure_classifier import PytestFailureClassifier
from executor.java_toolchain import JavaToolchain
from executor.mutation_runner import MutationResult, MutationRunner
from executor.pytest_runner import PytestRunner
from executor.runtime_observer import RuntimeObserver
from executor.test_quality import analyze_test_usability
from core.state_flow_event import StateFlowEvent, append_state_flow_event
from graph.graph_defect_propagation import GraphDefectPropagation
from graph.graph_updater import GraphUpdater
from llm_client import OpenAICompatibleLLM
from memory.shared_graph import SharedGraphMemory
from metrics.assertion_metric import AssertionEffectivenessMetric
from metrics.coverage_metric import StateFlowCoverageMetric
from metrics.state_quality import StateFlowQualityMetric
from repair.minimal_target_repair import StatePositioningRepairStrategy


@dataclass
class ExecutionReport:
    pytest_passed: bool
    coverage_percent: float
    mutation_score: float | None
    sfc: float
    ae: float
    sfq: float
    test_path: str
    tsq: float = 0.0
    pv: float = 0.0
    nrr: float = 1.0
    failure_category: str = "valid_baseline"
    failed_tests: list[dict[str, object]] | None = None
    coverage_valid: bool = True
    mutation_valid: bool = True
    quality_status: str = "valid"
    next_allowed_agents: list[str] | None = None
    line_coverage: float = 0.0
    branch_coverage: float | None = None
    combined_coverage: float = 0.0
    collected_count: int = 0
    timed_out: bool = False
    effective_mutants: int = 0
    mutation_reliability: str = "unavailable"
    weak_assertions_only: bool = False
    exception_only: bool = False
    normal_behavior_tests: int = 0
    assertion_count: int = 0
    weak_assertions: int = 0
    effective_assertions: int = 0
    sfc_components: dict[str, float | int] | None = None
    timeout_kind: str | None = None
    timeout_reason: str | None = None
    timeout_likely_location: str | None = None
    timeout_should_repair: bool = True
    weak_quality_flags: list[str] | None = None
    threshold_failures: dict[str, float] | None = None
    failure_location: str | None = None
    repair_failure_reason: str | None = None
    evaluation_profile: str = "full"
    mutation_complete: bool = True
    cache_hit: bool = False
    mutation_evidence: dict[str, object] | None = None
    quality_assessment: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "pytest_passed": self.pytest_passed,
            "coverage_percent": self.coverage_percent,
            "line_coverage": self.line_coverage,
            "branch_coverage": self.branch_coverage,
            "combined_coverage": self.combined_coverage,
            "mutation_score": self.mutation_score,
            "sfc": self.sfc,
            "ae": self.ae,
            "sfq": self.sfq,
            "tsq": self.tsq or self.sfq,
            "pv": self.pv,
            "nrr": self.nrr,
            "test_path": self.test_path,
            "failure_category": self.failure_category,
            "failed_tests": self.failed_tests or [],
            "coverage_valid": self.coverage_valid,
            "mutation_valid": self.mutation_valid,
            "quality_status": self.quality_status,
            "next_allowed_agents": self.next_allowed_agents or [],
            "collected_count": self.collected_count,
            "timed_out": self.timed_out,
            "effective_mutants": self.effective_mutants,
            "mutation_reliability": self.mutation_reliability,
            "weak_assertions_only": self.weak_assertions_only,
            "exception_only": self.exception_only,
            "normal_behavior_tests": self.normal_behavior_tests,
            "assertion_count": self.assertion_count,
            "weak_assertions": self.weak_assertions,
            "effective_assertions": self.effective_assertions,
            "assertion_quality_explanation": self._assertion_quality_explanation(),
            "sfc_components": self.sfc_components or {},
            "timeout_kind": self.timeout_kind,
            "timeout_reason": self.timeout_reason,
            "timeout_likely_location": self.timeout_likely_location,
            "timeout_should_repair": self.timeout_should_repair,
            "weak_quality_flags": self.weak_quality_flags or [],
            "threshold_failures": self.threshold_failures or {},
            "failure_location": self.failure_location,
            "repair_failure_reason": self.repair_failure_reason,
            "evaluation_profile": self.evaluation_profile,
            "mutation_complete": self.mutation_complete,
            "cache_hit": self.cache_hit,
            "mutation_evidence": self.mutation_evidence or {},
            "quality_assessment": self.quality_assessment or {},
        }

    def _assertion_quality_explanation(self) -> str:
        if self.assertion_count <= 0:
            return "没有检测到断言，测试无法证明返回行为是否正确"
        if self.effective_assertions <= 0:
            return "断言存在但全部被识别为弱断言，例如恒真、None 比较或只检查对象存在"
        if self.weak_assertions:
            return f"共 {self.assertion_count} 个断言，其中 {self.effective_assertions} 个有效、{self.weak_assertions} 个偏弱"
        return f"共 {self.assertion_count} 个断言，均为有效断言"


class ExecutionAgent:
    name = "测试生成智能体/执行与验证模块"

    def __init__(
        self,
        memory: SharedGraphMemory,
        llm: OpenAICompatibleLLM | None = None,
        llm_required: bool = False,
        retry_rounds: int = 2,
        isolate_humaneval_java: bool = False,
    ):
        self.memory = memory
        self.llm = llm
        self.llm_required = llm_required
        self.retry_rounds = retry_rounds
        self.isolate_humaneval_java = isolate_humaneval_java
        self.pytest_runner = PytestRunner()
        self.java_toolchain = JavaToolchain()
        self.coverage_runner = CoverageRunner()
        self.mutation_runner = MutationRunner()
        self.runtime_observer = RuntimeObserver()
        self.evaluation_cache = EvaluationCache(self.memory.storage_path.parent / "evaluation_cache.json")
        self.failure_classifier = PytestFailureClassifier()
        self.state_positioning_repair = StatePositioningRepairStrategy()
        self.analysis_agent = SourceAnalysisAgent(memory, llm=llm, llm_required=llm_required)
        self.generation_agent = TestGenerationAgent(memory, llm=llm, llm_required=llm_required)

    def prepare_initial_test(
        self,
        source_path: str | Path,
        output_dir: str | Path,
        experience: Any,
        model: str | None = None,
    ) -> tuple[Path, bool]:
        self.analyze_for_generation(source_path, experience, model=model)
        return self.generate_initial_test(source_path, output_dir, experience, model=model)

    def analyze_for_generation(self, source_path: str | Path, experience: Any, model: str | None = None):
        graph = self.analysis_agent.run(source_path)
        graph.metadata["experience_hints"] = experience.hints(graph, model=model)
        graph.metadata["execution_subagents"] = {
            "analysis": self.analysis_agent.name,
            "generation": self.generation_agent.name,
            "repair": "Repair Optimization Agent",
        }
        graph.metadata["collaboration_model"] = {
            "runtime": "LangGraph StateGraph",
            "agents": ["TestStateAgent", "TestGenerationAgent", "TestKnowledgeAgent"],
            "test_generation_internal_modules": [
                "StructuredContextExtractor",
                "ContextRewriter",
                "StructuredGraphBuilder",
                "TestGenerationModule",
                "JavaToolchain",
                "StatePositioningRepairStrategy",
                "StateTargetOptimizationStrategy",
            ],
            "test_state_internal_modules": ["DynamicValidator", "TestStateQualityMetric", "GraphDefectPropagation"],
            "shared_state": ["Test-State-Center", "StateFlowEvent", "knowledge_store/test_knowledge_guidance.json"],
            "loop": "TestStateAgent <-> TestKnowledgeAgent <-> TestGenerationAgent <-> TestStateAgent",
        }
        self._record_exchange(graph, "AnalysisResult", {"source_path": str(Path(source_path).resolve()), "graph_version": graph.version})
        self.memory.save(graph)
        return graph

    def generate_initial_test(
        self,
        source_path: str | Path,
        output_dir: str | Path,
        experience: Any,
        model: str | None = None,
    ) -> tuple[Path, bool]:
        try:
            test_path = self.generation_agent.run(output_dir)
            generation_failed = False
        except RuntimeError as exc:
            if Path(source_path).suffix.lower() == ".java":
                raise RuntimeError(f"Java generation failed; refusing a Python repair seed: {exc}") from exc
            test_path = self._seed_repairable_candidate(output_dir, source_path, exc)
            generation_failed = True
        graph = self.memory.load()
        if not generation_failed:
            self._record_generation_experience(graph, test_path, experience, model=model)
        self._record_exchange(
            graph,
            "GenerationResult",
            {
                "test_path": str(Path(test_path).resolve()),
                "generation_failed": generation_failed,
                "validation": graph.metadata.get("test_generation_validation", {}),
            },
        )
        append_state_flow_event(
            graph,
            StateFlowEvent(
                event_type="TestGenerated",
                source_agent="测试生成智能体",
                target_agent="测试状态智能体",
                payload={"test_path": str(Path(test_path).resolve()), "generation_failed": generation_failed},
                iteration=graph.iteration,
            ),
        )
        graph.bump_version("execution prepared initial analysis and test")
        self.memory.save(graph)
        return Path(test_path), generation_failed

    def _seed_repairable_candidate(self, output_dir: str | Path, source_path: str | Path, exc: RuntimeError) -> Path:
        graph = self.memory.load()
        output_dir = Path(output_dir)
        test_path = output_dir / f"test_{Path(source_path).stem}_stateflow.py"
        rejected = sorted((output_dir / "candidate_artifacts").glob("rejected_candidate_*.py"))
        if rejected:
            seed_text = rejected[-1].read_text(encoding="utf-8")
            seed_source = str(rejected[-1])
        else:
            seed_text = self._minimal_repair_seed(Path(source_path))
            seed_source = "minimal_repair_seed"
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.write_text(seed_text.rstrip() + "\n", encoding="utf-8")
        graph.metadata["generation_seeded_for_repair"] = {
            "reason": f"{exc.__class__.__name__}: {exc}",
            "seed_source": seed_source,
            "test_path": str(test_path.resolve()),
            "next_agent": "TestValidityAgent",
        }
        graph.metadata["generated_test_path"] = str(test_path.resolve())
        graph.bump_version("generation failed candidate seeded for repair")
        self.memory.save(graph)
        return test_path

    def _minimal_repair_seed(self, source_path: Path) -> str:
        lines = [
            "from __future__ import annotations",
            "",
            *PYTHON_TARGET_LOADER_IMPORTS,
            "",
            "import pytest",
            "",
            *python_target_loader_body(source_path),
            "",
            "@pytest.fixture(scope='module')",
            "def target():",
            "    return _load_target()",
            "",
            "def test_stateflow_seed_loads_target(target):",
            "    assert target is not None",
        ]
        return "\n".join(lines).rstrip() + "\n"

    def _java_project_root(self, source_path: str | Path, cwd: str | Path | None = None) -> Path:
        human_eval_workspace = self._human_eval_java_workspace(source_path)
        if human_eval_workspace is not None:
            return human_eval_workspace
        if cwd is not None and self._has_java_build_file(Path(cwd).resolve()):
            return Path(cwd).resolve()
        current = Path(source_path).resolve()
        if current.is_file():
            current = current.parent
        for candidate in (current, *current.parents):
            if self._has_java_build_file(candidate):
                return candidate
        bundled_workspace = self._bundled_java_workspace()
        if self._has_java_build_file(bundled_workspace):
            return bundled_workspace
        return current

    def _run_java(
        self,
        test_path: str | Path,
        cwd: str | Path | None = None,
        *,
        evaluation_profile: str = "full",
    ) -> ExecutionReport:
        graph = self.memory.load()
        profile = "screening" if evaluation_profile == "screening" else "full"
        project_root = self._java_project_root(graph.source_path, cwd=cwd)
        target_class = self._java_target_class(graph)
        has_build_file = self._has_java_build_file(project_root)
        if has_build_file:
            staged_test_path: Path | None = None
            staged_original: bytes | None = None
            staged_was_external = False
            try:
                prepared_source = self._prepare_bundled_java_workspace(
                    project_root,
                    graph.source_path,
                    graph.metadata.get("structured_context", {}),
                )
                staged_target = self._java_test_target_path(test_path, project_root)
                source_test = Path(test_path).resolve()
                staged_was_external = staged_target.resolve() != source_test
                if staged_was_external and staged_target.is_file():
                    staged_original = staged_target.read_bytes()
                staged_test_path = self._stage_java_test(test_path, project_root)
                cache_files = [
                    test_path,
                    *source_dependency_files(graph.source_path, project_root, "java"),
                    *dependency_files(project_root, "java"),
                ]
                cache_settings = {"target_class": target_class, "test_selector": staged_test_path.stem}
                cache_key = self.evaluation_cache.key(
                    language="java",
                    stage="toolchain",
                    profile=profile,
                    files=cache_files,
                    settings=cache_settings,
                )
                cached_toolchain = self.evaluation_cache.get("java_toolchain", cache_key)
                cache_hit = cached_toolchain is not None
                reused_validation = False
                toolchain_result = cached_toolchain
                if toolchain_result is None and profile == "full" and hasattr(self.java_toolchain, "run_pit_only"):
                    screening_key = self.evaluation_cache.key(
                        language="java",
                        stage="toolchain",
                        profile="screening",
                        files=cache_files,
                        settings=cache_settings,
                    )
                    screening_result = self.evaluation_cache.get("java_toolchain", screening_key)
                    if screening_result and screening_result.get("passed"):
                        pit_result = self.java_toolchain.run_pit_only(
                            project_root,
                            test_selector=staged_test_path.stem,
                            target_class=target_class,
                        )
                        toolchain_result = dict(screening_result)
                        toolchain_result.update(
                            {
                                "pit": pit_result,
                                "mutation_score": pit_result.get("mutation_score"),
                                "evaluation_profile": "full",
                                "reused_screening_validation": True,
                            }
                        )
                        reused_validation = True
                if toolchain_result is None:
                    toolchain_result = self._call_java_toolchain(
                        project_root=project_root,
                        test_selector=staged_test_path.stem,
                        target_class=target_class,
                        evaluation_profile=profile,
                    )
                transient_failure = any(
                    toolchain_result.get(stage, {}).get(flag, False)
                    for stage in ("junit", "pit") for flag in ("tool_missing", "timed_out")
                )
                if not cache_hit and not transient_failure:
                    self.evaluation_cache.put("java_toolchain", cache_key, toolchain_result)
                toolchain_result["cache_hit"] = cache_hit
                toolchain_result["reused_screening_validation"] = reused_validation
                toolchain_result["staged_test_path"] = str(staged_test_path)
                toolchain_result["target_class"] = target_class
                if prepared_source:
                    toolchain_result["prepared_source"] = prepared_source
            except OSError as exc:
                toolchain_result = {
                    "language": "java",
                    "build_system": self.java_toolchain._build_system(project_root),
                    "passed": False,
                    "junit": {"passed": False, "stderr": f"{exc.__class__.__name__}: {exc}"},
                    "line_coverage": None,
                    "branch_coverage": None,
                    "method_coverage": None,
                    "mutation_score": None,
                    "jacoco": {"valid": False, "reason": "test_staging_failed"},
                    "pit": {"valid": False, "reason": "test_staging_failed", "mutation_score": None},
                    "evaluation_profile": profile,
                }
                cache_hit = False
            finally:
                if staged_test_path is not None and staged_was_external:
                    try:
                        if staged_original is None:
                            staged_test_path.unlink(missing_ok=True)
                        else:
                            staged_test_path.write_bytes(staged_original)
                    except OSError:
                        pass
            junit_result = toolchain_result.get("junit", {}) if isinstance(toolchain_result.get("junit"), dict) else {}
            if toolchain_result.get("passed"):
                failure_category = "valid_baseline"
            elif junit_result.get("tool_missing"):
                failure_category = "java_tool_missing"
            elif junit_result.get("timed_out"):
                failure_category = "java_timeout"
            else:
                failure_category = "java_execution_failed"
        else:
            toolchain_result = {
                "language": "java",
                "build_system": "missing",
                "passed": False,
                "junit": {
                    "passed": False,
                    "stderr": "No Maven or Gradle build file found. Java execution requires a project root with pom.xml or build.gradle.",
                },
                "line_coverage": None,
                "branch_coverage": None,
                "method_coverage": None,
                "mutation_score": None,
                "jacoco": {"valid": False, "reason": "missing_build_file"},
                "pit": {"valid": False, "reason": "missing_build_file", "mutation_score": None},
                "evaluation_profile": profile,
            }
            cache_hit = False
            failure_category = "java_execution_unavailable"

        line_coverage = float(toolchain_result.get("line_coverage") or 0.0)
        branch_coverage = toolchain_result.get("branch_coverage")
        mutation_score = toolchain_result.get("mutation_score")
        pit_result = toolchain_result.get("pit", {}) if isinstance(toolchain_result.get("pit"), dict) else {}
        pit_result.setdefault("evaluation_profile", profile)
        pit_result.setdefault("final_evaluation", profile == "full")
        effective_mutants = int(pit_result.get("effective_mutants", 0) or 0)
        mutation_reliability = str(
            pit_result.get("score_reliability")
            or ("unavailable" if effective_mutants == 0 else "low" if effective_mutants < 3 else "normal")
        )
        junit_passed = bool(toolchain_result.get("passed"))
        failed_tests = [] if junit_passed else [self._java_failure_item(toolchain_result, failure_category)]
        coverage_valid = toolchain_result.get("line_coverage") is not None
        mutation_valid = bool(pit_result.get("valid")) and mutation_score is not None
        assertion_profile = self._java_assertion_profile(test_path)
        assertion_count = assertion_profile["total"]
        weak_assertions = assertion_profile["weak"]
        effective_assertions = assertion_profile["effective"]
        exception_assertions = assertion_profile["exception"]
        normal_assertions = assertion_profile["normal"]
        test_count = self._java_test_count(test_path)
        exception_only = test_count > 0 and exception_assertions > 0 and normal_assertions <= 0
        normal_behavior_tests = test_count if junit_passed and normal_assertions > 0 else 0
        sfc = line_coverage if coverage_valid else 0.0
        ae = 1.0 if assertion_count > 0 else 0.0
        sfc_components: dict[str, float | int] | None = None
        if junit_passed:
            updater = GraphUpdater(graph)
            updater.reset_current_run_state()
            covered_lines = set(toolchain_result.get("jacoco", {}).get("covered_lines", []) if isinstance(toolchain_result.get("jacoco"), dict) else [])
            updater.apply_coverage_lines({int(line) for line in covered_lines})
            updater.apply_java_coverage_edges()
            execution_id = updater.record_java_execution_evidence(test_path, toolchain_result, {int(line) for line in covered_lines})
            updater.apply_java_mutation_results(pit_result)
            updater.record_java_mutation_evidence(execution_id, pit_result)
            graph.record_flow(Path(test_path).name, [node.node_id for node in graph.nodes.values() if node.visited])
            GraphDefectPropagation(graph).run()
            sfc_metric = StateFlowCoverageMetric()
            sfc = sfc_metric.calculate(graph)
            sfc_components = sfc_metric.components(graph)
            sfc = self._coverage_backed_state_flow_score(sfc, line_coverage, branch_coverage if branch_coverage is not None else None, coverage_valid)
            ae = AssertionEffectivenessMetric().calculate(graph, total_asserts=assertion_count, mutation_score=float(mutation_score or 0.0))
        last_test_usability = {
            "language": "java",
            "test_count": test_count,
            "assertion_count": assertion_count,
            "weak_assertions": weak_assertions,
            "effective_assertions": effective_assertions,
            "weak_assertions_only": assertion_count <= 0 or effective_assertions <= 0,
            "exception_only": exception_only,
            "normal_behavior_tests": normal_behavior_tests,
        }
        pv = 1.0 if junit_passed and test_count > 0 and coverage_valid else 0.0
        positioning = graph.metadata.get("state_positioning_repair") or graph.metadata.get("minimal_target_repair", {})
        nrr = 0.0 if positioning.get("regression_detected") else 1.0
        reliable_mutation_score = (
            float(mutation_score)
            if mutation_score is not None and mutation_valid and mutation_reliability == "normal"
            else None
        )
        sfq = StateFlowQualityMetric().calculate(sfc, ae, reliable_mutation_score, pv=pv, nrr=nrr)
        report = ExecutionReport(
            pytest_passed=junit_passed,
            coverage_percent=line_coverage,
            mutation_score=None if mutation_score is None else float(mutation_score),
            sfc=sfc,
            ae=ae,
            sfq=sfq,
            test_path=str(Path(test_path).resolve()),
            tsq=sfq,
            pv=pv,
            nrr=nrr,
            failure_category=failure_category,
            failed_tests=failed_tests,
            coverage_valid=coverage_valid,
            mutation_valid=mutation_valid,
            quality_status="valid" if toolchain_result.get("passed") else "invalid",
            next_allowed_agents=[] if toolchain_result.get("passed") else ["TestValidityAgent"],
            line_coverage=line_coverage,
            branch_coverage=None if branch_coverage is None else float(branch_coverage),
            combined_coverage=line_coverage,
            collected_count=test_count if junit_passed else 0,
            timed_out=bool(toolchain_result.get("junit", {}).get("timed_out", False)),
            normal_behavior_tests=normal_behavior_tests,
            effective_mutants=effective_mutants,
            mutation_reliability=mutation_reliability,
            weak_assertions_only=assertion_count <= 0 or effective_assertions <= 0,
            exception_only=exception_only,
            assertion_count=assertion_count,
            weak_assertions=weak_assertions,
            effective_assertions=effective_assertions,
            sfc_components=sfc_components,
            evaluation_profile=profile,
            mutation_complete=profile == "full",
            cache_hit=cache_hit,
            mutation_evidence=self._java_mutation_metadata(
                pit_result,
                mutation_score=mutation_score,
                effective_mutants=effective_mutants,
                mutation_reliability=mutation_reliability,
            ),
        )
        graph.metadata["last_java_execution"] = self._java_execution_metadata(toolchain_result)
        jacoco_result = toolchain_result.get("jacoco", {}) if isinstance(toolchain_result.get("jacoco"), dict) else {}
        graph.metadata["last_coverage"] = {
            "percent": line_coverage,
            "line_coverage": line_coverage,
            "branch_coverage": branch_coverage,
            "method_coverage": toolchain_result.get("method_coverage"),
            "combined_coverage": line_coverage,
            "valid": coverage_valid,
            "covered_lines": list(jacoco_result.get("covered_lines", [])),
            "missing_lines": list(jacoco_result.get("missing_lines", [])),
            "target_class": target_class,
            "evidence_source": "jacoco",
        }
        graph.metadata["last_mutation"] = self._java_mutation_metadata(
            pit_result,
            mutation_score=mutation_score,
            effective_mutants=effective_mutants,
            mutation_reliability=mutation_reliability,
        )
        graph.metadata["last_evaluation_cache"] = {"profile": profile, "java_toolchain_hit": cache_hit}
        graph.metadata["last_test_usability"] = last_test_usability
        graph.metadata["last_sfc_components"] = sfc_components or {}
        graph.metadata["last_report"] = report.to_dict()
        graph.metadata["last_execution_result"] = report.to_dict()
        graph.metadata["execution_diagnosis"] = {
            "failure_category": failure_category,
            "root_candidates": [str(project_root)],
            "evidence_ids": [item.get("test_name", "java_execution") for item in failed_tests],
            "recommended_focus": self._java_recommended_focus(failure_category, toolchain_result),
            "confidence": 0.9,
        }
        self._record_exchange(graph, "ExecutionResult", report.to_dict())
        append_state_flow_event(
            graph,
            StateFlowEvent(
                event_type="ExecutionObserved" if toolchain_result.get("passed") else "ValidationFailed",
                source_agent="测试生成智能体",
                target_agent="测试状态智能体",
                payload={"report": report.to_dict(), "java_toolchain": self._java_toolchain_event_summary(toolchain_result)},
                iteration=graph.iteration,
            ),
        )
        append_state_flow_event(
            graph,
            StateFlowEvent(
                event_type="CoverageObserved",
                source_agent="测试生成智能体",
                target_agent="测试状态智能体",
                payload=dict(graph.metadata["last_coverage"]),
                iteration=graph.iteration,
            ),
        )
        append_state_flow_event(
            graph,
            StateFlowEvent(
                event_type="MutationObserved",
                source_agent="测试生成智能体",
                target_agent="测试状态智能体",
                payload=dict(graph.metadata["last_mutation"]),
                iteration=graph.iteration,
            ),
        )
        graph.bump_version("java execution completed")
        self.memory.save(graph)
        return report

    def _call_java_toolchain(
        self,
        *,
        project_root: Path,
        test_selector: str,
        target_class: str,
        evaluation_profile: str,
    ) -> dict[str, Any]:
        parameters = inspect.signature(self.java_toolchain.run).parameters
        kwargs: dict[str, object] = {
            "project_root": project_root,
            "test_selector": test_selector,
            "target_class": target_class,
        }
        if "evaluation_profile" in parameters:
            kwargs["evaluation_profile"] = evaluation_profile
        result = self.java_toolchain.run(**kwargs)
        if isinstance(result, dict):
            result.setdefault("evaluation_profile", evaluation_profile)
        return result

    def _java_execution_metadata(self, toolchain_result: dict[str, Any]) -> dict[str, object]:
        summary = self._java_toolchain_event_summary(toolchain_result)
        junit = toolchain_result.get("junit", {}) if isinstance(toolchain_result.get("junit"), dict) else {}
        summary["junit"] = {
            "passed": junit.get("passed"),
            "returncode": junit.get("returncode"),
            "tool_missing": junit.get("tool_missing", False),
            "timed_out": junit.get("timed_out", False),
            "error": str(junit.get("stderr") or junit.get("stdout") or "")[-2000:],
        }
        summary["staged_test_path"] = toolchain_result.get("staged_test_path")
        summary["prepared_source"] = toolchain_result.get("prepared_source")
        jacoco = toolchain_result.get("jacoco", {}) if isinstance(toolchain_result.get("jacoco"), dict) else {}
        pit = toolchain_result.get("pit", {}) if isinstance(toolchain_result.get("pit"), dict) else {}
        summary["jacoco"].update({"csv_path": jacoco.get("csv_path"), "xml_path": jacoco.get("xml_path")})
        summary["pit"].update({"xml_path": pit.get("xml_path"), "mutation_score": pit.get("mutation_score")})
        return summary

    def _java_mutation_metadata(
        self,
        pit_result: dict[str, Any],
        mutation_score: object,
        effective_mutants: int,
        mutation_reliability: str,
    ) -> dict[str, object]:
        fields = (
            "detected_count", "score_denominator",
            "valid", "total", "killed", "survived", "timeout", "invalid", "infra_error", "no_coverage",
            "killed_lines", "survived_lines", "timeout_lines", "invalid_lines", "infra_error_lines",
            "no_coverage_lines", "candidate_count", "attempted_mutants", "low_sample_reason", "autonomy_action",
            "xml_path", "target_class", "test_selector", "skipped", "skipped_reason", "evaluation_profile", "final_evaluation",
        )
        metadata = {field: pit_result.get(field) for field in fields if field in pit_result}
        metadata.update(
            {
                "score": mutation_score,
                "mutation_score": mutation_score,
                "effective_mutants": effective_mutants,
                "score_reliability": mutation_reliability,
                "evidence_source": "pit",
            }
        )
        return metadata

    def _java_toolchain_event_summary(self, toolchain_result: dict[str, Any]) -> dict[str, object]:
        pit = toolchain_result.get("pit", {}) if isinstance(toolchain_result.get("pit"), dict) else {}
        jacoco = toolchain_result.get("jacoco", {}) if isinstance(toolchain_result.get("jacoco"), dict) else {}
        return {
            "language": "java",
            "build_system": toolchain_result.get("build_system"),
            "passed": bool(toolchain_result.get("passed")),
            "target_class": toolchain_result.get("target_class"),
            "line_coverage": toolchain_result.get("line_coverage"),
            "branch_coverage": toolchain_result.get("branch_coverage"),
            "method_coverage": toolchain_result.get("method_coverage"),
            "mutation_score": toolchain_result.get("mutation_score"),
            "jacoco": {
                "valid": jacoco.get("valid"),
                "covered_lines": jacoco.get("covered_lines", []),
                "missing_lines": jacoco.get("missing_lines", []),
                "target_class": jacoco.get("target_class"),
            },
            "pit": {
                "valid": pit.get("valid"),
                "total": pit.get("total", 0),
                "effective_mutants": pit.get("effective_mutants", 0),
                "killed": pit.get("killed", 0),
                "survived": pit.get("survived", 0),
                "score_reliability": pit.get("score_reliability", "unavailable"),
                "survived_lines": pit.get("survived_lines", []),
            },
        }

    def _java_target_class(self, graph: Any) -> str:
        structured_context = graph.metadata.get("structured_context", {}) if hasattr(graph, "metadata") else {}
        package_name = str(structured_context.get("package_or_module") or "")
        source_stem = Path(graph.source_path).stem
        for cls in structured_context.get("classes", []) if isinstance(structured_context, dict) else []:
            if isinstance(cls, dict) and str(cls.get("name") or "") == source_stem:
                qualified = str(cls.get("qualified_name") or "")
                if qualified:
                    return qualified
        return f"{package_name}.{source_stem}" if package_name else source_stem

    def _java_failure_item(self, toolchain_result: dict[str, Any], failure_category: str) -> dict[str, object]:
        junit = toolchain_result.get("junit", {}) if isinstance(toolchain_result.get("junit"), dict) else {}
        message = str(junit.get("stderr") or junit.get("stdout") or failure_category)
        return {
            "test_name": "java_toolchain" if failure_category == "java_tool_missing" else "java_junit_execution",
            "message": message[-2000:],
            "failure_category": failure_category,
            "command": junit.get("command"),
            "staged_test_path": toolchain_result.get("staged_test_path"),
        }

    def _java_recommended_focus(self, failure_category: str, toolchain_result: dict[str, Any]) -> str:
        if failure_category == "valid_baseline":
            return "none"
        if failure_category in {"java_tool_missing", "java_execution_unavailable"}:
            return "java_project_toolchain"
        junit = toolchain_result.get("junit", {}) if isinstance(toolchain_result.get("junit"), dict) else {}
        message = str(junit.get("stderr") or junit.get("stdout") or "").lower()
        if "compilation failure" in message or "cannot find symbol" in message or "not public" in message:
            return "java_test_compile_repair"
        if "no tests matching pattern" in message or "no tests were executed" in message:
            return "java_test_selector_or_naming"
        return "java_test_validity_repair"

    def _java_assertion_count(self, test_path: str | Path) -> int:
        return self._java_assertion_profile(test_path)["total"]

    def _java_assertion_profile(self, test_path: str | Path) -> dict[str, int]:
        try:
            text = Path(test_path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return {"total": 0, "weak": 0, "effective": 0, "exception": 0, "normal": 0}
        assertions = re.findall(r"\b(assert[A-Za-z0-9_]*)\s*\(", text)
        weak = sum(name in {"assertDoesNotThrow", "assertNotNull"} for name in assertions)
        weak += len(re.findall(r"\bassertTrue\s*\(\s*true\s*\)", text, flags=re.IGNORECASE))
        weak += len(re.findall(r"\bassertFalse\s*\(\s*false\s*\)", text, flags=re.IGNORECASE))
        weak = min(len(assertions), weak)
        exception = sum(name in {"assertThrows", "assertThrowsExactly"} for name in assertions)
        normal = max(0, len(assertions) - exception - weak)
        return {
            "total": len(assertions),
            "weak": weak,
            "effective": max(0, len(assertions) - weak),
            "exception": exception,
            "normal": normal,
        }

    def _java_test_count(self, test_path: str | Path) -> int:
        try:
            text = Path(test_path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return 0
        return len(re.findall(r"@\s*(?:Test|ParameterizedTest|RepeatedTest|TestFactory|TestTemplate)\b", text))

    def _coverage_backed_state_flow_score(
        self,
        graph_score: float,
        line_coverage: float,
        branch_coverage: float | None,
        coverage_valid: bool,
    ) -> float:
        if not coverage_valid:
            return graph_score
        java_coverage_score = line_coverage
        if branch_coverage is not None:
            java_coverage_score = (line_coverage * 0.55) + (float(branch_coverage) * 0.45)
        return round(max(graph_score, java_coverage_score), 4)

    def _has_java_build_file(self, path: Path) -> bool:
        return any((path / name).exists() for name in ("pom.xml", "build.gradle", "build.gradle.kts", "gradlew", "gradlew.bat"))

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    def _bundled_java_workspace(self) -> Path:
        configured = os.environ.get("STATEFLOW_JAVA_WORKSPACE")
        return Path(configured).resolve() if configured else (self._repo_root() / "java_workspace").resolve()

    def _human_eval_java_workspace(self, source_path: str | Path) -> Path | None:
        if not getattr(self, "isolate_humaneval_java", False):
            return None
        source = Path(source_path).resolve()
        if not source.is_file() or source.suffix.lower() != ".java" or source.parent.name.casefold() != "humanevaljava":
            return None
        workspace = (self.memory.storage_path.parent / ".java_workspace").resolve()
        template = self._bundled_java_workspace() / "pom.xml"
        if not template.is_file():
            return None
        workspace.mkdir(parents=True, exist_ok=True)
        target_pom = workspace / "pom.xml"
        template_bytes = template.read_bytes()
        if not target_pom.exists() or target_pom.read_bytes() != template_bytes:
            target_pom.write_bytes(template_bytes)
        (workspace / ".stateflow_humanevaljava_workspace").write_text(
            str(source),
            encoding="utf-8",
        )
        return workspace

    def _is_bundled_java_workspace(self, project_root: Path) -> bool:
        try:
            resolved = project_root.resolve()
            return resolved == self._bundled_java_workspace().resolve() or (resolved / ".stateflow_humanevaljava_workspace").is_file()
        except OSError:
            return False

    def _prepare_bundled_java_workspace(
        self,
        project_root: Path,
        source_path: str | Path,
        structured_context: dict[str, object],
    ) -> dict[str, object] | None:
        if not self._is_bundled_java_workspace(project_root):
            return None
        source_file = Path(source_path).resolve()
        package_name = str(structured_context.get("package_or_module") or "")
        source_root, target_package = self._infer_java_dataset_root(source_file, package_name)
        main_root = project_root / "src" / "main" / "java"
        test_root = project_root / "src" / "test" / "java"
        isolated_humaneval = (project_root / ".stateflow_humanevaljava_workspace").is_file()
        if not isolated_humaneval:
            self._safe_clear_directory(main_root, project_root)
            self._safe_clear_directory(test_root, project_root)
        main_root.mkdir(parents=True, exist_ok=True)
        test_root.mkdir(parents=True, exist_ok=True)

        target_dir = main_root / target_package.replace(".", os.sep) if target_package else main_root
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if source_root.is_dir():
            self._copy_java_source_tree(source_root, target_dir)
        elif source_file.is_file():
            target_dir.mkdir(parents=True, exist_ok=True)
            target_source = target_dir / source_file.name
            source_text = source_file.read_text(encoding="utf-8-sig", errors="ignore")
            if not target_source.is_file() or target_source.read_text(encoding="utf-8", errors="ignore") != source_text:
                target_source.write_text(source_text, encoding="utf-8")
        self._write_java_workspace_support_sources(main_root, target_package)
        return {
            "workspace": str(project_root),
            "source_root": str(source_root),
            "target_source_dir": str(target_dir),
            "target_package": target_package,
            "workspace_mode": "isolated_humanevaljava" if isolated_humaneval else "shared_project",
        }

    def _infer_java_dataset_root(self, source_file: Path, package_name: str) -> tuple[Path, str]:
        package_parts = [part for part in package_name.split(".") if part]
        # HumanEvalJava is a flat collection of independent benchmark classes.  Copying
        # the whole directory makes every Maven invocation compile all 163 tasks, so one
        # task with a JDK-specific dependency can invalidate every unrelated task.
        if package_name.startswith("humaneval.") or source_file.parent.name.casefold() == "humanevaljava":
            return source_file, package_name
        if package_parts:
            root_package = package_parts[0]
            current = source_file.parent
            for candidate in (current, *current.parents):
                if candidate.name == root_package:
                    return candidate, root_package
            if len(current.parts) >= len(package_parts) and list(current.parts[-len(package_parts):]) == package_parts:
                return current.parents[len(package_parts) - 1], root_package
            return source_file.parent, root_package
        return source_file.parent, ""

    def _copy_java_source_tree(self, source_root: Path, target_dir: Path) -> None:
        for source in source_root.rglob("*"):
            relative = source.relative_to(source_root)
            target = target_dir / relative
            if source.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif source.suffix == ".java":
                self._copy_java_file(source, target)
            elif source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

    def _copy_java_file(self, source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        text = source.read_text(encoding="utf-8-sig", errors="ignore")
        target.write_text(text, encoding="utf-8")

    def _write_java_workspace_support_sources(self, main_root: Path, target_package: str) -> None:
        if target_package != "gson":
            return
        support = main_root / "com" / "google" / "gson" / "internal" / "GsonBuildConfig.java"
        support.parent.mkdir(parents=True, exist_ok=True)
        support.write_text(
            "package com.google.gson.internal;\n\n"
            "public final class GsonBuildConfig {\n"
            "    public static final String VERSION = \"2.10.1\";\n"
            "    private GsonBuildConfig() {}\n"
            "}\n",
            encoding="utf-8",
        )

    def _safe_clear_directory(self, directory: Path, project_root: Path) -> None:
        directory = directory.resolve()
        project_root = project_root.resolve()
        try:
            directory.relative_to(project_root)
        except ValueError as exc:
            raise OSError(f"Refusing to clear unsafe Java workspace directory: {directory}") from exc
        if directory == project_root:
            raise OSError(f"Refusing to clear unsafe Java workspace directory: {directory}")
        if directory.exists():
            shutil.rmtree(directory)

    def _stage_java_test(self, test_path: str | Path, project_root: Path) -> Path:
        source_path = Path(test_path).resolve()
        text = source_path.read_text(encoding="utf-8")
        target_path = self._java_test_target_path(source_path, project_root, text=text)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.resolve() != source_path:
            target_path.write_text(text, encoding="utf-8")
        return target_path

    def _java_test_target_path(self, test_path: str | Path, project_root: Path, text: str | None = None) -> Path:
        source_path = Path(test_path).resolve()
        text = text if text is not None else source_path.read_text(encoding="utf-8")
        package_match = re.search(r"^\s*package\s+([A-Za-z_][\w.]*)\s*;", text, flags=re.MULTILINE)
        package_parts = package_match.group(1).split(".") if package_match else []
        target_dir = project_root / "src" / "test" / "java"
        for part in package_parts:
            target_dir = target_dir / part
        return target_dir / source_path.name

    def _record_generation_experience(
        self,
        graph,
        test_path: str | Path,
        experience: Any,
        model: str | None = None,
    ) -> None:
        rejected_generation = graph.metadata.get("test_generation_candidate_rejected")
        if isinstance(rejected_generation, dict):
            experience.append_candidate(
                graph=graph,
                agent="ExecutionAgent/TestGenerationAgent",
                action="generate_test_plan",
                candidate_text=str(rejected_generation.get("candidate_path") or rejected_generation.get("pytest_output") or ""),
                accepted=False,
                pytest_passed=False,
                failure_category=rejected_generation.get("failure_category"),
                rejection_reason=str(rejected_generation.get("stage")),
                model=model,
                prompt_version=PROMPT_VERSION,
                schema_version=SCHEMA_VERSION,
            )
        generation_validation = graph.metadata.get("test_generation_validation", {})
        if generation_validation.get("stage") == "pending_compile_and_intent_validation":
            # Not yet evaluated is not a failed/negative experience.
            return
        path = Path(test_path)
        experience.append_candidate(
            graph=graph,
            agent="ExecutionAgent/TestGenerationAgent",
            action="generate_test_plan",
            candidate_text=path.read_text(encoding="utf-8") if path.exists() else "",
            accepted=bool(generation_validation.get("accepted", False)),
            pytest_passed=bool(generation_validation.get("accepted", False)),
            failure_category=generation_validation.get("failure_category"),
            rejection_reason=generation_validation.get("stage") if not generation_validation.get("accepted", False) else None,
            model=model,
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
        )

    def run(
        self,
        test_path: str | Path,
        cwd: str | Path | None = None,
        *,
        evaluation_profile: str = "full",
    ) -> ExecutionReport:
        graph = self.memory.load()
        if str(graph.metadata.get("language", "python")) == "java" or Path(test_path).suffix.lower() == ".java":
            return self._run_java(test_path, cwd=cwd, evaluation_profile=evaluation_profile)
        profile = "screening" if evaluation_profile == "screening" else "full"
        cwd = Path(cwd or Path(test_path).parent.parent).resolve()
        pytest_result, pytest_attempts = self._run_pytest_with_retries(test_path, cwd=cwd)
        usability = analyze_test_usability(test_path)
        collected_count = pytest_result.collected_count if pytest_result.collected_count is not None else usability.test_count
        if not pytest_result.passed:
            diagnosis = self.failure_classifier.classify(pytest_result, test_path, timed_out=pytest_result.timed_out)
            report = ExecutionReport(
                pytest_passed=False,
                coverage_percent=0.0,
                mutation_score=None,
                sfc=0.0,
                ae=0.0,
                sfq=0.0,
                test_path=str(Path(test_path).resolve()),
                failure_category=diagnosis.failure_category,
                failed_tests=[item.to_dict() for item in diagnosis.failed_tests],
                coverage_valid=False,
                mutation_valid=False,
                quality_status="invalid",
                next_allowed_agents=self._allowed_agents_for_failure(diagnosis.failure_category),
                collected_count=collected_count,
                timed_out=pytest_result.timed_out,
                weak_assertions_only=usability.weak_assertions_only,
                exception_only=usability.exception_only,
                normal_behavior_tests=usability.normal_behavior_tests,
                assertion_count=usability.assertion_count,
                weak_assertions=usability.weak_assertions,
                effective_assertions=max(0, usability.assertion_count - usability.weak_assertions),
                timeout_kind=diagnosis.timeout_kind,
                timeout_reason=diagnosis.timeout_reason,
                timeout_likely_location=diagnosis.likely_location,
                timeout_should_repair=diagnosis.should_repair,
                evaluation_profile=profile,
                mutation_complete=False,
            )
            graph.metadata["last_pytest"] = {
                "passed": False,
                "returncode": pytest_result.returncode,
                "stdout": pytest_result.stdout[-4000:],
                "stderr": pytest_result.stderr[-4000:],
                "attempts": pytest_attempts,
                "retry_rounds": max(0, pytest_attempts - 1),
                "timeout_kind": diagnosis.timeout_kind,
                "timeout_reason": diagnosis.timeout_reason,
                "timeout_likely_location": diagnosis.likely_location,
                "timeout_should_repair": diagnosis.should_repair,
            }
            graph.metadata["last_coverage"] = {"valid": False, "percent": 0.0}
            graph.metadata["last_mutation"] = {
                "valid": False,
                "score": None,
                "evaluation_profile": profile,
                "final_evaluation": False,
                "skipped": True,
                "skipped_reason": "baseline_not_runnable",
            }
            graph.metadata["last_test_usability"] = usability.to_dict()
            graph.metadata["last_runtime_observations"] = []
            graph.metadata["last_report"] = report.to_dict()
            graph.metadata["last_execution_result"] = report.to_dict()
            self._record_exchange(graph, "ExecutionResult", report.to_dict())
            graph.metadata["execution_diagnosis"] = {
                "failure_category": diagnosis.failure_category,
                "root_candidates": [item.get("test_name", "") for item in report.failed_tests or []],
                "evidence_ids": [],
                "recommended_focus": self._recommended_focus_for_failure(diagnosis),
                "confidence": 0.9,
                "timeout_kind": diagnosis.timeout_kind,
                "timeout_reason": diagnosis.timeout_reason,
                "likely_location": diagnosis.likely_location,
                "should_repair": diagnosis.should_repair,
            }
            graph.metadata["last_root_cause"] = {}
            append_state_flow_event(
                graph,
                StateFlowEvent(
                    event_type="ValidationFailed",
                    source_agent="测试生成智能体",
                    target_agent="测试状态智能体",
                    payload={"report": report.to_dict(), "diagnosis": graph.metadata["execution_diagnosis"]},
                    iteration=graph.iteration,
                ),
            )
            graph.bump_version("invalid execution completed")
            self.memory.save(graph)
            return report
        source_path = Path(graph.source_path).resolve()
        test_file = Path(test_path).resolve()
        cache_files = [
            test_file,
            *source_dependency_files(source_path, cwd or source_path.parent, "python"),
            *dependency_files(source_path.parent, "python"),
            *dependency_files(cwd, "python"),
        ]
        coverage_key = self.evaluation_cache.key(
            language="python",
            stage="coverage",
            profile="shared",
            files=cache_files,
        )
        cached_coverage = self.evaluation_cache.get("python_coverage", coverage_key)
        coverage_cache_hit = cached_coverage is not None
        coverage_result = self._coverage_from_cache(cached_coverage) if cached_coverage else self.coverage_runner.run(source_path, test_file, cwd=cwd)
        if not coverage_cache_hit and coverage_result.valid:
            self.evaluation_cache.put("python_coverage", coverage_key, self._coverage_to_cache(coverage_result))

        priority_lines = self._python_mutation_priority_lines(graph, coverage_result)
        mutation_key = self.evaluation_cache.key(
            language="python",
            stage="mutation",
            profile=profile,
            files=cache_files,
            settings={"max_mutants": self.mutation_runner.max_mutants, "screening_limit": 8, "priority_lines": sorted(priority_lines)},
        )
        cached_mutation = self.evaluation_cache.get("python_mutation", mutation_key)
        mutation_cache_hit = cached_mutation is not None
        prior_mutation = None
        previous_mutation = graph.metadata.get("last_mutation")
        previous_report = graph.metadata.get("last_report")
        if (
            profile == "full"
            and isinstance(previous_mutation, dict)
            and previous_mutation.get("evaluation_profile") == "screening"
            and isinstance(previous_report, dict)
            and Path(str(previous_report.get("test_path") or "")).resolve() == test_file
        ):
            prior_mutation = MutationResult.from_dict(previous_mutation)
        mutation_result = (
            MutationResult.from_dict(cached_mutation)
            if cached_mutation
            else self.mutation_runner.run(
                source_path,
                test_file,
                cwd=cwd,
                evaluation_profile=profile,
                priority_lines=priority_lines,
                prior_result=prior_mutation,
            )
        )
        if not mutation_cache_hit:
            self.evaluation_cache.put("python_mutation", mutation_key, mutation_result.to_dict())
        runtime_observations = self.runtime_observer.collect(graph.source_path, test_path)

        updater = GraphUpdater(graph)
        updater.reset_current_run_state()
        updater.apply_coverage_lines(coverage_result.covered_lines)
        updater.apply_coverage_arcs(getattr(coverage_result, "covered_arcs", set()))
        total_asserts = updater.apply_assert_counts(test_path)
        updater.record_execution_evidence(test_path, pytest_result, coverage_result, mutation_result, runtime_observations)
        updater.apply_mutation_results(mutation_result.to_dict())
        updater.mark_probable_edges()
        graph.record_flow(Path(test_path).name, [node.node_id for node in graph.nodes.values() if node.visited])
        defect_result = GraphDefectPropagation(graph).run()

        sfc_metric = StateFlowCoverageMetric()
        sfc = sfc_metric.calculate(graph)
        sfc_components = sfc_metric.components(graph)
        sfc = self._coverage_backed_state_flow_score(sfc, coverage_result.line_coverage, coverage_result.branch_coverage, coverage_result.valid)
        mutation_data = mutation_result.to_dict()
        effective_mutants = int(mutation_data.get("effective_mutants", 0) or 0)
        mutation_reliability = (
            "screening"
            if profile == "screening" and effective_mutants > 0
            else "unavailable" if effective_mutants == 0 else "low" if effective_mutants < 3 else "normal"
        )
        mutation_valid = not (
            mutation_result.total == 0
            and (mutation_result.invalid > 0 or mutation_result.infra_error > 0)
        )
        ae = AssertionEffectivenessMetric().calculate(graph, total_asserts=total_asserts, mutation_score=mutation_result.score or 0.0)
        pv = 1.0 if pytest_result.passed and collected_count > 0 and coverage_result.valid else 0.0
        positioning = graph.metadata.get("state_positioning_repair") or graph.metadata.get("minimal_target_repair", {})
        nrr = 0.0 if positioning.get("regression_detected") else 1.0
        reliable_mutation_score = mutation_result.score if mutation_valid and mutation_reliability == "normal" else None
        sfq = StateFlowQualityMetric().calculate(sfc, ae, reliable_mutation_score, pv=pv, nrr=nrr)
        report = ExecutionReport(
            pytest_passed=pytest_result.passed,
            coverage_percent=coverage_result.percent,
            mutation_score=mutation_result.score,
            sfc=sfc,
            ae=ae,
            sfq=sfq,
            test_path=str(Path(test_path).resolve()),
            tsq=sfq,
            pv=pv,
            nrr=nrr,
            failure_category="valid_baseline",
            failed_tests=[],
            coverage_valid=coverage_result.valid,
            mutation_valid=mutation_valid,
            quality_status="valid",
            next_allowed_agents=[],
            line_coverage=coverage_result.line_coverage,
            branch_coverage=coverage_result.branch_coverage,
            combined_coverage=coverage_result.combined_coverage,
            collected_count=collected_count,
            timed_out=pytest_result.timed_out,
            effective_mutants=effective_mutants,
            mutation_reliability=mutation_reliability,
            weak_assertions_only=usability.weak_assertions_only,
            exception_only=usability.exception_only,
            normal_behavior_tests=usability.normal_behavior_tests,
            assertion_count=usability.assertion_count,
            weak_assertions=usability.weak_assertions,
            effective_assertions=max(0, usability.assertion_count - usability.weak_assertions),
            sfc_components=sfc_components,
            evaluation_profile=profile,
            mutation_complete=profile == "full",
            cache_hit=coverage_cache_hit and mutation_cache_hit,
            mutation_evidence=mutation_data,
        )
        graph.metadata["last_pytest"] = {
            "passed": pytest_result.passed,
            "stdout": pytest_result.stdout[-4000:],
            "stderr": pytest_result.stderr[-4000:],
            "returncode": pytest_result.returncode,
            "attempts": pytest_attempts,
            "retry_rounds": max(0, pytest_attempts - 1),
        }
        graph.metadata["last_coverage"] = {
            "percent": coverage_result.percent,
            "line_coverage": coverage_result.line_coverage,
            "branch_coverage": coverage_result.branch_coverage,
            "combined_coverage": coverage_result.combined_coverage,
            "valid": coverage_result.valid,
            "covered_lines": sorted(coverage_result.covered_lines),
            "missing_lines": sorted(getattr(coverage_result, "missing_lines", set())),
            "covered_arcs": [list(arc) for arc in sorted(coverage_result.covered_arcs)],
            "covered_line_count": coverage_result.covered_line_count,
            "statement_count": coverage_result.statement_count,
            "covered_branch_count": coverage_result.covered_branch_count,
            "branch_count": coverage_result.branch_count,
            "stdout": coverage_result.stdout[-4000:],
            "stderr": coverage_result.stderr[-4000:],
        }
        mutation_data["score_reliability"] = mutation_reliability
        mutation_data["cache_hit"] = mutation_cache_hit
        mutation_data["final_evaluation"] = profile == "full"
        graph.metadata["last_mutation"] = mutation_data
        graph.metadata["last_evaluation_cache"] = {
            "profile": profile,
            "coverage_hit": coverage_cache_hit,
            "mutation_hit": mutation_cache_hit,
        }
        graph.metadata["last_test_usability"] = usability.to_dict()
        graph.metadata["last_sfc_components"] = sfc_components
        graph.metadata["last_runtime_observations"] = [item.to_dict() for item in runtime_observations]
        graph.metadata["last_report"] = report.to_dict()
        graph.metadata["last_execution_result"] = report.to_dict()
        self._record_exchange(graph, "ExecutionResult", report.to_dict())
        append_state_flow_event(
            graph,
            StateFlowEvent(
                event_type="ExecutionObserved",
                source_agent="测试生成智能体",
                target_agent="测试状态智能体",
                related_nodes=[node.node_id for node in graph.nodes.values() if node.visited][:120],
                payload={"report": report.to_dict(), "runtime_observation_count": len(runtime_observations)},
                iteration=graph.iteration,
            ),
        )
        append_state_flow_event(
            graph,
            StateFlowEvent(
                event_type="CoverageObserved",
                source_agent="测试生成智能体",
                target_agent="测试状态智能体",
                payload=graph.metadata["last_coverage"],
                iteration=graph.iteration,
            ),
        )
        append_state_flow_event(
            graph,
            StateFlowEvent(
                event_type="MutationObserved",
                source_agent="测试生成智能体",
                target_agent="测试状态智能体",
                payload=mutation_data,
                iteration=graph.iteration,
            ),
        )
        graph.metadata["last_root_cause"] = defect_result
        updater.finalize_execution_update("execution completed")
        self.memory.save(graph)
        return report

    def _coverage_to_cache(self, result: CoverageResult) -> dict[str, object]:
        return {
            "covered_lines": sorted(result.covered_lines),
            "missing_lines": sorted(result.missing_lines),
            "covered_arcs": [list(value) for value in sorted(result.covered_arcs)],
            "line_coverage": result.line_coverage,
            "branch_coverage": result.branch_coverage,
            "combined_coverage": result.combined_coverage,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "valid": result.valid,
            "covered_line_count": result.covered_line_count,
            "statement_count": result.statement_count,
            "covered_branch_count": result.covered_branch_count,
            "branch_count": result.branch_count,
        }

    def _coverage_from_cache(self, data: dict[str, object]) -> CoverageResult:
        return CoverageResult(
            covered_lines={int(value) for value in data.get("covered_lines", []) or []},
            missing_lines={int(value) for value in data.get("missing_lines", []) or []},
            covered_arcs={tuple(int(part) for part in value) for value in data.get("covered_arcs", []) or []},
            line_coverage=float(data.get("line_coverage", 0.0) or 0.0),
            branch_coverage=None if data.get("branch_coverage") is None else float(data["branch_coverage"]),
            combined_coverage=float(data.get("combined_coverage", 0.0) or 0.0),
            stdout=str(data.get("stdout") or ""),
            stderr=str(data.get("stderr") or ""),
            valid=bool(data.get("valid", True)),
            covered_line_count=int(data.get("covered_line_count", 0) or 0),
            statement_count=int(data.get("statement_count", 0) or 0),
            covered_branch_count=int(data.get("covered_branch_count", 0) or 0),
            branch_count=int(data.get("branch_count", 0) or 0),
        )

    def _python_mutation_priority_lines(self, graph: Any, coverage: CoverageResult) -> set[int]:
        lines = {int(value) for value in coverage.missing_lines}
        previous = graph.metadata.get("last_mutation") if isinstance(graph.metadata.get("last_mutation"), dict) else {}
        lines.update(int(value) for value in previous.get("survived_lines", []) or [])
        for node in graph.nodes.values():
            if node.repair_flag or node.mutation_status == "survived":
                if node.line_start:
                    lines.add(int(node.line_start))
        return lines

    def apply_plan(
        self,
        test_path: str | Path,
        plan: Any,
        repair_agent: Any,
        specialized_agents: dict[str, Any],
        evaluation_agent: Any,
        suggestions: list[Any] | None = None,
    ) -> bool:
        graph = self.memory.load()
        agent_name = str(getattr(plan, "selected_agent", None) or getattr(plan, "agent", ""))
        action = str(getattr(plan, "action", "repair"))
        graph.metadata["execution_plan_application"] = {
            "agent": self.name,
            "planned_action": action,
            "selected_tool": agent_name,
            "test_path": str(Path(test_path).resolve()),
        }
        self.memory.save(graph)
        if action not in {"repair", "optimize"} or agent_name in {"", "ExecutionAgent", "Stop"}:
            return False
        repair_suggestions = suggestions if suggestions is not None else (evaluation_agent.run() if evaluation_agent is not None else [])
        repair_decision = self.state_positioning_repair.locate(graph, test_path, repair_suggestions)
        graph.metadata["state_positioning_repair"] = repair_decision.to_dict()
        graph.metadata["minimal_target_repair"] = repair_decision.to_dict()
        append_state_flow_event(
            graph,
            StateFlowEvent(
                event_type="RepairScopeLocated",
                source_agent="测试生成智能体",
                target_agent="测试状态智能体",
                related_nodes=repair_decision.scope.related_nodes,
                payload=repair_decision.to_dict(),
                iteration=graph.iteration,
            ),
        )
        self.memory.save(graph)
        if str(graph.metadata.get("language", "")).lower() == "java" or Path(test_path).suffix.lower() == ".java":
            return self._apply_java_plan(test_path, plan, repair_suggestions, repair_decision)
        attempted: list[dict[str, object]] = []
        candidates = self._repair_candidate_order(agent_name)
        for candidate in candidates:
            before_hash = self._file_hash(test_path)
            try:
                specialized_agent = specialized_agents.get(candidate)
                if specialized_agent:
                    specialized_agent.run(test_path, suggestions=repair_suggestions)
                else:
                    repair_agent.run(test_path, repair_suggestions)
                graph = self.memory.load()
                applied = bool(graph.metadata.get("repair_agent", {}).get("applied"))
                after_hash = self._file_hash(test_path)
                changed = before_hash != after_hash
                attempted.append({"agent": candidate, "applied": applied, "changed": changed})
                if applied and changed:
                    graph.metadata["execution_repair_attempts"] = attempted
                    append_state_flow_event(
                        graph,
                        StateFlowEvent(
                            event_type="RepairApplied",
                            source_agent="测试生成智能体",
                            target_agent="测试状态智能体",
                            payload={"agent": candidate, "minimal_repair": repair_decision.to_dict(), "attempts": attempted},
                            iteration=graph.iteration,
                        ),
                    )
                    self.memory.save(graph)
                    return True
            except Exception as exc:
                graph = self.memory.load()
                attempted.append({"agent": candidate, "applied": False, "changed": False, "error": f"{exc.__class__.__name__}: {exc}"})
                graph.metadata["execution_repair_attempts"] = attempted
                self.memory.save(graph)
        graph = self.memory.load()
        graph.metadata["execution_repair_attempts"] = attempted
        self.memory.save(graph)
        return bool(graph.metadata.get("repair_agent", {}).get("applied"))

    def _apply_java_plan(self, test_path: str | Path, plan: Any, suggestions: list[Any], repair_decision: Any) -> bool:
        graph = self.memory.load()
        path = Path(test_path)
        original = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
        candidates = [self._sanitize_java_test(original, graph)]
        llm_candidate = self._java_llm_candidate(original, graph, plan, suggestions)
        if llm_candidate:
            candidates.insert(0, llm_candidate)
        for index, candidate in enumerate(candidates, start=1):
            candidate = self._ensure_java_test_name(candidate, path, graph)
            if not candidate.strip() or candidate.strip() == original.strip():
                continue
            path.write_text(candidate.rstrip() + "\n", encoding="utf-8")
            graph = self.memory.load()
            graph.metadata["repair_agent"] = {
                "applied": True,
                "repair_focus": str(getattr(plan, "focus", "java_quality_optimization")),
                "strategy": "java_llm_candidate" if index == 1 and llm_candidate else "java_static_sanitizer",
                "minimal_repair": repair_decision.to_dict() if hasattr(repair_decision, "to_dict") else {},
                "suggestion_count": len(suggestions),
                "test_path": str(path.resolve()),
            }
            graph.bump_version("java test repair or optimization completed")
            self.memory.save(graph)
            return True
        graph.metadata["repair_agent"] = {
            "applied": False,
            "repair_focus": str(getattr(plan, "focus", "java_quality_optimization")),
            "strategy": "java_no_change",
            "minimal_repair": repair_decision.to_dict() if hasattr(repair_decision, "to_dict") else {},
            "suggestion_count": len(suggestions),
            "test_path": str(path.resolve()),
        }
        self.memory.save(graph)
        return False

    def _sanitize_java_test(self, text: str, graph: Any) -> str:
        source_stem = Path(graph.source_path).stem
        expected_class = f"{source_stem}Test"
        repaired = text
        repaired = re.sub(r"\bnew\s+final\s+object\s*\[\]\s*\{\s*\}", "new Object[]{}", repaired)
        repaired = re.sub(r"\bnew\s+object\s*\[\]\s*\{\s*\}", "new Object[]{}", repaired)
        repaired = re.sub(r"\bnew\s+final\s+([A-Za-z_]\w*)\s*\[\]\s*\{\s*\}", r"new \1[]{}", repaired)
        repaired = re.sub(r"\bnew\s+([a-z][A-Za-z0-9_]*)\s*\[\]\s*\{\s*\}", self._java_array_type_case, repaired)
        for cls in self._java_private_constructor_classes(graph):
            repaired = re.sub(rf"new\s+{re.escape(cls)}\s*\(\)\.", f"{cls}.", repaired)
        repaired = re.sub(r"\bclass\s+\w+\s*\{", f"class {expected_class} {{", repaired, count=1)
        return repaired

    def _java_array_type_case(self, match: re.Match[str]) -> str:
        base = match.group(1)
        if base in {"boolean", "byte", "char", "short", "int", "long", "float", "double"}:
            return match.group(0)
        return f"new {base[:1].upper()}{base[1:]}[]{{}}"

    def _ensure_java_test_name(self, text: str, test_path: Path, graph: Any) -> str:
        source_stem = Path(graph.source_path).stem
        expected_class = f"{source_stem}Test"
        if re.search(r"\bclass\s+\w+\s*\{", text):
            return re.sub(r"\bclass\s+\w+\s*\{", f"class {expected_class} {{", text, count=1)
        package_name = str((graph.metadata.get("structured_context") or {}).get("package_or_module") or "")
        prefix = f"package {package_name};\n\n" if package_name else ""
        return prefix + "import org.junit.jupiter.api.Test;\n\nclass " + expected_class + " {\n}\n"

    def _java_private_constructor_classes(self, graph: Any) -> set[str]:
        classes: set[str] = set()
        context = graph.metadata.get("structured_context", {})
        for cls in context.get("classes", []) if isinstance(context, dict) else []:
            if not isinstance(cls, dict):
                continue
            constructors = cls.get("constructors", [])
            if constructors and all(isinstance(item, dict) and item.get("visibility") == "private" for item in constructors):
                classes.add(str(cls.get("name")))
        return {item for item in classes if item}

    def _java_llm_candidate(self, text: str, graph: Any, plan: Any, suggestions: list[Any]) -> str:
        if not self.llm or not self.llm.enabled():
            return ""
        source_path = Path(graph.source_path)
        try:
            source_text = source_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            source_text = ""
        prompt = f"""
You are the Java test optimization module inside ExecutionAgent.
Return one complete JUnit 5 test file only. Do not use Markdown.

Hard requirements:
- The test class name must be {source_path.stem}Test.
- Keep the package from the structured context.
- Prefer real assertions over assertDoesNotThrow when a return value or exception is observable.
- Do not instantiate classes whose constructor is private; call static methods statically.
- Generated Java must compile under Maven/JUnit 5.
- Target the planning focus and validation errors carried by the state flow.
- When a C-grade optimization contract is present, preserve every existing passing test and change at most one weak test method or add at most two focused test methods.
- Use concrete assertEquals/assertArrayEquals/assertThrows or relation assertions; do not add existence-only or smoke assertions.
- Directly target the listed missing lines, surviving mutation lines, target methods, or target states.
- Do not reduce any baseline coverage, assertion, mutation, SFC, or TSQ metric.

C-grade targeted optimization contract:
{graph.metadata.get("c_grade_optimization_contract")}

State positioning repair strategy:
{graph.metadata.get("state_positioning_repair") or graph.metadata.get("minimal_target_repair")}

Structured context:
{graph.metadata.get("structured_context")}

Planning decision:
{getattr(plan, "to_dict", lambda: {})()}

Last Java execution:
{graph.metadata.get("last_java_execution")}

Suggestions:
{[item.to_dict() if hasattr(item, "to_dict") else item for item in suggestions[:20]]}

Source:
```java
{source_text}
```

Current test:
```java
{text}
```
""".strip()
        try:
            raw = self.llm.chat(
                [
                    {"role": "system", "content": "Return only a complete JUnit 5 Java test file."},
                    {"role": "user", "content": prompt},
                ]
            )
            return self._extract_java_code(raw)
        except Exception as exc:
            graph.metadata["java_llm_repair_error"] = str(exc)
            if self.llm_required:
                raise
            return ""

    def _extract_java_code(self, raw: str) -> str:
        match = re.search(r"```(?:java)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
        return (match.group(1) if match else raw).strip()

    def _run_pytest_with_retries(self, test_path: str | Path, cwd: str | Path | None = None):
        attempts = 0
        timeout = 30
        result = None
        for retry in range(self.retry_rounds + 1):
            attempts += 1
            result = self.pytest_runner.run(test_path, cwd=cwd, timeout=timeout)
            if not result.timed_out:
                break
            timeout = max(timeout + 10, timeout * 2)
        assert result is not None
        return result, attempts

    def _repair_candidate_order(self, selected: str) -> list[str]:
        order = [selected]
        if selected == "TestValidityAgent":
            order.extend(["OracleRepairAgent", "RepairAgent"])
        elif selected == "OracleRepairAgent":
            order.extend(["TestValidityAgent", "RepairAgent"])
        elif selected in {"CoverageAgent", "AssertionAgent", "MutationAgent", "BoundaryAgent", "TypeAnalysisAgent"}:
            order.append("RepairAgent")
        else:
            order.extend(["TestValidityAgent", "OracleRepairAgent", "RepairAgent"])
        deduped: list[str] = []
        for item in order:
            if item and item not in deduped:
                deduped.append(item)
        return deduped

    def _file_hash(self, path: str | Path) -> str:
        file_path = Path(path)
        if not file_path.exists():
            return "missing"
        import hashlib

        return hashlib.sha256(file_path.read_bytes()).hexdigest()

    def _record_exchange(self, graph, artifact_type: str, payload: dict[str, Any]) -> None:
        exchange = graph.metadata.setdefault("agent_collaboration_exchange", [])
        if isinstance(exchange, list):
            exchange.append(
                {
                    "sender": "ExecutionAgent",
                    "artifact_type": artifact_type,
                    "payload": payload,
                }
            )

    def _allowed_agents_for_failure(self, failure_category: str) -> list[str]:
        if failure_category == "oracle_error":
            return ["OracleRepairAgent", "TestValidityAgent"]
        if failure_category == "timeout":
            return ["RepairAgent", "TestValidityAgent", "OracleRepairAgent"]
        return ["TestValidityAgent"]

    def _recommended_focus_for_failure(self, diagnosis: Any) -> str:
        failure_category = str(getattr(diagnosis, "failure_category", "runtime_error"))
        if failure_category == "oracle_error":
            return "oracle"
        if failure_category == "timeout":
            if not bool(getattr(diagnosis, "should_repair", True)):
                return "external_dependency_timeout"
            return "timeout_minimal_repair"
        return "test_validity"


def _is_connection_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "connection" in text or "not installed" in text or "http" in text
