from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from agents.evaluation_agent import TestStateEvaluationModule
from agents.execution_agent import ExecutionAgent, ExecutionReport
from agents.planning_agent import PlanningAgent, TestIntentPlan
from agents.repair_agent import RepairOptimizationAgent
from agents.specialized_agents import (
    AssertionStrengtheningAgent,
    BoundaryExceptionAgent,
    CoverageEnhancementAgent,
    MutationRepairAgent,
    OracleRepairAgent,
    RuntimeTypeAgent,
    TestValidityAgent,
)
from agents.test_knowledge_agent import TestKnowledgeAgent
from agents.test_generation_capability_agent import TestGenerationCapabilityAgent
from agents.test_state_agent import TestStateAgent
from core.public_metrics import public_result
from graph.experience_memory import AgentExperienceMemory, configure_knowledge_store_path, resolve_knowledge_store_path
from llm_client import OpenAICompatibleLLM
from llm_config import LLMConfig
from memory.shared_graph import SharedGraphMemory
from orchestration.test_workflow import LangGraphTestWorkflow


@dataclass
class Thresholds:
    line_coverage: float = 0.80
    branch_coverage: float = 0.70
    sfc: float = 0.65
    mutation: float = 0.60
    sfq: float = 0.70
    ae: float | None = 0.75
    high_line_coverage: float = 0.90
    high_branch_coverage: float = 0.80
    high_sfc: float = 0.75
    high_mutation: float = 0.75
    high_sfq: float = 0.82
    minimum_reliable_mutants: int = 3
    min_quality_delta: float = 0.005

    def reached(self, report: ExecutionReport) -> bool:
        return self.classify(report) == "quality_high"

    def usable(self, report: ExecutionReport) -> bool:
        return bool(
            report.pytest_passed
            and report.collected_count > 0
            and not report.timed_out
            and report.coverage_valid
            and report.normal_behavior_tests > 0
            and report.effective_assertions > 0
            and not report.weak_assertions_only
            and not report.exception_only
        )

    def mutation_applicable(self, report: ExecutionReport) -> bool:
        return bool(
            report.mutation_complete
            and
            report.mutation_score is not None
            and report.effective_mutants >= self.minimum_reliable_mutants
            and report.mutation_reliability == "normal"
        )

    def classify(self, report: ExecutionReport) -> str:
        return str(self.assess(report)["status"])

    def assess(self, report: ExecutionReport) -> dict[str, object]:
        if not self.usable(report):
            return {
                "status": "execution_invalid",
                "feasible": False,
                "objective": "first make the test compile, run, express normal behavior and contain effective assertions",
                "dimensions": {},
                "evidence_confidence": 0.0,
                "rationale": ["测试尚未形成可信的可执行结果"],
            }
        branch_acceptable = report.branch_coverage is None or report.branch_coverage >= self.branch_coverage
        branch_high = report.branch_coverage is None or report.branch_coverage >= max(self.high_branch_coverage, self.branch_coverage)
        mutation_available = self.mutation_applicable(report)
        mutation_acceptable = mutation_available and float(report.mutation_score) >= self.mutation
        mutation_high = mutation_available and float(report.mutation_score) >= max(self.high_mutation, self.mutation)
        ae_acceptable = self.ae is None or report.ae >= self.ae
        coverage_acceptable = report.line_coverage >= self.line_coverage and branch_acceptable and report.sfc >= self.sfc
        coverage_high = (
            report.line_coverage >= max(self.high_line_coverage, self.line_coverage)
            and branch_high
            and report.sfc >= max(self.high_sfc, self.sfc)
        )
        assertion_high = ae_acceptable and report.effective_assertions > 0 and not report.weak_assertions_only
        dimensions: dict[str, dict[str, object]] = {
            "coverage": {"available": True, "acceptable": coverage_acceptable, "high": coverage_high, "line": report.line_coverage, "branch": report.branch_coverage, "sfc": report.sfc},
            "assertion": {"available": True, "acceptable": ae_acceptable, "high": assertion_high, "ae": report.ae, "effective_assertions": report.effective_assertions},
            "mutation": {"available": mutation_available, "acceptable": mutation_acceptable, "high": mutation_high, "score": report.mutation_score, "effective_mutants": report.effective_mutants, "reliability": report.mutation_reliability},
        }
        available = [item for item in dimensions.values() if item["available"]]
        acceptable_count = sum(bool(item["acceptable"]) for item in available)
        high_count = sum(bool(item["high"]) for item in available)
        critical_weaknesses = []
        if report.line_coverage < 0.5 or report.sfc < 0.4:
            critical_weaknesses.append("coverage")
        if report.ae < 0.4:
            critical_weaknesses.append("assertion")
        if mutation_available and float(report.mutation_score or 0.0) < 0.3:
            critical_weaknesses.append("mutation")
        status = "valid_usable"
        if (
            report.sfq >= max(self.high_sfq, self.sfq)
            and high_count == len(available)
            and report.mutation_valid
            and not critical_weaknesses
        ):
            status = "quality_high"
        # B means that the suite is genuinely good, not merely that its weighted
        # aggregate hides one severe weakness.  In particular, a reliable mutation
        # score below 0.30 must remain C so that the dataset-level R3 optimization is
        # actually allowed to strengthen it.
        elif report.sfq >= self.sfq and acceptable_count >= 2 and not critical_weaknesses:
            status = "quality_acceptable"
        evidence_confidence = (2.0 + (1.0 if mutation_available else 0.0)) / 3.0
        rationale = [
            "测试可编译/运行，包含正常行为和有效断言",
            f"{acceptable_count}个可用核心质量维度达到可接受标准",
        ]
        if not mutation_available:
            rationale.append("可靠变异样本不足，等级不声称具有高变异检测证据")
        if critical_weaknesses:
            rationale.append("仍有明显弱项：" + "、".join(critical_weaknesses))
        return {
            "status": status,
            "feasible": True,
            "objective": "under executable test intent, maximize useful quality while controlling unnecessary test and iteration cost",
            "dimensions": dimensions,
            "evidence_confidence": round(evidence_confidence, 4),
            "assertion_signal_ratio": round(report.effective_assertions / max(1, report.assertion_count), 4),
            "acceptable_dimension_count": acceptable_count,
            "high_dimension_count": high_count,
            "critical_weaknesses": critical_weaknesses,
            "rationale": rationale,
        }

    def weak_quality_flags(self, report: ExecutionReport) -> list[str]:
        flags: list[str] = []
        if report.exception_only or report.normal_behavior_tests <= 0:
            flags.append("weak_normal_behavior")
        if not report.coverage_valid:
            flags.append("coverage_unavailable")
        elif report.line_coverage < self.line_coverage:
            flags.append("weak_line_coverage")
        if report.branch_coverage is not None and report.branch_coverage < self.branch_coverage:
            flags.append("weak_branch_coverage")
        if report.sfc < self.sfc:
            flags.append("weak_state_flow_coverage")
        if self.ae is not None and report.ae < self.ae:
            flags.append("weak_assertion_effectiveness")
        if report.weak_assertions_only or report.effective_assertions <= 0:
            flags.append("weak_assertions")
        if not report.mutation_valid:
            flags.append("mutation_unavailable")
        elif not self.mutation_applicable(report):
            flags.append("mutation_evidence_limited")
        elif self.mutation_applicable(report) and report.mutation_score is not None and report.mutation_score < self.mutation:
            flags.append("weak_mutation_score")
        if report.sfq < self.sfq:
            flags.append("weak_state_flow_quality")
        return sorted(set(flags))

    def threshold_failures(self, report: ExecutionReport) -> dict[str, float]:
        failures: dict[str, float] = {}
        if report.coverage_valid and report.line_coverage < self.line_coverage:
            failures["line_coverage_gap"] = round(self.line_coverage - report.line_coverage, 4)
        if report.branch_coverage is not None and report.branch_coverage < self.branch_coverage:
            failures["branch_coverage_gap"] = round(self.branch_coverage - report.branch_coverage, 4)
        if report.sfc < self.sfc:
            failures["sfc_gap"] = round(self.sfc - report.sfc, 4)
        if self.ae is not None and report.ae < self.ae:
            failures["ae_gap"] = round(self.ae - report.ae, 4)
        if self.mutation_applicable(report) and report.mutation_score is not None and report.mutation_score < self.mutation:
            failures["mutation_gap"] = round(self.mutation - report.mutation_score, 4)
        if report.sfq < self.sfq:
            failures["sfq_gap"] = round(self.sfq - report.sfq, 4)
        return failures

    def to_dict(self) -> dict[str, object]:
        return {
            "acceptable": {
                "line_coverage": self.line_coverage,
                "branch_coverage": self.branch_coverage,
                "sfc": self.sfc,
                "mutation": self.mutation,
                "sfq": self.sfq,
                "ae": self.ae,
            },
            "high": {
                "line_coverage": self.high_line_coverage,
                "branch_coverage": self.high_branch_coverage,
                "sfc": self.high_sfc,
                "mutation": self.high_mutation,
                "sfq": self.high_sfq,
            },
            "minimum_reliable_mutants": self.minimum_reliable_mutants,
            "min_quality_delta": self.min_quality_delta,
        }


class MultiAgentUnitTestSystem:
    def __init__(
        self,
        source_path: str | Path,
        output_dir: str | Path,
        graph_path: str | Path | None = None,
        thresholds: Thresholds | None = None,
        max_iterations: int = 3,
        retry_rounds: int = 2,
        llm_config: LLMConfig | None = None,
        llm_client: OpenAICompatibleLLM | None = None,
        experience_path: str | Path | None = None,
        experience_read_path: str | Path | None = None,
        experience_journal_dir: str | Path | None = None,
        execution_budget: int | None = None,
        llm_call_budget: int | None = None,
        token_budget: int | None = None,
        time_budget_seconds: float | None = None,
        resume: bool = True,
        rerun_existing: bool = False,
        experiment_round: int = 1,
        followup_strategy: str = "initial_generation",
        baseline_report: dict[str, object] | None = None,
        existing_test_path: str | Path | None = None,
        previous_grade: str | None = None,
        isolate_humaneval_java: bool = False,
    ):
        self.source_path = Path(source_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        task_seed = f"{self.source_path}|{self.output_dir}"
        if rerun_existing:
            task_seed += f"|rerun|{time.time_ns()}"
        self.task_id = hashlib.sha256(task_seed.encode("utf-8")).hexdigest()[:24]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.graph_path = Path(graph_path or self.output_dir / "state_flow_graph.json").resolve()
        self.memory = SharedGraphMemory(self.graph_path)
        self.thresholds = thresholds or Thresholds()
        self.max_repair_rounds = max_iterations
        self.retry_rounds = retry_rounds
        self.execution_budget = execution_budget
        self.llm_call_budget = llm_call_budget
        self.token_budget = token_budget
        self.time_budget_seconds = time_budget_seconds
        self.resume = resume
        self.rerun_existing = rerun_existing
        self.experiment_round = max(1, int(experiment_round or 1))
        self.followup_strategy = followup_strategy
        self.baseline_report = dict(baseline_report or {})
        self.existing_test_path = Path(existing_test_path).resolve() if existing_test_path else None
        self.previous_grade = previous_grade
        self.experience = AgentExperienceMemory(
            experience_path or resolve_knowledge_store_path(),
            read_base_path=experience_read_path,
            journal_dir=experience_journal_dir,
            publish_canonical=experience_read_path is None,
        )
        self.llm_config = llm_config or LLMConfig.from_env()
        self.llm = llm_client if llm_client is not None else OpenAICompatibleLLM(self.llm_config) if self.llm_config.enabled else None
        self.planning_agent = PlanningAgent(self.memory, llm=self.llm, llm_required=self.llm_config.required)
        self.execution_agent = ExecutionAgent(self.memory, llm=self.llm, llm_required=self.llm_config.required, retry_rounds=retry_rounds, isolate_humaneval_java=isolate_humaneval_java)
        self.evaluation_agent = TestStateEvaluationModule(self.memory, llm=self.llm, llm_required=self.llm_config.required)
        self.test_generation_agent = TestGenerationCapabilityAgent(self.execution_agent)
        self.test_state_agent = TestStateAgent(self.memory, self.planning_agent, self.evaluation_agent, self.execution_agent)
        self.test_knowledge_agent = TestKnowledgeAgent(self.memory, self.experience)
        self.repair_agent = RepairOptimizationAgent(self.memory, llm=self.llm, llm_required=self.llm_config.required)
        self.specialized_agents = {
            "CoverageAgent": CoverageEnhancementAgent(self.memory, llm=self.llm, llm_required=self.llm_config.required),
            "AssertionAgent": AssertionStrengtheningAgent(self.memory, llm=self.llm, llm_required=self.llm_config.required),
            "MutationAgent": MutationRepairAgent(self.memory, llm=self.llm, llm_required=self.llm_config.required),
            "BoundaryAgent": BoundaryExceptionAgent(self.memory, llm=self.llm, llm_required=self.llm_config.required),
            "TypeAnalysisAgent": RuntimeTypeAgent(self.memory, llm=self.llm, llm_required=self.llm_config.required),
            "TestValidityAgent": TestValidityAgent(self.memory, llm=self.llm, llm_required=self.llm_config.required),
            "OracleRepairAgent": OracleRepairAgent(self.memory, llm=self.llm, llm_required=self.llm_config.required),
        }

    def run(self) -> dict[str, object]:
        existing_summary = self._load_existing_summary()
        if existing_summary is not None:
            return public_result(existing_summary)
        return public_result(LangGraphTestWorkflow(self).run())

    def _load_existing_summary(self) -> dict[str, object] | None:
        if not self.resume or self.rerun_existing:
            return None
        summary_path = self.output_dir / "stateflow_summary.json"
        if not summary_path.exists():
            return None
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if str(Path(str(summary.get("source", ""))).resolve()) != str(self.source_path):
            return None
        if summary.get("source_sha256") and summary["source_sha256"] != hashlib.sha256(self.source_path.read_bytes()).hexdigest():
            return None
        if not self._summary_is_reusable(summary):
            return None
        public_summary = public_result(summary)
        if public_summary != summary:
            temporary_path = summary_path.with_name(f".{summary_path.name}.public.tmp")
            temporary_path.write_text(json.dumps(public_summary, indent=2, ensure_ascii=False), encoding="utf-8")
            temporary_path.replace(summary_path)
        summary = public_summary
        summary["resumed_existing"] = True
        return summary

    def _summary_is_reusable(self, summary: dict[str, object]) -> bool:
        final_report = summary.get("final_report") if isinstance(summary.get("final_report"), dict) else None
        if not final_report or not final_report.get("pytest_passed"):
            return False
        stop_reason = str(summary.get("stop_reason") or "")
        quality_status = str(summary.get("quality_status") or final_report.get("quality_status") or stop_reason)
        if stop_reason in {"plateau_reached", "metric_unreliable", "budget_limited"}:
            return False
        return quality_status in {"quality_high", "quality_acceptable", "valid_usable"}

    def _usage_snapshot(self) -> dict[str, int | float]:
        if self.llm is None:
            return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "duration_seconds": 0.0}
        snapshot = getattr(self.llm, "usage_snapshot", None)
        if callable(snapshot):
            return dict(snapshot())
        calls = getattr(self.llm, "calls", [])
        return {"calls": len(calls) if isinstance(calls, list) else 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "duration_seconds": 0.0}

    def _usage_delta(self, before: dict[str, int | float], after: dict[str, int | float]) -> dict[str, int | float]:
        return {key: round(float(after.get(key, 0)) - float(before.get(key, 0)), 4) for key in set(before) | set(after)}

    def _budget_reason(self, next_iteration: int, started: float) -> str | None:
        usage = self._usage_snapshot()
        if self.execution_budget is not None and next_iteration > self.execution_budget:
            return "budget_limited"
        if self.llm_call_budget is not None and int(usage.get("calls", 0)) >= self.llm_call_budget:
            return "budget_limited"
        if self.token_budget is not None and int(usage.get("total_tokens", 0)) >= self.token_budget:
            return "budget_limited"
        if self.time_budget_seconds is not None and time.perf_counter() - started >= self.time_budget_seconds:
            return "budget_limited"
        return None

    def _metric_delta(self, before: ExecutionReport | None, after: ExecutionReport) -> dict[str, float]:
        if before is None:
            return {key: 0.0 for key in ["line_coverage", "branch_coverage", "combined_coverage", "mutation_score", "sfc", "ae", "sfq"]}
        result: dict[str, float] = {}
        for key in ["line_coverage", "branch_coverage", "combined_coverage", "mutation_score", "sfc", "ae", "sfq"]:
            before_value = getattr(before, key, None)
            after_value = getattr(after, key, None)
            result[key] = round(float(after_value or 0.0) - float(before_value or 0.0), 4)
        return result

    def _file_hash(self, path: str | Path) -> str:
        file_path = Path(path)
        return hashlib.sha256(file_path.read_bytes()).hexdigest() if file_path.exists() else "missing"

    def _render_visualizations(self, graph) -> dict[str, object]:
        try:
            from visualization.graph_visualizer import StateFlowGraphVisualizer

            result = StateFlowGraphVisualizer().render(self.graph_path, self.output_dir)
            graph.metadata["visualization"] = result
            self.memory.save(graph)
            return result
        except Exception as exc:
            result = {"status": "visualization_failed", "error": f"{exc.__class__.__name__}: {exc}"}
            graph.metadata["visualization"] = result
            self.memory.save(graph)
            return result

    def _run_selected_agent(
        self,
        agent_name: str,
        test_path: Path,
        intent_plan: TestIntentPlan | None = None,
        feedback=None,
    ) -> bool:
        plan = intent_plan or type("Plan", (), {"selected_agent": agent_name, "action": "repair"})()
        suggestions = getattr(feedback, "suggestions", None)
        return self.test_generation_agent.apply_plan(
            test_path=test_path,
            plan=plan,
            repair_agent=self.repair_agent,
            specialized_agents=self.specialized_agents,
            suggestions=suggestions,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangGraph-based test-state-guided Python/Java unit test generation system.")
    parser.add_argument("--source", required=True, help="Python or Java source file to analyze.")
    parser.add_argument("--out", default="generated_tests", help="Output directory for generated tests and graph.")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum repair iterations.")
    parser.add_argument("--retry-rounds", type=int, default=2, help="Maximum pytest timeout retry rounds before repair.")
    parser.add_argument("--line-coverage", type=float, default=0.80, help="Acceptable line coverage threshold.")
    parser.add_argument("--branch-coverage", type=float, default=0.70, help="Acceptable branch coverage threshold.")
    parser.add_argument("--sfc", type=float, default=0.65, help="Acceptable test-state coverage threshold.")
    parser.add_argument("--ae", type=float, default=0.75, help="Minimum assertion-effectiveness threshold.")
    parser.add_argument("--mutation", type=float, default=0.60, help="Acceptable reliable mutation score threshold.")
    parser.add_argument("--tsq", "--sfq", dest="sfq", type=float, default=0.70, help="Acceptable Test State Quality threshold; --sfq is retained as a compatibility alias.")
    parser.add_argument("--execution-budget", type=int, default=None, help="Maximum number of execution iterations before budget_exhausted.")
    parser.add_argument("--llm-call-budget", type=int, default=None, help="Maximum LLM calls for one task.")
    parser.add_argument("--token-budget", type=int, default=None, help="Maximum recorded/estimated tokens for one task.")
    parser.add_argument("--time-budget-seconds", type=float, default=None, help="Maximum wall-clock seconds for one task.")
    parser.add_argument("--no-resume", action="store_true", help="Do not return an existing completed summary from the output directory.")
    parser.add_argument("--rerun-existing", action="store_true", help="Force rerunning even if stateflow_summary.json already exists.")
    parser.add_argument("--full-json", action="store_true", help="Print the full JSON summary instead of the concise terminal summary.")
    parser.add_argument("--llm", action="store_true", help="Use the configured OpenAI-compatible LLM for generation and repair.")
    parser.add_argument("--llm-required", action="store_true", help="Fail instead of falling back if the LLM call fails.")
    parser.add_argument("--llm-base-url", default=None, help="LLM host URL. For Ollama, use http://127.0.0.1:11434.")
    parser.add_argument("--llm-model", default=None, help="Model name, such as deepseek-r1:14b.")
    parser.add_argument("--llm-api-key", default=None, help="API key. For many local servers, any non-empty value is fine.")
    parser.add_argument("--llm-timeout", type=int, default=None, help="LLM request timeout in seconds.")
    parser.add_argument("--llm-temperature", type=float, default=None, help="LLM sampling temperature.")
    parser.add_argument("--knowledge-store", default=None, help="External long-term knowledge JSON path. Existing files are never overwritten by project defaults.")
    parser.add_argument("--no-ui", action="store_true", help="Do not start or open the real-time experiment dashboard.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from monitoring import finish_dashboard, launch_dashboard

    configure_knowledge_store_path(args.knowledge_store)
    llm_config = LLMConfig.from_env(
        enabled=args.llm,
        required=args.llm_required,
        base_url=args.llm_base_url,
        model=args.llm_model,
        api_key=args.llm_api_key,
        timeout=args.llm_timeout,
        temperature=args.llm_temperature,
    )
    launch_dashboard(args.out, label=Path(args.source).name, enabled=not args.no_ui, llm_config=llm_config)
    failure = None
    try:
        system = MultiAgentUnitTestSystem(
            source_path=args.source,
            output_dir=args.out,
            thresholds=Thresholds(
                line_coverage=args.line_coverage,
                branch_coverage=args.branch_coverage,
                sfc=args.sfc,
                ae=args.ae,
                mutation=args.mutation,
                sfq=args.sfq,
            ),
            max_iterations=args.max_iterations,
            retry_rounds=args.retry_rounds,
            execution_budget=args.execution_budget,
            llm_call_budget=args.llm_call_budget,
            token_budget=args.token_budget,
            time_budget_seconds=args.time_budget_seconds,
            resume=not args.no_resume,
            rerun_existing=args.rerun_existing,
            llm_config=llm_config,
        )
        summary = system.run()
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        finish_dashboard(args.out, success=failure is None, error=failure, enabled=not args.no_ui)
    print(json.dumps(public_result(summary if args.full_json else _terminal_summary(summary)), indent=2, ensure_ascii=False))


def _terminal_summary(summary: dict[str, object]) -> dict[str, object]:
    report = summary.get("final_report") if isinstance(summary.get("final_report"), dict) else {}
    return public_result({
        "source": summary.get("source"),
        "test_path": summary.get("test_path"),
        "graph_path": summary.get("graph_path"),
        "tsm_path": summary.get("tsm_path", summary.get("graph_path")),
        "summary_path": str(Path(str(summary.get("graph_path", ""))).with_name("stateflow_summary.json")) if summary.get("graph_path") else None,
        "resumed_existing": summary.get("resumed_existing", False),
        "iterations": summary.get("iterations"),
        "repair_rounds_used": summary.get("repair_rounds_used"),
        "stop_reason": summary.get("stop_reason"),
        "quality_status": summary.get("quality_status"),
        "metrics": {
            "pytest_passed": report.get("pytest_passed"),
            "line_coverage": report.get("line_coverage"),
            "branch_coverage": report.get("branch_coverage"),
            "assertion_effectiveness": report.get("ae"),
            "mutation_score": report.get("mutation_score"),
            "assertion_count": report.get("assertion_count"),
            "effective_mutants": report.get("effective_mutants"),
        },
        "visualization": summary.get("visualization"),
        "llm_usage": summary.get("llm_usage"),
    })


if __name__ == "__main__":
    main()
