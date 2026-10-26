from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.execution_agent import ExecutionAgent, ExecutionReport
from core.state_flow_event import StateFlowEvent, append_state_flow_event
from graph.completeness_metric import CompleteStateGraphMetric
from validation.dynamic_validator import DynamicValidator


class TestGenerationCapabilityAgent:
    """测试生成智能体：统一生成、验证、修复和优化能力。"""

    __test__ = False
    name = "测试生成智能体"

    def __init__(self, execution: ExecutionAgent):
        self.execution = execution
        self.dynamic_validator = DynamicValidator()
        self.completeness_metric = CompleteStateGraphMetric()

    def execute(
        self,
        test_path: str | Path,
        cwd: str | Path,
        *,
        evaluation_profile: str = "full",
    ) -> ExecutionReport:
        return self.execution.run(test_path, cwd=cwd, evaluation_profile=evaluation_profile)

    def validate(self, report: ExecutionReport) -> dict[str, object]:
        """验证执行产物并把证据交给测试状态智能体评估。"""
        graph = self.execution.memory.load()
        validation_report = self.dynamic_validator.validate(
            graph=graph,
            test_code=getattr(report, "test_path", None),
            context=graph.metadata.get("structured_context", {}),
        )
        completeness = self.completeness_metric.calculate(graph, validation_report=validation_report)
        feasible = bool(report.pytest_passed and report.collected_count > 0 and not report.timed_out)
        accepted = bool(feasible and validation_report.passed)
        graph.metadata["last_validation"] = validation_report.to_dict()
        graph.metadata["unresolved_validation_errors"] = [item.to_dict() for item in validation_report.errors]
        graph.metadata["complete_state_graph"] = completeness.to_dict()
        graph.metadata["test_generation_validation"] = {
            **(graph.metadata.get("test_generation_validation", {}) if isinstance(graph.metadata.get("test_generation_validation"), dict) else {}),
            "accepted": accepted,
            "status": "accepted" if accepted else "repair_required",
            "stage": "compiled_executed_and_intent_validated" if accepted else "execution_or_intent_validation_failed",
            "feasible": feasible,
            "dynamic_validation": validation_report.to_dict(),
            "test_path": str(Path(report.test_path).resolve()),
        }
        append_state_flow_event(
            graph,
            StateFlowEvent(
                event_type="ValidationPassed" if validation_report.passed else "ValidationFailed",
                source_agent=self.name,
                target_agent="测试状态智能体",
                related_nodes=[node for error in validation_report.errors for node in error.related_nodes],
                related_edges=[edge for error in validation_report.errors for edge in error.related_edges],
                payload={"validation": validation_report.to_dict(), "complete_state_graph": completeness.to_dict()},
                iteration=graph.iteration,
            ),
        )
        graph.bump_version("test generation validation completed")
        self.execution.memory.save(graph)
        return {
            "accepted": accepted,
            "feasible": feasible,
            "validation": validation_report.to_dict(),
            "complete_state_graph": completeness.to_dict(),
        }

    def apply_plan(
        self,
        test_path: Path,
        plan: object,
        repair_agent: object,
        specialized_agents: dict[str, object],
        suggestions: list[object] | None = None,
    ) -> bool:
        return self.execution.apply_plan(
            test_path=test_path,
            plan=plan,
            repair_agent=repair_agent,
            specialized_agents=specialized_agents,
            evaluation_agent=None,
            suggestions=suggestions,
        )
