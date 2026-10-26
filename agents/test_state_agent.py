from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.evaluation_agent import EvaluationFeedback, TestStateEvaluationModule
from agents.execution_agent import ExecutionAgent, ExecutionReport
from agents.planning_agent import PlanningAgent, TestIntentPlan
from core.state_flow_event import StateFlowEvent, append_state_flow_event
from graph.experience_memory import AgentExperienceMemory
from memory.shared_graph import SharedGraphMemory


@dataclass
class TestStateHealth:
    valid: bool
    issues: list[str]
    node_count: int
    edge_count: int
    language: str

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "issues": self.issues,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "language": self.language,
        }


class TestStateAgent:
    name = "测试状态智能体"

    def __init__(
        self,
        memory: SharedGraphMemory,
        planning: PlanningAgent,
        evaluation: TestStateEvaluationModule,
        execution: ExecutionAgent,
    ):
        self.memory = memory
        self.planning = planning
        self.evaluation = evaluation
        self.execution = execution

    def bootstrap(
        self,
        source_path: str | Path,
        output_dir: str | Path,
        experience: AgentExperienceMemory,
        model: str | None = None,
    ) -> tuple[Path, TestIntentPlan, TestStateHealth]:
        test_path, plan = self.planning.bootstrap(
            source_path,
            output_dir,
            experience,
            model=model,
            execution_agent=self.execution,
        )
        health = self.inspect()
        graph = self.memory.load()
        graph.metadata["collaboration_model"] = {
            "runtime": "LangGraph StateGraph",
            "primary_agents": ["TestStateAgent", "TestGenerationAgent", "TestKnowledgeAgent"],
            "primary_agent_names": ["测试状态智能体", "测试生成智能体", "测试知识智能体"],
            "state_owner": "TestStateAgent",
            "evaluation_owner": "TestStateAgent",
            "test_artifact_owner": "TestGenerationAgent",
            "knowledge_owner": "TestKnowledgeAgent",
            "non_linear_routes": ["state_self_repair", "knowledge_retrieval", "test_generation", "test_repair", "state_evaluation", "stop"],
        }
        graph.metadata["last_test_state_health"] = health.to_dict()
        self.memory.save(graph)
        return test_path, plan, health

    def inspect(self) -> TestStateHealth:
        issues: list[str] = []
        try:
            graph = self.memory.load()
        except (FileNotFoundError, ValueError, TypeError):
            return TestStateHealth(False, ["test_state_missing_or_unreadable"], 0, 0, "unknown")
        language = str(graph.metadata.get("language") or "")
        context = graph.metadata.get("structured_context")
        if language not in {"python", "java"}:
            issues.append("language_missing_or_unsupported")
        if not isinstance(context, dict):
            issues.append("structured_context_missing")
        if not graph.nodes:
            issues.append("test_state_nodes_missing")
        testable = [node for node in graph.nodes.values() if node.node_type in {"Function", "Async", "Method", "Constructor"} and node.metadata.get("visibility", "public") != "private"]
        if not testable:
            issues.append("testable_targets_missing")
        if not graph.flow_history:
            issues.append("test_state_events_missing")
        return TestStateHealth(not issues, issues, len(graph.nodes), len(graph.edges), language or "unknown")

    def self_repair(self, source_path: str | Path, reason: str) -> TestStateHealth:
        preserved: dict[str, Any] = {}
        try:
            old_graph = self.memory.load()
            for key in ("generated_test_path", "experience_hints", "last_knowledge_context", "llm_generation_model", "llm_execution_model"):
                if key in old_graph.metadata:
                    preserved[key] = old_graph.metadata[key]
        except (FileNotFoundError, ValueError, TypeError):
            pass
        graph = self.execution.analysis_agent.run(source_path)
        graph.metadata.update(preserved)
        health = self.inspect()
        graph = self.memory.load()
        graph.metadata["test_state_self_repair"] = {
            "reason": reason,
            "health": health.to_dict(),
            "action": "rebuild_from_source",
        }
        append_state_flow_event(
            graph,
            StateFlowEvent(
                event_type="TestStateRepaired" if health.valid else "TestStateRepairFailed",
                source_agent=self.name,
                target_agent=self.name,
                payload=graph.metadata["test_state_self_repair"],
                iteration=graph.iteration,
            ),
        )
        graph.bump_version("test state self repair completed")
        self.memory.save(graph)
        return health

    def evaluate_and_plan(
        self,
        report: ExecutionReport,
        thresholds: object,
        experience: AgentExperienceMemory,
        pending_decision: dict[str, object] | None = None,
        pending_before_report: ExecutionReport | None = None,
        pending_runtime: float = 0.0,
        pending_usage: dict[str, int | float] | None = None,
        model: str | None = None,
    ) -> tuple[EvaluationFeedback, Any, TestIntentPlan, TestStateHealth]:
        health = self.inspect()
        if not health.valid:
            raise RuntimeError("test_state_invalid:" + ",".join(health.issues))
        assessment = getattr(thresholds, "assess")(report)
        report.quality_assessment = assessment
        report.quality_status = str(assessment["status"])
        report.weak_quality_flags = getattr(thresholds, "weak_quality_flags")(report)
        report.threshold_failures = getattr(thresholds, "threshold_failures")(report)
        feedback = self.evaluation.run_feedback(
            report=report,
            thresholds=thresholds,
            experience=experience,
            pending_decision=pending_decision,
            pending_before_report=pending_before_report,
            pending_runtime=pending_runtime,
            pending_usage=pending_usage,
            model=model,
        )
        decision, plan = self.planning.replan(
            feedback=feedback,
            report=report,
            thresholds=thresholds,
            experience=experience,
            model=model,
        )
        graph = self.memory.load()
        graph.metadata["last_test_state_health"] = health.to_dict()
        graph.metadata["last_test_state_assignment"] = {
            "assigned_agent": plan.assigned_agent,
            "capability": plan.capability,
            "action": plan.action,
            "target_states": plan.target_states or [],
            "reason": plan.reason,
        }
        self.memory.save(graph)
        return feedback, decision, plan, health
