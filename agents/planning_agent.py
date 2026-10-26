from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.evaluation_agent import EvaluationFeedback
from agents.execution_agent import ExecutionAgent, ExecutionReport
from core.state_flow_event import StateFlowEvent, append_state_flow_event
from graph.adaptive_scheduler import AgentDecision, GraphDrivenAdaptiveScheduler
from graph.experience_memory import AgentExperienceMemory
from llm_client import OpenAICompatibleLLM
from memory.shared_graph import SharedGraphMemory
from metrics.measurement import CURRENT


@dataclass
class TestIntentPlan:
    phase: str
    action: str
    owner: str
    selected_agent: str
    reason: str
    gaps: dict[str, float]
    test_path: str | None = None
    focus: str = "initial_generation"
    feedback_summary: dict[str, Any] | None = None
    target_states: list[str] | None = None
    target_methods: list[str] | None = None
    target_branches: list[str] | None = None
    expected_gain: float = 0.0
    assigned_agent: str = "测试生成智能体"
    capability: str = "initial_generation"

    def to_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "action": self.action,
            "owner": self.owner,
            "selected_agent": self.selected_agent,
            "reason": self.reason,
            "gaps": self.gaps,
            "test_path": self.test_path,
            "focus": self.focus,
            "feedback_summary": self.feedback_summary or {},
            "target_states": self.target_states or [],
            "target_methods": self.target_methods or [],
            "target_branches": self.target_branches or [],
            "expected_gain": self.expected_gain,
            "assigned_agent": self.assigned_agent,
            "capability": self.capability,
        }


class PlanningAgent:
    name = "测试状态智能体/规划模块"

    def __init__(
        self,
        memory: SharedGraphMemory,
        llm: OpenAICompatibleLLM | None = None,
        llm_required: bool = False,
    ):
        self.memory = memory
        self.llm = llm
        self.llm_required = llm_required

    def bootstrap(
        self,
        source_path: str | Path,
        output_dir: str | Path,
        experience: AgentExperienceMemory,
        model: str | None = None,
        execution_agent: ExecutionAgent | None = None,
    ) -> tuple[Path, TestIntentPlan]:
        executor = execution_agent or ExecutionAgent(self.memory, llm=self.llm, llm_required=self.llm_required)
        try:
            graph = self.memory.load()
        except (FileNotFoundError, ValueError, TypeError):
            graph = executor.analyze_for_generation(source_path, experience, model=model)
        self._record_exchange(
            graph,
            "TestStateAgent",
            "TestIntentPlan",
            {
                "phase": "initial_orchestration",
                "action": "build_graph_and_generate_tests",
                "selected_agent": "TestGenerationAgent",
                "execution_internal_modules": ["StructuredContextExtractor", "ContextRewriter", "StructuredGraphBuilder", "TestGenerationModule"],
            },
        )
        targets = self._targets_from_graph(graph)
        plan = TestIntentPlan(
            phase="initial_test_intent",
            action="generate",
            owner="测试状态智能体",
            selected_agent="TestGenerationAgent",
            reason="initial test intent modeled from source analysis, shared TSG and knowledge retrieved before generation",
            gaps={},
            test_path=None,
            focus="normal_behavior_and_assertion_oracles",
            feedback_summary={"experience_hints": graph.metadata.get("experience_hints", {})},
            target_states=targets["states"],
            target_methods=targets["methods"],
            target_branches=targets["branches"],
            expected_gain=0.5,
            assigned_agent="测试生成智能体",
            capability="initial_generation",
        )
        graph.metadata["last_test_intent_plan"] = plan.to_dict()
        self._record_exchange(graph, "TestStateAgent", "TestIntentPlan", plan.to_dict())
        append_state_flow_event(
            graph,
            StateFlowEvent(
                event_type="TestIntentPlanned",
                source_agent="测试状态智能体",
                target_agent="测试生成智能体",
                related_nodes=plan.target_states or [],
                payload=plan.to_dict(),
                iteration=graph.iteration,
            ),
        )
        graph.bump_version("planning bootstrap completed")
        self.memory.save(graph)
        if CURRENT.get():
            CURRENT.get().begin_action(plan.to_dict(), graph, initial=True)
        test_path, generation_failed = executor.generate_initial_test(source_path, output_dir, experience, model=model)
        if generation_failed:
            plan.action = "repair"
            plan.selected_agent = "TestValidityAgent"
            plan.reason = "initial candidate was rejected; route the repairable evidence to validity repair"
            plan.focus = "validity_repair_plan"
            plan.capability = "validity_repair"
        plan.test_path = str(Path(test_path).resolve())
        graph = self.memory.load()
        graph.metadata["last_test_intent_plan"] = plan.to_dict()
        self.memory.save(graph)
        return Path(test_path), plan

    def replan(
        self,
        feedback: EvaluationFeedback,
        report: ExecutionReport,
        thresholds: object,
        experience: AgentExperienceMemory,
        model: str | None = None,
    ) -> tuple[AgentDecision, TestIntentPlan]:
        graph = self.memory.load()
        graph.metadata["experience_hints"] = experience.hints(graph, model=model)
        decision = GraphDrivenAdaptiveScheduler(graph, experience=experience).decide(report=report, thresholds=thresholds)
        targets = self._targets_from_feedback(graph, feedback)
        plan = TestIntentPlan(
            phase="feedback_replan",
            action=decision.action,
            owner="测试状态智能体",
            selected_agent=decision.agent,
            reason=self._planning_reason(decision.agent, decision.action, feedback),
            gaps=decision.gaps,
            test_path=str(Path(report.test_path).resolve()),
            focus=self._focus_for_agent(decision.agent, report),
            feedback_summary=feedback.to_dict(),
            target_states=targets["states"],
            target_methods=targets["methods"],
            target_branches=targets["branches"],
            expected_gain=max([float(item.priority) for item in feedback.suggestions[:5]], default=0.0),
            assigned_agent="测试状态智能体" if decision.action in {"stop", "verify"} else "测试生成智能体",
            capability=self._focus_for_agent(decision.agent, report),
        )
        graph.metadata["last_test_intent_plan"] = plan.to_dict()
        graph.metadata["last_agent_decision"] = decision.to_dict()
        self._record_exchange(graph, "TestStateAgent", "EvaluationFeedback", feedback.to_dict())
        self._record_exchange(graph, "TestStateAgent", "TestIntentPlan", plan.to_dict())
        append_state_flow_event(
            graph,
            StateFlowEvent(
                event_type="TestIntentPlanned",
                source_agent="测试状态智能体",
                target_agent="测试生成智能体",
                related_nodes=plan.target_states or [],
                payload=plan.to_dict(),
                iteration=graph.iteration,
            ),
        )
        graph.bump_version("planning feedback replan completed")
        self.memory.save(graph)
        return decision, plan


    def _planning_reason(self, agent: str, action: str, feedback: EvaluationFeedback) -> str:
        if action == "stop":
            return "quality target reached; no further test changes are planned"
        if action == "verify":
            return "evaluation found no meaningful remaining gap; request final verification"
        if action == "optimize":
            return f"test is executable but below the target threshold; evaluator focus={feedback.recommended_focus or agent}"
        mapping = {
            "CoverageAgent": "coverage is low; plan additional path-oriented tests",
            "AssertionAgent": "assertions are weak; plan stronger observable oracles",
            "MutationAgent": "survived mutants remain; plan mutation-killing assertions",
            "BoundaryAgent": "exception or boundary behavior is under-tested; plan boundary cases",
            "TypeAnalysisAgent": "normal behavior evidence is missing; plan executable normal input/output cases",
            "TestValidityAgent": "pytest evidence is invalid; plan local validity repair",
            "OracleRepairAgent": "pytest failed because expected behavior is wrong; plan oracle repair",
            "RepairAgent": "pytest timed out or failed at a localized root cause; plan minimal constrained repair",
        }
        if feedback.recommended_focus:
            return f"{mapping.get(agent, 'evaluation found remaining quality gaps')}; evaluator focus={feedback.recommended_focus}"
        return mapping.get(agent, "evaluation found remaining quality gaps")

    def _focus_for_agent(self, agent: str, report: ExecutionReport) -> str:
        if report.pytest_passed and report.quality_status == "valid_usable":
            return "quality_optimization_plan"
        if report.exception_only or report.normal_behavior_tests == 0:
            return "normal_behavior_plan"
        return {
            "CoverageAgent": "path_coverage_plan",
            "AssertionAgent": "assertion_strengthening_plan",
            "MutationAgent": "mutation_killing_plan",
            "BoundaryAgent": "boundary_exception_plan",
            "TypeAnalysisAgent": "normal_behavior_plan",
            "TestValidityAgent": "validity_repair_plan",
            "OracleRepairAgent": "oracle_repair_plan",
            "ExecutionAgent": "verification_plan",
            "Stop": "stop_plan",
        }.get(agent, "general_repair_plan")

    def _record_exchange(self, graph, sender: str, artifact_type: str, payload: dict[str, Any]) -> None:
        exchange = graph.metadata.setdefault("agent_collaboration_exchange", [])
        if isinstance(exchange, list):
            exchange.append(
                {
                    "sender": sender,
                    "artifact_type": artifact_type,
                    "payload": payload,
                }
            )

    def _targets_from_graph(self, graph) -> dict[str, list[str]]:
        methods = [
            node.node_id
            for node in graph.nodes.values()
            if node.node_type in {"Function", "Async", "Method", "Constructor"} and node.metadata.get("visibility", "public") != "private"
        ][:20]
        branches = [node.node_id for node in graph.nodes.values() if node.node_type in {"Branch", "Loop", "Match"}][:30]
        return {"states": methods + branches, "methods": methods, "branches": branches}

    def _targets_from_feedback(self, graph, feedback: EvaluationFeedback) -> dict[str, list[str]]:
        target_ids = [item.target_id for item in feedback.suggestions[:20]]
        existing = set(graph.nodes.keys())
        states = [item for item in target_ids if item in existing]
        if not states:
            states = self._targets_from_graph(graph)["states"][:20]
        methods = [node_id for node_id in states if graph.nodes.get(node_id) and graph.nodes[node_id].node_type in {"Function", "Async", "Method", "Constructor"}]
        branches = [node_id for node_id in states if graph.nodes.get(node_id) and graph.nodes[node_id].node_type in {"Branch", "Loop", "Match"}]
        return {"states": states, "methods": methods, "branches": branches}
