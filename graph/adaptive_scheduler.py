from __future__ import annotations

from dataclasses import dataclass

from .experience_memory import AGENT_TO_GAP, AgentExperienceMemory
from .graph_query import GraphQuery
from .state_graph import StateFlowGraph


@dataclass
class AgentDecision:
    agent: str
    reason: str
    gaps: dict[str, float]
    action: str = "repair"
    confidence: float = 0.0
    expected_reward: float = 0.0
    policy: str = "gap_only"

    def to_dict(self) -> dict[str, object]:
        return {
            "agent": self.agent,
            "reason": self.reason,
            "gaps": self.gaps,
            "action": self.action,
            "confidence": self.confidence,
            "expected_reward": self.expected_reward,
            "policy": self.policy,
        }


class GraphDrivenAdaptiveScheduler:
    def __init__(self, graph: StateFlowGraph, experience: AgentExperienceMemory | None = None):
        self.graph = graph
        self.query = GraphQuery(graph)
        self.experience = experience

    def decide(self, report: object | None = None, thresholds: object | None = None) -> AgentDecision:
        gaps = self.query.get_gaps()
        if report is not None and thresholds is not None:
            line_target = max(float(getattr(thresholds, "line_coverage", 0.0)), float(getattr(thresholds, "high_line_coverage", 0.0)))
            branch_target = max(float(getattr(thresholds, "branch_coverage", 0.0)), float(getattr(thresholds, "high_branch_coverage", 0.0)))
            sfc_target = max(float(getattr(thresholds, "sfc", 0.0)), float(getattr(thresholds, "high_sfc", 0.0)))
            mutation_target = max(float(getattr(thresholds, "mutation", 0.0)), float(getattr(thresholds, "high_mutation", 0.0)))
            line_ok = float(getattr(report, "line_coverage", 0.0) or 0.0) >= line_target
            branch_value = getattr(report, "branch_coverage", None)
            branch_ok = branch_value is None or float(branch_value) >= branch_target
            sfc_ok = float(getattr(report, "sfc", 0.0) or 0.0) >= sfc_target
            if line_ok and branch_ok and sfc_ok:
                gaps.coverage = 0.0
            else:
                branch_gap = 0.0 if branch_value is None else max(0.0, branch_target - float(branch_value))
                gaps.coverage = max(
                    gaps.coverage,
                    max(0.0, line_target - float(getattr(report, "line_coverage", 0.0) or 0.0)),
                    branch_gap,
                    max(0.0, sfc_target - float(getattr(report, "sfc", 0.0) or 0.0)),
                )
            mutation_applicable = getattr(thresholds, "mutation_applicable", lambda _: True)(report)
            mutation_value = getattr(report, "mutation_score", None)
            if not mutation_applicable or (mutation_value is not None and float(mutation_value) >= mutation_target):
                gaps.mutation = 0.0
            elif mutation_value is not None:
                gaps.mutation = max(gaps.mutation, mutation_target - float(mutation_value))
            ae_threshold = getattr(thresholds, "ae", None)
            if ae_threshold is None or float(getattr(report, "ae", 0.0) or 0.0) >= float(ae_threshold):
                gaps.assertion = 0.0
            else:
                gaps.assertion = max(gaps.assertion, float(ae_threshold) - float(getattr(report, "ae", 0.0) or 0.0))
        largest_gap, value = gaps.largest()
        mapping = {
            "coverage": "CoverageAgent",
            "mutation": "MutationAgent",
            "assertion": "AssertionAgent",
            "exception": "BoundaryAgent",
            "runtime_type": "TypeAnalysisAgent",
        }
        agent = mapping.get(largest_gap, "RepairAgent")
        action = "optimize"
        force_repair_agent = False
        if report is not None and thresholds is not None and thresholds.reached(report):
            action = "stop"
            agent = "Stop"
        elif report is not None and not bool(getattr(report, "pytest_passed", True)):
            action = "repair"
            failure_category = str(getattr(report, "failure_category", "runtime_error"))
            diagnosis = self.graph.metadata.get("execution_diagnosis", {})
            should_repair_timeout = bool(diagnosis.get("should_repair", True)) if isinstance(diagnosis, dict) else True
            if failure_category == "oracle_error":
                agent = "OracleRepairAgent"
            elif failure_category == "timeout" and should_repair_timeout:
                agent = "RepairAgent"
                gaps.coverage = max(1.0, float(getattr(gaps, "coverage", 0.0) or 0.0))
                largest_gap, value = "timeout", 1.0
            else:
                agent = "TestValidityAgent"
            if self.experience:
                model = str(self.graph.metadata.get("llm_generation_model") or self.graph.metadata.get("llm_execution_model") or "")
                validity_failure = self.experience.failure_rate("TestValidityAgent", model=model or None)
                oracle_failure = self.experience.failure_rate("OracleRepairAgent", model=model or None)
                if failure_category == "oracle_error" and oracle_failure > 0.75 and validity_failure < oracle_failure:
                    agent = "TestValidityAgent"
                predictions = {
                    "TestValidityAgent": {"failure_rate": validity_failure},
                    "OracleRepairAgent": {"failure_rate": oracle_failure},
                }
            force_repair_agent = True
        elif report is not None and (
            bool(getattr(report, "exception_only", False))
            or int(getattr(report, "normal_behavior_tests", 0) or 0) == 0
        ):
            action = "optimize"
            agent = "TypeAnalysisAgent"
            gaps.runtime_type = max(1.0, float(getattr(gaps, "runtime_type", 0.0) or 0.0))
            largest_gap, value = "runtime_type", gaps.runtime_type
            force_repair_agent = True
        elif report is not None and bool(getattr(report, "weak_assertions_only", False)):
            action = "optimize"
            agent = "AssertionAgent"
            gaps.assertion = max(1.0, float(getattr(gaps, "assertion", 0.0) or 0.0))
            largest_gap, value = "assertion", gaps.assertion
            force_repair_agent = True
        elif value <= 0.01:
            action = "verify"
            agent = "ExecutionAgent"
        predictions = predictions if "predictions" in locals() else {}
        expected_reward = 0.0
        confidence = round(min(1.0, value), 4)
        policy = "gap_only"
        if self.experience and action in {"repair", "optimize"} and not force_repair_agent:
            scored_agents: list[tuple[float, str]] = []
            gap_values = gaps.as_dict()
            diagnosis = self.graph.metadata.get("execution_diagnosis", {})
            recommended_focus = str(diagnosis.get("recommended_focus", ""))
            diagnosis_confidence = float(diagnosis.get("confidence", 0.0) or 0.0)
            last_agent = self._last_repair_agent()
            model = str(self.graph.metadata.get("llm_generation_model") or self.graph.metadata.get("llm_execution_model") or "")
            for candidate, gap_name in AGENT_TO_GAP.items():
                gap_score = gap_values.get(gap_name, 0.0)
                if gap_score + 1e-9 < max(0.01, value * 0.80):
                    continue
                prediction = self.experience.predict(candidate, gaps, self.graph, model=model or None)
                predictions[candidate] = prediction.to_dict()
                reward_score = max(0.0, prediction.predicted_reward)
                exploration_bonus = 0.08 if prediction.sample_count < 3 else 0.0
                confidence_score = diagnosis_confidence if recommended_focus in {gap_name, candidate} else 0.0
                cost_penalty = self._cost(candidate)
                failure_penalty = self.experience.failure_rate(candidate, model=model or None)
                repeat_penalty = 0.15 if candidate == last_agent else 0.0
                score = (0.45 * gap_score) + (0.30 * reward_score) + (0.15 * confidence_score) + exploration_bonus - (0.05 * cost_penalty) - (0.15 * failure_penalty) - repeat_penalty
                scored_agents.append((score, candidate))
            if scored_agents and any(item["sample_count"] for item in predictions.values()):
                best_score, agent = max(scored_agents, key=lambda item: item[0])
                selected_prediction = predictions[agent]
                expected_reward = float(selected_prediction["predicted_reward"])
                confidence = round(min(1.0, best_score), 4)
                policy = "gap_plus_experience"
        decision = AgentDecision(
            agent=agent,
            reason=self._decision_reason(action, agent, largest_gap, value, expected_reward),
            gaps=gaps.as_dict(),
            action=action,
            confidence=confidence,
            expected_reward=expected_reward,
            policy=policy,
        )
        self.graph.metadata["last_agent_decision"] = decision.to_dict()
        if predictions:
            self.graph.metadata["agent_reward_predictions"] = predictions
        self.graph.history.append(
            {
                "version": self.graph.version,
                "iteration": self.graph.iteration,
                "reason": "graph driven adaptive scheduling",
                "decision": decision.to_dict(),
            }
        )
        return decision

    def _last_repair_agent(self) -> str:
        for item in reversed(self.graph.history):
            decision = item.get("decision", {}) if isinstance(item, dict) else {}
            agent = decision.get("agent")
            if agent:
                return str(agent)
        return ""

    def _cost(self, agent: str) -> float:
        costs = {
            "CoverageAgent": 0.35,
            "AssertionAgent": 0.30,
            "MutationAgent": 0.75,
            "BoundaryAgent": 0.45,
            "TypeAnalysisAgent": 0.40,
        }
        return costs.get(agent, 0.5)

    def _decision_reason(self, action: str, agent: str, largest_gap: str, value: float, expected_reward: float) -> str:
        if action == "stop":
            return "quality thresholds are reached; scheduler stops the loop"
        if action == "verify":
            return "all graph gaps are near zero; scheduler requests verification"
        if action == "optimize":
            return f"{agent} selected to optimize remaining {largest_gap} gap {value} before the success threshold"
        if agent in {"TestValidityAgent", "OracleRepairAgent"}:
            return f"{agent} selected because pytest is invalid and only validity/oracle repair is allowed"
        if agent == "RepairAgent" and largest_gap == "timeout":
            return "RepairAgent selected because pytest timed out for a non-network local reason; use minimal root-cause repair"
        if agent == "TypeAnalysisAgent":
            return "TypeAnalysisAgent selected because no normal-behavior test was observed"
        if agent == "AssertionAgent" and largest_gap == "assertion" and value >= 1.0:
            return "AssertionAgent selected because the suite contains only weak assertions"
        if expected_reward:
            return f"{agent} selected from {largest_gap} gap {value} with historical expected reward {expected_reward}"
        return f"{agent} selected because {largest_gap} gap is highest at {value}"
