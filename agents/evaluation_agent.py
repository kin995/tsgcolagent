from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.state_flow_event import StateFlowEvent, append_state_flow_event
from graph.graph_query import GraphQuery
from graph.experience_memory import AgentExperienceMemory
from graph.state_graph import StateFlowGraph
from llm_client import OpenAICompatibleLLM, extract_json_data
from memory.shared_graph import SharedGraphMemory


@dataclass
class RepairSuggestion:
    target_id: str
    issue: str
    action: str
    priority: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "issue": self.issue,
            "action": self.action,
            "priority": self.priority,
            "reason": self.reason,
        }


@dataclass
class EvaluationFeedback:
    quality_status: str
    usable: bool
    recommended_focus: str
    reason: str
    gaps: dict[str, float]
    suggestions: list[RepairSuggestion]
    evidence: dict[str, Any]
    experience_record: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "quality_status": self.quality_status,
            "usable": self.usable,
            "recommended_focus": self.recommended_focus,
            "reason": self.reason,
            "gaps": self.gaps,
            "suggestions": [item.to_dict() for item in self.suggestions],
            "evidence": self.evidence,
            "experience_record": self.experience_record or {},
        }


class TestStateEvaluationModule:
    """测试状态智能体内部的质量评估与反馈模块。"""

    name = "测试状态智能体/评估模块"

    def __init__(self, memory: SharedGraphMemory, llm: OpenAICompatibleLLM | None = None, llm_required: bool = False):
        self.memory = memory
        self.llm = llm
        self.llm_required = llm_required

    def run(self) -> list[RepairSuggestion]:
        graph = self.memory.load()
        suggestions = self.evaluate(graph)
        suggestions = self._apply_llm_evaluation(graph, suggestions)
        graph.metadata["repair_suggestions"] = [item.to_dict() for item in suggestions]
        graph.bump_version("evaluation completed")
        self.memory.save(graph)
        return suggestions

    def run_feedback(
        self,
        report: Any,
        thresholds: Any,
        experience: AgentExperienceMemory | None = None,
        pending_decision: dict[str, Any] | None = None,
        pending_before_report: Any | None = None,
        pending_runtime: float = 0.0,
        pending_usage: dict[str, int | float] | None = None,
        model: str | None = None,
    ) -> EvaluationFeedback:
        graph = self.memory.load()
        graph.metadata["last_report"] = report.to_dict()
        validation_data = graph.metadata.get("last_validation", {})
        completeness_data = graph.metadata.get("complete_state_graph", {})
        if isinstance(completeness_data, dict) and completeness_data.get("complete"):
            append_state_flow_event(
                graph,
                StateFlowEvent(
                    event_type="CompletenessReached",
                    source_agent="测试状态智能体",
                    target_agent="测试状态智能体",
                    payload=completeness_data,
                    iteration=graph.iteration,
                ),
            )
        experience_record = None
        if experience and pending_decision and pending_before_report:
            experience_record = experience.append(
                graph=graph,
                agent=str(pending_decision["agent"]),
                gaps=dict(pending_decision.get("gaps", {})),
                before=pending_before_report,
                after=report,
                model=model,
                runtime_seconds=pending_runtime,
                llm_calls=int((pending_usage or {}).get("calls", 0)),
                token_count=int((pending_usage or {}).get("total_tokens", 0)),
                expected_reward=float(pending_decision.get("expected_reward", 0.0) or 0.0),
            )
            graph.metadata["last_agent_experience"] = experience_record
        if experience:
            graph.metadata["experience_hints"] = experience.hints(graph, model=model)
        suggestions = self.evaluate(graph)
        quality_status = str(getattr(report, "quality_status", "valid"))
        needs_repair_guidance = not (
            bool(getattr(thresholds, "usable")(report))
            and quality_status in {"quality_high", "quality_acceptable"}
        )
        if needs_repair_guidance:
            suggestions = self._apply_llm_evaluation(graph, suggestions)
        else:
            graph.metadata["evaluation_strategy"] = "local_evidence_terminal_quality"
        suggestions = self._failure_suggestions(report, graph) + suggestions
        gaps = GraphQuery(graph).get_gaps().as_dict()
        diagnosis = graph.metadata.get("execution_diagnosis", {})
        recommended_focus = str(diagnosis.get("recommended_focus") or self._focus_from_report(report, gaps, graph))
        feedback = EvaluationFeedback(
            quality_status=str(getattr(report, "quality_status", "valid")),
            usable=bool(getattr(thresholds, "usable")(report)),
            recommended_focus=recommended_focus,
            reason=self._feedback_reason(report, gaps, recommended_focus, graph),
            gaps=gaps,
            suggestions=suggestions,
            evidence={
                "report": report.to_dict(),
                "dynamic_validation": validation_data if isinstance(validation_data, dict) else {},
                "complete_state_graph": completeness_data if isinstance(completeness_data, dict) else {},
                "execution_diagnosis": diagnosis,
                "root_cause": graph.metadata.get("last_root_cause", {}),
                "experience_hints": graph.metadata.get("experience_hints", {}),
                "mutation_diagnostics": graph.metadata.get("last_mutation", {}),
            },
            experience_record=experience_record,
        )
        graph.metadata["repair_suggestions"] = [item.to_dict() for item in suggestions]
        graph.metadata["last_evaluation_feedback"] = feedback.to_dict()
        append_state_flow_event(
            graph,
            StateFlowEvent(
                event_type="EvaluationFeedback",
                source_agent="测试状态智能体",
                target_agent="测试生成智能体",
                payload=feedback.to_dict(),
                iteration=graph.iteration,
            ),
        )
        graph.bump_version("evaluation feedback completed")
        self.memory.save(graph)
        return feedback

    def evaluate(self, graph: StateFlowGraph) -> list[RepairSuggestion]:
        suggestions: list[RepairSuggestion] = []
        query = GraphQuery(graph)
        for candidate in query.get_candidate_repairs():
            node = candidate["node"]
            actions = candidate.get("actions", ["inspect"])
            suggestions.append(
                RepairSuggestion(
                    target_id=str(node["node_id"]),
                    issue="propagated_graph_defect",
                    action=", ".join(actions),
                    priority=float(node.get("defect_score", 0.0)),
                    reason=f"root-cause guided candidate: {node.get('node_type')} {node.get('name')} at line {node.get('line_start')}",
                )
            )
        for path in query.get_uncovered_paths():
            edge = path["edge"]
            suggestions.append(
                RepairSuggestion(
                    target_id=str(edge["source"]) + "->" + str(edge["target"]),
                    issue="uncovered_path",
                    action="Need Path",
                    priority=float(edge.get("weight", 1.0)),
                    reason="PTEG path edge has no observed execution evidence",
                )
            )
        for mutation in query.get_survived_mutations():
            suggestions.append(
                RepairSuggestion(
                    target_id=mutation.node_id,
                    issue="survived_mutation",
                    action="Mutation Failed",
                    priority=max(1.0, mutation.defect_score),
                    reason=f"survived mutation evidence near line {mutation.line_start}",
                )
            )
        suggestions.sort(key=lambda item: item.priority, reverse=True)
        return suggestions[:40]

    def _apply_llm_evaluation(self, graph: StateFlowGraph, suggestions: list[RepairSuggestion]) -> list[RepairSuggestion]:
        if not self.llm or not self.llm.enabled():
            graph.metadata["evaluation_strategy"] = "graph_query_only"
            return suggestions
        prompt = f"""
You are the quality evaluation module inside the Test State Agent in a multi-agent unit-test generation system.
Use the graph evidence and local metric suggestions to produce repair suggestions.

Return JSON list only. Each item must contain:
- target_id
- issue
- action
- priority
- reason

Local report:
{graph.metadata.get("last_report")}

Execution diagnosis:
{graph.metadata.get("execution_diagnosis")}

Root cause:
{graph.metadata.get("last_root_cause")}

Historical actionable lessons:
{(graph.metadata.get("experience_hints") or {}).get("actionable_lessons", [])}

Local suggestions:
{[item.to_dict() for item in suggestions[:20]]}
""".strip()
        try:
            response = self.llm.chat(
                [
                    {"role": "system", "content": "You rank and enrich unit-test repair suggestions. Return JSON only."},
                    {"role": "user", "content": prompt},
                ]
            )
            data = extract_json_data(response)
            llm_suggestions = self._suggestions_from_json(data)
            graph.metadata["llm_evaluation_raw"] = response
            graph.metadata["evaluation_strategy"] = "graph_query_plus_llm"
            graph.metadata["llm_evaluation_model"] = self.llm.config.model
            return (llm_suggestions + suggestions)[:40] if llm_suggestions else suggestions
        except Exception as exc:
            graph.metadata["llm_evaluation_error"] = str(exc)
            if self.llm_required and _is_connection_error(exc):
                raise
            graph.metadata["evaluation_strategy"] = "graph_query_after_llm_error"
            return suggestions

    def _suggestions_from_json(self, data: object) -> list[RepairSuggestion]:
        if not isinstance(data, list):
            return []
        suggestions: list[RepairSuggestion] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            suggestions.append(
                RepairSuggestion(
                    target_id=str(item.get("target_id", "llm")),
                    issue=str(item.get("issue", "llm_evaluation")),
                    action=str(item.get("action", "Repair")),
                    priority=float(item.get("priority", 1.0)),
                    reason=str(item.get("reason", "LLM-generated repair suggestion.")),
                )
            )
        suggestions.sort(key=lambda item: item.priority, reverse=True)
        return suggestions

    def _focus_from_report(self, report: Any, gaps: dict[str, float], graph: StateFlowGraph | None = None) -> str:
        if not bool(getattr(report, "pytest_passed", True)):
            failure_category = str(getattr(report, "failure_category", ""))
            if failure_category == "oracle_error":
                return "oracle"
            if failure_category == "timeout":
                return "timeout_minimal_repair" if bool(getattr(report, "timeout_should_repair", True)) else "external_dependency_timeout"
            return "test_validity"
        if bool(getattr(report, "exception_only", False)) or int(getattr(report, "normal_behavior_tests", 0) or 0) == 0:
            return "normal_behavior"
        if bool(getattr(report, "weak_assertions_only", False)):
            return "assertion"
        validation = graph.metadata.get("last_validation", {}) if graph else {}
        if isinstance(validation, dict) and not validation.get("passed", True):
            return "dynamic_validation"
        completeness = graph.metadata.get("complete_state_graph", {}) if graph else {}
        if isinstance(completeness, dict) and not completeness.get("complete", False):
            return "complete_state_graph"
        if gaps:
            return max(gaps.items(), key=lambda item: item[1])[0]
        return "verification"

    def _feedback_reason(self, report: Any, gaps: dict[str, float], recommended_focus: str, graph: StateFlowGraph | None = None) -> str:
        if not bool(getattr(report, "pytest_passed", True)):
            if str(getattr(report, "failure_category", "")) == "timeout":
                reason = str(getattr(report, "timeout_reason", "") or "pytest timed out")
                location = str(getattr(report, "timeout_likely_location", "") or "unknown location")
                return f"pytest timed out; reason={reason}; likely_location={location}"
            return f"pytest failed; failure category is {getattr(report, 'failure_category', 'unknown')}"
        if bool(getattr(report, "exception_only", False)):
            return "test suite only exercises exception behavior"
        if bool(getattr(report, "weak_assertions_only", False)):
            return "test suite contains only weak assertions"
        validation = graph.metadata.get("last_validation", {}) if graph else {}
        if isinstance(validation, dict) and not validation.get("passed", True):
            return f"dynamic validation failed; score={validation.get('score')}; focus={recommended_focus}"
        completeness = graph.metadata.get("complete_state_graph", {}) if graph else {}
        if isinstance(completeness, dict) and not completeness.get("complete", False):
            return f"complete state graph not reached; CSG={completeness.get('csg')}; focus={recommended_focus}"
        if int(getattr(report, "effective_mutants", 0) or 0) < 3:
            mutation = graph.metadata.get("last_mutation", {}) if graph is not None and isinstance(graph.metadata.get("last_mutation"), dict) else {}
            reason = str(mutation.get("low_sample_reason") or "effective mutants fewer than 3")
            action = str(mutation.get("autonomy_action") or "plan boundary and branch-distinguishing assertions")
            return f"mutation evidence has too few effective mutants; reason={reason}; action={action}"
        if gaps:
            largest_gap, value = max(gaps.items(), key=lambda item: item[1])
            return f"largest remaining evidence gap is {largest_gap}={round(float(value), 4)}; focus={recommended_focus}"
        return "quality evidence is sufficient for verification"

    def _failure_suggestions(self, report: Any, graph: StateFlowGraph) -> list[RepairSuggestion]:
        if bool(getattr(report, "pytest_passed", True)):
            return []
        failure_category = str(getattr(report, "failure_category", "runtime_error"))
        action = "OracleRepairAgent" if failure_category == "oracle_error" else "TestValidityAgent"
        if failure_category == "timeout" and bool(getattr(report, "timeout_should_repair", True)):
            action = "RepairAgent"
        failed_tests = getattr(report, "failed_tests", None) or []
        suggestions: list[RepairSuggestion] = []
        timeout_reason = str(getattr(report, "timeout_reason", "") or "")
        timeout_location = str(getattr(report, "timeout_likely_location", "") or "")
        for index, item in enumerate(failed_tests):
            if not isinstance(item, dict):
                continue
            test_name = str(item.get("test_name") or f"failed_test_{index + 1}")
            message = str(item.get("message") or item.get("error") or failure_category)
            if failure_category == "timeout" and timeout_reason:
                message = f"{timeout_reason}; likely_location={timeout_location}"
            suggestions.append(
                RepairSuggestion(
                    target_id=test_name,
                    issue=failure_category,
                    action=action,
                    priority=1.0,
                    reason=f"pytest failed in {test_name}: {message[:240]}",
                )
            )
        if not suggestions:
            diagnosis = graph.metadata.get("execution_diagnosis", {}) if isinstance(graph.metadata.get("execution_diagnosis"), dict) else {}
            suggestions.append(
                RepairSuggestion(
                    target_id="pytest_failure",
                    issue=failure_category,
                    action=action,
                    priority=1.0,
                    reason=f"pytest failed before quality tools; diagnosis={diagnosis}",
                )
            )
        return suggestions


def _is_connection_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "connection" in text or "not installed" in text or "http" in text


# 兼容旧导入名；评估模块当前归属于测试状态智能体。
EvaluationAgent = TestStateEvaluationModule
