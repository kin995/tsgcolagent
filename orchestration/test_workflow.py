from __future__ import annotations

import json
import hashlib
import sqlite3
import time
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from agents.execution_agent import ExecutionReport
from core.experiment_outcome import grade_for, round_label
from core.state_flow_event import append_state_flow_event
from orchestration.state_schema import TestWorkflowState
from metrics.measurement import CURRENT, Measurement


class LangGraphTestWorkflow:
    def __init__(self, system: Any):
        self.system = system
        checkpoint_dir = Path(system.output_dir) / ".langgraph"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = checkpoint_dir / "checkpoints.sqlite"
        self._checkpoint_connection: sqlite3.Connection | None = sqlite3.connect(self.checkpoint_path, check_same_thread=False)
        self._checkpointer = SqliteSaver(self._checkpoint_connection)
        self._checkpointer.setup()
        self.graph = self._build_graph()

    def run(self) -> dict[str, object]:
        started = time.perf_counter()
        optimize_existing = (
            self.system.followup_strategy == "optimize_existing"
            and self.system.existing_test_path is not None
            and self.system.existing_test_path.exists()
            and bool(self.system.baseline_report)
        )
        baseline = dict(self.system.baseline_report) if optimize_existing else None
        followup_intent = self._initial_followup_intent()
        initial: TestWorkflowState = {
            "task_id": self.system.task_id,
            "source_path": str(self.system.source_path),
            "output_dir": str(self.system.output_dir),
            "graph_path": str(self.system.graph_path),
            "tsm_path": str(self.system.graph_path),
            "reports": [dict(baseline)] if baseline else [],
            "plans": [],
            "decisions": [],
            "iteration": 1 if baseline else 0,
            "repair_rounds_used": 0,
            "state_repair_attempts": 0,
            "pending_before_report": baseline,
            "current_intent": followup_intent,
            "pending_decision": None,
            "pending_usage": None,
            "pending_runtime": 0.0,
            "next_action": "bootstrap",
            "stop_reason": "",
            "status": "running",
            "started_at": started,
            "last_error": "",
            "final_evaluation_done": False,
        }
        config = {
            "configurable": {"thread_id": self.system.task_id},
            "recursion_limit": max(20, (self._repair_limit() + 2) * 8),
        }
        measurement = Measurement(self.system.output_dir, self.system.source_path, self.system.experiment_round)
        if baseline:
            measurement.last_evidence = {"report": dict(baseline)}
        token = CURRENT.set(measurement)
        try:
            final_state = self.graph.invoke(initial, config=config)
            return self._summary(final_state, started)
        finally:
            CURRENT.reset(token)
            self.close()

    def close(self) -> None:
        connection = getattr(self, "_checkpoint_connection", None)
        if connection is not None:
            connection.close()
            self._checkpoint_connection = None

    def __del__(self) -> None:
        self.close()

    def _build_graph(self):
        builder = StateGraph(TestWorkflowState)
        builder.add_node("test_state_analyze", self._analyze_test_state)
        builder.add_node("test_state_bootstrap", self._bootstrap)
        builder.add_node("test_state_self_repair", self._repair_test_state)
        builder.add_node("test_knowledge_retrieve", self._retrieve_knowledge)
        builder.add_node("test_generation_agent", self._generate_repair_and_execute)
        builder.add_node("test_generation_finalize", self._finalize_execution)
        builder.add_node("test_state_evaluate_plan", self._evaluate_and_plan)
        builder.add_node("test_knowledge_update", self._update_knowledge)
        builder.add_edge(START, "test_state_analyze")
        builder.add_conditional_edges(
            "test_state_analyze",
            self._route,
            {
                "state_repair": "test_state_self_repair",
                "knowledge": "test_knowledge_retrieve",
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "test_state_bootstrap",
            self._route,
            {
                "state_repair": "test_state_self_repair",
                "generate": "test_generation_agent",
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "test_state_self_repair",
            self._route,
            {
                "bootstrap": "test_state_bootstrap",
                "state_repair": "test_state_self_repair",
                "knowledge": "test_knowledge_retrieve",
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "test_knowledge_retrieve",
            self._route,
            {
                "bootstrap": "test_state_bootstrap",
                "generate": "test_generation_agent",
                "state_repair": "test_state_self_repair",
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "test_generation_agent",
            self._route,
            {
                "evaluate": "test_state_evaluate_plan",
                "state_repair": "test_state_self_repair",
                "finalize": "test_generation_finalize",
                "end": END,
            },
        )
        builder.add_edge("test_state_evaluate_plan", "test_knowledge_update")
        builder.add_edge("test_generation_finalize", "test_state_evaluate_plan")
        builder.add_conditional_edges(
            "test_knowledge_update",
            self._route,
            {
                "knowledge": "test_knowledge_retrieve",
                "state_repair": "test_state_self_repair",
                "finalize": "test_generation_finalize",
                "end": END,
            },
        )
        return builder.compile(checkpointer=self._checkpointer)

    def _analyze_test_state(self, state: TestWorkflowState) -> dict[str, object]:
        try:
            self.system.execution_agent.analyze_for_generation(
                self.system.source_path,
                self.system.experience,
                model=self.system.llm_config.model,
            )
            health = self.system.test_state_agent.inspect()
        except Exception as exc:
            return {"next_action": "state_repair", "last_error": f"analysis:{exc.__class__.__name__}:{exc}"}
        graph = self.system.memory.load()
        graph.metadata["experiment_round"] = self.system.experiment_round
        graph.metadata["followup_strategy"] = self.system.followup_strategy
        graph.metadata["previous_grade"] = self.system.previous_grade
        if self.system.baseline_report:
            graph.metadata["previous_round_report"] = self.system.baseline_report
            mutation_evidence = self.system.baseline_report.get("mutation_evidence")
            if isinstance(mutation_evidence, dict) and mutation_evidence:
                graph.metadata["previous_mutation_evidence"] = mutation_evidence
                graph.metadata["last_mutation"] = mutation_evidence
        if self.system.followup_strategy in {"regenerate_after_failure", "restart_incomplete"}:
            category = str(self.system.baseline_report.get("failure_category") or "unknown")
            guidance = self.system.experience.failure_guidance(str(self.system.source_path), category)
            recovery = self.system.baseline_report.get("failure_recovery")
            recovery = dict(recovery) if isinstance(recovery, dict) else {}
            prior_test = ""
            if self.system.existing_test_path is not None and self.system.existing_test_path.is_file():
                prior_test = self.system.existing_test_path.read_text(encoding="utf-8", errors="ignore")[:8000]
            contract = {
                **recovery,
                "category": category,
                "knowledge_guidance": guidance,
                "previous_failed_test": prior_test,
                "require_different_candidate": bool(prior_test),
            }
            graph.metadata["failure_recovery_guidance"] = guidance
            graph.metadata["failure_recovery_contract"] = contract
            hints = graph.metadata.get("experience_hints") if isinstance(graph.metadata.get("experience_hints"), dict) else {}
            graph.metadata["experience_hints"] = {
                **hints,
                "followup_failure_guidance": guidance,
                "failure_recovery_contract": contract,
            }
        self.system.memory.save(graph)
        return {
            "tsm_version": graph.version,
            "next_action": "knowledge" if health.valid else "state_repair",
            "last_error": "" if health.valid else ",".join(health.issues),
        }

    def _bootstrap(self, state: TestWorkflowState) -> dict[str, object]:
        if (
            self.system.followup_strategy == "optimize_existing"
            and self.system.existing_test_path is not None
            and self.system.existing_test_path.exists()
            and self.system.baseline_report
        ):
            graph = self.system.memory.load()
            plan, feedback = self._state_target_optimization_plan(graph)
            graph.metadata["generated_test_path"] = str(self.system.existing_test_path)
            graph.metadata["last_report"] = dict(self.system.baseline_report)
            graph.metadata["last_test_intent_plan"] = plan
            graph.metadata["last_evaluation_feedback"] = feedback
            graph.metadata["followup_strategy"] = "optimize_existing"
            append_state_flow_event(
                graph,
                self._followup_plan_event(plan),
            )
            graph.bump_version("C-grade follow-up optimization planned")
            self.system.memory.save(graph)
            return {
                "test_path": str(self.system.existing_test_path),
                "current_intent": plan,
                "plans": [plan],
                "evaluation_feedback": feedback,
                "tsm_version": graph.version,
                "next_action": "generate",
                "last_error": "",
            }
        try:
            test_path, plan, health = self.system.test_state_agent.bootstrap(
                self.system.source_path,
                self.system.output_dir,
                self.system.experience,
                model=self.system.llm_config.model,
            )
        except Exception as exc:
            return {
                "next_action": "state_repair",
                "last_error": f"bootstrap:{exc.__class__.__name__}:{exc}",
            }
        graph = self.system.memory.load()
        return {
            "test_path": str(test_path.resolve()),
            "current_intent": plan.to_dict(),
            "plans": [plan.to_dict()],
            "tsm_version": graph.version,
            "next_action": "generate" if health.valid else "state_repair",
            "last_error": "" if health.valid else ",".join(health.issues),
        }

    def _initial_followup_intent(self) -> dict[str, object]:
        if self.system.experiment_round <= 1:
            return {}
        report = self.system.baseline_report
        gaps = {
            str(key): float(value)
            for key, value in (report.get("threshold_failures", {}) or {}).items()
            if isinstance(value, (int, float)) and float(value) > 0
        }
        if self.system.followup_strategy == "optimize_existing":
            focus = max(gaps.items(), key=lambda item: item[1])[0] if gaps else self._weak_focus(report)
            return {"action": "optimize", "focus": focus, "gaps": gaps, "previous_grade": self.system.previous_grade}
        category = str(report.get("failure_category") or "unknown")
        recovery = report.get("failure_recovery") if isinstance(report.get("failure_recovery"), dict) else {}
        return {
            "action": "regenerate_after_failure",
            "focus": category,
            "gaps": {category: 1.0},
            "previous_grade": self.system.previous_grade,
            "failure_reason": recovery.get("reason") or report.get("failure_detail"),
            "required_solutions": list(recovery.get("required_solutions", []) or []),
        }

    def _followup_plan_event(self, plan: dict[str, object]):
        from core.state_flow_event import StateFlowEvent

        return StateFlowEvent(
            event_type="TestIntentPlanned",
            source_agent="测试状态智能体",
            target_agent="测试生成智能体",
            related_nodes=list(plan.get("target_states", []) or []),
            payload=plan,
            iteration=self.system.memory.load().iteration,
        )

    def _state_target_optimization_plan(self, graph: Any) -> tuple[dict[str, object], dict[str, object]]:
        report = self.system.baseline_report
        gaps = {
            str(key): float(value)
            for key, value in (report.get("threshold_failures", {}) or {}).items()
            if isinstance(value, (int, float)) and float(value) > 0
        }
        dominant = max(gaps.items(), key=lambda item: item[1])[0] if gaps else self._weak_focus(report)
        selected_agent, action_name = self._optimization_agent(dominant)
        targets = [
            node.node_id
            for node in graph.nodes.values()
            if node.repair_flag or not node.visited or node.mutation_status == "survived"
        ][:30]
        mutation_evidence = report.get("mutation_evidence") if isinstance(report.get("mutation_evidence"), dict) else {}
        survived_lines = [int(value) for value in mutation_evidence.get("survived_lines", []) or []]
        coverage_evidence = graph.metadata.get("last_coverage") if isinstance(graph.metadata.get("last_coverage"), dict) else {}
        missing_lines = [int(value) for value in coverage_evidence.get("missing_lines", []) or []]
        target_methods = list(dict.fromkeys(
            node.name for node in graph.nodes.values()
            if node.node_id in targets and node.node_type in {"Function", "Async", "Method", "Constructor"}
        ))[:12]
        knowledge_context = graph.metadata.get("last_knowledge_context", {})
        optimization_contract = {
            "experiment_round": self.system.experiment_round,
            "baseline_grade": self.system.previous_grade,
            "baseline_metrics": {
                key: report.get(key)
                for key in ("line_coverage", "branch_coverage", "sfc", "ae", "mutation_score", "sfq", "tsq")
            },
            "dominant_gap": dominant,
            "threshold_gap": float(gaps.get(dominant, 0.0)),
            "target_states": targets,
            "target_methods": target_methods,
            "missing_lines": missing_lines[:40],
            "survived_mutation_lines": survived_lines[:40],
            "knowledge_context": knowledge_context,
            "constraints": [
                "preserve every currently passing test unless a specific weak oracle is replaced",
                "change only tests connected to the dominant gap and target evidence",
                "add concrete behavioral assertions; do not add smoke, existence, or tautological assertions",
                "do not reduce line coverage, branch coverage, SFC, AE, mutation score, or TSQ",
                "accept the candidate only after semantic patch search proves measurable improvement",
            ],
            "llm_required_for_strategy": bool(self.system.llm and self.system.llm.enabled()),
        }
        graph.metadata["state_target_optimization"] = optimization_contract
        graph.metadata["c_grade_optimization_contract"] = optimization_contract
        graph.metadata["optimization_prompt_version"] = "state-target-optimization-v3"
        self.system.memory.save(graph)
        suggestion = {
            "target_id": targets[0] if targets else "test_suite",
            "issue": dominant,
            "action": action_name,
            "priority": float(gaps.get(dominant, 1.0)),
            "reason": f"R{self.system.experiment_round} preserves the runnable C-grade suite and targets only its largest quality gap: {dominant}",
            "target_lines": list(dict.fromkeys([*survived_lines, *missing_lines]))[:60],
        }
        plan = {
            "phase": "followup_quality_optimization",
            "action": "optimize",
            "owner": "测试状态智能体",
            "selected_agent": selected_agent,
            "reason": "C-grade follow-up keeps passing tests, retrieves knowledge, and applies a focused quality enhancement instead of regenerating from scratch",
            "gaps": gaps,
            "test_path": str(self.system.existing_test_path),
            "focus": dominant,
            "feedback_summary": {"baseline_grade": self.system.previous_grade, "baseline_report": report},
            "target_states": targets,
            "target_methods": target_methods,
            "target_branches": [],
            "missing_lines": missing_lines,
            "survived_mutation_lines": survived_lines,
            "expected_gain": max(gaps.values(), default=0.1),
            "assigned_agent": "测试生成智能体",
            "capability": f"focused_{dominant}_optimization",
        }
        feedback = {
            "quality_status": report.get("quality_status", "valid_usable"),
            "usable": True,
            "recommended_focus": dominant,
            "reason": plan["reason"],
            "gaps": gaps,
            "suggestions": [suggestion],
            "evidence": {"baseline_report": report, "knowledge_context": knowledge_context, "optimization_contract": optimization_contract},
        }
        return plan, feedback

    def _weak_focus(self, report: dict[str, object]) -> str:
        flags = " ".join(str(value) for value in report.get("weak_quality_flags", []) or [])
        if "mutation" in flags:
            return "mutation"
        if "assert" in flags or "oracle" in flags:
            return "ae"
        if "branch" in flags:
            return "branch_coverage"
        if "line" in flags or "coverage" in flags:
            return "line_coverage"
        return "sfc"

    def _optimization_agent(self, gap: str) -> tuple[str, str]:
        if "mutation" in gap:
            return "MutationAgent", "Need Mutation"
        if "ae" in gap or "assert" in gap or "oracle" in gap:
            return "AssertionAgent", "Need Assert"
        if "branch" in gap or "boundary" in gap:
            return "BoundaryAgent", "Need Boundary"
        return "CoverageAgent", "Need Path"

    def _repair_test_state(self, state: TestWorkflowState) -> dict[str, object]:
        attempts = int(state.get("state_repair_attempts", 0)) + 1
        if attempts > 2:
            return {
                "state_repair_attempts": attempts,
                "next_action": "end",
                "stop_reason": "test_state_invalid",
                "status": "failed",
            }
        health = self.system.test_state_agent.self_repair(
            self.system.source_path,
            reason=str(state.get("last_error") or "test state health check failed"),
        )
        graph = self.system.memory.load()
        if not health.valid:
            return {
                "state_repair_attempts": attempts,
                "tsm_version": graph.version,
                "next_action": "state_repair" if attempts < 2 else "end",
                "stop_reason": "" if attempts < 2 else "test_state_invalid",
                "status": "running" if attempts < 2 else "failed",
                "last_error": ",".join(health.issues),
            }
        return {
            "state_repair_attempts": attempts,
            "tsm_version": graph.version,
            "next_action": "knowledge",
            "last_error": "",
        }

    def _retrieve_knowledge(self, state: TestWorkflowState) -> dict[str, object]:
        try:
            context = self.system.test_knowledge_agent.retrieve(
                state.get("current_intent", {}),
                model=self.system.llm_config.model,
            )
            return {"knowledge_context": context.to_dict(), "next_action": "generate" if state.get("test_path") else "bootstrap"}
        except Exception as exc:
            return {
                "knowledge_context": {"status": "degraded", "error": f"{exc.__class__.__name__}:{exc}"},
                "next_action": "generate",
            }

    def _generate_repair_and_execute(self, state: TestWorkflowState) -> dict[str, object]:
        budget_reason = self.system._budget_reason(int(state.get("iteration", 0)) + 1, float(state.get("started_at", time.perf_counter())))
        if budget_reason:
            if self._screening_result_needs_final(state.get("pending_before_report")):
                return {"next_action": "finalize", "status": "running"}
            return {"next_action": "end", "stop_reason": budget_reason, "status": "incomplete"}
        test_path = Path(str(state.get("test_path") or ""))
        if not test_path.exists():
            return {"next_action": "state_repair", "last_error": "generated_test_missing"}

        repair_rounds = int(state.get("repair_rounds_used", 0))
        pending_before = state.get("pending_before_report")
        reports = list(state.get("reports", []))
        pending_runtime = 0.0
        pending_usage: dict[str, int | float] | None = None
        if int(state.get("iteration", 0)) > 0:
            if repair_rounds >= self._repair_limit():
                if self._screening_result_needs_final(pending_before):
                    return {"next_action": "finalize", "status": "running"}
                return {"next_action": "end", "stop_reason": "quality_target_not_reached", "status": "incomplete"}
            plan_data = state.get("current_intent", {})
            measurement = CURRENT.get()
            if measurement:
                measurement.begin_action(plan_data, self.system.memory.load(), test_path)
            feedback_data = state.get("evaluation_feedback", {})
            plan = SimpleNamespace(**plan_data)
            suggestions = [SimpleNamespace(**item) for item in feedback_data.get("suggestions", []) if isinstance(item, dict)]
            feedback = SimpleNamespace(suggestions=suggestions)
            before_hash = self.system._file_hash(test_path)
            graph_before = self.system.memory.load()
            version_before = graph_before.version
            usage_before = self.system._usage_snapshot()
            repair_started = time.perf_counter()
            repaired = self.system._run_selected_agent(str(plan_data.get("selected_agent") or "TestValidityAgent"), test_path, plan, feedback)
            pending_runtime = time.perf_counter() - repair_started
            usage_after = self.system._usage_snapshot()
            pending_usage = self.system._usage_delta(usage_before, usage_after)
            after_hash = self.system._file_hash(test_path)
            graph_after = self.system.memory.load()
            file_changed = before_hash != after_hash
            repair_rounds += 1
            if reports:
                reports[-1]["repair"] = {
                    "repair_applied": repaired,
                    "repair_accepted": bool(repaired and file_changed),
                    "test_file_changed": file_changed,
                    "graph_changed": graph_after.version != version_before,
                    "before_hash": before_hash,
                    "after_hash": after_hash,
                    "llm_usage": pending_usage,
                    "runtime_seconds": round(pending_runtime, 4),
                }
            if not repaired or not file_changed:
                if CURRENT.get():
                    CURRENT.get().finish_action(unchanged=True)
                if self._screening_result_needs_final(pending_before):
                    return {
                        "reports": reports,
                        "repair_rounds_used": repair_rounds,
                        "next_action": "finalize",
                        "status": "running",
                    }
                if isinstance(pending_before, dict):
                    try:
                        preserved = self._report(pending_before)
                    except (TypeError, ValueError):
                        preserved = None
                    if preserved is not None and self.system.thresholds.usable(preserved):
                        preserved.quality_assessment = self.system.thresholds.assess(preserved)
                        preserved.quality_status = str(preserved.quality_assessment["status"])
                        return {
                            "current_report": preserved.to_dict(),
                            "reports": reports,
                            "repair_rounds_used": repair_rounds,
                            "next_action": "end",
                            "stop_reason": "plateau_reached",
                            "status": "incomplete",
                        }
                return {
                    "reports": reports,
                    "repair_rounds_used": repair_rounds,
                    "next_action": "end",
                    "stop_reason": "plateau_reached" if pending_before and pending_before.get("pytest_passed") else "repair_failed",
                    "status": "incomplete",
                }

        usage_before_execution = self.system._usage_snapshot()
        iteration_started = time.perf_counter()
        report = self.system.test_generation_agent.execute(
            test_path,
            cwd=self.system.output_dir.parent,
            evaluation_profile="screening",
        )
        graph = self.system.memory.load()
        graph.metadata["last_report"] = report.to_dict()
        self.system.memory.save(graph)
        if CURRENT.get():
            CURRENT.get().observe(report.to_dict(), graph, test_path)
        return {
            "current_report": report.to_dict(),
            "reports": reports,
            "iteration": int(state.get("iteration", 0)) + 1,
            "repair_rounds_used": repair_rounds,
            "pending_runtime": pending_runtime,
            "pending_usage": pending_usage,
            "next_action": "evaluate",
            "status": "running",
            "last_execution_runtime": round(time.perf_counter() - iteration_started, 4),
            "last_execution_usage": self.system._usage_delta(usage_before_execution, self.system._usage_snapshot()),
            "final_evaluation_done": False,
        }

    def _finalize_execution(self, state: TestWorkflowState) -> dict[str, object]:
        test_path = Path(str(state.get("test_path") or ""))
        if not test_path.exists():
            return {"next_action": "state_repair", "last_error": "final_test_missing"}
        usage_before = self.system._usage_snapshot()
        started = time.perf_counter()
        report = self.system.test_generation_agent.execute(
            test_path,
            cwd=self.system.output_dir.parent,
            evaluation_profile="full",
        )
        graph = self.system.memory.load()
        graph.metadata["last_report"] = report.to_dict()
        graph.metadata["final_evaluation"] = {
            "completed": True,
            "profile": "full",
            "test_path": str(test_path.resolve()),
        }
        self.system.memory.save(graph)
        if CURRENT.get():
            CURRENT.get().observe(report.to_dict(), graph, test_path)
        return {
            "current_report": report.to_dict(),
            "iteration": int(state.get("iteration", 0)) + 1,
            "next_action": "evaluate",
            "status": "running",
            "last_execution_runtime": round(time.perf_counter() - started, 4),
            "last_execution_usage": self.system._usage_delta(usage_before, self.system._usage_snapshot()),
            "final_evaluation_done": True,
        }

    def _evaluate_and_plan(self, state: TestWorkflowState) -> dict[str, object]:
        report = self._report(state.get("current_report", {}))
        before = self._report(state["pending_before_report"]) if state.get("pending_before_report") else None
        metric_delta = self.system._metric_delta(before, report)
        try:
            self.system.test_generation_agent.validate(report)
            feedback, decision, plan, health = self.system.test_state_agent.evaluate_and_plan(
                report=report,
                thresholds=self.system.thresholds,
                experience=self.system.experience,
                pending_decision=state.get("pending_decision"),
                pending_before_report=before,
                pending_runtime=float(state.get("pending_runtime", 0.0)),
                pending_usage=state.get("pending_usage"),
                model=self.system.llm_config.model,
            )
        except RuntimeError as exc:
            return {"next_action": "state_repair", "last_error": str(exc)}

        plan_data = plan.to_dict()
        feedback_data = feedback.to_dict()
        if (
            self.system.followup_strategy == "optimize_existing"
            and int(state.get("repair_rounds_used", 0)) < self._repair_limit()
            and not state.get("final_evaluation_done")
        ):
            graph = self.system.memory.load()
            plan_data, feedback_data = self._state_target_optimization_plan(graph)
            graph.metadata["last_test_intent_plan"] = plan_data
            graph.metadata["last_evaluation_feedback"] = feedback_data
            self.system.memory.save(graph)

        report_data = report.to_dict()
        # The public task summary omits process metrics. Keep the evaluated
        # report in the private graph so a resumed follow-up round has the
        # same candidate ranking and gap evidence as an uninterrupted run.
        graph = self.system.memory.load()
        graph.metadata["last_report"] = report_data
        self.system.memory.save(graph)
        report_data.update(
            {
                "iteration": int(state.get("iteration", 0)),
                "repair_round": int(state.get("repair_rounds_used", 0)),
                "scheduler_action": decision.action,
                "test_intent_plan": plan_data,
                "evaluation_feedback": feedback_data,
                "before_metrics": before.to_dict() if before else None,
                "after_metrics": report.to_dict(),
                "metric_delta": metric_delta,
                "state_health": health.to_dict(),
            }
        )
        reports = list(state.get("reports", []))
        reports.append(report_data)
        decisions = [*state.get("decisions", []), decision.to_dict()]
        plans = [*state.get("plans", []), plan_data]
        stop_reason = ""
        next_action = "knowledge"
        status = "running"
        repair_limit = self._repair_limit()
        if state.get("final_evaluation_done"):
            if self.system.thresholds.usable(report):
                should_continue_target_optimization = bool(
                    self.system.followup_strategy == "optimize_existing"
                    and report.quality_status == "valid_usable"
                    and int(state.get("repair_rounds_used", 0)) < repair_limit
                )
                if should_continue_target_optimization:
                    stop_reason = ""
                    next_action = "knowledge"
                    status = "running"
                else:
                    stop_reason = report.quality_status
                    next_action = "end"
                    status = "completed" if report.quality_status in {"quality_high", "quality_acceptable"} else "incomplete"
            else:
                stop_reason, next_action, status = "execution_invalid", "end", "failed"
        elif report.failure_category in {"java_tool_missing", "java_execution_unavailable"}:
            stop_reason, next_action, status = "execution_invalid", "end", "failed"
        elif not self.system.thresholds.usable(report) and (
            int(state.get("repair_rounds_used", 0)) >= repair_limit
            or decision.action in {"stop", "verify"}
        ):
            stop_reason, next_action, status = "execution_invalid", "end", "failed"
        elif report.quality_status == "quality_high":
            if self._screening_result_needs_final(report.to_dict()):
                stop_reason, next_action, status = "", "finalize", "running"
            else:
                stop_reason, next_action, status = "quality_high", "end", "completed"
        elif int(state.get("repair_rounds_used", 0)) >= repair_limit:
            if self._screening_result_needs_final(report.to_dict()):
                stop_reason, next_action, status = "", "finalize", "running"
            else:
                stop_reason, next_action, status = "quality_target_not_reached", "end", "incomplete"
        elif decision.action in {"stop", "verify"}:
            if self._screening_result_needs_final(report.to_dict()):
                stop_reason, next_action, status = "", "finalize", "running"
            else:
                stop_reason, next_action, status = "quality_target_not_reached", "end", "incomplete"
        return {
            "current_report": report.to_dict(),
            "reports": reports,
            "decisions": decisions,
            "plans": plans,
            "current_intent": plan_data,
            "evaluation_feedback": feedback_data,
            "pending_before_report": report.to_dict(),
            "pending_decision": decision.to_dict(),
            "next_action": next_action,
            "stop_reason": stop_reason,
            "status": status,
        }

    def _update_knowledge(self, state: TestWorkflowState) -> dict[str, object]:
        if state.get("next_action") == "finalize":
            return {}
        try:
            self.system.test_knowledge_agent.record_outcome(
                state.get("current_report", {}),
                state.get("current_intent", {}),
            )
        except Exception as exc:
            return {"last_error": f"knowledge_update:{exc.__class__.__name__}:{exc}"}
        return {}

    def _route(self, state: TestWorkflowState) -> Literal["bootstrap", "state_repair", "knowledge", "generate", "evaluate", "finalize", "end"]:
        action = str(state.get("next_action") or "end")
        if action in {"bootstrap", "state_repair", "knowledge", "generate", "evaluate", "finalize"}:
            return action  # type: ignore[return-value]
        return "end"

    def _screening_result_needs_final(self, report: dict[str, object] | None) -> bool:
        if not isinstance(report, dict):
            return False
        if str(report.get("evaluation_profile") or "full") != "screening":
            return False
        try:
            parsed = self._report(report)
        except (TypeError, ValueError):
            return False
        return self.system.thresholds.usable(parsed)

    def _repair_limit(self) -> int:
        return min(3, max(0, int(self.system.max_repair_rounds)))

    def _report(self, data: dict[str, Any]) -> ExecutionReport:
        allowed = {field.name for field in fields(ExecutionReport)}
        return ExecutionReport(**{key: value for key, value in data.items() if key in allowed})

    def _summary(self, state: TestWorkflowState, started: float) -> dict[str, object]:
        graph = self.system.memory.load()
        generation_validation = graph.metadata.get("test_generation_validation", {})
        generation_validation = generation_validation if isinstance(generation_validation, dict) else {}
        generation_rejection = graph.metadata.get("test_generation_candidate_rejected", {})
        generation_rejection = generation_rejection if isinstance(generation_rejection, dict) else {}
        last_error = str(state.get("last_error") or graph.metadata.get("llm_generation_error") or "").strip()
        failed_validation = generation_rejection or (generation_validation if generation_validation.get("accepted") is False else {})
        if not last_error and failed_validation:
            validation_detail = str(failed_validation.get("pytest_output") or "").strip().splitlines()
            detail = validation_detail[0] if validation_detail else str(failed_validation.get("stage") or "candidate validation failed")
            last_error = f"{failed_validation.get('failure_category') or 'candidate_rejected'}: {detail}"
        final_report = state.get("current_report") or None
        final_grade = grade_for((final_report or {}).get("quality_status", "generation_failed"), final_report, execute=True)
        reports = list(state.get("reports", []))
        if reports:
            reports[-1]["stop_reason"] = state.get("stop_reason")
        summary = {
            "source": str(self.system.source_path),
            "source_sha256": hashlib.sha256(self.system.source_path.read_bytes()).hexdigest(),
            "test_path": state.get("test_path"),
            "graph_path": str(self.system.graph_path),
            "orchestration": {
                "runtime": "LangGraph StateGraph",
                "workflow": "non_linear_test_state_guidance",
                "primary_agents": ["TestStateAgent", "TestGenerationAgent", "TestKnowledgeAgent"],
                "primary_agent_names": ["测试状态智能体", "测试生成智能体", "测试知识智能体"],
                "test_generation_internal_modules": ["generation", "candidate_validation", "execution_validation", "repair", "optimization"],
                "test_state_internal_modules": ["state_analysis", "state_repair", "quality_evaluation", "gap_analysis", "next_task_planning"],
                "validation_owner": "TestGenerationAgent",
                "result_evaluation_owner": "TestStateAgent",
                "evaluation_strategy": "screening_then_full_final",
                "state_repair_attempts": state.get("state_repair_attempts", 0),
                "checkpoint_path": str(self.checkpoint_path.resolve()),
            },
            "iterations": state.get("iteration", 0),
            "repair_rounds_used": state.get("repair_rounds_used", 0),
            "max_repair_rounds": self.system.max_repair_rounds,
            "effective_repair_rounds": self._repair_limit(),
            "retry_rounds": self.system.retry_rounds,
            "thresholds": self.system.thresholds.to_dict(),
            "final_report": final_report,
            "reports": reports,
            "agent_decisions": state.get("decisions", []),
            "test_intent_plans": state.get("plans", []),
            "stop_reason": state.get("stop_reason") or "incomplete",
            "last_error": last_error or None,
            "failure_evidence": {
                "stage": failed_validation.get("stage") or ("source_analysis" if str(last_error).startswith("analysis:") else None),
                "generation_strategy": graph.metadata.get("test_generation_strategy"),
                "llm_generation_error": graph.metadata.get("llm_generation_error"),
                "candidate_validation": failed_validation or None,
                "state_repair_attempts": state.get("state_repair_attempts", 0),
            },
            "quality_status": (final_report or {}).get("quality_status", "generation_failed"),
            "final_grade": final_grade,
            "experiment_round": self.system.experiment_round,
            "grade_label": round_label(final_grade, self.system.experiment_round),
            "followup_strategy": self.system.followup_strategy,
            "experience": graph.metadata.get("experience_hints", {}),
            "knowledge_context": graph.metadata.get("last_knowledge_context", {}),
            "knowledge_history": graph.metadata.get("knowledge_retrieval_history", []),
            "knowledge_outcome": graph.metadata.get("last_knowledge_outcome", {}),
            "experience_summary": self.system.experience.summary(),
            "visualization": {},
            "graph_version": graph.version,
            "llm": self.system.llm_config.public_dict(),
            "llm_usage": self.system._usage_snapshot(),
            "runtime_seconds": round(time.perf_counter() - started, 4),
        }
        summary_path = self.system.output_dir / "stateflow_summary.json"
        if CURRENT.get():
            summary["paper_metrics"] = CURRENT.get().summary(final_report or {}, summary["runtime_seconds"], completed=True)
        # The renderer reads paper_metrics/summary.json. Finalize that sidecar
        # first so Time and other measured values agree across JSON and HTML.
        summary["visualization"] = self.system._render_visualizations(graph)
        from core.public_metrics import public_result

        public_summary = public_result(summary)
        summary_path.write_text(json.dumps(public_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return public_summary
