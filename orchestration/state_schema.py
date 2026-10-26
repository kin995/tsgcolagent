from __future__ import annotations

from typing import Any, TypedDict


class TestWorkflowState(TypedDict, total=False):
    task_id: str
    source_path: str
    output_dir: str
    graph_path: str
    tsm_version: int
    test_path: str
    current_intent: dict[str, Any]
    knowledge_context: dict[str, Any]
    current_report: dict[str, Any]
    pending_before_report: dict[str, Any] | None
    pending_decision: dict[str, Any] | None
    pending_usage: dict[str, int | float] | None
    pending_runtime: float
    evaluation_feedback: dict[str, Any]
    reports: list[dict[str, Any]]
    plans: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    iteration: int
    repair_rounds_used: int
    state_repair_attempts: int
    next_action: str
    stop_reason: str
    status: str
    started_at: float
    last_error: str
    final_evaluation_done: bool
