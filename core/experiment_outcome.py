from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


GRADE_BY_STATUS = {
    "generated": "A",
    "quality_high": "A",
    "quality_acceptable": "B",
    "valid_usable": "C",
    "quality_target_not_reached": "C",
    "plateau_reached": "C",
    "metric_unreliable": "C",
    "budget_limited": "C",
    "execution_invalid": "D",
    "failed": "D",
    "generation_failed": "E",
    "not_started": "E",
    "interrupted": "E",
}

GRADE_LABELS = {
    "A": "高质量",
    "B": "良好",
    "C": "可用",
    "D": "无效",
    "E": "未完成",
}

VALID_GRADES = {"A", "B", "C"}
SECOND_ROUND_GRADES = {"D", "E"}
THIRD_ROUND_GRADES = {"C"}

FAILURE_SOLUTIONS = {
    "syntax_error": "检查并修复生成测试的语法，再重新进行候选验证。",
    "compile_error": "根据编译器错误修复类型、包名、方法签名和依赖声明。",
    "import_error": "核对源码模块路径、测试导入方式和缺失的Python依赖。",
    "collection_error": "修复测试文件命名、测试函数结构或框架收集配置。",
    "fixture_error": "补充或替换缺失夹具，避免依赖当前任务不存在的测试环境。",
    "runtime_error": "根据首个运行异常定位输入、调用方式或环境问题，并执行状态定位修复。",
    "oracle_error": "重新推导期望值，优先使用明确的输入输出、状态变化或异常类型断言。",
    "assertion_failure": "核对测试意图和实际行为，修正错误预期并保留有效断言。",
    "timeout": "缩小输入规模，规避长循环、深递归和外部等待，并限制单次执行时间。",
    "exception_only": "补充正常输入输出测试，不能只验证异常路径。",
    "no_normal_behavior": "为每个可测试公开目标补充至少一个正常行为测试。",
    "weak_assertions_only": "将恒真、非空或对象存在性断言替换为具体行为断言。",
    "coverage_unavailable": "修复覆盖工具配置，确保测试通过后能够生成有效覆盖数据。",
    "java_tool_missing": "配置可用的JDK、Maven或Gradle后再执行Java测试。",
    "java_execution_unavailable": "检查Java构建入口、依赖和测试插件配置。",
    "candidate_rejected": "根据候选验证报告修复语法、导入、正常行为和断言质量。",
    "test_state_invalid": "由测试状态智能体重建源码结构和测试状态后再规划。",
    "generation_failed": "读取生成阶段和候选验证证据，修复明确原因后重新生成可执行测试。",
    "llm_generation_failed": "检查模型调用错误、响应格式和预算，再结合已有知识重新生成测试。",
    "source_analysis_failed": "检查源码可解析性和结构提取结果，由测试状态智能体修复状态后重新规划。",
    "budget_limited": "核对时间、模型调用和令牌预算，保留已有证据后继续未完成任务。",
    "unknown": "保留完整错误和执行证据，由测试状态智能体重新归因后再选择修复策略。",
}


def grade_for(status: object, report: dict[str, Any] | None = None, execute: bool = True) -> str:
    status_text = str(status or "")
    report = report if isinstance(report, dict) else {}
    quality_status = str(report.get("quality_status") or status_text)
    if not execute and status_text == "generated":
        return "A"
    if execute:
        if not report:
            return "D" if status_text == "execution_invalid" else "E"
        if not report.get("pytest_passed", False):
            return "D"
    return GRADE_BY_STATUS.get(quality_status, GRADE_BY_STATUS.get(status_text, "E"))


def round_label(grade: str, experiment_round: int) -> str:
    return f"{grade}\u00b7R{max(1, int(experiment_round or 1))}"


def failure_category(item: dict[str, Any]) -> str:
    report = item.get("final_report") if isinstance(item.get("final_report"), dict) else {}
    diagnostic = item.get("round_failure_diagnostic") if isinstance(item.get("round_failure_diagnostic"), dict) else {}
    outcome = item.get("experiment_outcome") if isinstance(item.get("experiment_outcome"), dict) else {}
    direct = diagnostic.get("failure_category") or outcome.get("failure_category") or report.get("failure_category") or item.get("failure_category")
    if direct:
        return str(direct)
    error = " ".join(
        str(value or "")
        for value in (
            diagnostic.get("error"),
            outcome.get("error"),
            report.get("failure_detail"),
            report.get("error"),
            item.get("error"),
            item.get("last_error"),
            item.get("stop_reason"),
        )
    ).lower()
    checks = [
        "syntax_error",
        "compile_error",
        "import_error",
        "collection_error",
        "fixture_error",
        "runtime_error",
        "oracle_error",
        "assertion_failure",
        "timeout",
        "exception_only",
        "no_normal_behavior",
        "weak_assertions_only",
        "java_tool_missing",
        "java_execution_unavailable",
        "candidate_rejected",
        "test_state_invalid",
    ]
    for category in checks:
        if category in error or category.replace("_", " ") in error:
            return category
    if "no valid generated test candidate" in error:
        return "candidate_rejected"
    if "llm" in error or "model" in error or "response schema" in error:
        return "llm_generation_failed"
    if "analysis:" in error or "source analysis" in error:
        return "source_analysis_failed"
    if "budget" in error:
        return "budget_limited"
    if str(item.get("status") or report.get("quality_status") or "") == "generation_failed":
        return "generation_failed"
    return "unknown"


def failure_reason(item: dict[str, Any]) -> str:
    report = item.get("final_report") if isinstance(item.get("final_report"), dict) else {}
    diagnostic = item.get("round_failure_diagnostic") if isinstance(item.get("round_failure_diagnostic"), dict) else {}
    outcome = item.get("experiment_outcome") if isinstance(item.get("experiment_outcome"), dict) else {}
    category = failure_category(item)
    detail = (
        diagnostic.get("failure_reason")
        or outcome.get("failure_reason")
        or report.get("failure_detail")
        or report.get("error")
        or report.get("stderr")
        or item.get("error")
        or item.get("last_error")
        or item.get("stop_reason")
        or f"任务在形成可执行测试和完整评估证据前停止（状态：{item.get('status') or 'unknown'}）"
    )
    first_line = str(detail).strip().splitlines()[0] if str(detail).strip() else category
    if first_line.startswith(f"{category}:"):
        return first_line[:320]
    return f"{category}: {first_line[:300]}"


def default_solution(category: str) -> str:
    return FAILURE_SOLUTIONS.get(category, FAILURE_SOLUTIONS["unknown"])


def aggregate_results(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = [item for item in results if isinstance(item, dict)]
    grade_counts: Counter[str] = Counter(str(item.get("final_grade") or "E") for item in items)
    round_counts: Counter[int] = Counter(int(item.get("experiment_round", 1) or 1) for item in items)
    grade_round_counts = {
        grade: {f"R{round_no}": sum(1 for item in items if item.get("final_grade") == grade and int(item.get("experiment_round", 1) or 1) == round_no) for round_no in (1, 2, 3)}
        for grade in "ABCDE"
    }
    valid = sum(grade_counts[grade] for grade in VALID_GRADES)
    total = len(items)
    return {
        "total_tasks": total,
        "valid_results": valid,
        "invalid_results": grade_counts["D"],
        "unfinished_results": grade_counts["E"],
        "completed_execution": valid,
        "not_completed_execution": grade_counts["D"] + grade_counts["E"],
        "completion_rate": round(valid / total, 4) if total else 0.0,
        "grade_counts": {grade: grade_counts[grade] for grade in "ABCDE"},
        "grade_round_counts": grade_round_counts,
        "final_round_counts": {f"R{round_no}": round_counts[round_no] for round_no in (1, 2, 3)},
        "balance_valid": total == valid + grade_counts["D"] + grade_counts["E"],
    }
