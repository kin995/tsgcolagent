from __future__ import annotations

import argparse
import html
import json
import math
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from batch_generate_tests import generate_tests_for_directory
from core.project_paths import resolve_datasets_root, resolve_runs_root
from core.public_metrics import public_result
from graph.experience_memory import configure_knowledge_store_path
from llm_config import LLMConfig
from main import Thresholds


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source_dir: Path
    output_dir: Path
    language: str
    execute: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source_dir": str(self.source_dir),
            "output_dir": str(self.output_dir),
            "language": self.language,
            "execute": self.execute,
        }


def run_all_datasets(
    datasets_root: str | Path | None = None,
    output_root: str | Path | None = None,
    llm_config: LLMConfig | None = None,
    max_iterations: int = 3,
    retry_rounds: int = 2,
    retry_failed_rounds: int = 2,
    limit_per_dataset: int | None = None,
    execute_java: bool = True,
    resume: bool = True,
    rerun_existing: bool = False,
    fail_fast: bool = False,
    quiet: bool = False,
    python_workers: int = 4,
    human_eval_java_workers: int = 2,
    knowledge_store: str | Path | None = None,
) -> dict[str, object]:
    configure_knowledge_store_path(knowledge_store)
    datasets_root = resolve_datasets_root(datasets_root)
    output_root = resolve_runs_root(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    llm_config = llm_config or LLMConfig.from_env()
    specs = discover_dataset_specs(datasets_root, output_root, execute_java=execute_java)
    started_at = _utc_now()
    results: list[dict[str, object]] = []

    for index, spec in enumerate(specs, start=1):
        if not quiet:
            print(f"[{index}/{len(specs)}] {spec.name} -> {spec.output_dir}")
        try:
            summary = generate_tests_for_directory(
                source_dir=spec.source_dir,
                output_dir=spec.output_dir,
                language=spec.language,
                execute=spec.execute,
                max_iterations=max_iterations,
                retry_rounds=retry_rounds,
                thresholds=Thresholds(),
                llm_config=llm_config,
                limit=limit_per_dataset,
                fail_fast=False,
                verbose=not quiet,
                retry_failed_rounds=retry_failed_rounds,
                resume=resume,
                rerun_existing=rerun_existing,
                workers=(
                    python_workers
                    if spec.language == "python"
                    else human_eval_java_workers
                    if spec.name == "HumanEvalJava"
                    else 1
                ),
                knowledge_store=knowledge_store,
            )
            results.append(
                {
                    "dataset": spec.to_dict(),
                    "status": "skipped_completed" if summary.get("dataset_skipped") else "completed",
                    "summary_path": str((spec.output_dir / "batch_summary.json").resolve()),
                    "batch_report": _batch_report_path(summary),
                    "summary": _compact_dataset_summary(summary),
                }
            )
        except Exception as exc:
            result = {
                "dataset": spec.to_dict(),
                "status": "failed",
                "error": f"{exc.__class__.__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            }
            results.append(result)
            if fail_fast:
                break

        master_summary = _build_master_summary(
            datasets_root=datasets_root,
            output_root=output_root,
            llm_config=llm_config,
            started_at=started_at,
            results=results,
            total_specs=len(specs),
        )
        _write_master_summary(output_root, master_summary)

    master_summary = _build_master_summary(
        datasets_root=datasets_root,
        output_root=output_root,
        llm_config=llm_config,
        started_at=started_at,
        results=results,
        total_specs=len(specs),
    )
    _write_master_summary(output_root, master_summary)
    return public_result(master_summary)


def discover_dataset_specs(datasets_root: Path, output_root: Path, execute_java: bool = True) -> list[DatasetSpec]:
    if not datasets_root.is_dir():
        raise FileNotFoundError(f"Datasets root not found: {datasets_root}")

    specs: list[DatasetSpec] = []
    known_python = [("TestEval", "TestEval"), ("HumanEval", "HumanEval")]
    for name, dirname in known_python:
        source = datasets_root / dirname
        if source.is_dir():
            specs.append(DatasetSpec(name=name, source_dir=source, output_dir=output_root / name, language="python", execute=True))

    human_eval_java = datasets_root / "HumanEvalJava"
    if human_eval_java.is_dir():
        specs.append(DatasetSpec(name="HumanEvalJava", source_dir=human_eval_java, output_dir=output_root / "HumanEvalJava", language="java", execute=execute_java))

    defects_root = datasets_root / "Defects4J"
    for project in ["cli", "csv", "lang3", "gson"]:
        source = defects_root / project
        if source.is_dir():
            specs.append(
                DatasetSpec(
                    name=f"Defects4J_{project}",
                    source_dir=source,
                    output_dir=output_root / f"Defects4J_{project}",
                    language="java",
                    execute=execute_java,
                )
            )

    known_sources = {str(spec.source_dir.resolve()) for spec in specs}
    for source in sorted(datasets_root.iterdir()):
        if not source.is_dir() or str(source.resolve()) in known_sources or source.name == "Defects4J":
            continue
        language = _infer_language(source)
        if language:
            specs.append(
                DatasetSpec(
                    name=_safe_name(source.name),
                    source_dir=source,
                    output_dir=output_root / _safe_name(source.name),
                    language=language,
                    execute=True if language == "python" else execute_java,
                )
            )
    return specs


def _infer_language(source: Path) -> str | None:
    py_count = sum(1 for _ in source.rglob("*.py"))
    java_count = sum(1 for _ in source.rglob("*.java"))
    if py_count == 0 and java_count == 0:
        return None
    return "java" if java_count > py_count else "python"


def _compact_dataset_summary(summary: dict[str, object]) -> dict[str, object]:
    quality = summary.get("quality_overview") if isinstance(summary.get("quality_overview"), dict) else {}
    outcome = summary.get("outcome_summary") if isinstance(summary.get("outcome_summary"), dict) else {}
    return {
        "paper_metrics": summary.get("paper_metrics", {}),
        "total": summary.get("total"),
        "discovered_files": summary.get("discovered_files"),
        "selected_tasks": summary.get("selected_tasks"),
        "resumed_existing": summary.get("resumed_existing"),
        "started_at": summary.get("started_at"),
        "finished_at": summary.get("finished_at"),
        "session_duration_seconds": summary.get("session_duration_seconds"),
        "duration_seconds": summary.get("cumulative_duration_seconds", summary.get("duration_seconds")),
        "cumulative_duration_seconds": summary.get("cumulative_duration_seconds", summary.get("duration_seconds")),
        "succeeded": summary.get("succeeded"),
        "weak_success": summary.get("weak_success"),
        "failed": summary.get("failed"),
        "incomplete": summary.get("incomplete"),
        "mode": summary.get("mode"),
        "language": summary.get("language"),
        "effective_pattern": summary.get("effective_pattern"),
        "retry_summary": summary.get("retry_summary"),
        "experiment_round_summary": summary.get("experiment_round_summary"),
        "outcome_summary": outcome,
        "dataset_skipped": summary.get("dataset_skipped", False),
        "failure_tasks": [
            {
                "source": item.get("source"),
                "grade_label": item.get("grade_label"),
                "failure_reason": item.get("failure_reason"),
                "previous_solutions": item.get("previous_solutions", []),
                "failure_report": item.get("failure_report"),
            }
            for item in summary.get("results", [])
            if isinstance(item, dict) and item.get("final_grade") in {"D", "E"}
        ],
        "metrics": {
            "executed_tasks": quality.get("executed_tasks"),
            "average_line_coverage": quality.get("average_line_coverage"),
            "average_branch_coverage": quality.get("average_branch_coverage"),
            "average_ae": quality.get("average_ae"),
            "average_mutation_score": quality.get("average_mutation_score"),
        },
    }


def _batch_report_path(summary: dict[str, object]) -> str | None:
    report = summary.get("batch_report")
    if isinstance(report, dict):
        path = report.get("batch_report_html")
        return str(path) if path else None
    if report:
        return str(report)
    return None


def _build_master_summary(
    datasets_root: Path,
    output_root: Path,
    llm_config: LLMConfig,
    started_at: str,
    results: list[dict[str, object]],
    total_specs: int,
) -> dict[str, object]:
    finished_at = _utc_now()
    compact = [item.get("summary", {}) for item in results if isinstance(item.get("summary"), dict)]

    def total(key: str) -> int:
        return sum(int(item.get(key, 0) or 0) for item in compact)

    grade_counts = {
        grade: sum(
            int(((item.get("outcome_summary") or {}).get("grade_counts") or {}).get(grade, 0) or 0)
            for item in compact
            if isinstance(item.get("outcome_summary"), dict)
        )
        for grade in "ABCDE"
    }
    total_tasks = sum(grade_counts.values())
    valid_results = grade_counts["A"] + grade_counts["B"] + grade_counts["C"]
    session_duration = _duration_seconds(started_at, finished_at)
    cumulative_duration = round(sum(float(item.get("cumulative_duration_seconds") or item.get("duration_seconds") or 0.0) for item in compact), 4)
    first_starts = [str(item.get("started_at")) for item in compact if item.get("started_at")]
    return {
        "datasets_root": str(datasets_root),
        "output_root": str(output_root),
        "started_at": min(first_starts) if first_starts else started_at,
        "first_started_at": min(first_starts) if first_starts else started_at,
        "current_session_started_at": started_at,
        "finished_at": finished_at,
        "session_duration_seconds": session_duration,
        "duration_seconds": cumulative_duration or session_duration,
        "cumulative_duration_seconds": cumulative_duration or session_duration,
        "llm": llm_config.public_dict(),
        "total_datasets": total_specs,
        "completed_datasets": sum(1 for item in results if item.get("status") in {"completed", "skipped_completed"}),
        "skipped_completed_datasets": sum(1 for item in results if item.get("status") == "skipped_completed"),
        "failed_datasets": sum(1 for item in results if item.get("status") == "failed"),
        "total_tasks": total_tasks or total("total"),
        "selected_tasks": total("selected_tasks"),
        "resumed_existing": total("resumed_existing"),
        "succeeded": valid_results,
        "weak_success": grade_counts["C"],
        "failed": grade_counts["D"],
        "unfinished": grade_counts["E"],
        "incomplete": grade_counts["D"] + grade_counts["E"],
        "outcome_summary": {
            "total_tasks": total_tasks,
            "valid_results": valid_results,
            "completed_execution": valid_results,
            "not_completed_execution": grade_counts["D"] + grade_counts["E"],
            "grade_counts": grade_counts,
            "balance_valid": total_tasks == valid_results + grade_counts["D"] + grade_counts["E"],
        },
        "test_knowledge_guidance_library": _knowledge_library_paths(),
        "results": results,
        "summary_path": str((output_root / "all_datasets_summary.json").resolve()),
    }


def _write_master_summary(output_root: Path, summary: dict[str, object]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    public_summary = public_result(summary)
    (output_root / "all_datasets_summary.json").write_text(json.dumps(public_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "all_datasets_report.html").write_text(_render_master_report(public_summary), encoding="utf-8")


def _knowledge_library_paths() -> dict[str, str]:
    try:
        from graph.experience_memory import resolve_knowledge_store_path

        knowledge_path = resolve_knowledge_store_path()
        store_dir = knowledge_path.parent
        return {
            "name": "测试知识引导库",
            "html": str((store_dir / "test_knowledge_guidance_library.html").resolve()),
            "summary": str((store_dir / "test_knowledge_guidance_summary.json").resolve()),
            "knowledge": str(knowledge_path.resolve()),
        }
    except Exception:
        return {}


def _duration_seconds(started_at: str, finished_at: str) -> float:
    try:
        started = datetime.fromisoformat(started_at)
        finished = datetime.fromisoformat(finished_at)
    except ValueError:
        return 0.0
    return round(max(0.0, (finished - started).total_seconds()), 4)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_")
    return cleaned or "dataset"


def _render_master_report(summary: dict[str, object]) -> str:
    rows: list[str] = []
    failures: list[str] = []
    output_root = Path(str(summary.get("output_root") or ".")).resolve()
    paper_rows: list[tuple[int, dict[str, object]]] = []
    for item in summary.get("results", []):
        if not isinstance(item, dict):
            continue
        dataset = item.get("dataset") if isinstance(item.get("dataset"), dict) else {}
        compact = item.get("summary") if isinstance(item.get("summary"), dict) else {}
        metrics = compact.get("metrics") if isinstance(compact.get("metrics"), dict) else {}
        paper = compact.get("paper_metrics") if isinstance(compact.get("paper_metrics"), dict) else {}
        paper_rows.append((int(compact.get("selected_tasks") or 0), paper))
        outcome = compact.get("outcome_summary") if isinstance(compact.get("outcome_summary"), dict) else {}
        grades = outcome.get("grade_counts") if isinstance(outcome.get("grade_counts"), dict) else {}
        rows.append(
            "<tr>"
            f"<td>{_esc(dataset.get('name'))}</td>"
            f"<td>{_esc(dataset.get('language'))}</td>"
            f"<td>{_esc(item.get('status'))}</td>"
            f"<td>{_esc(compact.get('selected_tasks'))}</td>"
            f"<td>{_esc(compact.get('resumed_existing'))}</td>"
            f"<td>{_esc(_fmt_duration(compact.get('duration_seconds')))}</td>"
            f"<td>{_esc(grades.get('A', 0))}</td><td>{_esc(grades.get('B', 0))}</td><td>{_esc(grades.get('C', 0))}</td>"
            f"<td>{_esc(grades.get('D', 0))}</td><td>{_esc(grades.get('E', 0))}</td>"
            f"<td>{_esc(_fmt_metric(metrics.get('average_line_coverage')))}</td>"
            f"<td>{_esc(_fmt_metric(metrics.get('average_branch_coverage')))}</td>"
            f"<td>{_esc(_fmt_metric(metrics.get('average_ae')))}</td>"
            f"<td>{_esc(_fmt_metric(metrics.get('average_mutation_score')))}</td>"
            f"<td>{_esc(_fmt_paper_rate(paper.get('tir')))}</td>"
            f"<td>{_esc(_fmt_paper_number(paper.get('avg_exec')))}</td>"
            f"<td>{_link(item.get('summary_path'), 'JSON', output_root)}</td>"
            f"<td>{_link(item.get('batch_report'), 'HTML', output_root)}</td>"
            "</tr>"
        )
        for failure in compact.get("failure_tasks", []):
            if not isinstance(failure, dict):
                continue
            solutions = failure.get("previous_solutions", []) if isinstance(failure.get("previous_solutions"), list) else []
            failures.append(
                "<tr>"
                f"<td>{_esc(dataset.get('name'))}</td><td>{_esc(Path(str(failure.get('source') or '')).name)}</td>"
                f"<td><b>{_esc(failure.get('grade_label'))}</b></td><td>{_esc(failure.get('failure_reason'))}</td>"
                f"<td>{_esc('；'.join(str(value) for value in solutions))}</td>"
                f"<td>{_link(failure.get('failure_report'), '失败报告', output_root)}</td></tr>"
            )
    if not rows:
        rows.append("<tr><td colspan=\"19\">没有运行数据集</td></tr>")
    failure_body = "".join(failures) or '<tr><td colspan="6">暂无D/E任务</td></tr>'
    outcome = summary.get("outcome_summary") if isinstance(summary.get("outcome_summary"), dict) else {}
    grades = outcome.get("grade_counts") if isinstance(outcome.get("grade_counts"), dict) else {}
    knowledge = summary.get("test_knowledge_guidance_library") if isinstance(summary.get("test_knowledge_guidance_library"), dict) else {}
    knowledge_link = _link(knowledge.get("html"), "打开测试知识引导库", output_root)
    selected_total = sum(selected for selected, _paper in paper_rows)
    comparable = sum(int(paper.get("tir_comparable") or 0) for _selected, paper in paper_rows)
    improved = sum(int(paper.get("tir_improved") or 0) for _selected, paper in paper_rows)
    exec_complete = sum(int(paper.get("exec_complete_tasks") or 0) for _selected, paper in paper_rows)
    overall_tir = improved / comparable if comparable else None
    observed_exec = [
        (int(paper.get("exec_complete_tasks") or 0), float(paper["avg_exec"]))
        for _selected, paper in paper_rows
        if int(paper.get("exec_complete_tasks") or 0) > 0
        and isinstance(paper.get("avg_exec"), (int, float))
        and math.isfinite(paper["avg_exec"])
    ]
    observed_total = sum(count for count, _average in observed_exec)
    overall_avg_exec = (
        sum(average * count for count, average in observed_exec) / observed_total
        if observed_total else None
    )

    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>全数据集测试总报告</title>"
        "<style>"
        "body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#17202a;background:#f8fafc}"
        "h1{font-size:24px;margin:0 0 8px}.sub{color:#5d6879}.panel{margin-top:20px}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:16px 0 24px}"
        ".metric{background:white;border:1px solid #d8dee9;border-radius:8px;padding:12px}"
        ".metric b{display:block;font-size:22px;margin-top:6px}"
        "table{width:100%;border-collapse:collapse;background:white;border:1px solid #d8dee9}"
        "th,td{padding:8px 10px;border-bottom:1px solid #e5e9f0;text-align:left;font-size:13px;vertical-align:top}"
        "th{background:#edf2f7}"
        "a{color:#1f5fbf;text-decoration:none}"
        "</style></head><body>"
        "<h1>全数据集测试总报告</h1><p class=\"sub\">总任务 = A + B + C + D + E；已完整执行 = A + B + C。</p>"
        "<div class=\"grid\">"
        f"{_metric_card('数据集', summary.get('completed_datasets'), summary.get('total_datasets'))}"
        f"{_metric_card('整目录跳过', summary.get('skipped_completed_datasets'))}"
        f"{_metric_card('总任务', outcome.get('total_tasks', summary.get('total_tasks')))}"
        f"{_metric_card('累计实验时长', _fmt_duration(summary.get('duration_seconds')))}"
        f"{_metric_card('已完整执行', outcome.get('completed_execution'))}"
        f"{_metric_card('A 高质量', grades.get('A', 0))}{_metric_card('B 良好', grades.get('B', 0))}{_metric_card('C 可用', grades.get('C', 0))}"
        f"{_metric_card('D 无效', grades.get('D', 0))}{_metric_card('E 未完成', grades.get('E', 0))}"
        f"{_metric_card('TIR', _fmt_paper_rate(overall_tir))}{_metric_card('AvgExec', _fmt_paper_number(overall_avg_exec))}"
        "</div>"
        f"<p class=\"sub\">TIR 改善 {improved}/{comparable} 个可比较目标；AvgExec 按 {observed_total}/{selected_total} 项有完整计数的任务计算，缺证任务未纳入。</p>"
        f"<p>{knowledge_link}</p><section class=\"panel\"><h2>数据集结果</h2><table><thead><tr>"
        "<th>数据集</th><th>语言</th><th>状态</th><th>任务</th><th>恢复</th><th>累计耗时</th>"
        "<th>A</th><th>B</th><th>C</th><th>D</th><th>E</th><th>平均行覆盖</th><th>平均分支覆盖</th>"
        "<th>平均AE</th><th>平均变异</th><th>TIR</th><th>AvgExec</th><th>JSON</th><th>报告</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></section><section class=\"panel\"><h2>D/E失败与未完成任务</h2><table><thead><tr><th>数据集</th><th>任务</th><th>最终等级</th><th>具体原因</th><th>之前的解决措施</th><th>失败报告</th></tr></thead><tbody>"
        + failure_body
        + "</tbody></table></section></body></html>"
    )


def _metric_card(label: str, value: object, total: object | None = None) -> str:
    text = f"{value}/{total}" if total is not None else value
    return f"<div class=\"metric\"><span>{_esc(label)}</span><b>{_esc(text)}</b></div>"


def _link(path: object, label: str, output_root: Path) -> str:
    if not path:
        return ""
    href = str(path)
    try:
        href_path = Path(href).resolve()
        href = href_path.relative_to(output_root).as_posix()
    except (OSError, ValueError):
        href = str(path)
    return f"<a href=\"{_esc(href)}\">{_esc(label)}</a>"


def _fmt_metric(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def _fmt_paper_rate(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1:
        return f"{value * 100:.2f}%"
    return "—"


def _fmt_paper_number(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return f"{value:.2f}"
    return "—"


def _fmt_duration(value: object) -> str:
    try:
        total = max(0, int(float(value or 0)))
    except (TypeError, ValueError):
        return ""
    days, rest = divmod(total, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, seconds = divmod(rest, 60)
    clock = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}天 {clock}" if days else clock


def _esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run every bundled dataset with resume-safe dataset-level orchestration.")
    parser.add_argument("--datasets-root", default=str(resolve_datasets_root()), help="Dataset root. Defaults to the project datasets directory, TSG_DATASETS_ROOT, or <DATASETS>\\datasets.")
    parser.add_argument("--out-root", default=str(resolve_runs_root()), help="Root output directory for all dataset runs.")
    parser.add_argument("--llm", action="store_true", help="Use the configured local DeepSeek service.")
    parser.add_argument("--llm-required", action="store_true", help="Fail instead of falling back if the LLM call fails.")
    parser.add_argument("--llm-base-url", default=None, help="Local model service URL.")
    parser.add_argument("--llm-model", default=None, help="Local model name.")
    parser.add_argument("--llm-api-key", default=None, help="API key placeholder for local compatible servers.")
    parser.add_argument("--llm-timeout", type=int, default=None, help="LLM request timeout in seconds.")
    parser.add_argument("--llm-temperature", type=float, default=None, help="LLM sampling temperature.")
    parser.add_argument("--execute-java", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-java-execution", action="store_true", help="Only generate Java JUnit tests and skip Java JUnit/JaCoCo/PIT execution.")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum repair iterations per task.")
    parser.add_argument("--retry-rounds", type=int, default=2, help="Maximum pytest timeout retry rounds per task.")
    parser.add_argument("--retry-failed-rounds", type=int, default=2, help="Compatibility option: 2 means R1 baseline, R2 D/E recovery, and R3 C optimization; no R4.")
    parser.add_argument("--limit-per-dataset", type=int, default=None, help="Only process the first N files of each dataset.")
    parser.add_argument("--python-workers", type=int, default=4, help="Parallel task count inside Python datasets.")
    parser.add_argument("--humanevaljava-workers", type=int, default=2, help="Isolated HumanEvalJava class workers (1-4). Other Java datasets remain sequential.")
    parser.add_argument("--knowledge-store", default=None, help="External long-term knowledge JSON path. Existing files are never overwritten by project defaults.")
    parser.add_argument("--no-resume", action="store_true", help="Do not reuse existing per-task summaries.")
    parser.add_argument("--rerun-existing", action="store_true", help="Force rerunning tasks even if summaries already exist.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed dataset.")
    parser.add_argument("--quiet", action="store_true", help="Hide per-file progress inside each dataset.")
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
    launch_dashboard(args.out_root, label="全部数据集实验", enabled=not args.no_ui, llm_config=llm_config)
    failure = None
    try:
        summary = run_all_datasets(
            datasets_root=args.datasets_root,
            output_root=args.out_root,
            llm_config=llm_config,
            max_iterations=args.max_iterations,
            retry_rounds=args.retry_rounds,
            retry_failed_rounds=args.retry_failed_rounds,
            limit_per_dataset=args.limit_per_dataset,
            execute_java=not args.skip_java_execution,
            resume=not args.no_resume,
            rerun_existing=args.rerun_existing,
            fail_fast=args.fail_fast,
            quiet=args.quiet,
            python_workers=args.python_workers,
            human_eval_java_workers=args.humanevaljava_workers,
            knowledge_store=args.knowledge_store,
        )
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        finish_dashboard(args.out_root, success=failure is None, error=failure, enabled=not args.no_ui)
    print(json.dumps(_terminal_summary(summary), indent=2, ensure_ascii=False))


def _terminal_summary(summary: dict[str, object]) -> dict[str, object]:
    return {
        "total_datasets": summary.get("total_datasets"),
        "completed_datasets": summary.get("completed_datasets"),
        "skipped_completed_datasets": summary.get("skipped_completed_datasets"),
        "failed_datasets": summary.get("failed_datasets"),
        "total_tasks": summary.get("total_tasks"),
        "selected_tasks": summary.get("selected_tasks"),
        "resumed_existing": summary.get("resumed_existing"),
        "outcome_summary": summary.get("outcome_summary"),
        "test_knowledge_guidance_library": summary.get("test_knowledge_guidance_library"),
        "output_root": summary.get("output_root"),
        "summary_path": summary.get("summary_path"),
        "datasets": [
            {
                "name": item.get("dataset", {}).get("name") if isinstance(item.get("dataset"), dict) else None,
                "status": item.get("status"),
                "summary": item.get("summary"),
                "summary_path": item.get("summary_path"),
                "batch_report": item.get("batch_report"),
                "error": item.get("error"),
            }
            for item in summary.get("results", [])
            if isinstance(item, dict)
        ],
    }


if __name__ == "__main__":
    main()
