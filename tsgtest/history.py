"""本地运行历史：扫描 runs 目录、读取详情、生成可导出的报告。"""
from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

ARTIFACTS = ("state_flow_graph.html", "source_coverage.html", "state_flow_graph.svg")


def _read(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _mtime(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return ""


def _first_source(job: dict[str, Any]) -> str:
    result = job.get("result") or {}
    items = result.get("items") if isinstance(result, dict) else None
    if isinstance(items, list) and items:
        first = items[0]
        if isinstance(first, dict):
            return str(first.get("source") or "")
    return ""


def list_runs(runs_dir: Path) -> list[dict[str, Any]]:
    if not runs_dir.is_dir():
        return []
    runs = []
    for folder in runs_dir.iterdir():
        if not folder.is_dir():
            continue
        summary_path = folder / "stateflow_summary.json"
        job_path = folder / "tsgtest_job.json"
        if not summary_path.is_file() and not job_path.is_file():
            continue
        summary = _read(summary_path)
        job = _read(job_path)
        final = summary.get("final_report") or {}
        runs.append({
            "name": folder.name,
            "source": summary.get("source") or _first_source(job),
            "status": job.get("status") or summary.get("quality_status") or "",
            "quality": summary.get("quality_status") or "",
            "created_at": job.get("created_at") or _mtime(folder),
            "iterations": summary.get("iterations"),
            "line_coverage": final.get("line_coverage"),
            "branch_coverage": final.get("branch_coverage"),
            "mutation_score": final.get("mutation_score"),
        })
    runs.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return runs


def _safe_dir(runs_dir: Path, name: str) -> Path | None:
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        return None
    candidate = (runs_dir / name).resolve()
    try:
        if candidate.parent != runs_dir.resolve():
            return None
    except OSError:
        return None
    return candidate if candidate.is_dir() else None


def resolve_run(runs_dir: Path, name: str) -> Path | None:
    return _safe_dir(runs_dir, name)


def load_run(runs_dir: Path, name: str) -> dict[str, Any] | None:
    folder = _safe_dir(runs_dir, name)
    if folder is None:
        return None
    summary = _read(folder / "stateflow_summary.json")
    job = _read(folder / "tsgtest_job.json")
    final = summary.get("final_report") or {}
    paper = summary.get("paper_metrics") or {}
    rounds = []
    for report in summary.get("reports") or []:
        if not isinstance(report, dict):
            continue
        rounds.append({
            "round": report.get("iteration"),
            "passed": report.get("pytest_passed"),
            "line_coverage": report.get("line_coverage"),
            "branch_coverage": report.get("branch_coverage"),
            "ae": report.get("ae"),
            "mutation_score": report.get("mutation_score"),
            "tests": report.get("collected_count"),
        })
    decisions = []
    for decision in summary.get("agent_decisions") or []:
        if isinstance(decision, dict):
            decisions.append({
                "agent": decision.get("agent"),
                "action": decision.get("action"),
                "reason": decision.get("reason"),
            })
    test_text = None
    test_path = summary.get("test_path")
    if test_path and Path(str(test_path)).is_file():
        try:
            test_text = Path(str(test_path)).read_text(encoding="utf-8", errors="replace")
        except OSError:
            test_text = None
    if test_text is None:
        for candidate in sorted(folder.glob("test_*")):
            if candidate.suffix in {".py", ".java"}:
                test_text = candidate.read_text(encoding="utf-8", errors="replace")
                break
    artifacts = [name for name in ARTIFACTS if (folder / name).is_file()]
    return {
        "name": name,
        "source": summary.get("source") or _first_source(job),
        "status": job.get("status") or summary.get("quality_status") or "",
        "quality": summary.get("quality_status") or "",
        "stop_reason": summary.get("stop_reason") or "",
        "iterations": summary.get("iterations"),
        "created_at": job.get("created_at") or _mtime(folder),
        "model": (summary.get("llm") or {}).get("model") or "",
        "metrics": {
            "line_coverage": final.get("line_coverage"),
            "branch_coverage": final.get("branch_coverage"),
            "ae": final.get("ae"),
            "mutation_score": final.get("mutation_score"),
            "pass_rate": paper.get("pass_rate"),
            "tir": paper.get("tir"),
            "avg_exec": paper.get("avg_exec"),
            "duration_seconds": paper.get("duration_seconds"),
            "collected_count": final.get("collected_count"),
            "effective_assertions": final.get("effective_assertions"),
            "assertion_count": final.get("assertion_count"),
        },
        "llm": summary.get("llm_usage") or {},
        "rounds": rounds,
        "decisions": decisions,
        "test_text": test_text,
        "artifacts": artifacts,
        "error": job.get("error") or summary.get("last_error") or "",
    }


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def render_markdown(run: dict[str, Any]) -> str:
    metrics = run.get("metrics") or {}
    llm = run.get("llm") or {}
    lines = [
        "# TSGColAgent 运行报告",
        "",
        f"- 运行目录：`{run['name']}`",
        f"- 源文件：`{run.get('source') or '—'}`",
        f"- 状态：{run.get('status') or '—'}（{run.get('stop_reason') or '—'}）",
        f"- 模型：{run.get('model') or '—'}",
        f"- 轮次：{run.get('iterations')}",
        "",
        "## 指标",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 行覆盖 LC | {_pct(metrics.get('line_coverage'))} |",
        f"| 分支覆盖 BC | {_pct(metrics.get('branch_coverage'))} |",
        f"| 断言有效性 AE | {_pct(metrics.get('ae'))} |",
        f"| 变异得分 MS | {_pct(metrics.get('mutation_score'))} |",
        f"| 通过率 | {_pct(metrics.get('pass_rate'))} |",
        f"| 目标改善率 TIR | {_pct(metrics.get('tir'))} |",
        f"| 平均执行次数 | {metrics.get('avg_exec') if metrics.get('avg_exec') is not None else '—'} |",
        f"| 耗时 | {metrics.get('duration_seconds') if metrics.get('duration_seconds') is not None else '—'} s |",
        f"| 收集用例 | {metrics.get('collected_count') if metrics.get('collected_count') is not None else '—'} |",
        "",
        "## 每轮结果",
        "",
        "| 轮次 | 通过 | LC | BC | AE | MS | 用例 |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in run.get("rounds") or []:
        lines.append(
            f"| {item.get('round')} | {item.get('passed')} | {_pct(item.get('line_coverage'))} | "
            f"{_pct(item.get('branch_coverage'))} | {_pct(item.get('ae'))} | "
            f"{_pct(item.get('mutation_score'))} | {item.get('tests')} |"
        )
    lines += [
        "",
        "## AI 调用",
        "",
        f"- 调用次数：{llm.get('calls', '—')}",
        f"- 输入 tokens：{llm.get('prompt_tokens', '—')}",
        f"- 输出 tokens：{llm.get('completion_tokens', '—')}",
        f"- 模型耗时：{llm.get('duration_seconds', '—')} s",
        "",
    ]
    decisions = run.get("decisions") or []
    if decisions:
        lines += ["## 决策记录", ""]
        for decision in decisions:
            lines.append(f"- {decision.get('agent') or '—'} / {decision.get('action') or '—'}：{decision.get('reason') or ''}")
        lines.append("")
    if run.get("error"):
        lines += ["## 错误", "", str(run["error"]), ""]
    if run.get("test_text"):
        lines += ["## 生成的测试", "", "```", str(run["test_text"]).strip(), "```", ""]
    return "\n".join(lines)


def render_html(run: dict[str, Any]) -> str:
    metrics = run.get("metrics") or {}
    llm = run.get("llm") or {}
    rows = "".join(
        f"<tr><td>{item.get('round')}</td><td>{item.get('passed')}</td>"
        f"<td>{_pct(item.get('line_coverage'))}</td><td>{_pct(item.get('branch_coverage'))}</td>"
        f"<td>{_pct(item.get('ae'))}</td><td>{_pct(item.get('mutation_score'))}</td>"
        f"<td>{item.get('tests')}</td></tr>"
        for item in run.get("rounds") or []
    )
    decisions = "".join(
        f"<li><b>{html.escape(str(d.get('agent') or ''))}</b> / {html.escape(str(d.get('action') or ''))}："
        f"{html.escape(str(d.get('reason') or ''))}</li>"
        for d in run.get("decisions") or []
    )
    test_block = ""
    if run.get("test_text"):
        test_block = f"<h2>生成的测试</h2><pre>{html.escape(str(run['test_text']))}</pre>"
    error_block = f"<h2>错误</h2><pre>{html.escape(str(run['error']))}</pre>" if run.get("error") else ""
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>TSGColAgent 运行报告</title>
<style>
body{{font-family:"Microsoft YaHei",system-ui,sans-serif;max-width:900px;margin:32px auto;padding:0 16px;color:#1c2430;line-height:1.6}}
h1{{font-size:22px}} h2{{font-size:16px;margin-top:28px;border-bottom:1px solid #e3e8ee;padding-bottom:6px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border:1px solid #d7dee8;padding:6px 8px;text-align:left}}
th{{background:#eef3fa}} pre{{background:#f5f7fa;padding:12px;overflow:auto;font-size:12px}}
.muted{{color:#6b7785;font-size:13px}} .cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:12px 0}}
.card{{border:1px solid #e3e8ee;border-radius:8px;padding:8px 10px}} .card b{{display:block;font-size:18px}}
</style></head><body>
<h1>TSGColAgent 运行报告</h1>
<p class="muted">{html.escape(run.get('name') or '')} · {html.escape(run.get('source') or '')}</p>
<p>状态：{html.escape(run.get('status') or '—')}（{html.escape(run.get('stop_reason') or '—')}） · 模型：{html.escape(run.get('model') or '—')} · 轮次：{run.get('iterations')}</p>
<div class="cards">
<div class="card">LC<b>{_pct(metrics.get('line_coverage'))}</b></div>
<div class="card">BC<b>{_pct(metrics.get('branch_coverage'))}</b></div>
<div class="card">AE<b>{_pct(metrics.get('ae'))}</b></div>
<div class="card">MS<b>{_pct(metrics.get('mutation_score'))}</b></div>
</div>
<h2>每轮结果</h2>
<table><thead><tr><th>轮次</th><th>通过</th><th>LC</th><th>BC</th><th>AE</th><th>MS</th><th>用例</th></tr></thead>
<tbody>{rows}</tbody></table>
<h2>AI 调用</h2>
<p>调用 {llm.get('calls', '—')} 次 · 输入 {llm.get('prompt_tokens', '—')} tokens · 输出 {llm.get('completion_tokens', '—')} tokens · 耗时 {llm.get('duration_seconds', '—')} s</p>
<h2>决策记录</h2><ul>{decisions}</ul>
{error_block}
{test_block}
</body></html>"""
