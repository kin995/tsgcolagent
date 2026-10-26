from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.public_metrics import public_result, public_text


PROGRAM_TYPES = {
    "Function",
    "Class",
    "Interface",
    "Enum",
    "Record",
    "AnnotationType",
    "Branch",
    "Loop",
    "Return",
    "Call",
    "Exception",
    "Raise",
    "Throw",
    "End",
    "Async",
    "With",
    "Match",
    "Lambda",
    "Generator",
}

EVIDENCE_TYPES = {"TestCase", "Assertion", "Mutation", "RuntimeState", "RuntimeType", "Execution"}


class TestStateModelVisualizer:
    """Render offline reports from the shared TSG and execution evidence."""

    def render(self, graph: Any, output_dir: str | Path) -> dict[str, object]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        model = self._load_model(graph)
        paper_path = output / "paper_metrics" / "summary.json"
        if paper_path.is_file():
            try:
                paper_metrics = json.loads(paper_path.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                paper_metrics = {}
            if isinstance(paper_metrics, dict):
                model.setdefault("metadata", {})["paper_metrics"] = paper_metrics

        model = self._public_model(model)
        svg = self._svg(model)
        svg_path = output / "state_flow_graph.svg"
        html_path = output / "state_flow_graph.html"
        source_path = output / "source_coverage.html"
        svg_path.write_text(svg, encoding="utf-8")
        html_path.write_text(self._interactive_html(model, svg), encoding="utf-8")
        source_path.write_text(self._source_html(model), encoding="utf-8")
        return {
            "status": "generated",
            "graph_html": str(html_path.resolve()),
            "graph_svg": str(svg_path.resolve()),
            "source_coverage_html": str(source_path.resolve()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def render_batch_summary(self, batch_summary: dict[str, Any] | str | Path, output_dir: str | Path | None = None) -> dict[str, object]:
        if isinstance(batch_summary, (str, Path)):
            summary_path = Path(batch_summary)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            output = Path(output_dir or summary_path.parent)
        else:
            summary = batch_summary
            output = Path(output_dir or summary.get("output_dir") or ".")
        output.mkdir(parents=True, exist_ok=True)
        html_path = output / "batch_quality_report.html"
        html_path.write_text(self._batch_html(summary), encoding="utf-8")
        return {
            "status": "generated",
            "batch_report_html": str(html_path.resolve()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _load_model(self, graph: Any) -> dict[str, Any]:
        if isinstance(graph, (str, Path)):
            model = json.loads(Path(graph).read_text(encoding="utf-8"))
        else:
            model = graph.to_dict(include_snapshots=False)
        model.pop("snapshots", None)
        return model

    def _public_model(self, model: dict[str, Any]) -> dict[str, Any]:
        """Keep the internal verdict while removing process scores before rendering."""
        verdict = self._summary(model)
        visible = public_result(model)
        if not isinstance(visible.get("metadata"), dict):
            visible["metadata"] = {}
        visible["metadata"]["_display_verdict"] = {
            "label": verdict["label"],
            "tone": verdict["tone"],
            "explanation": verdict["explanation"],
        }
        return visible

    def _summary(self, model: dict[str, Any]) -> dict[str, Any]:
        nodes = list(model.get("nodes", {}).values())
        edges = list(model.get("edges", []))
        metadata = model.get("metadata", {}) if isinstance(model.get("metadata"), dict) else {}
        report = metadata.get("last_report", {}) if isinstance(metadata.get("last_report"), dict) else {}
        coverage = metadata.get("last_coverage", {}) if isinstance(metadata.get("last_coverage"), dict) else {}
        mutation = metadata.get("last_mutation", {}) if isinstance(metadata.get("last_mutation"), dict) else {}
        usability = metadata.get("last_test_usability", {}) if isinstance(metadata.get("last_test_usability"), dict) else {}
        paper = metadata.get("paper_metrics", {}) if isinstance(metadata.get("paper_metrics"), dict) else {}

        program = [item for item in nodes if item.get("node_type") in PROGRAM_TYPES]
        tests = [item for item in nodes if item.get("node_type") == "TestCase"]
        assertions = [item for item in nodes if item.get("node_type") == "Assertion"]
        runtime = [item for item in nodes if item.get("node_type") == "RuntimeState"]
        mutations = [item for item in nodes if item.get("node_type") == "Mutation"]
        covered = sum(bool(item.get("visited")) for item in program)

        has_report = bool(report)
        data_status = "executed" if has_report else "not_executed"
        if has_report and "line_coverage" not in report:
            data_status = "legacy_executed"

        line_coverage = self._number(report.get("line_coverage"))
        branch_coverage = self._number(report.get("branch_coverage"))
        combined_coverage = self._number(report.get("combined_coverage", report.get("coverage_percent")))
        if line_coverage is None:
            line_coverage = self._number(coverage.get("line_coverage"))
        if branch_coverage is None:
            branch_coverage = self._number(coverage.get("branch_coverage"))
        if combined_coverage is None:
            combined_coverage = self._number(coverage.get("combined_coverage", coverage.get("percent")))

        ae = self._number(report.get("ae"))
        mutation_score = self._number(report.get("mutation_score", mutation.get("score")))
        tsq = self._number(report.get("tsq", report.get("sfq")))
        passed = bool(report.get("pytest_passed", False))
        normal_tests = int(report.get("normal_behavior_tests", usability.get("normal_behavior_tests", 0)) or 0)

        if not has_report:
            label, tone = "仅生成，未执行", "warn"
            explanation = "当前报告只有源码结构和生成文件信息，没有 pytest、coverage 或变异测试的真实执行结果。"
        elif not passed:
            label, tone = "执行无效", "danger"
            explanation = "pytest 未通过，覆盖率和变异检出不能作为有效质量结论。"
        elif tsq is not None and tsq >= 0.85 and normal_tests > 0:
            label, tone = "质量良好", "good"
            explanation = "测试已通过，并且覆盖、断言、运行证据和变异检出形成了可用证据。"
        else:
            label, tone = "可用，仍需优化", "warn"
            explanation = "测试可以执行，但仍可能存在未覆盖路径、断言不足或变异检出不足。"

        visible_verdict = metadata.get("_display_verdict", {}) if isinstance(metadata.get("_display_verdict"), dict) else {}
        if visible_verdict:
            label = str(visible_verdict.get("label") or label)
            tone = str(visible_verdict.get("tone") or tone)
            explanation = str(visible_verdict.get("explanation") or explanation)

        return {
            "nodes": nodes,
            "edges": edges,
            "program": program,
            "tests": tests,
            "assertions": assertions,
            "runtime": runtime,
            "mutations": mutations,
            "covered": covered,
            "line_coverage": line_coverage,
            "branch_coverage": branch_coverage,
            "combined_coverage": combined_coverage,
            "mutation": mutation_score,
            "ae": ae,
            "tir": self._number(paper.get("tir")),
            "tir_improved": int(paper.get("tir_improved", 0) or 0),
            "tir_comparable": int(paper.get("tir_comparable", 0) or 0),
            "tir_excluded": int(paper.get("tir_excluded", 0) or 0),
            "avg_exec": paper.get("avg_exec"),
            "exec_count_status": paper.get("exec_count_status"),
            "duration_seconds": paper.get("duration_seconds"),
            "pass_rate": paper.get("pass_rate"),
            "normal_tests": normal_tests,
            "label": label,
            "tone": tone,
            "explanation": explanation,
            "data_status": data_status,
            "report": report,
            "coverage": coverage,
            "metadata": metadata,
        }

    def _functions(self, summary: dict[str, Any]) -> list[dict[str, Any]]:
        functions = [item for item in summary["program"] if item.get("node_type") in {"Function", "Async"}]
        return sorted(
            functions,
            key=lambda item: (-self._function_risk(item, summary)[0], int(item.get("line_start", 0) or 0)),
        )

    def _function_risk(self, item: dict[str, Any], summary: dict[str, Any]) -> tuple[float, str]:
        start = int(item.get("line_start", 0) or 0)
        end = int(item.get("line_end", start) or start)
        program_nodes = [
            node for node in summary.get("program", [])
            if start <= int(node.get("line_start", 0) or 0) <= end
        ]
        node_ids = {str(node.get("node_id")) for node in program_nodes}
        edges = [
            edge for edge in summary.get("edges", [])
            if edge.get("source") in node_ids and edge.get("target") in node_ids
        ]
        traceable_edges = [
            edge for edge in edges
            if edge.get("metadata", {}).get("traceable", True)
        ]
        edge_gap = (
            sum(1 for edge in traceable_edges if not edge.get("visited")) / len(traceable_edges)
            if traceable_edges
            else 0.0
        )
        node_gap = (
            sum(1 for node in program_nodes if not node.get("visited")) / len(program_nodes)
            if program_nodes
            else 0.0
        )
        direct = max(
            float(item.get("coverage_defect", 0.0) or 0.0),
            float(item.get("assertion_defect", 0.0) or 0.0),
            float(item.get("mutation_defect", 0.0) or 0.0),
            float(item.get("exception_defect", 0.0) or 0.0),
            float(item.get("type_defect", 0.0) or 0.0),
        )
        missing_lines = set(summary.get("coverage", {}).get("missing_lines", []) or [])
        line_gap = 1.0 if any(start <= int(line) <= end for line in missing_lines) else 0.0
        survived_lines = set(summary.get("metadata", {}).get("last_mutation", {}).get("survived_lines", []) or [])
        mutation_gap = 1.0 if any(start <= int(line) <= end for line in survived_lines) else 0.0
        assertion_gap = 0.35 if item.get("visited") and int(item.get("assert_count", 0) or 0) <= 0 else 0.0
        risk = max(direct, line_gap, mutation_gap, node_gap, edge_gap * 0.65, assertion_gap)
        reasons = []
        if line_gap:
            reasons.append("存在未执行源码行")
        if mutation_gap:
            reasons.append("存在存活变异")
        if edge_gap >= 0.05:
            reasons.append("结构边证据不足")
        if node_gap >= 0.05:
            reasons.append("图节点未覆盖")
        if assertion_gap:
            reasons.append("函数缺少有效断言")
        if not reasons:
            reasons.append("暂无明显残留风险")
        return round(max(0.0, min(1.0, risk)), 4), "；".join(reasons[:3])

    def _svg(self, model: dict[str, Any]) -> str:
        summary = self._summary(model)
        width, height = 1200, 760
        tone = {"good": ("#dcfce7", "#166534"), "warn": ("#fef3c7", "#92400e"), "danger": ("#fee2e2", "#991b1b")}[summary["tone"]]
        metrics = [
            ("行覆盖", self._percent(summary["line_coverage"])),
            ("分支覆盖", self._percent(summary["branch_coverage"])),
            ("断言有效性", self._percent(summary["ae"])),
            ("变异检出", self._percent(summary["mutation"])),
            ("TIR", self._percent(summary["tir"])),
            ("AvgExec", self._decimal(summary["avg_exec"])),
        ]
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="测试质量摘要">',
            '<style>text{font-family:Inter,"Segoe UI","Microsoft YaHei",sans-serif}.muted{fill:#64748b}.title{fill:#0f172a;font-weight:700}</style>',
            '<rect width="1200" height="760" rx="24" fill="#f8fafc"/>',
            '<rect x="34" y="30" width="1132" height="100" rx="18" fill="#0f172a"/>',
            '<text x="62" y="68" fill="#94a3b8" font-size="14">ValiFixTest &amp; TSGColAgent 测试质量摘要</text>',
            f'<text x="62" y="104" fill="white" font-size="24" font-weight="700">{self._esc_literal(Path(str(model.get("source_path", "项目"))).name)}</text>',
            f'<rect x="946" y="58" width="190" height="42" rx="21" fill="{tone[0]}"/>',
            f'<text x="1041" y="85" text-anchor="middle" fill="{tone[1]}" font-size="16" font-weight="700">{summary["label"]}</text>',
        ]
        for index, (name, value) in enumerate(metrics):
            x = 34 + (index % 3) * 378
            y = 156 + (index // 3) * 112
            parts.extend([
                f'<rect x="{x}" y="{y}" width="350" height="92" rx="14" fill="white" stroke="#e2e8f0"/>',
                f'<text x="{x + 18}" y="{y + 31}" class="muted" font-size="13">{name}</text>',
                f'<text x="{x + 18}" y="{y + 70}" class="title" font-size="30">{value}</text>',
            ])
        parts.extend([
            '<text x="34" y="405" class="title" font-size="18">证据链路</text>',
            '<text x="34" y="430" class="muted" font-size="13">从源码结构、测试用例、断言、运行状态到变异样本的证据数量。</text>',
        ])
        stages = [
            ("分析代码", len(summary["program"]), "程序节点"),
            ("执行测试", len(summary["tests"]), "测试用例"),
            ("检查结果", len(summary["assertions"]), "断言"),
            ("记录运行", len(summary["runtime"]), "运行状态"),
            ("注入缺陷", len(summary["mutations"]), "变异样本"),
        ]
        for index, (name, count, unit) in enumerate(stages):
            x = 34 + index * 226
            parts.extend([
                f'<rect x="{x}" y="456" width="208" height="92" rx="14" fill="#eef2ff" stroke="#c7d2fe"/>',
                f'<text x="{x + 16}" y="486" fill="#4338ca" font-size="14" font-weight="700">{name}</text>',
                f'<text x="{x + 16}" y="524" class="title" font-size="26">{count}<tspan font-size="13" class="muted"> {unit}</tspan></text>',
            ])
        parts.extend([
            '<text x="34" y="602" class="title" font-size="18">最需要关注的函数</text>',
            '<text x="34" y="626" class="muted" font-size="13">以下函数仍有未覆盖或测试证据不足的情况。</text>',
        ])
        for index, item in enumerate(self._functions(summary)[:3]):
            y = 650 + index * 34
            state = "已覆盖" if item.get("visited") else "未覆盖"
            color = "#16a34a" if item.get("visited") else "#dc2626"
            _, reason = self._function_risk(item, summary)
            parts.extend([
                f'<rect x="34" y="{y}" width="1132" height="28" rx="8" fill="white"/>',
                f'<circle cx="52" cy="{y + 14}" r="5" fill="{color}"/>',
                f'<text x="68" y="{y + 19}" class="title" font-size="13">{self._esc_literal(item.get("name", ""))}</text>',
                f'<text x="770" y="{y + 19}" class="muted" font-size="12">第 {int(item.get("line_start", 0) or 0)} 行</text>',
                f'<text x="920" y="{y + 19}" fill="{color}" font-size="12" font-weight="700">{state}</text>',
                f'<text x="1060" y="{y + 19}" class="muted" font-size="12">{self._esc(reason[:18])}</text>',
            ])
        parts.append("</svg>")
        return "".join(parts)

    def _interactive_html(self, model: dict[str, Any], svg: str) -> str:
        summary = self._summary(model)
        source_name = self._esc_literal(Path(str(model.get("source_path", "项目"))).name)
        function_cards = "".join(self._function_card(item, summary) for item in self._functions(summary)[:10]) or '<p class="empty">暂无函数级数据。</p>'
        rows = self._evidence_rows(summary["nodes"])
        generated = datetime.now(timezone.utc).isoformat()
        report = summary["report"]
        test_path = str(report.get("test_path") or summary["metadata"].get("generated_test_path") or "未记录")
        data_label = {
            "executed": "真实执行数据",
            "legacy_executed": "旧格式执行数据",
            "not_executed": "未执行，仅生成",
        }.get(summary["data_status"], summary["data_status"])
        return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>单任务测试质量报告</title>{self._style()}</head>
<body><main class="wrap"><section class="hero"><div class="eyebrow">ValiFixTest &amp; TSGColAgent 单任务测试质量报告</div><h1>{source_name}</h1><p>{html.escape(summary["explanation"])}</p><p>以下LC、BC、AE、MS、PassRate、AvgExec、Time和TIR均为当前任务的实际记录，不是批量平均值；缺少采集证据时显示“未执行”。</p><span class="verdict {summary["tone"]}">{summary["label"]}</span></section>
<section class="metrics">
{self._metric("数据来源", data_label)}
{self._metric("测试用例", str(len(summary["tests"])) if summary["tests"] else self._dash())}
{self._metric("LC", self._percent(summary["line_coverage"]))}
{self._metric("BC", self._percent(summary["branch_coverage"]))}
{self._metric("AE", self._percent(summary["ae"]))}
{self._metric("MS", self._percent(summary["mutation"]))}
</section>
<section class="metrics secondary">
{self._metric("PassRate", self._percent(summary["pass_rate"]))}
{self._metric("AvgExec", self._decimal(summary["avg_exec"]))}
{self._metric("Time/s", self._decimal(summary["duration_seconds"]))}
{self._metric("TIR", self._percent(summary["tir"]))}
{self._metric("TIR改善/可比较", f'{summary["tir_improved"]}/{summary["tir_comparable"]}')}
{self._metric("TIR缺证", str(summary["tir_excluded"]))}
</section>
<section class="panel"><h2>数据来源</h2><p class="sub">本页优先读取测试生成智能体执行模块写入共享TSG的 <code>metadata.last_report</code>、<code>metadata.last_coverage</code> 和 <code>metadata.last_mutation</code>。如果状态是“未执行，仅生成”，就不能把页面当成质量结论。</p><div class="kv"><b>测试状态文件</b><span>{html.escape(str(summary["metadata"].get("graph_path", "")) or "当前 state_flow_graph.json")}</span><b>测试文件</b><span>{html.escape(str(test_path))}</span></div></section>
<section class="panel"><h2>评估指标说明</h2><p class="sub">LC和BC来自覆盖工具，AE反映断言质量，MS来自变异测试。AvgExec统计工具级测试执行会话；TIR按行动前登记的评价目标与实际保留结果汇总。</p><div class="formula"><b>TIR</b><span>改善目标数 ÷ 可比较目标数；同时保留缺证、未行动或中断记录。</span><b>AvgExec</b><span>当前任务的累计测试工具会话数；联合测试与覆盖计一次，缓存及仅编译或收集不计。</span></div></section>
<section class="panel"><h2>最需要关注的函数</h2><p class="sub">“高风险”表示该函数附近仍有覆盖、断言、异常、类型或变异方面的残留风险。它不是说代码一定有 bug，而是说测试证据还不够强。</p><div class="risks">{function_cards}</div></section>
<section class="panel snapshot"><h2>一页摘要</h2>{svg}</section>
<section class="panel"><h2>如何阅读</h2><div class="help"><div><b>已覆盖</b><br><small>测试执行到这段代码，但还要看断言和变异检出是否足够。</small></div><div><b>未覆盖</b><br><small>没有运行证据。若前面行覆盖显示 100%，请优先相信 coverage.py 的行覆盖，并把这里当作图节点级提示。</small></div><div><b>专业证据明细</b><br><small>下面表格是原始图节点，方便追查每个函数、断言、运行状态和变异证据。</small></div></div>
<details><summary>查看专业证据明细</summary><input id="search" class="search" placeholder="搜索节点名称、类型或 ID"><div class="table-wrap"><table><thead><tr><th>状态</th><th>类型</th><th>名称</th><th>行</th><th>证据 ID</th></tr></thead><tbody>{rows}</tbody></table></div></details></section>
<footer>离线报告，生成时间 {generated}</footer></main>{self._script()}</body></html>'''

    def _batch_html(self, summary: dict[str, Any]) -> str:
        summary = public_result(summary)
        results = summary.get("results", []) if isinstance(summary.get("results"), list) else []
        overview = summary.get("quality_overview", {}) if isinstance(summary.get("quality_overview"), dict) else self._batch_quality_overview(results)
        paper = summary.get("paper_metrics", {}) if isinstance(summary.get("paper_metrics"), dict) else {}
        outcome = summary.get("outcome_summary", {}) if isinstance(summary.get("outcome_summary"), dict) else {}
        grades = outcome.get("grade_counts", {}) if isinstance(outcome.get("grade_counts"), dict) else {}
        round_summary = summary.get("experiment_round_summary", {}) if isinstance(summary.get("experiment_round_summary"), dict) else {}
        followup_task_count = round_summary.get("followup_task_count", round_summary.get("unique_tasks", 0))
        executed_rows = []
        failure_rows = []
        for item in results:
            if not isinstance(item, dict):
                continue
            report = self._report_for_batch_item(item)
            item_paper = self._paper_metrics_for_batch_item(item)
            grade = str(item.get("final_grade") or ("C" if report and report.get("pytest_passed") else "E"))
            experiment_round = int(item.get("experiment_round", 1) or 1)
            grade_label = str(item.get("grade_label") or f"{grade}·R{experiment_round}")
            trajectory = " → ".join(
                str(entry.get("grade") or "E")
                for entry in item.get("grade_history", [])
                if isinstance(entry, dict)
            ) or grade
            reason = self._batch_item_reason(item, report)
            task_visualization = item.get("visualization", {}) if isinstance(item.get("visualization"), dict) else {}
            task_report_path = task_visualization.get("graph_html") or item.get("summary_path")
            executed_row = (
                "<tr>"
                f"<td>{self._esc(Path(str(item.get('source', ''))).name)}</td>"
                f"<td><b>{html.escape(grade_label)}</b></td>"
                f"<td>{html.escape(trajectory)}</td>"
                f"<td>{self._percent(report.get('line_coverage') if report else None)}</td>"
                f"<td>{self._percent(report.get('branch_coverage') if report else None)}</td>"
                f"<td>{self._percent(report.get('ae') if report else None)}</td>"
                f"<td>{html.escape(str(report.get('effective_assertions', '未执行') if report else '未执行'))}</td>"
                f"<td>{self._percent(report.get('mutation_score') if report else None)}</td>"
                f"<td>{self._percent(item_paper.get('tir'))}</td>"
                f"<td>{self._decimal(item_paper.get('avg_exec'))}</td>"
                f"<td>{self._decimal(item_paper.get('duration_seconds', item.get('duration_seconds')))}</td>"
                f"<td>{html.escape(reason)}</td>"
                f"<td>{self._link(task_report_path, '单任务报告')}</td>"
                "</tr>"
            )
            solutions = item.get("previous_solutions", []) if isinstance(item.get("previous_solutions"), list) else []
            failure_reason = str(item.get("failure_reason") or reason)
            failure_row = (
                "<tr>"
                f"<td>{self._esc(Path(str(item.get('source', ''))).name)}</td>"
                f"<td><b>{html.escape(grade_label)}</b></td>"
                f"<td>{html.escape(trajectory)}</td>"
                f"<td>{html.escape(failure_reason)}</td>"
                f"<td>{html.escape('；'.join(str(value) for value in solutions) or '没有匹配到历史解决措施')}</td>"
                f"<td>{self._link(task_report_path, '单任务报告')}</td>"
                f"<td>{self._link(item.get('failure_report'), '失败报告')}</td>"
                f"<td><code>{self._esc(item.get('output_dir', ''))}</code></td>"
                "</tr>"
            )
            if grade in {"A", "B", "C"}:
                executed_rows.append(executed_row)
            else:
                failure_rows.append(failure_row)
        generated = datetime.now(timezone.utc).isoformat()
        batch_panel = self._batch_experience_panel(summary)
        started_at = str(summary.get("started_at", ""))
        finished_at = str(summary.get("finished_at", ""))
        duration = self._duration_text(summary.get("duration_seconds"))
        executed_body = "".join(executed_rows) or '<tr><td colspan="13">暂无已通过并产生完整质量指标的任务</td></tr>'
        failure_body = "".join(failure_rows) or '<tr><td colspan="8">暂无D/E任务</td></tr>'
        knowledge = summary.get("test_knowledge_guidance_library", {}) if isinstance(summary.get("test_knowledge_guidance_library"), dict) else {}
        knowledge_link = self._link(knowledge.get("html"), "打开测试知识引导库")
        return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>批量测试质量总报告</title>{self._style()}</head><body><main class="wrap">
<section class="hero"><div class="eyebrow">ValiFixTest &amp; TSGColAgent 批量总报告</div><h1>批量测试质量总览</h1><p>总任务 = A + B + C + D + E；已完整执行 = A + B + C。轮次标记表示任务最后参与的实验轮次；本页带“平均”的指标按全部有效任务汇总，表格与单任务报告显示每个任务自己的实际值。</p></section>
<section class="metrics">{self._metric("总任务", str(outcome.get("total_tasks", len(results))))}{self._metric("已完整执行 A+B+C", str(outcome.get("completed_execution", 0)))}{self._metric("未完整执行 D+E", str(outcome.get("not_completed_execution", 0)))}{self._metric("A 高质量", str(grades.get("A", 0)))}{self._metric("B 良好", str(grades.get("B", 0)))}{self._metric("C 可用", str(grades.get("C", 0)))}</section>
<section class="metrics secondary">{self._metric("D 无效", str(grades.get("D", 0)))}{self._metric("E 未完成", str(grades.get("E", 0)))}{self._metric("进入后续轮次的任务数", str(followup_task_count))}{self._metric("后续轮次累计执行次数", str(round_summary.get("attempts", 0)))}{self._metric("恢复为有效结果", str(round_summary.get("recovered", 0)))}{self._metric("质量提升", str(round_summary.get("improved", 0)))}</section>
<section class="metrics secondary">{self._metric("有效断言总数", str(overview.get("total_effective_assertions", 0)))}{self._metric("弱断言总数", str(overview.get("total_weak_assertions", 0)))}{self._metric("变异评分任务", str(overview.get("mutation_scored_tasks", 0)))}</section>
<section class="metrics secondary">{self._metric("平均 LC", self._percent(overview.get("average_line_coverage")))}{self._metric("平均 BC", self._percent(overview.get("average_branch_coverage")))}{self._metric("平均 AE", self._percent(overview.get("average_ae")))}{self._metric("平均 MS", self._percent(overview.get("average_mutation_score")))}{self._metric("PassRate", self._percent(paper.get("pass_rate")))}{self._metric("TIR", self._percent(paper.get("tir")))}</section>
<section class="metrics secondary">{self._metric("AvgExec", self._decimal(paper.get("avg_exec")))}{self._metric("平均 Time/s", self._decimal(paper.get("average_time_seconds")))}{self._metric("TIR改善/可比较", f'{int(paper.get("tir_improved", 0) or 0)}/{int(paper.get("tir_comparable", 0) or 0)}')}{self._metric("TIR缺证", str(int(paper.get("tir_excluded", 0) or 0)))}{self._metric("执行计数完整任务", f'{int(paper.get("exec_complete_tasks", 0) or 0)}/{int(paper.get("selected_tasks", outcome.get("total_tasks", len(results))) or 0)}')}</section>
<section class="panel"><h2>运行与知识</h2><div class="kv"><b>开始时间</b><span>{html.escape(started_at or '未记录')}</span><b>结束时间</b><span>{html.escape(finished_at or '未记录')}</span><b>运行时长</b><span>{html.escape(duration)}</span><b>测试知识引导库</b><span>{knowledge_link or '未生成'}</span></div></section>
{batch_panel}<section class="panel"><h2>已完整执行任务（A/B/C）</h2><div class="table-wrap"><table class="compact"><thead><tr><th>源码</th><th>最终等级</th><th>等级轨迹</th><th>LC</th><th>BC</th><th>AE</th><th>有效断言</th><th>MS</th><th>TIR</th><th>AvgExec</th><th>Time/s</th><th>说明</th><th>报告</th></tr></thead><tbody>{executed_body}</tbody></table></div></section>
<section class="panel"><h2>D/E失败与未完成任务</h2><div class="table-wrap"><table class="compact fail-table"><thead><tr><th>源码</th><th>最终等级</th><th>等级轨迹</th><th>具体失败原因</th><th>之前的解决措施</th><th>任务报告</th><th>失败报告</th><th>输出目录</th></tr></thead><tbody>{failure_body}</tbody></table></div></section>
<section class="panel"><h2>计算说明</h2><p class="sub">TIR按改善目标数与可比较目标数计算；AvgExec按有完整测试工具会话计数的任务求均值，上方同时列出完整计数任务数。缺证任务不纳入均值，也不按0处理。只有测试可运行、存在正常行为和有效断言时才能进入A/B/C。</p></section><footer>离线报告，生成时间 {generated}</footer></main></body></html>'''

    def _batch_item_reason(self, item: dict[str, Any], report: dict[str, Any] | None) -> str:
        if str(item.get("final_grade") or "") in {"D", "E"} and item.get("failure_reason"):
            return str(item.get("failure_reason"))
        if report:
            if bool(report.get("timed_out")) or str(report.get("failure_category", "")) == "timeout":
                return "测试执行超时；系统会把该任务写入长期经验，下轮优先使用小输入并规避长循环、深递归和外部等待"
            if not report.get("pytest_passed"):
                category = report.get("failure_category") or item.get("status") or "unknown"
                detail = report.get("failure_detail") or report.get("error") or item.get("error") or category
                return f"{category}: {str(detail).splitlines()[0][:220]}"
            reliability = str(report.get("mutation_reliability") or "")
            effective = int(report.get("effective_mutants", 0) or 0)
            if reliability in {"low", "unavailable"} or effective < 3:
                reason = str(report.get("mutation_low_sample_reason") or "有效变异体少于 3 个")
                action = str(report.get("mutation_autonomy_action") or "下一轮应补边界、分支区分和杀变异断言")
                return f"测试可运行；变异样本偏少，有效变异体 {effective}。原因：{reason}。处理：{action}"
            return "测试可运行，指标来自真实执行"
        error = str(item.get("error") or "")
        if "exception_only" in error:
            return "低质量测试被拒绝：候选只有异常测试，缺少正常输入输出断言"
        if "no_normal_behavior" in error:
            return "低质量测试被拒绝：缺少正常行为测试"
        if "weak_assertions_only" in error:
            return "低质量测试被拒绝：断言过弱"
        if error:
            return error.splitlines()[0][:180]
        category = item.get("failure_category") or item.get("status") or "unknown"
        return f"没有可用执行报告；系统归因为 {category}，需要查看任务日志或生成阶段拒绝原因"

    def _duration_text(self, value: object) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "未记录"
        seconds = int(round(number))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}小时{minutes}分{secs}秒"
        if minutes:
            return f"{minutes}分{secs}秒"
        return f"{secs}秒"

    def _batch_quality_overview(self, results: list[Any]) -> dict[str, Any]:
        reports = [
            self._report_for_batch_item(item)
            for item in results
            if isinstance(item, dict)
        ]
        reports = [report for report in reports if report and report.get("pytest_passed")]

        def avg(key: str) -> float | None:
            values = [float(report[key]) for report in reports if report.get(key) is not None]
            return round(sum(values) / len(values), 4) if values else None

        def total(key: str) -> int:
            return sum(int(report.get(key, 0) or 0) for report in reports)

        return {
            "average_line_coverage": avg("line_coverage"),
            "average_branch_coverage": avg("branch_coverage"),
            "average_ae": avg("ae"),
            "average_mutation_score": avg("mutation_score"),
            "total_effective_assertions": total("effective_assertions"),
            "total_weak_assertions": total("weak_assertions"),
            "assertion_explanation": "有效断言=总断言数减去弱断言；弱断言包括恒真、None 比较、对象存在性检查等。",
        }

    def _batch_experience_panel(self, summary: dict[str, Any]) -> str:
        experience = summary.get("batch_experience")
        if not isinstance(experience, dict) or experience.get("status") == "experience_update_failed":
            return ""
        categories = experience.get("failure_categories", {}) if isinstance(experience.get("failure_categories"), dict) else {}
        guidance = experience.get("guidance", []) if isinstance(experience.get("guidance"), list) else []
        analysis = summary.get("failure_analysis", {}) if isinstance(summary.get("failure_analysis"), dict) else {}
        unknown_details = analysis.get("unknown_details", []) if isinstance(analysis.get("unknown_details"), list) else []
        category_text = "，".join(f"{html.escape(str(name))}: {int(count)}" for name, count in categories.items()) or "暂无失败类别"
        guidance_items = "".join(f"<li>{html.escape(self._translate_guidance(str(item)))}</li>" for item in guidance) or "<li>暂无新的批量经验</li>"
        unknown_items = "".join(
            f"<li>{self._esc(Path(str(item.get('source', ''))).name)}：{self._esc(item.get('error_summary', '没有错误摘要'))}</li>"
            for item in unknown_details[:10]
            if isinstance(item, dict)
        )
        unknown_panel = f"<p class=\"sub\">仍有 unknown 失败，表示执行报告和错误文本都不足以稳定归类；系统已保留摘要供下一轮经验修正。</p><ul>{unknown_items}</ul>" if unknown_items else ""
        timeout_tasks = experience.get("timeout_tasks", []) if isinstance(experience.get("timeout_tasks"), list) else []
        mutation_timeout_tasks = experience.get("mutation_timeout_tasks", []) if isinstance(experience.get("mutation_timeout_tasks"), list) else []
        timeout_text = ""
        if timeout_tasks or mutation_timeout_tasks:
            timeout_text = (
                f'<p class="sub">长期经验已记录超时任务：测试执行超时 {len(timeout_tasks)} 个，'
                f'变异超时 {len(mutation_timeout_tasks)} 个。后续生成会优先使用小输入、短序列，并规避可能卡住的循环、递归或外部等待。</p>'
            )
        return (
            '<section class="panel"><h2>批量经验</h2>'
            f'<p class="sub">本次运行已写入机器经验库。失败类别：{category_text}。</p>'
            f'{timeout_text}'
            f'<ul>{guidance_items}</ul>'
            f'{unknown_panel}'
            '</section>'
        )

    def _translate_guidance(self, text: str) -> str:
        translations = {
            "Mutation evidence was often low-sample; add boundary and branch-distinguishing assertions so more mutants become meaningful.": "变异证据经常是低样本：原因通常是源码可变异位置少、变异超时、无效变异或测试没有区分分支。下一轮应补边界输入、分支区分断言和杀变异断言，让更多变异体变得有意义。",
            "Avoid weak assertions such as result is not None; use exact return values, length, boolean identity, or approximate numeric checks.": "避免弱断言，例如只检查 result is not None；应使用明确返回值、长度、布尔值或近似数值断言。",
        }
        return translations.get(text, text)

    def _report_for_batch_item(self, item: dict[str, Any]) -> dict[str, Any] | None:
        report = item.get("final_report")
        if isinstance(report, dict):
            return public_result(report)
        for key in ("summary_path", "graph_path"):
            path_value = item.get(key)
            if not path_value:
                continue
            path = Path(str(path_value))
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if key == "summary_path" and isinstance(data.get("final_report"), dict):
                return public_result(data["final_report"])
            if key == "graph_path":
                metadata = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
                if isinstance(metadata.get("last_report"), dict):
                    return public_result(metadata["last_report"])
        return None

    def _paper_metrics_for_batch_item(self, item: dict[str, Any]) -> dict[str, Any]:
        embedded = item.get("paper_metrics")
        if isinstance(embedded, dict):
            return embedded
        summary_path = item.get("summary_path")
        candidates: list[Path] = []
        if summary_path:
            path = Path(str(summary_path))
            candidates.extend([path.parent / "paper_metrics" / "summary.json", path])
        output_dir = item.get("output_dir")
        if output_dir:
            candidates.append(Path(str(output_dir)) / "paper_metrics" / "summary.json")
        for path in candidates:
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                continue
            if path.name == "stateflow_summary.json":
                data = data.get("paper_metrics", {}) if isinstance(data, dict) else {}
            if isinstance(data, dict) and data:
                return data
        return {}

    def _evidence_rows(self, nodes: list[dict[str, Any]]) -> str:
        rows = []
        for node in sorted(nodes, key=lambda item: (int(item.get("line_start", 0) or 0), str(item.get("node_id", "")))):
            is_program = node.get("node_type") in PROGRAM_TYPES
            if is_program:
                state = "已覆盖" if node.get("visited") else "未覆盖"
                tone = "ok" if node.get("visited") else "off"
            else:
                state = self._evidence_status(str(node.get("node_type", "")))
                tone = "evidence"
            rows.append(
                '<tr data-row>'
                f'<td><span class="dot {tone}"></span>{state}</td>'
                f'<td>{self._esc(node.get("node_type", ""))}</td>'
                f'<td>{self._esc_literal(node.get("name", ""))}</td>'
                f'<td>{int(node.get("line_start", 0) or 0)}</td>'
                f'<td><code>{self._esc_literal(node.get("node_id", ""))}</code></td>'
                "</tr>"
            )
        return "".join(rows)

    def _evidence_status(self, node_type: str) -> str:
        labels = {
            "Execution": "执行记录",
            "RuntimeState": "运行证据",
            "RuntimeType": "类型证据",
            "TestCase": "测试用例",
            "Assertion": "断言证据",
            "Mutation": "变异证据",
        }
        return labels.get(node_type, "证据节点")

    def _function_card(self, item: dict[str, Any], summary: dict[str, Any]) -> str:
        visited = bool(item.get("visited"))
        state = "已覆盖" if visited else "未覆盖"
        _, reason = self._function_risk(item, summary)
        return (
            f'<div class="risk"><b>{self._esc_literal(item.get("name", ""))}</b>'
            f'<span class="badge {"ok" if visited else "off"}">{state}</span>'
            f'<small>第 {int(item.get("line_start", 0) or 0)} 行 · {html.escape(reason)}</small></div>'
        )

    def _concern(self, item: dict[str, Any], short: bool = False) -> str:
        defects = {
            "覆盖": float(item.get("coverage_defect", 0.0) or 0.0),
            "断言": float(item.get("assertion_defect", 0.0) or 0.0),
            "变异": float(item.get("mutation_defect", 0.0) or 0.0),
            "异常路径": float(item.get("exception_defect", 0.0) or 0.0),
            "类型证据": float(item.get("type_defect", 0.0) or 0.0),
        }
        name, value = max(defects.items(), key=lambda pair: pair[1])
        if value <= 0.05:
            return "暂无明显缺口"
        if short:
            return f"关注：{name}"
        explanations = {
            "覆盖": "仍有代码路径缺少执行证据",
            "断言": "结果检查还不够具体",
            "变异": "仍有模拟缺陷没有被测试发现",
            "异常路径": "异常分支证据不足",
            "类型证据": "参数或返回类型证据不足",
        }
        return explanations[name]

    def _source_html(self, model: dict[str, Any]) -> str:
        source = Path(str(model.get("source_path", "")))
        try:
            lines = source.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = ["源文件不可用：" + str(source)]
        summary = self._summary(model)
        coverage = summary["coverage"]
        covered_lines = {int(item) for item in coverage.get("covered_lines", []) if isinstance(item, int)}
        missing_lines = {int(item) for item in coverage.get("missing_lines", []) if isinstance(item, int)}
        has_line_data = bool(covered_lines or missing_lines) or "covered_lines" in coverage or "missing_lines" in coverage
        mutation = summary["metadata"].get("last_mutation", {})
        killed = set(mutation.get("killed_lines", [])) if isinstance(mutation, dict) else set()
        survived = set(mutation.get("survived_lines", [])) if isinstance(mutation, dict) else set()
        timeout = set(mutation.get("timeout_lines", [])) if isinstance(mutation, dict) else set()
        invalid = set(mutation.get("invalid_lines", [])) if isinstance(mutation, dict) else set()
        infra_error = set(mutation.get("infra_error_lines", [])) if isinstance(mutation, dict) else set()
        equivalent = set(mutation.get("equivalent_lines", [])) if isinstance(mutation, dict) else set()
        for mutation_node in summary.get("mutations", []):
            line_number = int(mutation_node.get("line_start", 0) or 0)
            if line_number <= 0:
                continue
            status = str(mutation_node.get("metadata", {}).get("status") or mutation_node.get("mutation_status") or "")
            if status == "killed":
                killed.add(line_number)
            elif status == "survived":
                survived.add(line_number)
            elif status == "timeout":
                timeout.add(line_number)
            elif status == "invalid":
                invalid.add(line_number)
            elif status == "infra_error":
                infra_error.add(line_number)
            elif status == "equivalent":
                equivalent.add(line_number)
        mutation_summary = self._mutation_summary(mutation if isinstance(mutation, dict) else {}, summary.get("mutations", []))
        functions = [item for item in summary["program"] if item.get("node_type") in {"Function", "Async"}]
        function_risks = {str(item.get("node_id")): self._function_risk(item, summary) for item in functions}
        rows = []
        for number, line in enumerate(lines, 1):
            stripped = line.strip()
            related = [
                node for node in summary["program"]
                if int(node.get("line_start", 0) or 0) <= number <= int(node.get("line_end", 0) or 0)
            ]
            branch_nodes = [node for node in related if node.get("node_type") in {"Branch", "Loop", "Match", "Exception"} and int(node.get("line_start", 0) or 0) == number]
            owner = self._owning_function(functions, number)
            _, owner_reason = function_risks.get(str(owner.get("node_id")) if owner else "", (0.0, ""))
            if has_line_data:
                covered = number in covered_lines
                missing = number in missing_lines
                neutral = not covered and not missing
                css = "covered" if covered else "uncovered" if missing else "neutral"
                state = "已执行" if covered else "未执行" if missing else "非执行行"
                source_note = "真实行/分支覆盖 + 共享TSG证据"
            else:
                covered = any(node.get("visited") for node in related)
                css = "covered" if covered else "uncovered"
                state = "图节点已覆盖" if covered else "图节点未覆盖"
                source_note = "共享TSG节点估算证据"
            badges = []
            if branch_nodes:
                complete, covered_count, total_count = self._branch_evidence(branch_nodes, summary)
                label = "分支已全" if complete else f"分支未全 {covered_count}/{total_count}"
                badges.append(f'<span class="tag branch {"ok" if complete else "warn"}" data-metric="branch">{label}</span>')
            if owner and number == int(owner.get("line_start", 0) or 0):
                assert_count = int(owner.get("assert_count", 0) or 0)
                if assert_count > 0:
                    badges.append(f'<span class="tag assertion count" data-metric="assertion">断言数 {assert_count}</span>')
                    badges.append(f'<span class="tag assertion strong" data-metric="assertion">强断言 {assert_count}</span>')
                elif owner.get("visited"):
                    badges.append('<span class="tag weak" data-metric="assertion">弱断言/缺断言</span>')
            if number in killed:
                badges.append('<span class="tag killed" data-metric="mutation">变异已杀</span>')
            if number in survived:
                badges.append('<span class="tag mutation" data-metric="mutation">存活变异</span>')
            if number in timeout:
                badges.append('<span class="tag timeout" data-metric="mutation">变异超时</span>')
            if number in invalid:
                badges.append('<span class="tag invalid-mut" data-metric="mutation">无效变异</span>')
            if number in infra_error:
                badges.append('<span class="tag infra-mut" data-metric="mutation">环境错误</span>')
            if number in equivalent:
                badges.append('<span class="tag equiv-mut" data-metric="mutation">疑似等价</span>')
            evidence = "".join(badges) or '<span class="status">无额外证据</span>'
            title = html.escape(owner_reason or self._line_reason(branch_nodes, summary))
            rows.append(
                f'<tr class="{css}" data-row data-line="{number}" title="{title}"><td>{number}</td><td><span class="status" data-metric="line">{state}</span></td>'
                f'<td>{evidence}</td><td><code>{self._esc_literal(line)}</code></td></tr>'
            )
        return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>源码证据热图</title>{self._style()}</head><body><main class="wrap"><section class="hero"><div class="eyebrow">ValiFixTest &amp; TSGColAgent 源码证据热图</div><h1>源码证据热图</h1><p>源码覆盖热力图升级版：{html.escape(str(source))}</p><span class="verdict warn">数据来源：{source_note}</span></section><section class="metrics">{self._metric("LC", self._percent(summary["line_coverage"]))}{self._metric("BC", self._percent(summary["branch_coverage"]))}{self._metric("AE", self._percent(summary["ae"]))}{self._metric("MS", self._percent(summary["mutation"]))}{self._metric("TIR", self._percent(summary["tir"]))}{self._metric("AvgExec", self._decimal(summary["avg_exec"]))}</section><section class="panel"><h2>目标改善与执行开销</h2><p class="sub">TIR：{summary["tir_improved"]}/{summary["tir_comparable"]}（排除或缺证 {summary["tir_excluded"]}）；Time/s：{self._decimal(summary["duration_seconds"])}。指标缺少完整采集证据时显示“未执行”。</p></section><section class="panel"><h2>变异结果说明</h2><p class="sub">{html.escape(mutation_summary)}</p></section><section class="panel"><h2>这张图怎么看</h2><p class="sub">选择某一类视图后，只显示该类证据。行覆盖只看绿色/灰色背景；分支覆盖只看黄色已覆盖和灰色未覆盖；有效断言只看断言数、强断言、弱断言；变异检出只看变异已杀、存活变异、变异超时、无效变异等状态。</p><div class="modebar"><button data-mode="all" class="active">全部指标</button><button data-mode="line">只看行覆盖</button><button data-mode="branch">只看分支覆盖</button><button data-mode="assertion">只看有效断言</button><button data-mode="mutation">只看变异检出</button></div></section><section class="panel"><div class="legend"><span class="l-covered">行覆盖：绿色覆盖</span><span class="l-uncovered">行覆盖：灰色未覆盖</span><span class="branch ok">分支覆盖</span><span class="branch warn">分支未覆盖</span><span class="assertion strong">强断言</span><span class="weak">弱断言/缺断言</span><span class="killed">变异已杀</span><span class="mutation">存活变异</span><span class="timeout">变异超时</span><span class="invalid-mut">无效变异</span></div><div class="table-wrap source"><table><thead><tr><th>行</th><th>行覆盖</th><th>证据标签</th><th>源码</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section></main>{self._source_script()}</body></html>'''

    def _owning_function(self, functions: list[dict[str, Any]], line_number: int) -> dict[str, Any] | None:
        owners = [
            item for item in functions
            if int(item.get("line_start", 0) or 0) <= line_number <= int(item.get("line_end", 0) or 0)
        ]
        if not owners:
            return None
        return min(owners, key=lambda item: int(item.get("line_end", 0) or 0) - int(item.get("line_start", 0) or 0))

    def _branch_evidence(self, branch_nodes: list[dict[str, Any]], summary: dict[str, Any]) -> tuple[bool, int, int]:
        branch_ids = {str(node.get("node_id")) for node in branch_nodes}
        branch_edge_types = {"true", "false", "loop_body", "loop_exit", "loop_else", "case", "except", "try_body", "try_else"}
        edges = [
            edge for edge in summary.get("edges", [])
            if edge.get("source") in branch_ids and edge.get("edge_type") in branch_edge_types
        ]
        if not edges:
            covered = sum(1 for node in branch_nodes if node.get("visited"))
            total = len(branch_nodes)
            return covered >= total, covered, total
        covered = sum(1 for edge in edges if edge.get("visited"))
        return covered >= len(edges), covered, len(edges)

    def _line_reason(self, branch_nodes: list[dict[str, Any]], summary: dict[str, Any]) -> str:
        if not branch_nodes:
            return ""
        complete, covered, total = self._branch_evidence(branch_nodes, summary)
        return "分支证据完整" if complete else f"该行存在未覆盖分支：{covered}/{total}"

    def _mutation_summary(self, mutation: dict[str, Any], mutation_nodes: list[dict[str, Any]] | None = None) -> str:
        node_counts = {"killed": 0, "survived": 0, "timeout": 0, "invalid": 0, "infra_error": 0, "equivalent": 0}
        for mutation_node in mutation_nodes or []:
            status = str(mutation_node.get("metadata", {}).get("status") or mutation_node.get("mutation_status") or "")
            if status in node_counts:
                node_counts[status] += 1
        if not mutation and not any(node_counts.values()):
            return "本次报告没有变异测试记录，不能判断是全部杀死还是没有生成变异。"
        killed = int(mutation.get("killed", node_counts["killed"]) or 0)
        survived = int(mutation.get("survived", node_counts["survived"]) or 0)
        timeout = int(mutation.get("timeout", node_counts["timeout"]) or 0)
        invalid = int(mutation.get("invalid", node_counts["invalid"]) or 0)
        infra_error = int(mutation.get("infra_error", node_counts["infra_error"]) or 0)
        equivalent = int(mutation.get("equivalent", node_counts["equivalent"]) or 0)
        effective = int(mutation.get("effective_mutants", killed + survived) or 0)
        low_reason = str(mutation.get("low_sample_reason") or "")
        autonomy_action = str(mutation.get("autonomy_action") or "")
        timeout_retries = int(mutation.get("timeout_retries", 0) or 0)
        total_observed = killed + survived + timeout + invalid + infra_error + equivalent
        if total_observed == 0:
            reason = low_reason or "没有生成可报告的变异样本"
            return f"本次没有生成可报告的变异样本。原因：{reason}。处理：{autonomy_action or '报告标记为低样本，不把变异高分当作强证据。'}"
        if survived == 0:
            status = "没有存活变异；已产生的有效变异都被测试杀死。"
        else:
            status = f"存在 {survived} 个存活变异，需要补强测试。"
        reason_text = f" 原因：{low_reason}。" if low_reason else ""
        retry_text = f" 已自动重试超时样本 {timeout_retries} 次。" if timeout_retries else ""
        action_text = f" 处理：{autonomy_action}" if autonomy_action else ""
        return (
            f"{status} 统计：已杀 {killed}，存活 {survived}，超时 {timeout}，"
            f"无效 {invalid}，环境错误 {infra_error}，等价 {equivalent}，有效样本 {effective}。"
            f"{reason_text}{retry_text}{action_text}"
        )

    def _metric(self, label: str, value: str) -> str:
        return f'<div class="metric"><small>{self._esc(label)}</small><strong>{self._esc(value)}</strong></div>'

    def _esc(self, value: object) -> str:
        return html.escape(public_text("" if value is None else str(value)))

    def _esc_literal(self, value: object) -> str:
        """Escape source identifiers and code without changing their meaning."""
        return html.escape("" if value is None else str(value))

    def _link(self, path_value: object, label: str) -> str:
        if not path_value:
            return "无"
        href = str(path_value)
        if "://" not in href:
            try:
                href = Path(href).resolve().as_uri()
            except (OSError, ValueError):
                pass
        return f'<a href="{html.escape(href)}">{self._esc(label)}</a>'

    def _number(self, value: object) -> float | None:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number / 100.0 if number > 1.0 else number

    def _percent(self, value: object) -> str:
        number = self._number(value)
        return self._dash() if number is None else f"{max(0.0, min(1.0, number)):.0%}"

    def _decimal(self, value: object) -> str:
        if value is None:
            return self._dash()
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return self._dash()

    def _dash(self) -> str:
        return "未执行"

    def _style(self) -> str:
        return '''<style>
:root{--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--bg:#f6f8fc}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,"Segoe UI","Microsoft YaHei",sans-serif;line-height:1.55}
.wrap{max-width:1180px;margin:auto;padding:28px 22px 60px}
.hero{background:linear-gradient(135deg,#111827,#312e81);color:#fff;border-radius:22px;padding:32px;box-shadow:0 20px 45px #1e1b4b20}
.eyebrow{font-size:12px;letter-spacing:.12em;color:#c7d2fe}
h1{margin:7px 0 10px;font-size:31px}
.hero p{margin:0;color:#e0e7ff;overflow-wrap:anywhere}
.verdict{display:inline-block;margin-top:18px;padding:8px 14px;border-radius:999px;font-weight:700}
.verdict.good{background:#dcfce7;color:#166534}.verdict.warn{background:#fef3c7;color:#92400e}.verdict.danger{background:#fee2e2;color:#991b1b}
.metrics{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:18px 0}.secondary{margin-top:-6px}
.metric,.panel{background:#fff;border:1px solid var(--line);border-radius:14px;box-shadow:0 7px 25px #0f172a0b}
.metric{padding:16px;min-width:0}.metric small{display:block;color:var(--muted);font-size:12px}.metric strong{display:block;font-size:23px;margin-top:6px;overflow-wrap:anywhere}
.panel{padding:22px;margin-top:16px}h2{font-size:20px;margin:0 0 6px}.sub{color:var(--muted);margin:0 0 16px}
.risks{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.risk{border:1px solid var(--line);border-radius:12px;padding:14px;display:grid;grid-template-columns:1fr auto;gap:4px}.risk b{overflow-wrap:anywhere}.risk small{color:var(--muted);grid-column:1/-1}
.badge{align-self:center;padding:5px 9px;border-radius:999px;font-size:12px;font-weight:700}.badge.ok{background:#dcfce7;color:#166534}.badge.off{background:#fee2e2;color:#991b1b}
.snapshot svg{width:100%;height:auto;border-radius:16px}details{margin-top:16px}summary{cursor:pointer;font-weight:700;padding:8px 0}
.search{width:100%;padding:11px 13px;border:1px solid #cbd5e1;border-radius:10px;margin:10px 0}
.table-wrap{overflow:auto;max-height:520px}.table-wrap.source{max-height:76vh}
table{width:100%;border-collapse:collapse;background:#fff;font-size:13px}th,td{padding:9px 10px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}th{position:sticky;top:0;background:#f8fafc;white-space:nowrap}.compact{min-width:1120px}.fail-table{min-width:880px}
code{font-size:11px;color:#475569}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px}.dot.ok{background:#16a34a}.dot.off{background:#cbd5e1}.dot.evidence{background:#6366f1}
.help{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.help div,.formula,.kv{padding:14px;background:#f8fafc;border-radius:12px}.formula,.kv{display:grid;grid-template-columns:140px minmax(0,1fr);gap:10px 14px}.kv span{min-width:0;overflow-wrap:anywhere}
.legend{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}.legend span,.tag,.status{display:inline-block;padding:4px 8px;border-radius:999px;font-size:12px;margin:2px}
.l-covered{background:#dcfce7;color:#166534}.l-uncovered{background:#f1f5f9;color:#64748b}
.branch{background:#fef3c7;color:#92400e}.branch.warn{background:#e5e7eb;color:#4b5563}
.assertion{background:#ede9fe;color:#5b21b6}.assertion.strong{background:#dbeafe;color:#1d4ed8}
.killed{background:#dcfce7;color:#166534}.mutation{background:#ffedd5;color:#9a3412}.timeout{background:#e0e7ff;color:#3730a3}.invalid-mut{background:#e5e7eb;color:#374151}.infra-mut{background:#fee2e2;color:#991b1b}.equiv-mut{background:#f1f5f9;color:#475569}
.weak{background:#fee2e2;color:#991b1b}.attention{background:#fef9c3;color:#854d0e}.defect{background:#fee2e2;color:#991b1b}
.modebar{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}.modebar button{appearance:none;border:1px solid #cbd5e1;background:#fff;color:#334155;border-radius:999px;padding:7px 12px;font:inherit;font-size:13px;cursor:pointer}.modebar button.active{border-color:#2563eb;background:#dbeafe;color:#1d4ed8;font-weight:700}.modebar button:hover{border-color:#2563eb}
tr.covered td:last-child{background:#f0fdf4;border-left:4px solid #22c55e}
tr.uncovered td:last-child{background:#f8fafc;color:#94a3b8;border-left:4px solid #cbd5e1}
tr.neutral td:last-child{background:#fff;color:#94a3b8;border-left:4px solid #e2e8f0}
body[data-view="branch"] tr td:last-child,body[data-view="assertion"] tr td:last-child,body[data-view="mutation"] tr td:last-child{background:#fff;color:inherit;border-left:4px solid #e2e8f0}
body[data-view="line"] .tag:not([data-metric="line"]),body[data-view="line"] td:nth-child(3){display:none}
footer{color:var(--muted);font-size:12px;margin-top:24px}a{color:#4338ca}
@media(max-width:950px){.metrics{grid-template-columns:repeat(3,1fr)}.risks,.help{grid-template-columns:1fr}.formula,.kv{grid-template-columns:1fr}}
@media(max-width:560px){.metrics{grid-template-columns:1fr}h1{font-size:25px}}
</style>'''

    def _script(self) -> str:
        return '''<script>(()=>{const input=document.querySelector('#search');if(!input)return;const rows=[...document.querySelectorAll('[data-row]')];input.addEventListener('input',()=>{const q=input.value.toLowerCase();rows.forEach(row=>row.hidden=!row.textContent.toLowerCase().includes(q))})})();</script>'''

    def _source_script(self) -> str:
        return '''<script>(()=>{const buttons=[...document.querySelectorAll('[data-mode]')];const rows=[...document.querySelectorAll('[data-row]')];function apply(mode){document.body.dataset.view=mode;buttons.forEach(button=>button.classList.toggle('active',button.dataset.mode===mode));rows.forEach(row=>{const tags=[...row.querySelectorAll('[data-metric]')];if(mode==='all'){row.hidden=false;tags.forEach(tag=>tag.hidden=false);return}if(mode==='line'){row.hidden=false;tags.forEach(tag=>tag.hidden=tag.dataset.metric!=='line');return}tags.forEach(tag=>tag.hidden=tag.dataset.metric!==mode);row.hidden=!tags.some(tag=>tag.dataset.metric===mode)})}buttons.forEach(button=>button.addEventListener('click',()=>apply(button.dataset.mode)));apply('all')})();</script>'''


StateFlowGraphVisualizer = TestStateModelVisualizer
