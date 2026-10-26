"""把内置示例运行导出成可核验的证据（Markdown + JSON）。

用法：python scripts/export_demo_evidence.py
输出：docs/evidence/single_task_demo.md 与 .json
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "demo" / "report" / "stateflow_summary.json"
OUT_DIR = ROOT / "docs" / "evidence"

FIELDS = [
    ("行覆盖率 LC", "line_coverage", "{:.1%}"),
    ("分支覆盖率 BC", "branch_coverage", "{:.1%}"),
    ("断言有效性 AE", "ae", "{:.1%}"),
    ("变异得分 MS", "mutation_score", "{:.1%}"),
    ("通过率", "pass_rate", "{:.1%}"),
    ("目标改善率 TIR", "tir", "{:.1%}"),
    ("平均工具执行次数", "exec_count", "{:.0f}"),
    ("平均耗时（秒）", "duration_seconds", "{:.2f}"),
]


def main() -> int:
    if not SUMMARY.is_file():
        print(f"缺少示例运行：{SUMMARY}")
        return 1
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    final = summary.get("final_report") or {}
    paper = summary.get("paper_metrics") or {}
    llm = summary.get("llm") or {}

    evidence = {
        "kind": "historical_run",
        "note": "仓库内置的脱敏示例运行，不是本次现场生成",
        "source": summary.get("source"),
        "test_path": summary.get("test_path"),
        "iterations": summary.get("iterations"),
        "stop_reason": summary.get("stop_reason"),
        "quality_status": summary.get("quality_status"),
        "pytest_passed": final.get("pytest_passed"),
        "collected_count": final.get("collected_count"),
        "effective_mutants": final.get("effective_mutants"),
        "effective_assertions": final.get("effective_assertions"),
        "assertion_count": final.get("assertion_count"),
        "paper_metrics": {k: paper.get(k) for k in ("line_coverage", "branch_coverage", "ae", "mutation_score",
                                                   "pass_rate", "tir", "tir_comparable", "tir_improved",
                                                   "exec_count", "avg_exec", "duration_seconds")},
        "llm": {k: llm.get(k) for k in ("model", "calls", "total_tokens", "duration_seconds") if k in llm},
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "single_task_demo.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 单任务证据（内置示例运行）",
        "",
        f"- 来源：`{evidence['source']}`",
        f"- 质量状态：`{evidence['quality_status']}`（stop_reason: `{evidence['stop_reason']}`）",
        f"- 轮次：{evidence['iterations']}",
        f"- 测试通过：{evidence['pytest_passed']}，收集用例：{evidence['collected_count']}",
        f"- 有效断言：{evidence['effective_assertions']} / {evidence['assertion_count']}",
        f"- 有效变异体：{evidence['effective_mutants']}",
        "",
        "| 指标 | 值 |",
        "|---|---|",
    ]
    for label, key, fmt in FIELDS:
        value = paper.get(key)
        lines.append(f"| {label} | {'—' if value is None else fmt.format(value)} |")
    lines += [
        "",
        "> 这是仓库内置的历史运行记录，已脱敏；现场演示请以实际运行输出为准。",
        "> 复现命令：`python -m tsgtest gen <source> -o runs/<name>` 或界面「开始生成」。",
        "",
    ]
    (OUT_DIR / "single_task_demo.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"已导出：{OUT_DIR / 'single_task_demo.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
