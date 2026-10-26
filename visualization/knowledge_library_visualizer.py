from __future__ import annotations

import html
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.experience_memory import AgentExperienceMemory, resolve_knowledge_store_path
from core.public_metrics import public_result, public_text


class TestKnowledgeGuidanceLibraryVisualizer:
    __test__ = False

    def render(self, memory: AgentExperienceMemory | None = None) -> dict[str, object]:
        knowledge_path = resolve_knowledge_store_path()
        output_dir = knowledge_path.parent
        memory = memory or AgentExperienceMemory(knowledge_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        records = memory.load()
        knowledge = self._knowledge_rows(records)
        applications = self._application_rows(records)
        summary = {
            "name": "测试知识引导库",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "knowledge_path": str(knowledge_path.resolve()),
            "knowledge_count": len(knowledge),
            "adopted": sum(item["learning_status"] == "已掌握" for item in knowledge),
            "learning": sum(item["learning_status"] == "正在学习" for item in knowledge),
            "pending_review": sum(item["learning_status"] == "待复核" for item in knowledge),
            "avoid": sum(item["learning_status"] == "应避免" for item in knowledge),
            "application_count": sum(max(1, int(item.get("application_count", 1) or 1)) for item in applications),
            "knowledge": knowledge,
            "applications": applications,
        }
        summary_path = output_dir / "test_knowledge_guidance_summary.json"
        html_path = output_dir / "test_knowledge_guidance_library.html"
        public_summary = public_result(summary)
        summary_path.write_text(json.dumps(public_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        html_path.write_text(self._html(public_summary), encoding="utf-8")
        return {
            "status": "generated",
            "name": "测试知识引导库",
            "html": str(html_path.resolve()),
            "summary": str(summary_path.resolve()),
            "knowledge_path": str(knowledge_path.resolve()),
        }

    def _knowledge_rows(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            if record.get("record_type") == "batch_summary":
                continue
            signature = record.get("task_signature", {}) if isinstance(record.get("task_signature"), dict) else {}
            language = str(signature.get("language") or record.get("language") or "通用")
            category = str(record.get("failure_category") or record.get("dominant_gap") or record.get("action") or "general")
            lessons = record.get("learning_notes", []) if isinstance(record.get("learning_notes"), list) else []
            strategies = []
            for note in lessons:
                if isinstance(note, dict):
                    value = note.get("solution") or note.get("reminder") or note.get("action")
                    if value:
                        strategies.append(str(value))
            if not strategies:
                strategies = [str(record.get("action") or "保留可执行测试并针对最大质量缺口进行最小修改")]
            for strategy in dict.fromkeys(strategies):
                groups[(language, category, strategy)].append(record)

        rows: list[dict[str, Any]] = []
        for index, ((language, category, strategy), samples) in enumerate(groups.items(), start=1):
            weights = [max(1, int(item.get("support_count", 1) or 1)) for item in samples]
            application_count = sum(weights)
            rewards = [float(item.get("reward", 0.0) or 0.0) for item in samples]
            positive = sum(weight for value, weight in zip(rewards, weights) if value > 0)
            negative = sum(weight for value, weight in zip(rewards, weights) if value < 0)
            success_rate = positive / application_count if application_count else 0.0
            average_reward = sum(value * weight for value, weight in zip(rewards, weights)) / application_count if application_count else 0.0
            ordinary_sources = {str(item.get("source") or "") for item in samples if item.get("source") and item.get("record_type") != "historical_pattern"}
            historical_sources = sum(int(item.get("independent_task_count", 0) or 0) for item in samples if item.get("record_type") == "historical_pattern")
            independent_sources = len(ordinary_sources) + historical_sources
            stages = {str(item.get("learning_stage") or "") for item in samples}
            real_validations = sum(int(item.get("validation_count", 0) or 0) for item in samples)
            if "avoid" in stages and negative / application_count >= 0.70:
                status = "应避免"
            elif {"adopted", "mastered"} & stages or (real_validations >= 2 and independent_sources >= 2 and success_rate >= 0.70 and average_reward > 0):
                status = "已掌握"
            elif "learning" in stages or application_count >= 2:
                status = "正在学习"
            else:
                status = "待复核"
            rows.append(
                {
                    "knowledge_id": f"TKG-{index:04d}",
                    "language": language,
                    "problem_type": category,
                    "strategy": strategy,
                    "learning_status": status,
                    "application_count": application_count,
                    "independent_tasks": independent_sources,
                    "last_updated": max(float(item.get("timestamp", 0.0) or 0.0) for item in samples),
                }
            )
        order = {"已掌握": 0, "正在学习": 1, "待复核": 2, "应避免": 3}
        return sorted(rows, key=lambda item: (order[item["learning_status"]], -item["application_count"], item["problem_type"]))

    def _application_rows(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for record in records:
            if record.get("record_type") == "batch_summary":
                continue
            before = record.get("before") or record.get("before_metrics") or {}
            after = record.get("after") or record.get("after_metrics") or {}
            rows.append(
                {
                    "source": Path(str(record.get("source") or "")).name,
                    "agent": record.get("agent"),
                    "action": record.get("action"),
                    "failure_category": record.get("failure_category"),
                    "experience_id": record.get("experience_id"),
                    "before_quality": (before or {}).get("quality_status") if isinstance(before, dict) else None,
                    "after_quality": (after or {}).get("quality_status") if isinstance(after, dict) else None,
                    "timestamp": record.get("timestamp"),
                    "application_count": max(1, int(record.get("support_count", 1) or 1)),
                }
            )
        return sorted(rows, key=lambda item: float(item.get("timestamp", 0.0) or 0.0), reverse=True)[:1000]

    def _html(self, summary: dict[str, Any]) -> str:
        knowledge_rows = "".join(
            "<tr>"
            f"<td>{self._esc(item['knowledge_id'])}</td><td>{self._esc(item['language'])}</td>"
            f"<td>{self._esc(item['problem_type'])}</td><td>{self._esc(item['strategy'])}</td>"
            f"<td>{self._esc(item['learning_status'])}</td><td>{item['application_count']}</td>"
            f"<td>{item['independent_tasks']}</td></tr>"
            for item in summary["knowledge"]
        ) or '<tr><td colspan="7">暂无知识记录</td></tr>'
        application_rows = "".join(
            "<tr>"
            f"<td>{self._esc(item['source'])}</td><td>{self._esc(item['agent'])}</td>"
            f"<td>{self._esc(item['action'])}</td><td>{self._esc(item['failure_category'])}</td>"
            f"<td>{self._esc(item['before_quality'])}</td><td>{self._esc(item['after_quality'])}</td>"
            f"<td>{item['application_count']}</td></tr>"
            for item in summary["applications"][:300]
        ) or '<tr><td colspan="7">暂无应用记录</td></tr>'
        return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>测试知识引导库</title>
<style>body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;margin:0;background:#f6f8fb;color:#172033}}main{{max-width:1500px;margin:auto;padding:28px}}h1{{margin:0 0 8px}}.sub{{color:#5d6879}}.metrics{{display:grid;grid-template-columns:repeat(5,minmax(130px,1fr));gap:10px;margin:18px 0}}.metric,.panel{{background:white;border:1px solid #dce2ea;border-radius:8px;padding:14px}}.metric b{{display:block;font-size:24px;margin-top:5px}}.panel{{margin-top:14px}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px 10px;border-bottom:1px solid #e7ebf0;text-align:left;font-size:13px;vertical-align:top}}th{{background:#edf2f7;position:sticky;top:0}}td:nth-child(4){{min-width:420px}}code{{word-break:break-all}}</style></head><body><main>
<h1>测试知识引导库</h1><p class="sub">以真实应用和失败证据区分已掌握、正在学习、待复核或应避免的策略；所有知识仍持续接受新证据。</p>
<section class="metrics"><div class="metric">知识条目<b>{summary['knowledge_count']}</b></div><div class="metric">已掌握<b>{summary['adopted']}</b></div><div class="metric">正在学习<b>{summary['learning']}</b></div><div class="metric">待复核<b>{summary['pending_review']}</b></div><div class="metric">应避免<b>{summary['avoid']}</b></div></section>
<section class="panel"><h2>知识总表</h2><div class="table-wrap"><table><thead><tr><th>知识ID</th><th>语言</th><th>问题类型</th><th>解决策略</th><th>学习状态</th><th>使用次数</th><th>独立任务数</th></tr></thead><tbody>{knowledge_rows}</tbody></table></div></section>
<section class="panel"><h2>知识应用记录</h2><div class="table-wrap"><table><thead><tr><th>任务</th><th>执行模块</th><th>操作</th><th>问题类型</th><th>使用前</th><th>使用后</th><th>支持次数</th></tr></thead><tbody>{application_rows}</tbody></table></div></section>
<section class="panel"><h2>存储位置</h2><code>{self._esc(summary['knowledge_path'])}</code></section></main></body></html>'''

    def _esc(self, value: object) -> str:
        return html.escape(public_text("" if value is None else str(value)))
