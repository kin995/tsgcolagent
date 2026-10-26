"""Register the bundled demo run so the dashboard has something to show.

Only the registry file (`.monitoring/experiments.json`) is written; the demo
run under `demo/` is never modified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
REGISTRY = ROOT / ".monitoring" / "experiments.json"


def experiment_id(output_dir: str | Path) -> str:
    """Same id scheme as monitoring.dashboard_data.experiment_id."""
    return hashlib.sha256(str(Path(output_dir).resolve()).encode("utf-8")).hexdigest()[:16]


def main() -> None:
    report = DEMO / "report"
    if not report.is_dir():
        raise SystemExit(f"缺少示例运行目录：{report}")
    item = {
        "id": experiment_id(report),
        "output_dir": str(report.resolve()),
        "label": "示例运行 · absolute_value（单任务）",
        "registered_at": "2026-09-23T06:14:25.702915+00:00",
        "execution_status": "completed",
        "knowledge_store": str((DEMO / "knowledge.json").resolve()),
    }
    data = {"active_id": item["id"], "experiments": [item]}
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已注册示例实验：{item['output_dir']}")
    print(f"注册表：{REGISTRY}")


if __name__ == "__main__":
    main()
