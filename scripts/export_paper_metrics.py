"""Export per-dataset paper metrics without altering runs or manuscript tables.

Run from the project: python scripts/export_paper_metrics.py ../runs-new
Separate roots remain separate experimental conditions. Output is JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.public_metrics import public_result
from metrics.paper_metrics import aggregate_metrics, direct_metrics


def read(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def export(root):
    root = Path(root)
    batches = [root / "batch_summary.json"] if (root / "batch_summary.json").exists() else sorted(root.glob("*/batch_summary.json"))
    if not batches:
        raise ValueError(f"No batch results at {root}")
    rows = []
    for path in batches:
        batch = read(path)
        tasks = []
        for item in batch.get("results", []):
            case_name = str(item.get("output_dir") or "").replace("\\", "/").rstrip("/").split("/")[-1]
            case = path.parent / case_name
            summary = read(case / "stateflow_summary.json")
            values = read(case / "paper_metrics/summary.json") or summary.get("paper_metrics") or {}
            if values.get("source_hash") and summary.get("source_sha256") and values["source_hash"] != summary["source_sha256"]:
                values = {}
            duration = item.get("duration_seconds")
            if duration is None:
                duration = summary.get("cumulative_duration_seconds", summary.get("duration_seconds", summary.get("runtime_seconds")))
            values.update(direct_metrics(item.get("final_report") or {}, duration))
            tasks.append({"metrics": values, "grade": item.get("final_grade", "E")})
        selected = batch.get("selected_tasks")
        rows.append({"dataset": path.parent.name, "model": (batch.get("llm") or {}).get("model"),
                     "metrics": aggregate_metrics(tasks, selected if isinstance(selected, int) else len(tasks))})
    return public_result({"method": "ValiFixTest & TSGColAgent", "condition_root": str(root.resolve()), "datasets": rows})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([export(root) for root in args.roots], ensure_ascii=False, indent=2))
