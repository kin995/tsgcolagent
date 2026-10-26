"""Read-only evidence inventory; never infer TIR/AvgExec from workflow iterations.

Usage: python scripts/audit_experiment_metrics.py ../runs-DeepSeek-R1 ../runs-GPT-4
Outputs JSON to stdout. Does not change experiment artifacts or paper values.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.public_metrics import internal_metric_key


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def audit(roots: list[Path]) -> dict:
    rows = []
    fingerprints: dict[str, list[str]] = {}
    for root in roots:
        batches = sorted(root.glob("*/batch_summary.json"))
        if not batches:
            raise ValueError(f"No dataset batch_summary.json: {root}")
        for path in batches:
            batch = read(path)
            results = batch.get("results", [])
            passed = sum(
                bool((item.get("final_report") or {}).get("pytest_passed"))
                and ((item.get("final_report") or {}).get("collected_count") or 0) > 0
                for item in results
            )
            models: Counter = Counter()
            fields: Counter = Counter()
            plan_fields: Counter = Counter()
            summaries = sorted(path.parent.glob("*/stateflow_summary.json"))
            for summary in summaries:
                data = read(summary)
                models[str(data.get("llm", {}).get("model", "unknown"))] += 1
                fields.update(data.keys())
                for plan in data.get("test_intent_plans", []):
                    if isinstance(plan, dict):
                        plan_fields.update(plan.keys())
                digest = hashlib.sha256(summary.read_bytes()).hexdigest()
                fingerprints.setdefault(digest, []).append(str(summary))
            selected = batch.get("selected_tasks")
            rows.append({
                "folder": root.name, "dataset": path.parent.name,
                "batch_model": batch.get("llm", {}).get("model"),
                "task_models": dict(models), "selected_tasks": selected,
                "result_count": len(results), "summary_count": len(summaries),
                "passed_and_collected": passed,
                "batch_pass_rate_percent": round(passed / selected * 100, 4)
                if isinstance(selected, int) and selected > 0 and selected == len(results) else None,
                "summary_fields": sorted(key for key in fields if not internal_metric_key(key)),
                "plan_fields": sorted(key for key in plan_fields if not internal_metric_key(key)),
                "tir": None, "avg_exec": None,
                "metric_status": "historical_evidence_insufficient",
            })
    duplicate_groups = [paths for paths in fingerprints.values()
                        if len({Path(p).parts[-4] for p in paths}) > 1]
    return {"method": "ValiFixTest & TSGColAgent", "rows": rows,
            "identical_summary_groups_across_roots": duplicate_groups,
            "notes": [
                "PassRate here is a batch-final-report cross-check, not a replacement for paper values.",
                "Historical plans/metric deltas are not preregistered goal-level independent before/retained-after evaluations.",
                "Reports, repair rounds, attempts and mutant counts are not an exhaustive actual test-invocation ledger.",
                "Missing historical evidence is not measured zero and not proof that no comparable goals existed.",
            ]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.roots), ensure_ascii=False, indent=2))
