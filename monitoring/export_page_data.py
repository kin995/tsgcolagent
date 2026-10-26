"""Export a completed experiment as separate, page-ready JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .page_data import write_page_data
from .server import DashboardHandler, REGISTRY, _public_snapshot


def export_experiment(experiment_id: str, destination: str | Path) -> dict[str, object]:
    record = REGISTRY.get(experiment_id)
    if record is None or record.get("execution_status") != "completed":
        raise ValueError("Select a registered, completed experiment")
    run_root = Path(str(record["output_dir"])).resolve()
    target = Path(destination).resolve()
    if target == run_root or run_root in target.parents:
        raise ValueError("Page data must be stored separately from raw run results")
    existing = target / "manifest.json"
    if existing.is_file():
        manifest = json.loads(existing.read_text(encoding="utf-8"))
        if (manifest.get("source") or {}).get("experiment_id") != experiment_id:
            raise ValueError("Destination already contains a different experiment")

    handler = DashboardHandler.__new__(DashboardHandler)
    runtime = _public_snapshot(handler._snapshot(experiment_id))
    knowledge = handler._knowledge_snapshot(experiment_id)
    master_manifest = write_page_data(target, runtime, knowledge)
    datasets = []
    for card in runtime.get("experiments", []):
        if not isinstance(card, dict) or card.get("parent_id") != experiment_id:
            continue
        name = str(card.get("label") or "")
        child_id = str(card.get("id") or "")
        if not name or Path(name).name != name or not child_id or any(char in name for char in ("/", "\\", ":")):
            raise ValueError("Unsafe dataset label in experiment")
        child_runtime = _public_snapshot(handler._snapshot(child_id))
        child_folder = target / "datasets" / name
        child_manifest = write_page_data(child_folder, child_runtime, knowledge)
        datasets.append({
            "id": child_id,
            "name": name,
            "manifest": f"datasets/{name}/manifest.json",
            "tasks": child_manifest["counts"]["tasks"],
        })

    index: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "master_manifest": "manifest.json",
        "tasks": master_manifest["counts"]["tasks"],
        "knowledge_records": master_manifest["counts"]["knowledge_records"],
        "datasets": datasets,
    }
    content = json.dumps(index, indent=2, ensure_ascii=False) + "\n"
    index_path = target / "index.json"
    if not index_path.is_file() or index_path.read_text(encoding="utf-8") != content:
        index_path.write_text(content, encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(export_experiment(args.experiment_id, args.out), ensure_ascii=False))


if __name__ == "__main__":
    main()
