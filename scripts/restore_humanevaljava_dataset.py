from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def restore(source_root: Path, output_root: Path) -> tuple[int, int]:
    recovered = 0
    unchanged = 0
    output_root.mkdir(parents=True, exist_ok=True)
    case_dirs = sorted(path for path in source_root.iterdir() if path.is_dir())
    if not case_dirs:
        raise RuntimeError(f"no HumanEvalJava result directories found under {source_root}")

    for case_dir in case_dirs:
        graph_path = case_dir / "state_flow_graph.json"
        if not graph_path.is_file():
            raise RuntimeError(f"missing state graph: {graph_path}")
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        context = graph.get("metadata", {}).get("structured_context", {})
        classes = context.get("classes") if isinstance(context.get("classes"), list) else []
        if len(classes) != 1 or not str(classes[0].get("code") or "").strip():
            raise RuntimeError(f"incomplete Java source context: {graph_path}")

        class_name = str(classes[0].get("name") or case_dir.name).strip()
        package_name = str(context.get("package_or_module") or "").strip()
        imports = sorted(
            {
                str(item.get("name") or "").strip()
                for item in context.get("dependencies", [])
                if isinstance(item, dict) and item.get("kind") == "import" and item.get("name")
            }
        )
        sections = []
        if package_name:
            sections.append(f"package {package_name};")
        if imports:
            sections.append("\n".join(f"import {name};" for name in imports))
        sections.append(str(classes[0]["code"]).strip())
        content = "\n\n".join(sections) + "\n"
        target = output_root / f"{class_name}.java"
        if target.exists():
            if target.read_text(encoding="utf-8") != content:
                raise RuntimeError(f"refusing to overwrite different source: {target}")
            unchanged += 1
            continue
        target.write_text(content, encoding="utf-8")
        recovered += 1

    return recovered, unchanged


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore HumanEvalJava sources from preserved state-flow artifacts.")
    parser.add_argument("--runs", type=Path, default=PROJECT_ROOT / "runs" / "HumanEvalJava")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "datasets" / "HumanEvalJava")
    args = parser.parse_args()
    recovered, unchanged = restore(args.runs.resolve(), args.output.resolve())
    print(f"HumanEvalJava restored: {recovered}; already identical: {unchanged}")


if __name__ == "__main__":
    main()
