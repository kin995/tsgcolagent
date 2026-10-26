from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from batch_generate_tests import generate_tests_for_directory
from core.public_metrics import public_result
from graph.experience_memory import configure_knowledge_store_path, resolve_knowledge_store_path
from llm_config import LLMConfig
from main import MultiAgentUnitTestSystem, Thresholds


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "generated_tests_llm"


def run_target(
    target: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    max_iterations: int = 3,
    limit: int | None = None,
    quiet: bool = False,
    retry_failed_rounds: int = 2,
    resume: bool = True,
    rerun_existing: bool = False,
    knowledge_store: str | Path | None = None,
) -> dict[str, object]:
    configure_knowledge_store_path(knowledge_store)
    target_path = Path(target).resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"Target not found: {target_path}")

    llm_config = LLMConfig.from_env(enabled=True, required=True)
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if target_path.is_dir():
        output_dir = output_root / _safe_name(target_path.name)
        result = generate_tests_for_directory(
            source_dir=target_path,
            output_dir=output_dir,
            language="auto",
            execute=True,
            max_iterations=max_iterations,
            thresholds=Thresholds(),
            llm_config=llm_config,
            limit=limit,
            fail_fast=False,
            verbose=not quiet,
            retry_failed_rounds=retry_failed_rounds,
            resume=resume,
            rerun_existing=rerun_existing,
            knowledge_store=knowledge_store,
        )
        return public_result(result)

    if target_path.suffix.lower() not in {".py", ".java"}:
        raise ValueError(f"Target must be a Python/Java file or a directory: {target_path}")

    output_dir = output_root / _safe_name(target_path.stem)
    system = MultiAgentUnitTestSystem(
        source_path=target_path,
        output_dir=output_dir,
        thresholds=Thresholds(),
        max_iterations=max_iterations,
        llm_config=llm_config,
        experience_path=resolve_knowledge_store_path(),
        resume=resume,
        rerun_existing=rerun_existing,
    )
    return public_result(system.run())


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_")
    return cleaned or datetime.now().strftime("run_%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the LLM multi-agent unit-test generation system with project defaults."
    )
    parser.add_argument("target", help="Python/Java source file or directory to test.")
    parser.add_argument("--out-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root directory.")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum generation-repair iterations.")
    parser.add_argument("--limit", type=int, default=None, help="For directories, only process the first N files.")
    parser.add_argument("--quiet", action="store_true", help="Hide per-file progress for directory runs.")
    parser.add_argument("--retry-failed-rounds", type=int, default=2, help="Compatibility option: 2 enables R1/R2/R3 (R2 recovers D/E; R3 optimizes C; no R4).")
    parser.add_argument("--no-resume", action="store_true", help="Do not reuse existing per-task summaries.")
    parser.add_argument("--rerun-existing", action="store_true", help="Force rerunning tasks even if summaries already exist.")
    parser.add_argument("--knowledge-store", default=None, help="External long-term knowledge JSON path. Existing files are never overwritten by project defaults.")
    parser.add_argument("--no-ui", action="store_true", help="Do not start or open the real-time experiment dashboard.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from monitoring import finish_dashboard, launch_dashboard

    configure_knowledge_store_path(args.knowledge_store)
    target_path = Path(args.target).resolve()
    dashboard_out = Path(args.out_root).resolve() / _safe_name(target_path.name if target_path.is_dir() else target_path.stem)
    launch_dashboard(
        dashboard_out,
        label=target_path.name,
        enabled=not args.no_ui,
        llm_config=LLMConfig.from_env(enabled=True, required=True),
    )
    failure = None
    try:
        summary = run_target(
            target=args.target,
            output_root=args.out_root,
            max_iterations=args.max_iterations,
            limit=args.limit,
            quiet=args.quiet,
            retry_failed_rounds=args.retry_failed_rounds,
            resume=not args.no_resume,
            rerun_existing=args.rerun_existing,
            knowledge_store=args.knowledge_store,
        )
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        finish_dashboard(dashboard_out, success=failure is None, error=failure, enabled=not args.no_ui)
    print(json.dumps(_brief_summary(summary), indent=2, ensure_ascii=False))


def _brief_summary(summary: dict[str, object]) -> dict[str, object]:
    keys = [
        "source",
        "source_dir",
        "output_dir",
        "test_path",
        "graph_path",
        "iterations",
        "stop_reason",
        "quality_status",
        "total",
        "discovered_files",
        "official_tasks",
        "selected_tasks",
        "skipped",
        "skipped_reason_counts",
        "skipped_examples",
        "succeeded",
        "failed",
        "retry_summary",
        "retry_history",
        "final_report",
        "visualization",
        "llm_usage",
        "llm",
    ]
    return public_result({key: summary[key] for key in keys if key in summary})


if __name__ == "__main__":
    main()
