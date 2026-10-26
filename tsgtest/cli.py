"""命令行入口：serve / gen / check。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_settings
from .engine import collect_sources, dependency_status, ollama_models, run_source, write_job_record
from .server import serve

ROOT = Path(__file__).resolve().parents[1]
QUALITY_OK = {"quality_high", "quality_acceptable", "valid_usable"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tsgtest", description="TSGColAgent 单元测试生成工具")
    sub = parser.add_subparsers(dest="command")

    serve_cmd = sub.add_parser("serve", help="启动带前端的本地服务")
    serve_cmd.add_argument("--host", default=None)
    serve_cmd.add_argument("--port", type=int, default=None)
    serve_cmd.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    serve_cmd.add_argument("--config", default=None)

    check = sub.add_parser("check", help="检查依赖、Ollama 与 Java 工具链")
    check.add_argument("--base-url", default=None)

    gen = sub.add_parser("gen", help="命令行直接生成（不起前端）")
    gen.add_argument("source")
    gen.add_argument("-o", "--out", default=None)
    gen.add_argument("--model", default=None)
    gen.add_argument("--base-url", default=None)
    gen.add_argument("--rounds", type=int, default=None)
    gen.add_argument("--temperature", type=float, default=None)
    gen.add_argument("--json", action="store_true", help="只输出 JSON")
    gen.add_argument("--config", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command or "serve"
    settings = load_settings(ROOT, Path(args.config) if getattr(args, "config", None) else None)

    if command == "serve":
        for attr in ("host", "port"):
            value = getattr(args, attr, None)
            if value is not None:
                setattr(settings, attr, value)
        serve(settings, open_browser=getattr(args, "open", False))
        return 0

    if command == "check":
        deps = dependency_status()
        base_url = args.base_url or settings.ollama_base_url
        models, error = ollama_models(base_url)
        print(f"Ollama（{base_url}）：" + ("不可达：" + error if error else f"{len(models)} 个模型"))
        for name in models:
            print(f"  - {name}")
        print("依赖：")
        for name, ok in deps.items():
            print(f"  {'OK  ' if ok else 'MISS'}  {name}")
        missing = [name for name, ok in deps.items() if not ok]
        return 1 if missing else 0

    if command == "gen":
        source = Path(args.source).expanduser()
        if not source.exists():
            print(f"路径不存在：{source}", file=sys.stderr)
            return 1
        for attr, key in (("model", "model"), ("base_url", "ollama_base_url"), ("rounds", "max_rounds"), ("temperature", "temperature")):
            value = getattr(args, attr, None)
            if value is not None:
                setattr(settings, key, value)
        out_dir = Path(args.out).expanduser() if args.out else settings.runs_dir / f"{source.stem or 'run'}_manual"
        out_dir = out_dir.resolve()

        def log(message: str) -> None:
            if not args.json:
                print(f"[tsgtest] {message}", flush=True)

        if source.is_dir():
            count = len(collect_sources(source, settings.max_files))
            log(f"目录模式：{count} 个文件")
        try:
            result = run_source(source, out_dir, settings, log)
        except Exception as exc:  # noqa: BLE001 - 命令行需要给出可读错误
            print(f"执行失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        payload = {"source": str(source), "out_dir": str(out_dir), "result": result}
        write_job_record(out_dir, payload)
        items = result.get("items", [])
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for item in items:
                metrics = item.get("metrics") or {}
                print(f"\n{item['source']}")
                print(f"  状态：{item.get('status')}  轮次：{item.get('iterations')}")
                print("  LC={:.1%}  BC={:.1%}  AE={:.1%}  MS={}".format(
                    float(metrics.get("line_coverage") or 0),
                    float(metrics.get("branch_coverage") or 0),
                    float(metrics.get("ae") or 0),
                    "n/a" if metrics.get("mutation_score") is None else f"{float(metrics['mutation_score']):.1%}",
                ))
                print(f"  测试文件：{item.get('test_path')}")
        ok = bool(items) and all(
            item.get("pytest_passed") and str(item.get("status")) in QUALITY_OK for item in items
        )
        return 0 if ok else 2

    print(f"未知命令：{command}", file=sys.stderr)
    return 1
