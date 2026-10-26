"""Check whether this machine can run the bundled code.

Usage:  python scripts/check_env.py
Exit code 0 = ready, 1 = something is missing.
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["pytest", "coverage", "sortedcontainers", "langgraph", "tree_sitter", "tree_sitter_java", "ollama"]


def main() -> int:
    problems: list[str] = []
    print(f"Python: {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info < (3, 11):
        problems.append("Python 版本过低，需要 3.11 及以上（当前 %s）" % sys.version.split()[0])

    print("\n依赖：")
    for name in REQUIRED:
        ok = importlib.util.find_spec(name) is not None
        print(f"  {'OK  ' if ok else 'MISS'}  {name}")
        if not ok:
            problems.append(f"缺少依赖 {name}")

    print("\n前端构建产物：")
    index = ROOT / "frontend" / "dist" / "index.html"
    if index.is_file():
        print(f"  OK    {index.relative_to(ROOT)}")
    else:
        problems.append("frontend/dist 缺失，请执行 cd frontend && npm install && npm run build")
        print("  MISS  frontend/dist/index.html")

    print("\n可选外部工具（仅在跑 Java / 变异任务时需要）：")
    for tool in ("java", "mvn", "npm"):
        path = shutil.which(tool)
        print(f"  {'OK  ' if path else 'MISS'}  {tool}" + (f"  {path}" if path else ""))

    print("\n数据集目录：datasets/（当前%s）" % ("存在" if (ROOT / "datasets").is_dir() else "不存在，需自行准备 TestEval / HumanEval / HumanEvalJava / Defects4J"))

    if problems:
        print("\n需要先处理：")
        for item in problems:
            print("  - " + item)
        print("\n安装依赖：python -m pip install -r requirements.txt")
        return 1
    print("\n环境检查通过。启动界面：python -m monitoring.server --port 8765")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
