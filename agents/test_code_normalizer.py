from __future__ import annotations

import re
from pathlib import Path

from agents.test_plan import PYTHON_TARGET_LOADER_IMPORTS, python_target_loader_body


def normalize_generated_test_code(
    code: str,
    source_path: str | Path | None = None,
    function_names: list[str] | None = None,
) -> str:
    """Fix common LLM-generated pytest loader mistakes before execution."""
    fixed = code.strip() + "\n"
    if _uses_path(fixed) and "from pathlib import Path" not in fixed:
        fixed = _insert_import(fixed, "from pathlib import Path")
    if "import os" not in fixed and ("os.environ" in fixed or "os.path" in fixed):
        fixed = _insert_import(fixed, "import os")
    fixed = re.sub(
        r"(?m)^(\s*target_module\s*=\s*)TARGET_PATH\.resolve\(\)\s*$",
        r"\1Path(TARGET_PATH).resolve()",
        fixed,
    )
    fixed = re.sub(
        r"(?m)^(\s*target_module\s*=\s*)Path\(TARGET_PATH\)\.resolve\(\)\s*$",
        r"\1Path(TARGET_PATH).resolve()",
        fixed,
    )
    if source_path and function_names:
        fixed = _remove_unsafe_target_imports(fixed, Path(source_path), function_names)
        fixed = _ensure_target_loader(fixed, Path(source_path), function_names)
    return fixed


def _uses_path(code: str) -> bool:
    return bool(re.search(r"\bPath\s*\(", code) or re.search(r"\bPath\.", code))


def _insert_import(code: str, import_line: str) -> str:
    lines = code.splitlines()
    insert_at = 0
    if lines and lines[0].startswith("from __future__"):
        insert_at = 1
        if len(lines) > 1 and lines[1].strip() == "":
            insert_at = 2
    else:
        for index, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                insert_at = index + 1
    lines.insert(insert_at, import_line)
    return "\n".join(lines).rstrip() + "\n"


def _remove_unsafe_target_imports(code: str, source_path: Path, function_names: list[str]) -> str:
    module_name = source_path.stem
    names = "|".join(re.escape(name) for name in function_names)
    if not names:
        return code
    lines: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if re.fullmatch(rf"import\s+({names})", stripped):
            continue
        if re.fullmatch(rf"from\s+({re.escape(module_name)}|{names})\s+import\s+.*", stripped):
            continue
        if re.fullmatch(rf"import\s+{re.escape(module_name)}", stripped):
            continue
        lines.append(line)
    return "\n".join(lines).rstrip() + "\n"


def _ensure_target_loader(code: str, source_path: Path, function_names: list[str]) -> str:
    if "STATEFLOW_TARGET" in code and all(f"{name} = target_module.{name}" in code for name in function_names):
        return code
    fixed = code
    fixed = _remove_legacy_loader_lines(fixed)
    for import_line in PYTHON_TARGET_LOADER_IMPORTS:
        if import_line not in fixed:
            fixed = _insert_import(fixed, import_line)
    loader = [
        "",
        "# Stable loader added by the multi-agent test normalizer.",
        *python_target_loader_body(source_path),
        "",
        "target_module = _load_target()",
    ]
    for name in function_names:
        loader.append(f"{name} = target_module.{name}")
    loader.append("")
    return _insert_after_import_block(fixed, "\n".join(loader))


def _remove_legacy_loader_lines(code: str) -> str:
    lines: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("DEFAULT_TARGET ="):
            continue
        if stripped.startswith("TARGET_PATH ="):
            continue
        if stripped.startswith("target_module ="):
            continue
        if stripped.startswith("sys.path.append("):
            continue
        lines.append(line)
    return "\n".join(lines).rstrip() + "\n"


def _insert_after_import_block(code: str, block: str) -> str:
    if "Stable loader added by the multi-agent test normalizer." in code:
        return code
    lines = code.splitlines()
    insert_at = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from ") or stripped == "" or stripped.startswith("#"):
            insert_at = index + 1
            continue
        break
    return "\n".join(lines[:insert_at]).rstrip() + "\n" + block.rstrip() + "\n" + "\n".join(lines[insert_at:]).lstrip() + "\n"
