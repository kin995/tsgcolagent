from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RuntimeObservation:
    function: str
    test_name: str
    nodeid: str
    arg_types: list[str]
    class_name: str | None = None
    return_type: str | None = None
    exception_type: str | None = None
    exception_message_pattern: str | None = None
    value_summary: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "function": self.function,
            "test_name": self.test_name,
            "nodeid": self.nodeid,
            "arg_types": self.arg_types,
            "class_name": self.class_name,
            "return_type": self.return_type,
            "exception_type": self.exception_type,
            "exception_message_pattern": self.exception_message_pattern,
            "value_summary": self.value_summary,
        }


class RuntimeObserver:
    """Collects lightweight runtime evidence from literal target calls in tests."""

    def collect(self, source_path: str | Path, test_path: str | Path, timeout: int = 10) -> list[RuntimeObservation]:
        source_path = Path(source_path).resolve()
        test_path = Path(test_path).resolve()
        calls = self._extract_literal_calls(test_path)
        if not calls:
            return []
        payload = json.dumps(calls, ensure_ascii=False)
        script = r"""
from __future__ import annotations
import ast
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

source, payload = sys.argv[1], sys.argv[2]
calls = json.loads(payload)
source_path = Path(source).resolve()
parts = [source_path.stem]
cursor = source_path.parent
while (cursor / "__init__.py").is_file():
    parts.insert(0, cursor.name)
    cursor = cursor.parent
sys.path.insert(0, str(cursor if len(parts) > 1 else source_path.parent))
module_name = ".".join(parts) if len(parts) > 1 else "stateflow_runtime_target"
spec = importlib.util.spec_from_file_location(module_name, source_path)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[module_name] = module
spec.loader.exec_module(module)

def summarize(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        text = repr(value)
    else:
        size = ""
        try:
            size = f", len={len(value)}"
        except Exception:
            size = ""
        text = f"<{type(value).__name__}{size}>"
    return text[:120]

rows = []
for item in calls:
    owner = getattr(module, item["class_name"])() if item.get("class_name") else module
    func = getattr(owner, item["function"])
    args = [ast.literal_eval(value) for value in item["args"]]
    row = {
        "function": item["function"],
        "class_name": item.get("class_name"),
        "test_name": item["test_name"],
        "nodeid": item.get("nodeid", item["test_name"]),
        "arg_types": [type(value).__name__ for value in args],
        "return_type": None,
        "exception_type": None,
        "exception_message_pattern": None,
        "value_summary": None,
    }
    try:
        result = func(*args)
        if hasattr(result, "__await__"):
            result = asyncio.run(result)
        row["return_type"] = type(result).__name__
        row["value_summary"] = summarize(result)
    except Exception as exc:
        row["exception_type"] = exc.__class__.__name__
        message = str(exc)
        row["exception_message_pattern"] = message[:120] if message else ""
    rows.append(row)
print(json.dumps(rows, ensure_ascii=False))
"""
        try:
            completed = subprocess.run(
                [sys.executable, "-c", script, str(source_path), payload],
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return []
        if completed.returncode != 0:
            return []
        try:
            rows = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return []
        observations: list[RuntimeObservation] = []
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            observations.append(
                RuntimeObservation(
                    function=str(row.get("function", "")),
                    test_name=str(row.get("test_name", "")),
                    nodeid=str(row.get("nodeid", row.get("test_name", ""))),
                    arg_types=[str(item) for item in row.get("arg_types", [])],
                    class_name=str(row.get("class_name")) if row.get("class_name") else None,
                    return_type=row.get("return_type"),
                    exception_type=row.get("exception_type"),
                    exception_message_pattern=row.get("exception_message_pattern"),
                    value_summary=row.get("value_summary"),
                )
            )
        return [item for item in observations if item.function and item.test_name]

    def _extract_literal_calls(self, test_path: Path) -> list[dict[str, object]]:
        try:
            tree = ast.parse(test_path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            return []
        calls: list[dict[str, object]] = []
        for item in tree.body:
            if not isinstance(item, ast.FunctionDef) or not item.name.startswith("test_"):
                continue
            receiver_call_ids = {
                id(child.func.value)
                for child in ast.walk(item)
                if isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Call)
            }
            for child in ast.walk(item):
                if id(child) in receiver_call_ids:
                    continue
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                    call_info = self._target_call_info(child)
                    if not call_info:
                        continue
                    function_name = call_info["function"]
                    if function_name in {"raises", "fixture", "mark"}:
                        continue
                    arg_texts: list[str] = []
                    literal = True
                    for arg in child.args:
                        try:
                            ast.literal_eval(arg)
                        except Exception:
                            literal = False
                            break
                        arg_texts.append(ast.unparse(arg))
                    if literal:
                        calls.append({
                            "test_name": item.name,
                            "nodeid": item.name,
                            "function": function_name,
                            "class_name": call_info.get("class_name"),
                            "args": arg_texts,
                        })
        return calls[:100]

    def _target_call_info(self, node: ast.Call) -> dict[str, str | None] | None:
        if not isinstance(node.func, ast.Attribute):
            return None
        receiver = node.func.value
        if self._looks_like_target_call(receiver):
            return {"function": node.func.attr, "class_name": None}
        if isinstance(receiver, ast.Call) and isinstance(receiver.func, ast.Attribute):
            if self._looks_like_target_call(receiver.func.value):
                return {"function": node.func.attr, "class_name": receiver.func.attr}
        return None

    def _looks_like_target_call(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name) and node.id == "target":
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_load_target":
            return True
        return False
