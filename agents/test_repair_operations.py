from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RepairOperation:
    op: str
    test_name: str | None = None
    new_code: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {"op": self.op, "test_name": self.test_name, "new_code": self.new_code}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RepairOperation":
        raw_op = data.get("op") or data.get("operation") or data.get("action")
        aliases = {
            "add": "add_test",
            "append_test": "add_test",
            "replace": "replace_test",
            "rewrite_test": "replace_test",
            "delete": "delete_test",
            "remove_test": "delete_test",
        }
        op = aliases.get(str(raw_op), str(raw_op)) if raw_op is not None else ""
        if op not in {"replace_test", "delete_test", "add_test"}:
            raise ValueError(f"unsupported repair operation: {raw_op!r}")
        test_name = data.get("test_name") or data.get("target_test") or data.get("name")
        new_code = data.get("new_code") or data.get("code") or data.get("test_code")
        if op in {"replace_test", "delete_test"} and not test_name:
            raise ValueError(f"{op} requires test_name")
        if op in {"replace_test", "add_test"} and not new_code:
            raise ValueError(f"{op} requires new_code")
        return cls(
            op=op,
            test_name=str(test_name) if test_name else None,
            new_code=str(new_code) if new_code else None,
        )


class TestRepairOperationApplier:
    def apply(self, test_file: str | Path, operations: list[RepairOperation]) -> str:
        path = Path(test_file)
        text = path.read_text(encoding="utf-8")
        for operation in operations:
            if operation.op == "replace_test":
                text = self._replace_test(text, operation.test_name or "", operation.new_code or "")
            elif operation.op == "delete_test":
                text = self._delete_test(text, operation.test_name or "")
            elif operation.op == "add_test":
                text = text.rstrip() + "\n\n" + (operation.new_code or "").strip() + "\n"
            else:
                raise ValueError(f"unsupported repair operation: {operation.op}")
        ast.parse(text)
        return text

    def _replace_test(self, text: str, test_name: str, new_code: str) -> str:
        ranges = self._function_ranges(text)
        if test_name not in ranges:
            return text.rstrip() + "\n\n" + new_code.strip() + "\n"
        start, end = ranges[test_name]
        lines = text.splitlines()
        new_lines = lines[:start] + new_code.strip().splitlines() + lines[end:]
        return "\n".join(new_lines).rstrip() + "\n"

    def _delete_test(self, text: str, test_name: str) -> str:
        ranges = self._function_ranges(text)
        if test_name not in ranges:
            return text
        start, end = ranges[test_name]
        lines = text.splitlines()
        return "\n".join(lines[:start] + lines[end:]).rstrip() + "\n"

    def _function_ranges(self, text: str) -> dict[str, tuple[int, int]]:
        tree = ast.parse(text)
        lines = text.splitlines()
        ranges: dict[str, tuple[int, int]] = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                start = node.lineno - 1
                end = getattr(node, "end_lineno", node.lineno)
                while end < len(lines) and lines[end].strip() == "":
                    end += 1
                ranges[node.name] = (start, end)
        return ranges
