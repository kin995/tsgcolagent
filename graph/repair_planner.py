from __future__ import annotations

import ast
import itertools
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .graph_query import GraphQuery
from .state_graph import StateFlowGraph


@dataclass
class MinimalRepairPlan:
    root_node_id: str
    root_name: str
    target_test: str | None
    action: str
    patch: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "root_node_id": self.root_node_id,
            "root_name": self.root_name,
            "target_test": self.target_test,
            "action": self.action,
            "patch": self.patch,
            "reason": self.reason,
        }


class RootCauseRepairPlanner:
    def __init__(self, graph: StateFlowGraph, focus: str = "general"):
        self.graph = graph
        self.query = GraphQuery(graph)
        self.focus = focus

    def plan(self, test_path: str | Path) -> MinimalRepairPlan | None:
        root_info = self.query.get_root_cause()
        if not root_info or not root_info.get("root"):
            return None
        root = self._repairable_root(root_info)
        if not root:
            return None
        root_name = str(root["name"])
        related = self.query.get_candidate_repairs()
        target_test = None
        for item in related:
            node = item.get("node", {})
            if node.get("node_id") == root.get("node_id"):
                tests = item.get("related_tests", [])
                if tests:
                    target_test = str(tests[0].get("name"))
                break
        action = self._choose_action(root)
        patch = self._build_patch(root_name, action, root)
        if not patch:
            return None
        return MinimalRepairPlan(
            root_node_id=str(root["node_id"]),
            root_name=root_name,
            target_test=target_test,
            action=action,
            patch=patch,
            reason=f"highest propagated defect is {root_name} with score {root.get('defect_score')}",
        )

    def _repairable_root(self, root_info: dict[str, object]) -> dict[str, object] | None:
        root = root_info.get("root")
        if isinstance(root, dict) and root.get("node_type") in {"Function", "Async", "Method", "Constructor"}:
            return root
        for node_id in root_info.get("path", []):
            node = self.graph.get_node(str(node_id))
            if node and node.node_type in {"Function", "Async", "Method", "Constructor"}:
                return node.to_dict()
        highest = self.query.get_highest_defect(program_only=True)
        if highest and highest.node_type in {"Function", "Async", "Method", "Constructor"}:
            return highest.to_dict()
        functions = [node for node in self.graph.nodes.values() if node.node_type in {"Function", "Async", "Method", "Constructor"}]
        if functions:
            return max(functions, key=lambda node: node.defect_score).to_dict()
        return None

    def apply(self, test_path: str | Path, plan: MinimalRepairPlan) -> bool:
        path = Path(test_path)
        if not path.exists():
            return False
        text = path.read_text(encoding="utf-8")
        marker = f"def test_stateflow_minimal_repair_{self._safe(plan.root_name)}"
        if marker in text:
            return False
        path.write_text(text.rstrip() + "\n\n" + plan.patch + "\n", encoding="utf-8")
        return True

    def _choose_action(self, root: dict[str, object]) -> str:
        if self.focus in {"coverage", "assertion", "mutation", "exception", "runtime_type"}:
            return self.focus
        scores = {
            "coverage": float(root.get("coverage_defect", 0.0)),
            "assertion": float(root.get("assertion_defect", 0.0)),
            "mutation": float(root.get("mutation_defect", 0.0)),
            "exception": float(root.get("exception_defect", 0.0)),
            "runtime_type": float(root.get("type_defect", 0.0)),
        }
        return max(scores.items(), key=lambda item: item[1])[0]

    def _build_patch(self, function_name: str, action: str, root: dict[str, object]) -> str:
        safe = self._safe(function_name)
        metadata = root.get("metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        class_name = self._class_name(metadata)
        receiver = f"target.{class_name}()" if class_name else "target"
        examples = self._observe_examples(function_name, root, action)
        blocks: list[str] = []
        for index, (arguments, expected) in enumerate(examples, start=1):
            call = f"{receiver}.{function_name}({', '.join(repr(value) for value in arguments)})"
            assertion = f"assert result == {expected!r}"
            if isinstance(expected, float):
                assertion = f"assert result == pytest.approx({expected!r})"
            lines = [
                f"def test_stateflow_minimal_repair_{safe}_{index}(target):",
                f"    result = {call}",
                f"    {assertion}",
            ]
            blocks.append("\n".join(lines))
        if not blocks:
            return ""
        prefix = "import pytest\n\n" if any(isinstance(expected, float) for _, expected in examples) else ""
        return prefix + "\n\n".join(blocks)

    def _observe_examples(self, function_name: str, root: dict[str, object], action: str) -> list[tuple[tuple[object, ...], object]]:
        metadata = root.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        arguments = [str(item) for item in metadata.get("args", []) if str(item) != "self"]
        annotations = metadata.get("arg_annotations", {})
        annotations = annotations if isinstance(annotations, dict) else {}
        value_sets = [self._values_for_argument(name, str(annotations.get(name) or ""), action) for name in arguments]
        combinations = itertools.product(*value_sets) if value_sets else [tuple()]
        observed: list[tuple[tuple[object, ...], object]] = []
        class_name = self._class_name(metadata)
        for values in itertools.islice(combinations, 16):
            result = self._run_observation(function_name, tuple(values), root.get("node_type") == "Async", class_name=class_name)
            if result is not _NO_RESULT and (tuple(values), result) not in observed:
                observed.append((tuple(values), result))
            if len(observed) >= 3:
                break
        return observed

    def _values_for_argument(self, name: str, annotation: str, action: str = "general") -> tuple[object, ...]:
        lowered = annotation.lower()
        normalized_name = name.lower()
        if "str" in lowered or any(token in normalized_name for token in ("text", "name", "word", "string", "pattern")) or normalized_name in {"s", "p", "c", "ch"}:
            if action == "mutation":
                return ("", "a", "*", "?", "a*", "ab")
            return ("", "abc", "a b")
        if "dict" in lowered or "mapping" in lowered:
            return ({}, {"a": 1})
        if any(token in lowered for token in ("list", "sequence", "iterable", "tuple", "set")) or name.lower() in {"items", "values", "numbers"}:
            return ([], [0], [1, 2, 3])
        if "bool" in lowered:
            return (False, True)
        if "float" in lowered or any(token in name.lower() for token in ("threshold", "ratio", "rate")):
            return (0.0, 0.5, 1.0)
        if "int" in lowered or any(token in normalized_name for token in ("count", "index", "size", "number")):
            return (-1, 0, 1, 2) if action == "mutation" else (0, 1, -1)
        return (0, "", [], None)

    def _run_observation(self, function_name: str, arguments: tuple[object, ...], is_async: bool, class_name: str | None = None) -> object:
        source = Path(self.graph.source_path).resolve()
        owner_expr = f"getattr(module, {class_name!r})()" if class_name else "module"
        script = (
            "import ast, asyncio, importlib.util\n"
            f"spec=importlib.util.spec_from_file_location('stateflow_observe', {str(source)!r})\n"
            "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n"
            f"owner={owner_expr}\n"
            f"value=getattr(owner, {function_name!r})(*ast.literal_eval({repr(repr(arguments))}))\n"
            + ("value=asyncio.run(value)\n" if is_async else "")
            + "print(repr(value))\n"
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=source.parent,
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if completed.returncode != 0 or not completed.stdout.strip():
                return _NO_RESULT
            value = ast.literal_eval(completed.stdout.strip().splitlines()[-1])
            return value if self._literal_safe(value) else _NO_RESULT
        except (OSError, subprocess.SubprocessError, SyntaxError, ValueError):
            return _NO_RESULT

    def _literal_safe(self, value: object) -> bool:
        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            return True
        if isinstance(value, (list, tuple)):
            return all(self._literal_safe(item) for item in value)
        if isinstance(value, dict):
            return all(self._literal_safe(key) and self._literal_safe(item) for key, item in value.items())
        return False

    def _safe(self, name: str) -> str:
        return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_") or "target"

    def _class_name(self, metadata: dict[str, object]) -> str | None:
        scope = str(metadata.get("scope") or "")
        return scope if scope and "::" not in scope else None


_NO_RESULT = object()
