from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from agents.candidate_validator import CandidateTestValidator
from agents.repair_agent import RepairOptimizationAgent
from agents.test_code_normalizer import normalize_generated_test_code
from agents.test_repair_operations import RepairOperation, TestRepairOperationApplier
from llm_client import OpenAICompatibleLLM
from memory.shared_graph import SharedGraphMemory


class FocusedRepairAgent:
    name = "测试生成智能体/定向修复模块"
    focus = "general"

    def __init__(self, memory: SharedGraphMemory, llm: OpenAICompatibleLLM | None = None, llm_required: bool = False):
        self.memory = memory
        self.llm = llm
        self.llm_required = llm_required

    def run(self, test_path: str | Path, suggestions=None) -> Path:
        graph = self.memory.load()
        graph.metadata["active_specialized_agent"] = self.name
        graph.metadata["active_repair_focus"] = self.focus
        self.memory.save(graph)
        if suggestions is None:
            stored = graph.metadata.get("last_evaluation_feedback", {})
            raw_suggestions = stored.get("suggestions", []) if isinstance(stored, dict) else []
            suggestions = [SimpleNamespace(**item) for item in raw_suggestions if isinstance(item, dict)]
        return RepairOptimizationAgent(
            self.memory,
            llm=self.llm,
            llm_required=self.llm_required,
            focus=self.focus,
        ).run(test_path, suggestions)

class CoverageEnhancementAgent(FocusedRepairAgent):
    name = "测试生成智能体/覆盖增强模块"
    focus = "coverage"


class AssertionStrengtheningAgent(FocusedRepairAgent):
    name = "测试生成智能体/断言增强模块"
    focus = "assertion"


class MutationRepairAgent(FocusedRepairAgent):
    name = "测试生成智能体/变异增强模块"
    focus = "mutation"


class BoundaryExceptionAgent(FocusedRepairAgent):
    name = "测试生成智能体/边界异常模块"
    focus = "exception"


class RuntimeTypeAgent(FocusedRepairAgent):
    name = "测试生成智能体/运行类型模块"
    focus = "runtime_type"


class TestValidityAgent(FocusedRepairAgent):
    name = "测试生成智能体/测试有效性模块"
    focus = "test_validity"

    def run(self, test_path: str | Path, suggestions=None) -> Path:
        test_path = Path(test_path)
        before = test_path.read_bytes() if test_path.exists() else b""
        if self.llm and self.llm.enabled():
            RepairOptimizationAgent(
                self.memory,
                llm=self.llm,
                llm_required=self.llm_required,
                focus=self.focus,
            ).run(test_path, suggestions or [])
            graph = self.memory.load()
            after = test_path.read_bytes() if test_path.exists() else b""
            if before != after and bool(graph.metadata.get("repair_agent", {}).get("applied")):
                return test_path
        graph = self.memory.load()
        function_names = [
            node.name for node in graph.nodes.values()
            if node.node_type in {"Function", "Async"} and not node.metadata.get("scope")
        ]
        original = test_path.read_text(encoding="utf-8")
        repaired = normalize_generated_test_code(original, source_path=graph.source_path, function_names=function_names)
        operations = [RepairOperation(op="replace_test", test_name=name, new_code=self._extract_test(repaired, name)) for name in self._test_names(original) if self._extract_test(repaired, name)]
        if not operations:
            operations = [RepairOperation(op="add_test", new_code=repaired)]
        return self._validate_and_record(test_path, repaired, operations)

    def _validate_and_record(self, test_path: Path, candidate: str, operations: list[RepairOperation]) -> Path:
        graph = self.memory.load()
        validator = CandidateTestValidator(test_path.parent / "candidate_artifacts")
        validation = validator.validate_and_write(candidate, test_path, cwd=test_path.parent.parent, metadata={"agent": self.name, "operations": [op.to_dict() for op in operations]})
        graph.metadata["repair_agent"] = {
            "applied": validation.accepted,
            "repair_focus": self.focus,
            "operations": [op.to_dict() for op in operations],
            "validation": validation.to_dict(),
            "test_path": str(test_path.resolve()),
        }
        graph.bump_version("test validity repair completed")
        self.memory.save(graph)
        return test_path

    def _test_names(self, text: str) -> list[str]:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return []
        return [node.name for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")]

    def _extract_test(self, text: str, test_name: str) -> str:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return ""
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == test_name:
                return ast.get_source_segment(text, node) or ""
        return ""


class OracleRepairAgent(TestValidityAgent):
    name = "测试生成智能体/Oracle修复模块"
    focus = "oracle"

    def run(self, test_path: str | Path, suggestions=None) -> Path:
        graph = self.memory.load()
        test_path = Path(test_path)
        text = test_path.read_text(encoding="utf-8")
        failed_tests = graph.metadata.get("last_report", {}).get("failed_tests", [])
        failed_names = [str(item.get("test_name")) for item in failed_tests if item.get("test_name")]
        operations: list[RepairOperation] = []
        for name in failed_names:
            replacement = self._replacement_test(graph, name)
            if replacement:
                operations.append(RepairOperation(op="replace_test", test_name=name, new_code=replacement))
            else:
                operations.append(RepairOperation(op="delete_test", test_name=name))
        if not operations:
            operations.append(RepairOperation(op="add_test", new_code=self._fallback_observed_test(graph)))
        candidate = TestRepairOperationApplier().apply(test_path, operations)
        return self._validate_and_record(test_path, candidate, operations)

    def _replacement_test(self, graph, test_name: str) -> str:
        function = self._function_for_failed_test(graph, test_name) or self._first_function(graph)
        if not function:
            return ""
        observed = self._observe(graph.source_path, function)
        if not observed:
            return ""
        args, result_repr = observed
        return (
            f"def {test_name}(target):\n"
            f"    result = {self._call_expression(function, args)}\n"
            f"    assert result == {result_repr}\n"
        )

    def _fallback_observed_test(self, graph) -> str:
        function = self._first_function(graph)
        if not function:
            return "def test_stateflow_module_loads(target):\n    assert target is not None\n"
        observed = self._observe(graph.source_path, function)
        if not observed:
            return "def test_stateflow_module_loads(target):\n    assert target is not None\n"
        args, result_repr = observed
        return (
            f"def test_stateflow_oracle_repaired_{function.name}(target):\n"
            f"    result = {self._call_expression(function, args)}\n"
            f"    assert result == {result_repr}\n"
        )

    def _first_function(self, graph):
        for node in graph.nodes.values():
            scope = str(node.metadata.get("scope") or "")
            if node.node_type in {"Function", "Async"} and "::" not in scope:
                return node
        return None

    def _function_for_failed_test(self, graph, test_name: str):
        test_path = Path(graph.metadata.get("generated_test_path") or graph.metadata.get("last_report", {}).get("test_path", ""))
        if not test_path.exists():
            return None
        try:
            tree = ast.parse(test_path.read_text(encoding="utf-8"))
        except SyntaxError:
            return None
        called: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == test_name:
                receiver_call_ids = {
                    id(child.func.value)
                    for child in ast.walk(node)
                    if isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and isinstance(child.func.value, ast.Call)
                }
                for child in ast.walk(node):
                    if id(child) in receiver_call_ids:
                        continue
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                        function_name = self._target_call_name(child)
                        if function_name:
                            called.append(function_name)
                break
        for function_name in called:
            for graph_node in graph.nodes.values():
                scope = str(graph_node.metadata.get("scope") or "")
                if graph_node.node_type in {"Function", "Async"} and graph_node.name == function_name and "::" not in scope:
                    return graph_node
        return None

    def _observe(self, source_path: str, function_node) -> tuple[str, str] | None:
        args = self._args_for(function_node)
        class_name = self._class_name(function_node)
        script = r"""
import ast
import importlib.util
import sys
from pathlib import Path
source, function_name, args_repr, class_name = sys.argv[1:5]
args = ast.literal_eval(args_repr)
source_path = Path(source).resolve()
parts = [source_path.stem]
cursor = source_path.parent
while (cursor / "__init__.py").is_file():
    parts.insert(0, cursor.name)
    cursor = cursor.parent
sys.path.insert(0, str(cursor if len(parts) > 1 else source_path.parent))
module_name = ".".join(parts) if len(parts) > 1 else "stateflow_target"
spec = importlib.util.spec_from_file_location(module_name, source_path)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[module_name] = module
spec.loader.exec_module(module)
try:
    owner = getattr(module, class_name)() if class_name else module
    print("OK|" + repr(getattr(owner, function_name)(*args)))
except Exception as exc:
    print("EXC|" + exc.__class__.__name__)
"""
        completed = subprocess.run([sys.executable, "-c", script, source_path, function_node.name, repr(args), class_name or ""], text=True, capture_output=True, timeout=5)
        if completed.returncode != 0:
            return None
        out = completed.stdout.strip()
        if not out.startswith("OK|"):
            return None
        return ", ".join(repr(item) for item in args), out[3:]

    def _args_for(self, function_node) -> list[object]:
        args: list[object] = []
        annotations = function_node.metadata.get("arg_annotations", {})
        for arg in function_node.metadata.get("args", []):
            if arg == "self":
                continue
            ann = str(annotations.get(arg) if isinstance(annotations, dict) else "").lower()
            lower = arg.lower()
            if "list" in ann or "numbers" in lower or "nums" in lower:
                args.append([1.0, 2.0, 3.0])
            elif "str" in ann or "text" in lower or "string" in lower:
                args.append("")
            elif "bool" in ann:
                args.append(False)
            elif "float" in ann or "threshold" in lower:
                args.append(0.5)
            elif "int" in ann or lower in {"n", "x", "y"}:
                args.append(1)
            else:
                args.append(None)
        return args

    def _class_name(self, function_node) -> str | None:
        scope = str(function_node.metadata.get("scope") or "")
        return scope if scope and "::" not in scope else None

    def _call_expression(self, function_node, args: str) -> str:
        class_name = self._class_name(function_node)
        receiver = f"target.{class_name}()" if class_name else "target"
        return f"{receiver}.{function_node.name}({args})"

    def _target_call_name(self, node: ast.Call) -> str | None:
        if not isinstance(node.func, ast.Attribute):
            return None
        value = node.func.value
        if isinstance(value, ast.Name) and value.id in {"target", "target_module"}:
            return node.func.attr
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
            receiver = value.func.value
            if isinstance(receiver, ast.Name) and receiver.id in {"target", "target_module"}:
                return node.func.attr
        return None
