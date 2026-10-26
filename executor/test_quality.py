from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TestUsabilityResult:
    test_count: int
    assertion_count: int
    normal_behavior_tests: int
    exception_tests: int
    weak_assertions: int

    @property
    def weak_assertions_only(self) -> bool:
        return self.assertion_count > 0 and self.weak_assertions >= self.assertion_count

    @property
    def exception_only(self) -> bool:
        return self.test_count > 0 and self.normal_behavior_tests == 0 and self.exception_tests == self.test_count

    def to_dict(self) -> dict[str, object]:
        return {
            "test_count": self.test_count,
            "assertion_count": self.assertion_count,
            "normal_behavior_tests": self.normal_behavior_tests,
            "exception_tests": self.exception_tests,
            "weak_assertions": self.weak_assertions,
            "weak_assertions_only": self.weak_assertions_only,
            "exception_only": self.exception_only,
        }


def analyze_test_usability(test_path: str | Path) -> TestUsabilityResult:
    path = Path(test_path)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return TestUsabilityResult(0, 0, 0, 0, 0)

    tests = _collected_test_functions(tree)
    assertion_count = 0
    weak_assertions = 0
    exception_tests = 0
    normal_behavior_tests = 0
    for test in tests:
        assertions = [node for node in ast.walk(test) if isinstance(node, ast.Assert)]
        raises = [node for node in ast.walk(test) if _is_pytest_raises(node)]
        assertion_count += len(assertions) + len(raises)
        weak_assertions += sum(1 for node in assertions if _is_weak_assertion(node.test))
        if raises:
            exception_tests += 1
        if assertions and _calls_target(test):
            normal_behavior_tests += 1
    return TestUsabilityResult(
        test_count=len(tests),
        assertion_count=assertion_count,
        normal_behavior_tests=normal_behavior_tests,
        exception_tests=exception_tests,
        weak_assertions=weak_assertions,
    )


def _collected_test_functions(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    tests: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            tests.append(node)
            continue
        if not isinstance(node, ast.ClassDef) or not node.name.startswith("Test"):
            continue
        tests.extend(
            child
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test_")
        )
    return tests


def _is_pytest_raises(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pytest"
        and node.func.attr == "raises"
    )


def _calls_target(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in {"target", "target_module"}:
            return True
    return False


def _is_weak_assertion(test: ast.AST) -> bool:
    if isinstance(test, ast.Constant) and isinstance(test.value, bool):
        return True
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and len(test.comparators) == 1:
        right = test.comparators[0]
        if isinstance(test.ops[0], (ast.Is, ast.IsNot)) and isinstance(right, ast.Constant) and right.value is None:
            return True
        if isinstance(test.ops[0], (ast.Eq, ast.NotEq)) and isinstance(right, ast.Constant) and right.value is None:
            return True
        if ast.dump(test.left) == ast.dump(right):
            return True
    if isinstance(test, ast.Call) and isinstance(test.func, ast.Name) and test.func.id in {"callable", "hasattr"}:
        return True
    if isinstance(test, ast.Name) and test.id in {"target", "target_module"}:
        return True
    return False
