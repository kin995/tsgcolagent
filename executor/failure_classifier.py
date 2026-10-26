from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


FAILURE_CATEGORIES = {
    "syntax_error",
    "import_error",
    "collection_error",
    "fixture_error",
    "no_tests_collected",
    "oracle_error",
    "runtime_error",
    "timeout",
    "valid_baseline",
}


@dataclass
class FailedTest:
    nodeid: str = ""
    test_name: str = ""
    line: int | None = None
    actual: str | None = None
    expected: str | None = None
    message: str | None = None
    code: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "nodeid": self.nodeid,
            "test_name": self.test_name,
            "line": self.line,
            "actual": self.actual,
            "expected": self.expected,
            "message": self.message,
            "code": self.code,
        }


@dataclass
class FailureDiagnosis:
    failure_category: str
    failed_tests: list[FailedTest] = field(default_factory=list)
    pytest_returncode: int | None = None
    stdout_summary: str = ""
    stderr_summary: str = ""
    timeout_kind: str | None = None
    timeout_reason: str | None = None
    likely_location: str | None = None
    should_repair: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "failure_category": self.failure_category,
            "failed_tests": [item.to_dict() for item in self.failed_tests],
            "pytest_returncode": self.pytest_returncode,
            "stdout_summary": self.stdout_summary,
            "stderr_summary": self.stderr_summary,
            "timeout_kind": self.timeout_kind,
            "timeout_reason": self.timeout_reason,
            "likely_location": self.likely_location,
            "should_repair": self.should_repair,
        }


class PytestFailureClassifier:
    def classify(self, pytest_result: Any, test_path: str | Path | None = None, timed_out: bool = False) -> FailureDiagnosis:
        if timed_out:
            return self._timeout_diagnosis(pytest_result, test_path)
        stdout = str(getattr(pytest_result, "stdout", "") or "")
        stderr = str(getattr(pytest_result, "stderr", "") or "")
        returncode = getattr(pytest_result, "returncode", None)
        output = f"{stdout}\n{stderr}"
        if returncode == 0:
            return FailureDiagnosis("valid_baseline", pytest_returncode=returncode, stdout_summary=_tail(stdout), stderr_summary=_tail(stderr))
        lowered = output.lower()
        if "syntaxerror" in lowered:
            category = "syntax_error"
        elif "modulenotfounderror" in lowered or "importerror" in lowered:
            category = "import_error"
        elif "fixture" in lowered and ("not found" in lowered or "fixture" in lowered):
            category = "fixture_error"
        elif "collected 0 items" in lowered or "no tests ran" in lowered:
            category = "no_tests_collected"
        elif "error during collection" in lowered or "error collecting" in lowered or returncode in {2, 3, 4, 5}:
            category = "collection_error"
        elif "assertionerror" in lowered or " assert " in lowered or "e       assert" in lowered:
            category = "oracle_error"
        else:
            category = "runtime_error"
        return FailureDiagnosis(
            category,
            failed_tests=self._failed_tests(output, test_path),
            pytest_returncode=returncode,
            stdout_summary=_tail(stdout),
            stderr_summary=_tail(stderr),
        )

    def _timeout_diagnosis(self, pytest_result: Any, test_path: str | Path | None) -> FailureDiagnosis:
        stdout = str(getattr(pytest_result, "stdout", "") or "")
        stderr = str(getattr(pytest_result, "stderr", "") or "")
        output = f"{stdout}\n{stderr}"
        kind = "pytest_execution_timeout"
        reason = "pytest exceeded the configured timeout after retry"
        likely_location = "pytest runtime"
        should_repair = True
        failed_tests: list[FailedTest] = []

        timeout_probe = _inspect_timeout_sources(Path(test_path)) if test_path else {}
        external_hits = timeout_probe.get("external_wait_markers", [])
        if external_hits:
            kind = "external_dependency_timeout"
            reason = f"timeout is likely caused by network or external service access: {', '.join(external_hits[:4])}"
            likely_location = str(timeout_probe.get("likely_location") or "external dependency")
            should_repair = False
        elif timeout_probe.get("sleep_markers"):
            kind = "local_wait_timeout"
            reason = "timeout is likely caused by local sleep or polling in the generated test"
            likely_location = str(timeout_probe.get("likely_location") or "test wait")
        elif timeout_probe.get("target_calls"):
            kind = "target_call_timeout"
            calls = ", ".join(str(item) for item in timeout_probe.get("target_calls", [])[:4])
            reason = f"timeout is likely caused by a target function call that did not return: {calls}"
            likely_location = str(timeout_probe.get("likely_location") or calls)
        elif timeout_probe.get("test_names"):
            kind = "test_body_timeout"
            reason = "timeout occurred while executing generated pytest test bodies"
            likely_location = str(timeout_probe.get("likely_location") or timeout_probe.get("test_names", ["pytest"])[0])

        for item in timeout_probe.get("failed_tests", []):
            if isinstance(item, FailedTest):
                item.message = reason
                failed_tests.append(item)
        if not failed_tests:
            failed_tests = [FailedTest(test_name="pytest_timeout", message=reason, code=_tail(output, 1000))]
        return FailureDiagnosis(
            "timeout",
            failed_tests=failed_tests[:5],
            pytest_returncode=None,
            stdout_summary=_tail(stdout),
            stderr_summary=_tail(stderr),
            timeout_kind=kind,
            timeout_reason=reason,
            likely_location=likely_location,
            should_repair=should_repair,
        )

    def _failed_tests(self, output: str, test_path: str | Path | None) -> list[FailedTest]:
        tests: list[FailedTest] = []
        for match in re.finditer(r"FAILED\s+([^\s]+)", output):
            nodeid = match.group(1)
            name_match = re.search(r"::([A-Za-z_][A-Za-z0-9_]*)", nodeid)
            tests.append(FailedTest(nodeid=nodeid, test_name=name_match.group(1) if name_match else ""))
        if not tests:
            name_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*=", output)
            tests.append(FailedTest(test_name=name_match.group(1) if name_match else ""))
        if test_path:
            code_by_name = _test_functions(Path(test_path))
            for item in tests:
                if item.test_name in code_by_name:
                    item.code = code_by_name[item.test_name]
        return tests[:10]


def _test_functions(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}
    result: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            result[node.name] = ast.get_source_segment(text, node) or ""
    return result


def _inspect_timeout_sources(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    lowered = text.lower()
    external_markers = [
        marker
        for marker in ["requests", "urllib", "socket", "httpx", "aiohttp", "openai", "ollama", "client.chat", "api_key"]
        if marker in lowered
    ]
    sleep_markers = [marker for marker in ["time.sleep", "sleep(", "asyncio.sleep"] if marker in lowered]
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {
            "external_wait_markers": external_markers,
            "sleep_markers": sleep_markers,
            "likely_location": "syntax error before timeout inspection",
        }
    test_names: list[str] = []
    target_calls: list[str] = []
    failed_tests: list[FailedTest] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        test_names.append(node.name)
        code = ast.get_source_segment(text, node) or ""
        call_names: list[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and isinstance(child.func.value, ast.Name):
                if child.func.value.id in {"target", "target_module"}:
                    call_names.append(child.func.attr)
        for name in call_names:
            target_calls.append(name)
        failed_tests.append(
            FailedTest(
                test_name=node.name,
                nodeid=f"{path.name}::{node.name}",
                line=getattr(node, "lineno", None),
                actual=", ".join(call_names) if call_names else None,
                code=code,
            )
        )
    likely_location = ""
    if failed_tests:
        first = failed_tests[0]
        likely_location = f"{first.test_name}:line {first.line}" if first.line else first.test_name
    return {
        "external_wait_markers": external_markers,
        "sleep_markers": sleep_markers,
        "target_calls": target_calls,
        "test_names": test_names,
        "failed_tests": failed_tests,
        "likely_location": likely_location,
    }


def _tail(text: str, limit: int = 3000) -> str:
    return text[-limit:]
