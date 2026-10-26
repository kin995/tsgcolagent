from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROMPT_VERSION = "test-plan-v2"
SCHEMA_VERSION = "test-plan-schema-v1"
PYTHON_TARGET_LOADER_IMPORTS = ("import importlib.util", "import os", "import sys", "from pathlib import Path")


def python_target_loader_body(source_path: str | Path) -> list[str]:
    """Render a loader that preserves Python package context, including for mutants."""
    source_path = Path(source_path).resolve()
    return [
        f"DEFAULT_TARGET = Path(r'''{source_path}''')",
        "TARGET_PATH = Path(os.environ.get('STATEFLOW_TARGET', DEFAULT_TARGET)).resolve()",
        "",
        "def _target_module_name():",
        "    parts = [DEFAULT_TARGET.stem]",
        "    cursor = DEFAULT_TARGET.parent",
        "    while (cursor / '__init__.py').is_file():",
        "        parts.insert(0, cursor.name)",
        "        cursor = cursor.parent",
        "    search_root = cursor if len(parts) > 1 else DEFAULT_TARGET.parent",
        "    search_text = str(search_root)",
        "    if search_text not in sys.path:",
        "        sys.path.insert(0, search_text)",
        "    return '.'.join(parts) if len(parts) > 1 else 'stateflow_target'",
        "",
        "def _load_target():",
        "    module_name = _target_module_name()",
        "    spec = importlib.util.spec_from_file_location(module_name, TARGET_PATH)",
        "    assert spec and spec.loader",
        "    module = importlib.util.module_from_spec(spec)",
        "    sys.modules[module_name] = module",
        "    spec.loader.exec_module(module)",
        "    return module",
    ]


@dataclass
class TestOracle:
    kind: str
    value: Any = None
    exception: str | None = None
    property: str | None = None

    @classmethod
    def from_dict(cls, data: Any) -> "TestOracle":
        if not isinstance(data, dict):
            raise ValueError("oracle must be an object")
        kind = str(data.get("kind", "")).strip()
        if kind not in {"equals", "raises", "is_true", "is_false", "approx", "length_equals", "type_is"}:
            raise ValueError(f"unsupported oracle kind: {kind}")
        confidence_value = data.get("confidence")
        if confidence_value is not None:
            _validate_confidence(confidence_value)
        return cls(
            kind=kind,
            value=data.get("value"),
            exception=str(data.get("exception")) if data.get("exception") is not None else None,
            property=str(data.get("property")) if data.get("property") is not None else None,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "value": self.value,
            "exception": self.exception,
            "property": self.property,
        }


@dataclass
class TestCasePlan:
    id: str
    function: str
    args: list[Any]
    oracle: TestOracle
    evidence_ids: list[str] = field(default_factory=list)
    rationale: str = ""
    confidence: float = 0.0

    @classmethod
    def from_dict(cls, data: Any, allowed_functions: set[str], evidence_ids: set[str]) -> "TestCasePlan":
        if not isinstance(data, dict):
            raise ValueError("case must be an object")
        case_id = _safe_case_id(str(data.get("id", "")).strip())
        if not case_id:
            raise ValueError("case id is required")
        function = str(data.get("function", "")).strip()
        if function not in allowed_functions:
            raise ValueError(f"unknown function: {function}")
        args = data.get("args", [])
        if not isinstance(args, list):
            raise ValueError("args must be a list")
        oracle = TestOracle.from_dict(data.get("oracle"))
        confidence = _validate_confidence(data.get("confidence", 0.0))
        case_evidence = [str(item) for item in data.get("evidence_ids", []) if str(item) in evidence_ids]
        _reject_weak_oracle(oracle)
        return cls(
            id=case_id,
            function=function,
            args=args,
            oracle=oracle,
            evidence_ids=case_evidence,
            rationale=str(data.get("rationale", ""))[:500],
            confidence=confidence,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "function": self.function,
            "args": self.args,
            "oracle": self.oracle.to_dict(),
            "evidence_ids": self.evidence_ids,
            "rationale": self.rationale,
            "confidence": self.confidence,
        }


@dataclass
class TestPlan:
    cases: list[TestCasePlan]

    @classmethod
    def from_dict(cls, data: Any, allowed_functions: set[str], evidence_ids: set[str]) -> "TestPlan":
        if not isinstance(data, dict):
            raise ValueError("test plan must be an object")
        raw_cases = data.get("cases")
        if not isinstance(raw_cases, list):
            raise ValueError("cases must be a list")
        cases: list[TestCasePlan] = []
        seen: set[str] = set()
        for item in raw_cases[:80]:
            case = TestCasePlan.from_dict(item, allowed_functions, evidence_ids)
            if case.id in seen:
                continue
            seen.add(case.id)
            cases.append(case)
        if not cases:
            raise ValueError("no valid test cases")
        return cls(cases=cases)

    def to_dict(self) -> dict[str, object]:
        return {"cases": [case.to_dict() for case in self.cases]}


class TestPlanRenderer:
    def render(self, source_path: str | Path, plan: TestPlan, function_metadata: dict[str, dict[str, Any]] | None = None) -> str:
        source_path = Path(source_path).resolve()
        function_metadata = function_metadata or {}
        needs_async = any(
            function_metadata.get(case.function, {}).get("node_type") == "Async"
            for case in plan.cases
        )
        lines = [
            "from __future__ import annotations",
            "",
        ]
        if needs_async:
            lines.extend(["import asyncio", ""])
        lines.extend([
            *PYTHON_TARGET_LOADER_IMPORTS,
            "",
            "import pytest",
            "",
            *python_target_loader_body(source_path),
            "",
            "@pytest.fixture(scope='module')",
            "def target():",
            "    return _load_target()",
            "",
        ])
        for case in plan.cases:
            lines.extend(self._render_case(case, function_metadata.get(case.function, {})))
        return "\n".join(lines).rstrip() + "\n"

    def _render_case(self, case: TestCasePlan, metadata: dict[str, Any]) -> list[str]:
        test_name = f"test_{_safe_case_id(case.id)}"
        receiver = f"target.{metadata['class_name']}()" if metadata.get("class_name") else "target"
        raw_call = f"{receiver}.{case.function}({', '.join(_literal(arg) for arg in case.args)})"
        if metadata.get("node_type") == "Async":
            call = f"asyncio.run({raw_call})"
        elif metadata.get("is_generator"):
            call = f"list({raw_call})"
        else:
            call = raw_call
        oracle = case.oracle
        lines = [f"def {test_name}(target):"]
        if oracle.kind == "raises":
            exception = oracle.exception or "Exception"
            lines.append(f"    with pytest.raises({exception}):")
            lines.append(f"        {call}")
        else:
            lines.append(f"    result = {call}")
            if oracle.kind == "equals":
                lines.append(f"    assert result == {_literal(oracle.value)}")
            elif oracle.kind == "is_true":
                lines.append("    assert result is True")
            elif oracle.kind == "is_false":
                lines.append("    assert result is False")
            elif oracle.kind == "approx":
                lines.append(f"    assert result == pytest.approx({_literal(oracle.value)})")
            elif oracle.kind == "length_equals":
                lines.append(f"    assert len(result) == {_literal(oracle.value)}")
            elif oracle.kind == "type_is":
                type_name = str(oracle.value or "object")
                lines.append(f"    assert type(result).__name__ == {_literal(type_name)}")
            else:
                raise ValueError(f"unsupported oracle kind: {oracle.kind}")
        lines.append("")
        return lines


def parse_test_plan_json(text: str, allowed_functions: set[str], evidence_ids: set[str]) -> TestPlan:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidate = fenced.group(1).strip() if fenced else text.strip()
    data = json.loads(candidate)
    return TestPlan.from_dict(data, allowed_functions=allowed_functions, evidence_ids=evidence_ids)


def candidate_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _literal(value: Any) -> str:
    return repr(value)


def _safe_case_id(value: str) -> str:
    safe = re.sub(r"\W+", "_", value).strip("_").lower()
    return safe[:80]


def _validate_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be numeric") from exc
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    return confidence


def _reject_weak_oracle(oracle: TestOracle) -> None:
    if oracle.kind == "equals":
        if isinstance(oracle.value, str) and oracle.value.strip() in {"value", "result"}:
            raise ValueError("weak self-referential equals oracle")
    if oracle.kind == "type_is" and oracle.value in {None, "object"}:
        raise ValueError("weak type oracle")
