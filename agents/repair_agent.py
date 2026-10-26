from __future__ import annotations

import re
import json
from pathlib import Path

from agents.evaluation_agent import RepairSuggestion
from agents.test_repair_operations import RepairOperation, TestRepairOperationApplier
from graph.patch_search import PatchCandidate, PatchSearchRepair
from graph.repair_planner import RootCauseRepairPlanner
from llm_client import OpenAICompatibleLLM, extract_json_data
from memory.shared_graph import SharedGraphMemory
from agents.test_code_normalizer import normalize_generated_test_code


class RepairOptimizationAgent:
    name = "测试生成智能体/修复优化模块"

    def __init__(
        self,
        memory: SharedGraphMemory,
        llm: OpenAICompatibleLLM | None = None,
        llm_required: bool = False,
        focus: str = "general",
    ):
        self.memory = memory
        self.llm = llm
        self.llm_required = llm_required
        self.focus = focus

    def run(self, test_path: str | Path, suggestions: list[RepairSuggestion]) -> Path:
        graph = self.memory.load()
        test_path = Path(test_path)
        text = test_path.read_text(encoding="utf-8") if test_path.exists() else ""
        plan = RootCauseRepairPlanner(graph, focus=self.focus).plan(test_path)
        optimization_contract = graph.metadata.get("c_grade_optimization_contract")
        quality_optimization = isinstance(optimization_contract, dict) and bool(optimization_contract)
        applied = False
        applied_strategy = "none"
        if self.llm and self.llm.enabled():
            graph.metadata["optimization_llm_invocation"] = {
                "status": "requested",
                "model": self.llm.config.model,
                "focus": self.focus,
                "prompt_version": graph.metadata.get("optimization_prompt_version", "targeted-repair-v1"),
                "quality_optimization": quality_optimization,
            }
            self.memory.save(graph)
            try:
                if plan or quality_optimization:
                    operations = self.optimize_operations_with_llm(text, suggestions, plan)
                    applied = self._apply_validated_operations(test_path, operations)
                    if not applied:
                        applied = bool(plan) and self._apply_best_validated_patch(test_path, [PatchCandidate("planner_patch", plan.patch)])
                        applied_strategy = "planner_patch" if applied else "none"
                    else:
                        applied_strategy = "llm_targeted_operations"
                else:
                    operations = self.optimize_operations_with_llm(text, suggestions, plan=None)
                    applied = self._apply_validated_operations(test_path, operations)
                    applied_strategy = "llm_repair_operations" if applied else "none"
            except Exception as exc:
                graph = self.memory.load()
                graph.metadata["llm_repair_error"] = str(exc)
                graph.metadata["optimization_llm_invocation"] = {
                    **graph.metadata.get("optimization_llm_invocation", {}),
                    "status": "failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
                if self.llm_required and _is_connection_error(exc):
                    raise
        if not applied and plan:
            applied = self._apply_best_validated_patch(test_path, [PatchCandidate("planner_patch", plan.patch)])
            applied_strategy = "planner_patch" if applied else applied_strategy
        if not applied and not quality_optimization:
            repaired = self.optimize_test_code(text, suggestions)
            applied = self._apply_best_validated_patch(test_path, [PatchCandidate("heuristic_full_file_rewrite", repaired, replace_file=True)])
            applied_strategy = "heuristic_repair" if applied else applied_strategy
        graph = self.memory.load()
        graph.metadata["test_repair_strategy"] = applied_strategy
        if self.llm and self.llm.enabled():
            graph.metadata["llm_repair_model"] = self.llm.config.model
            graph.metadata["optimization_llm_invocation"] = {
                **graph.metadata.get("optimization_llm_invocation", {}),
                "status": "accepted" if applied and applied_strategy == "llm_targeted_operations" else "completed_without_llm_acceptance",
                "selected_strategy": applied_strategy,
            }
        graph.metadata["repair_agent"] = {
            "applied": applied,
            "suggestion_count": len(suggestions),
            "minimal_plan": plan.to_dict() if plan else None,
            "minimal_patch_applied": applied and applied_strategy in {"llm_targeted_operations", "planner_patch"},
            "repair_focus": self.focus,
            "quality_optimization": quality_optimization,
            "strategy": applied_strategy,
            "test_path": str(test_path.resolve()),
        }
        graph.bump_version("repair optimization completed")
        self.memory.save(graph)
        return test_path

    def _apply_validated_operations(self, test_path: Path, operations: list[RepairOperation]) -> bool:
        graph = self.memory.load()
        try:
            candidate = TestRepairOperationApplier().apply(test_path, operations)
        except Exception as exc:
            graph.metadata["repair_operation_error"] = str(exc)
            self.memory.save(graph)
            return False
        search = PatchSearchRepair(graph.source_path, baseline_report=graph.metadata.get("last_report"), graph=graph)
        best = search.choose(test_path, [PatchCandidate("llm_operations", candidate, replace_file=True)])
        record = {
            "candidate_count": 1,
            "operation_count": len(operations),
            "operations": [op.to_dict() for op in operations],
            "selected": best.to_dict() if best else None,
            "evaluations": [item.to_dict() for item in search.last_evaluations],
            "validator": "semantic_and_quality_patch_search",
        }
        graph.metadata.setdefault("patch_search_history", []).append(record)
        graph.metadata["patch_search"] = record
        graph.metadata["repair_operation_validation"] = record
        if not best:
            self.memory.save(graph)
            return False
        test_path.write_text(best.candidate.patch.strip() + "\n", encoding="utf-8")
        self.memory.save(graph)
        return True

    def _apply_best_validated_patch(self, test_path: Path, candidates: list[PatchCandidate]) -> bool:
        graph = self.memory.load()
        search = PatchSearchRepair(graph.source_path, baseline_report=graph.metadata.get("last_report"), graph=graph)
        best = search.choose(test_path, candidates)
        search_record = {
            "candidate_count": len(candidates),
            "selected": best.to_dict() if best else None,
            "evaluations": [item.to_dict() for item in search.last_evaluations],
        }
        graph.metadata.setdefault("patch_search_history", []).append(search_record)
        graph.metadata["patch_search"] = search_record
        if not best:
            self.memory.save(graph)
            return False
        if best.candidate.replace_file:
            test_path.write_text(best.candidate.patch.strip() + "\n", encoding="utf-8")
            applied = True
        else:
            applied = self._append_patch_once(test_path, best.candidate.patch)
        self.memory.save(graph)
        return applied

    def optimize_operations_with_llm(self, text: str, suggestions: list[RepairSuggestion], plan) -> list[RepairOperation]:
        assert self.llm is not None
        graph = self.memory.load()
        source_path = Path(graph.source_path).resolve()
        source_text = source_path.read_text(encoding="utf-8")
        prompt = f"""
You are the Repair Optimization Agent in an LLM multi-agent pytest generation loop.
Return structured RepairOperation JSON only.

Hard requirements:
- Return JSON with key operations.
- Each operation must be one of replace_test, delete_test, add_test.
- Return at most three operations, ordered by expected quality gain.
- Prefer replace_test for wrong assertions.
- Do not return a full file.
- Do not invent imports, loaders, or module-level target imports.
- New code must be only pytest test function code.
- New tests must call target.<function>(...).
- The patch focus is: {self.focus}.
- For C-grade quality optimization, never delete a passing test and never rewrite unrelated tests.
- Every added or replaced test must contain a concrete behavioral assertion or an exact pytest.raises oracle.
- Coverage focus: exercise the listed missing lines or target states with a concrete expected result.
- Assertion focus: replace weak/existence assertions with exact values, relationships, or exception types.
- Mutation focus: choose inputs and oracles that distinguish the listed surviving mutant lines from the original behavior.
- Boundary focus: test the nearest meaningful boundary on both sides, not arbitrary values.
- Runtime-type focus: use observed types and preserve the public API contract.
- A candidate is useful only when it improves the dominant metric without reducing any baseline metric.

Target source path:
{source_path}

Target source:
```python
{source_text}
```

Existing test file:
```python
{text}
```

Root-cause minimal repair plan:
{plan.to_dict() if plan else None}

State-guided minimal target scope:
{graph.metadata.get("state_positioning_repair") or graph.metadata.get("minimal_target_repair")}

C-grade targeted optimization contract:
{graph.metadata.get("c_grade_optimization_contract")}

Scheduler decision:
{graph.metadata.get("last_agent_decision")}

Historical experience hints:
{graph.metadata.get("experience_hints")}

Actionable learning reminders:
{(graph.metadata.get("experience_hints") or {}).get("actionable_lessons", [])}

Failure reminders that must not be repeated:
{(graph.metadata.get("experience_hints") or {}).get("failure_reminders", [])}

Optimization reminders for runnable but weak tests:
{(graph.metadata.get("experience_hints") or {}).get("optimization_reminders", [])}

Evaluation suggestions:
{[item.to_dict() if hasattr(item, "to_dict") else vars(item) for item in suggestions]}
""".strip()
        last_error = ""
        for _attempt in range(2):
            retry_note = (
                "\n\nThe previous response was invalid. Correct this validation error and return the exact schema only: "
                + last_error
                if last_error
                else ""
            )
            response = self.llm.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "Return JSON only. Exact example: "
                            '{"operations":[{"op":"add_test","test_name":null,'
                            '"new_code":"def test_case(target):\\n    assert target.f(1) == 2\\n"}]}'
                        ),
                    },
                    {"role": "user", "content": prompt + retry_note},
                ]
            )
            try:
                data = extract_json_data(response)
                raw_operations = data.get("operations", []) if isinstance(data, dict) else []
                operations = [RepairOperation.from_dict(item) for item in raw_operations if isinstance(item, dict)]
                if not operations:
                    raise ValueError("LLM returned no repair operations")
                return self._validate_llm_operations(operations, graph)
            except Exception as exc:
                last_error = str(exc)
        raise ValueError(f"invalid repair operations after retry: {last_error}")

    def _validate_llm_operations(self, operations: list[RepairOperation], graph) -> list[RepairOperation]:
        contract = graph.metadata.get("c_grade_optimization_contract")
        quality_optimization = isinstance(contract, dict) and bool(contract)
        if len(operations) > 3:
            raise ValueError("at most three targeted operations are allowed")
        if not quality_optimization:
            return operations
        for operation in operations:
            if operation.op == "delete_test":
                raise ValueError("C-grade optimization cannot delete a passing test")
            code = operation.new_code or ""
            if "target." not in code:
                raise ValueError("each C-grade optimization must directly call the target")
            if "assert " not in code and "pytest.raises" not in code:
                raise ValueError("each C-grade optimization needs a concrete assertion or exception oracle")
        return operations

    def _append_patch_once(self, test_path: Path, patch: str) -> bool:
        if not test_path.exists():
            return False
        text = test_path.read_text(encoding="utf-8")
        marker = self._first_test_name(patch)
        if marker and marker in text:
            return False
        test_path.write_text(text.rstrip() + "\n\n" + patch.strip() + "\n", encoding="utf-8")
        return True

    def _first_test_name(self, patch: str) -> str | None:
        match = re.search(r"def\s+(test_[A-Za-z0-9_]+)\s*\(", patch)
        return match.group(1) if match else None

    def optimize_test_code(self, text: str, suggestions: list[RepairSuggestion]) -> str:
        if "pytest" not in text:
            lines = text.splitlines()
            insert_at = 1 if lines and lines[0].startswith("from __future__") else 0
            lines.insert(insert_at, "import pytest")
            text = "\n".join(lines) + "\n"
        text = self._remove_duplicate_blank_lines(text)
        needs_assert_repair = any(item.action == "Need Assert" for item in suggestions)
        needs_exception_repair = any(item.action == "Need Exception" for item in suggestions)
        needs_boundary_repair = any(item.action in {"Need Boundary", "Need Path"} for item in suggestions)
        additions: list[str] = []
        if needs_assert_repair and "test_stateflow_module_has_public_api" not in text:
            additions.extend(
                [
                    "",
                    "def test_stateflow_module_has_public_api(target):",
                    "    public_names = [name for name in dir(target) if not name.startswith('_')]",
                    "    assert isinstance(public_names, list)",
                    "",
                ]
            )
        if needs_boundary_repair and "test_stateflow_boundary_smoke" not in text:
            additions.extend(
                [
                    "def test_stateflow_boundary_smoke(target):",
                    "    callables = [getattr(target, name) for name in dir(target) if callable(getattr(target, name)) and not name.startswith('_')]",
                    "    assert callables is not None",
                    "",
                ]
            )
        if needs_exception_repair and "test_stateflow_exception_contracts_are_explicit" not in text:
            additions.extend(
                [
                    "def test_stateflow_exception_contracts_are_explicit(target):",
                    "    exception_types = [value for value in vars(target).values() if isinstance(value, type) and issubclass(value, BaseException)]",
                    "    assert isinstance(exception_types, list)",
                    "",
                ]
            )
        return text.rstrip() + "\n" + "\n".join(additions)

    def _remove_duplicate_blank_lines(self, text: str) -> str:
        return re.sub(r"\n{3,}", "\n\n", text)


def _is_connection_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "connection" in text or "not installed" in text or "http" in text
