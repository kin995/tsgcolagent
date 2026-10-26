from __future__ import annotations

import ast
import json
import keyword
import re
import subprocess
import sys
from pathlib import Path

from agents.candidate_validator import CandidateTestValidator
from agents.java_test_renderer import JavaJUnitTestRenderer
from agents.test_plan import TestCasePlan, TestOracle, TestPlan, TestPlanRenderer, parse_test_plan_json
from graph.state_graph import StateFlowGraph
from llm_client import OpenAICompatibleLLM
from memory.shared_graph import SharedGraphMemory


class TestGenerationAgent:
    __test__ = False
    name = "测试生成智能体/生成模块"

    def __init__(self, memory: SharedGraphMemory, llm: OpenAICompatibleLLM | None = None, llm_required: bool = False):
        self.memory = memory
        self.llm = llm
        self.llm_required = llm_required

    def run(self, output_dir: str | Path) -> Path:
        graph = self.memory.load()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if str(graph.metadata.get("language", "python")) == "java":
            return self._run_java(output_dir, graph)
        test_path = output_dir / f"test_{Path(graph.source_path).stem}_stateflow.py"
        test_code, plan, generation_metadata = self.generate_pytest(graph)
        validator = CandidateTestValidator(output_dir / "candidate_artifacts")
        validation = validator.validate_and_write(
            test_code,
            test_path,
            cwd=output_dir.parent,
            metadata={"agent": self.name, "plan": plan.to_dict(), **generation_metadata},
        )
        if not validation.accepted:
            graph.metadata["test_generation_candidate_rejected"] = validation.to_dict()
            fallback_plan = self._deterministic_plan(graph)
            fallback_code = TestPlanRenderer().render(graph.source_path, fallback_plan, self._function_metadata(graph))
            validation = validator.validate_and_write(
                fallback_code,
                test_path,
                cwd=output_dir.parent,
                metadata={"agent": self.name, "plan": fallback_plan.to_dict(), "strategy": "deterministic_fallback"},
            )
            test_code = fallback_code
            plan = fallback_plan
        graph.metadata["test_generation_validation"] = validation.to_dict()
        if not validation.accepted:
            raise RuntimeError(f"no valid generated test candidate: {validation.to_dict()}")
        graph.metadata["generated_test_path"] = str(test_path.resolve())
        graph.metadata["generated_test_plan"] = plan.to_dict()
        graph.bump_version("test generation completed")
        self.memory.save(graph)
        return test_path

    def _run_java(self, output_dir: Path, graph: StateFlowGraph) -> Path:
        baseline_code, test_class, metadata = JavaJUnitTestRenderer().render(
            graph.source_path,
            graph.metadata.get("structured_context", {}),
            knowledge_hints=graph.metadata.get("experience_hints", {}),
        )
        code = baseline_code
        strategy = "structured_junit_fallback"
        llm_error = None
        if self.llm and self.llm.enabled():
            try:
                code = self.generate_java_test_with_llm(graph, baseline_code, test_class)
                strategy = "llm_state_guided_junit"
                graph.metadata["llm_generation_model"] = self.llm.config.model
            except Exception as exc:
                llm_error = f"{exc.__class__.__name__}: {exc}"
                graph.metadata["llm_generation_error"] = llm_error
                if self.llm_required:
                    self.memory.save(graph)
                    raise RuntimeError(f"required LLM Java test generation failed: {exc}") from exc
        test_path = output_dir / f"{test_class}.java"
        test_path.write_text(code, encoding="utf-8")
        graph.metadata["generated_test_path"] = str(test_path.resolve())
        graph.metadata["generated_test_plan"] = {
            "language": "java",
            "framework": "JUnit 5",
            "test_class": test_class,
            **metadata,
            "strategy": strategy,
        }
        graph.metadata["test_generation_validation"] = {
            "accepted": False,
            "stage": "pending_compile_and_intent_validation",
            "status": "pending_execution_validation",
            "language": "java",
            "framework": "JUnit 5",
            "strategy": strategy,
            "llm_error": llm_error,
            **metadata,
        }
        graph.metadata["test_generation_strategy"] = strategy
        graph.metadata["generation_knowledge_application"] = metadata.get("knowledge_application", {})
        graph.bump_version("java junit candidate generated and awaiting toolchain validation")
        self.memory.save(graph)
        return test_path

    def generate_java_test_with_llm(self, graph: StateFlowGraph, baseline_code: str, test_class: str) -> str:
        assert self.llm is not None
        source_path = Path(graph.source_path).resolve()
        source_text = source_path.read_text(encoding="utf-8", errors="ignore")
        context = graph.metadata.get("structured_context", {})
        intent = graph.metadata.get("last_test_intent_plan", {})
        knowledge = graph.metadata.get("experience_hints", {})
        knowledge_context = graph.metadata.get("last_knowledge_context", {})
        prompt = f"""
You are the generation module inside the Test Generation Agent. Generate one complete JUnit 5 test file.

The shared TSG has decomposed the target into testable states, methods, branches, calls, exceptions and test intent. Assemble a feasible, focused and high-quality test suite from that evidence.

Hard requirements:
- Return Java code only, without Markdown or explanation.
- The test class name must be {test_class}.
- Preserve the source package exactly.
- The code must compile and run in the target Maven/Gradle project with JUnit 5.
- Cover normal behavior first, then boundaries and meaningful exceptions.
- Prefer concrete value, collection, state-change and exact exception assertions.
- Do not use smoke-only, existence-only, tautological or blanket assertDoesNotThrow tests when behavior is observable.
- Respect constructors, static methods, visibility, generics and declared parameter types from the structured context.
- Use retrieved knowledge as guidance, but never copy an incompatible historical solution.
- Keep the suite concise; every test must express a distinct test intent.

Test-state intent:
{json_safe(intent)}

Structured target decomposition:
{json_safe(context)}

Retrieved test knowledge:
{json_safe(knowledge)}

Selected knowledge context:
{json_safe(knowledge_context)}

Deterministic structural baseline that may be improved or replaced:
```java
{baseline_code}
```

Target source:
```java
{source_text}
```
""".strip()
        raw = self.llm.chat(
            [
                {"role": "system", "content": "Return exactly one complete compilable JUnit 5 Java test file."},
                {"role": "user", "content": prompt},
            ]
        )
        code = self._extract_java_code(raw)
        if "@Test" not in code or not re.search(rf"\bclass\s+{re.escape(test_class)}\b", code):
            raise ValueError("Java LLM response does not contain the required JUnit test class")
        return code.rstrip() + "\n"

    def _extract_java_code(self, text: str) -> str:
        fenced = re.search(r"```(?:java)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        return (fenced.group(1) if fenced else text).strip()

    def generate_pytest(self, graph: StateFlowGraph) -> tuple[str, TestPlan, dict[str, object]]:
        functions = self._top_level_functions(graph)
        function_names = {str(function["name"]) for function in functions}
        evidence_ids = set(graph.nodes.keys())
        if self.llm and self.llm.enabled():
            try:
                plan = self.generate_test_plan_with_llm(graph, function_names, evidence_ids)
                code = TestPlanRenderer().render(graph.source_path, plan, self._function_metadata(graph))
                ast.parse(code)
                graph.metadata["test_generation_strategy"] = "llm"
                graph.metadata["llm_generation_model"] = self.llm.config.model
                return code, plan, {"strategy": "llm_test_plan", "model": self.llm.config.model}
            except Exception as exc:
                graph.metadata["llm_generation_error"] = str(exc)
                if self.llm_required:
                    self.memory.save(graph)
                    raise RuntimeError(f"required LLM test generation failed: {exc}") from exc
                graph.metadata["test_generation_strategy"] = "heuristic_fallback_after_llm_error"
        plan = self._deterministic_plan(graph)
        graph.metadata["test_generation_strategy"] = graph.metadata.get("test_generation_strategy", "deterministic")
        return TestPlanRenderer().render(graph.source_path, plan, self._function_metadata(graph)), plan, {"strategy": "deterministic"}

    def generate_test_plan_with_llm(self, graph: StateFlowGraph, function_names: set[str], evidence_ids: set[str]) -> TestPlan:
        assert self.llm is not None
        source_path = Path(graph.source_path).resolve()
        source_text = source_path.read_text(encoding="utf-8")
        functions = self._top_level_functions(graph)
        experience = graph.metadata.get("experience_hints", {})
        prompt = f"""
You are the generation module inside the Test Generation Agent in a multi-agent unit test generation system.
Generate a structured pytest test plan for the target Python source.

Hard requirements:
- Output JSON only with this schema: {{"cases":[{{"id":"unique_case_id","function":"target_function","args":[],"oracle":{{"kind":"equals","value":null}},"evidence_ids":[],"rationale":"","confidence":0.0}}]}}
- Do not output Python code.
- Do not create imports, fixtures, loaders, or module names.
- Every function call will be rendered by the system; use only discovered function names.
- Each case must verify exactly one behavior.
- Valid oracle kinds: equals, raises, is_true, is_false, approx, length_equals, type_is.
- Reject weak oracle ideas such as result is not None or result is None, callable checks, or value == value.
- Use explicit expected output, exception type, or verifiable property.
- Prefer normal behavior cases over exception cases. Every public function must have at least one non-raises case whenever a normal call is possible.
- At least 70 percent of cases should be non-raises cases with exact expected values, unless the function truly has no normal behavior.
- Mine examples from the docstring first, including ">>> f(...)" and "f(...) => value" examples.
- Treat <, <=, >, >= equality boundaries carefully.
- Prefer exactly representable numeric boundary values such as integers, halves, and quarters.

Top-level functions discovered from the analysis graph:
{json_safe(functions)}

Structured analysis artifact:
{json_safe(graph.metadata.get("analysis_artifact", {}))}

Historical experience hints:
{json_safe(experience)}

Actionable learning reminders from previous failures and weak-but-runnable tests:
{json_safe(experience.get("actionable_lessons", []))}

Failure reminders that must be avoided:
{json_safe(experience.get("failure_reminders", []))}

Optimization reminders for runnable but below-threshold tests:
{json_safe(experience.get("optimization_reminders", []))}

Unit-test concepts learned through Test Knowledge Agent conversations. Apply only concepts relevant to this target and preserve executable evidence:
{json_safe(experience.get("conversation_concepts", []))}

Batch-level guidance:
{json_safe(experience.get("batch_guidance", {}))}

Failure-recovery contract from the previous D/E round. When present, the new candidate must address the exact cause, apply the required solutions, and must not repeat the previous failed test unchanged:
{json_safe(graph.metadata.get("failure_recovery_contract", {}))}

Target source:
```python
{source_text}
```
""".strip()
        last_error = ""
        for attempt in range(2):
            response = self.llm.chat(
                [
                    {
                        "role": "system",
                        "content": "You generate structured JSON test plans. Return only JSON.",
                    },
                    {"role": "user", "content": prompt + (f"\n\nPrevious schema error: {last_error}" if last_error else "")},
                ]
            )
            try:
                plan = parse_test_plan_json(response, allowed_functions=function_names, evidence_ids=evidence_ids)
                return self._ground_and_strengthen_llm_plan(graph, plan, functions)
            except Exception as exc:
                last_error = str(exc)
        raise ValueError(f"invalid LLM test plan after retry: {last_error}")

    def _ground_and_strengthen_llm_plan(
        self,
        graph: StateFlowGraph,
        plan: TestPlan,
        functions: list[dict[str, object]],
    ) -> TestPlan:
        """Validate LLM inputs against the real target and ground their oracles.

        TestEval methods rarely contain docstring examples.  An LLM can therefore
        choose useful inputs but still guess an incorrect exact result; rejecting the
        whole suite then falls back to one trivial empty-input test.  We keep diverse,
        runnable inputs, obtain the same deterministic observation evidence already
        used by the fallback generator, and discard unusable guesses.
        """
        function_by_name = {str(item["name"]): item for item in functions}
        grounded: list[TestCasePlan] = []
        seen: set[tuple[str, str]] = set()

        def add_grounded(case: TestCasePlan) -> None:
            function = function_by_name.get(case.function)
            if function is None:
                return
            key = (case.function, repr(case.args))
            if key in seen:
                return
            observed = self._observe_call(case.function, [repr(value) for value in case.args], function, graph)
            if not observed or observed.get("status") == "timeout":
                return
            if observed.get("status") == "ok":
                try:
                    expected = ast.literal_eval(str(observed.get("value")))
                except (SyntaxError, ValueError):
                    return
                grounded.append(
                    TestCasePlan(
                        id=case.id,
                        function=case.function,
                        args=case.args,
                        oracle=TestOracle(kind="equals", value=expected),
                        evidence_ids=case.evidence_ids,
                        rationale=(case.rationale + "; oracle grounded by deterministic target observation").strip("; "),
                        confidence=max(case.confidence, 0.9),
                    )
                )
                seen.add(key)
            elif case.oracle.kind == "raises" and observed.get("status") == "exception":
                grounded.append(
                    TestCasePlan(
                        id=case.id,
                        function=case.function,
                        args=case.args,
                        oracle=TestOracle(kind="raises", exception=str(observed.get("exception") or "Exception")),
                        evidence_ids=case.evidence_ids,
                        rationale=case.rationale,
                        confidence=case.confidence,
                    )
                )
                seen.add(key)

        for case in plan.cases:
            add_grounded(case)

        normal_count = sum(case.oracle.kind != "raises" for case in grounded)
        if normal_count < 3:
            deterministic = self._deterministic_plan(graph)
            for case in deterministic.cases:
                add_grounded(case)
                normal_count = sum(item.oracle.kind != "raises" for item in grounded)
                if normal_count >= 6:
                    break
        has_variable_inputs = any(
            any(str(arg) != "self" for arg in function.get("args", []))
            for function in functions
        )
        minimum_normal = 2 if has_variable_inputs or len(functions) > 1 else 1
        if normal_count < minimum_normal:
            raise ValueError(f"test plan did not produce at least {minimum_normal} distinct runnable normal-behavior cases")
        return TestPlan(cases=grounded[:24])

    def _deterministic_plan(self, graph: StateFlowGraph) -> TestPlan:
        cases: list[TestCasePlan] = []
        for function in self._top_level_functions(graph):
            name = str(function["name"])
            args = [arg for arg in function.get("args", []) if arg != "self"]
            normal_cases: list[TestCasePlan] = []
            exception_cases: list[TestCasePlan] = []
            for index, values in enumerate(self._candidate_input_sets(args, graph, function)[:16]):
                observed = self._observe_call(name, values, function, graph)
                literal_values = [ast.literal_eval(value) for value in values]
                if observed and observed["status"] == "ok":
                    value = ast.literal_eval(observed["value"])
                    oracle = TestOracle(kind="equals", value=value)
                    normal_cases.append(
                        TestCasePlan(
                            id=f"{self._safe_name(name)}_case_{index + 1}",
                            function=name,
                            args=literal_values,
                            oracle=oracle,
                            confidence=0.9,
                            rationale="deterministic observed normal behavior",
                        )
                    )
                elif observed and observed["status"] == "exception":
                    oracle = TestOracle(kind="raises", exception=observed.get("exception", "Exception"))
                    exception_cases.append(
                        TestCasePlan(
                            id=f"{self._safe_name(name)}_exception_case_{index + 1}",
                            function=name,
                            args=literal_values,
                            oracle=oracle,
                            confidence=0.75,
                            rationale="deterministic observed exception behavior",
                        )
                    )
            cases.extend(normal_cases[:10])
            if normal_cases:
                cases.extend(exception_cases[:2])
        if not cases:
            raise RuntimeError("no deterministic test plan could be produced")
        return TestPlan(cases=cases)

    def _top_level_functions(self, graph: StateFlowGraph) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        exported = self._exported_function_keys(graph)
        for node in graph.nodes.values():
            scope = str(node.metadata.get("scope") or "")
            if node.node_type in {"Function", "Async"} and (scope, node.name) in exported:
                class_name = scope or None
                result.append(
                    {
                        "name": node.name,
                        "node_type": node.node_type,
                        "args": node.metadata.get("args", []),
                        "arg_annotations": node.metadata.get("arg_annotations", {}),
                        "is_generator": bool(node.metadata.get("is_generator")),
                        "scope": scope,
                        "class_name": class_name,
                        "qualified_name": f"{class_name}.{node.name}" if class_name else node.name,
                        "line_start": node.line_start,
                        "line_end": node.line_end,
                    }
                )
        return result

    def _exported_function_keys(self, graph: StateFlowGraph) -> set[tuple[str, str]]:
        source_path = Path(graph.source_path)
        try:
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            return {
                (str(node.metadata.get("scope") or ""), node.name)
                for node in graph.nodes.values()
                if node.node_type in {"Function", "Async"} and "::" not in str(node.metadata.get("scope") or "")
            }
        exported: set[tuple[str, str]] = set()
        for item in tree.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and not item.name.startswith("_"):
                exported.add(("", item.name))
            elif isinstance(item, ast.ClassDef) and not item.name.startswith("_"):
                for child in item.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and not child.name.startswith("_"):
                        exported.add((item.name, child.name))
        return exported

    def _function_metadata(self, graph: StateFlowGraph) -> dict[str, dict[str, object]]:
        return {str(item["name"]): item for item in self._top_level_functions(graph)}

    def _tests_for_function(self, function: dict[str, object], graph: StateFlowGraph) -> list[str]:
        name = str(function["name"])
        safe = self._safe_name(name)
        args = [arg for arg in function.get("args", []) if arg != "self"]
        input_sets = self._candidate_input_sets(args, graph, function)
        call_lines: list[str] = []
        for index, values in enumerate(input_sets):
            literal_args = ", ".join(values)
            test_name = f"test_{safe}_stateflow_case_{index + 1}"
            observed = self._observe_call(name, values, function, graph)
            receiver = f"target.{function['class_name']}()" if function.get("class_name") else "target"
            if function["node_type"] == "Async":
                call = f"asyncio.run({receiver}.{name}({literal_args}))"
            elif function.get("is_generator"):
                call = f"list({receiver}.{name}({literal_args}))"
            else:
                call = f"{receiver}.{name}({literal_args})"
            if observed and observed["status"] == "ok":
                call_lines.extend(
                    [
                        f"def {test_name}(target):",
                        f"    result = {call}",
                        f"    assert result == {observed['value']}",
                        "",
                    ]
                )
            elif observed and observed["status"] == "exception":
                exc_name = observed.get("exception", "Exception")
                call_lines.extend(
                    [
                        f"def {test_name}(target):",
                        f"    with pytest.raises({exc_name if exc_name in {'Exception', 'ValueError', 'TypeError', 'KeyError', 'IndexError', 'ZeroDivisionError'} else 'Exception'}):",
                        f"        {call}",
                        "",
                    ]
                )
            else:
                call_lines.extend(
                    [
                        f"def {test_name}(target):",
                        f"    result = {call}",
                        "    assert result is not None or result is None",
                        "",
                    ]
                )

        if self._has_raise_node(graph, function):
            values = ", ".join(self._exception_candidate_args(args))
            receiver = f"target.{function['class_name']}()" if function.get("class_name") else "target"
            call_lines.extend(
                [
                    f"def test_{safe}_exception_path_stateflow(target):",
                    "    with pytest.raises(Exception):",
                    f"        {receiver}.{name}({values})",
                    "",
                ]
            )
        return call_lines

    def _observe_call(
        self,
        function_name: str,
        values: list[str],
        function: dict[str, object],
        graph: StateFlowGraph,
    ) -> dict[str, str] | None:
        script = r"""
from __future__ import annotations
import ast
import asyncio
import importlib.util
import sys
from pathlib import Path

source, function_name, values_repr, is_async, is_generator, class_name = sys.argv[1:7]
values = ast.literal_eval(values_repr)
source_path = Path(source).resolve()
parts = [source_path.stem]
cursor = source_path.parent
while (cursor / "__init__.py").is_file():
    parts.insert(0, cursor.name)
    cursor = cursor.parent
search_root = cursor if len(parts) > 1 else source_path.parent
sys.path.insert(0, str(search_root))
module_name = ".".join(parts) if len(parts) > 1 else "stateflow_observe_target"
spec = importlib.util.spec_from_file_location(module_name, source_path)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[module_name] = module
spec.loader.exec_module(module)
func = getattr(getattr(module, class_name)(), function_name) if class_name else getattr(module, function_name)
try:
    if is_async == "1":
        result = asyncio.run(func(*values))
    elif is_generator == "1":
        result = list(func(*values))
    else:
        result = func(*values)
    print("OK|" + repr(result))
except Exception as exc:
    print("EXC|" + exc.__class__.__name__)
"""
        try:
            literal_values = [ast.literal_eval(value) for value in values]
        except Exception:
            return None
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    graph.source_path,
                    function_name,
                    repr(literal_values),
                    "1" if function["node_type"] == "Async" else "0",
                    "1" if function.get("is_generator") else "0",
                    str(function.get("class_name") or ""),
                ],
                text=True,
                capture_output=True,
                timeout=3,
            )
        except subprocess.TimeoutExpired:
            graph.metadata.setdefault("observation_timeouts", []).append(
                {"function": function_name, "class_name": function.get("class_name"), "args": literal_values, "timeout_seconds": 3}
            )
            return {"status": "timeout"}
        if completed.returncode != 0:
            return None
        output = completed.stdout.strip()
        if output.startswith("OK|"):
            return {"status": "ok", "value": output[3:]}
        if output.startswith("EXC|"):
            return {"status": "exception", "exception": output[4:]}
        return None

    def _candidate_input_sets(self, args: list[str], graph: StateFlowGraph, function: dict[str, object]) -> list[list[str]]:
        if not args:
            return [[]]
        annotations = function.get("arg_annotations", {})
        candidates = self._docstring_input_sets(graph, function)
        modes = ["normal", "low", "high", "negative", "threshold", "very_high", "empty", "text"]
        candidates.extend([[self._value_for_arg(arg, mode, annotations, graph, function) for arg in args] for mode in modes])
        candidates.extend(self._paired_boundary_sets(args, annotations, graph, function))
        deduped: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for item in candidates:
            if len(item) != len(args):
                continue
            key = tuple(item)
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        return deduped[:24]

    def _value_for_arg(
        self,
        arg: str,
        mode: str,
        annotations: object | None = None,
        graph: StateFlowGraph | None = None,
        function: dict[str, object] | None = None,
    ) -> str:
        lower = arg.lower()
        annotation = ""
        if isinstance(annotations, dict):
            annotation = str(annotations.get(arg) or "").lower()
        context = self._argument_context(arg, graph, function)
        if annotation in {"int", "float"}:
            return {
                "low": "0",
                "high": "10",
                "negative": "-1",
                "threshold": "60",
                "very_high": "90",
                "empty": "0",
                "text": "1",
            }.get(mode, "1")
        if annotation in {"str", "path"}:
            return "''" if mode == "empty" else "'abc'"
        if annotation == "bool":
            return "False" if mode in {"low", "empty"} else "True"
        if annotation in {"object", "any"}:
            return {
                "low": "None",
                "high": "'abc'",
                "negative": "-1",
                "threshold": "{}",
                "very_high": "[1, 2, 3]",
                "empty": "None",
                "text": "'abc'",
            }.get(mode, "1")
        if any(token in annotation.replace(" ", "") for token in {"list[list[", "list[tuple[", "sequence[sequence["}):
            if mode == "empty":
                return "[]"
            lower_arg = arg.lower()
            if any(token in lower_arg for token in {"query", "queries", "pair", "edge", "interval"}):
                return {
                    "low": "[[1, 1]]",
                    "high": "[[2, 2], [3, 4], [4, 8]]",
                    "negative": "[[1, 1], [2, 1]]",
                    "threshold": "[[3, 12], [5, 16]]",
                    "very_high": "[[10, 64]]",
                    "text": "[[2, 6], [3, 10]]",
                }.get(mode, "[[1, 1], [2, 2]]")
            return {
                "low": "[[1]]",
                "high": "[[1, 2, 3], [8, 9, 4], [7, 6, 5]]",
                "negative": "[[-3, -2], [-1, 0]]",
                "threshold": "[[1, 2], [4, 3]]",
                "very_high": "[[10, 10], [10, 10]]",
                "text": "[[3, 1, 2], [2, 4, 1]]",
            }.get(mode, "[[1, 2], [3, 4]]")
        if "list[str" in annotation.replace(" ", "") or "list[string" in annotation.replace(" ", ""):
            if mode == "empty":
                return "[]"
            return {
                "low": "['']",
                "high": "['alpha', 'beta', 'gamma']",
                "negative": "['A', 'a']",
                "threshold": "['ab', 'abc']",
                "very_high": "['longer word', 'x']",
                "text": "['foo', 'bar']",
            }.get(mode, "['a', 'bb', 'ccc']")
        if any(token in annotation for token in {"list", "tuple", "set"}):
            if mode == "empty":
                return "[]"
            return {
                "low": "[0]",
                "high": "[1, 2, 3, 4]",
                "negative": "[-2, -1, 0, 1]",
                "threshold": "[59, 60, 61]",
                "very_high": "[10, 100, 1000]",
                "text": "[1, 1, 2, 3]",
            }.get(mode, "[1, 2, -1, 3]" if any(token in context for token in ["integer", "number", "sum", "positive"]) else "[1, 2, 3]")
        if "dict" in annotation or "mapping" in annotation:
            return "{}" if mode == "empty" else "{'a': 1}"
        if "str" in annotation or "string" in context or "word" in context or "text" in context:
            return "''" if mode == "empty" else "'abc'"
        if "list" in annotation or "tuple" in annotation or "set" in annotation or "list" in context or "array" in context:
            if mode == "empty":
                return "[]"
            if any(token in context for token in ["integer", "number", "sum", "pair"]):
                return "[1, 2, -1, 3]"
            return "[1, 2, 3]"
        if "dict" in annotation or "dict" in context or "mapping" in context:
            return "{}" if mode == "empty" else "{'a': 1}"
        if lower in {"flag", "enabled", "active", "valid", "ok"} or lower.startswith("is_") or lower.startswith("has_"):
            return "False" if mode in {"low", "empty"} else "True"
        if any(token in lower for token in ["items", "values", "nums", "numbers", "list", "arr"]):
            return "[]" if mode == "empty" else "[1, 2, 3]"
        if any(token in lower for token in ["mapping", "dict", "config", "options"]):
            return "{}" if mode == "empty" else "{'a': 1}"
        if any(token in lower for token in ["name", "text", "string", "word", "path"]):
            return "''" if mode == "empty" else "'abc'"
        if any(token in lower for token in ["score", "number", "num", "count", "index", "idx", "age", "size"]) or lower in {"n", "x", "y", "a", "b"}:
            return {
                "low": "0",
                "high": "10",
                "negative": "-1",
                "threshold": "60",
                "very_high": "90",
                "empty": "0",
                "text": "1",
            }.get(mode, "1")
        return "None" if mode == "empty" else "1"

    def _paired_boundary_sets(
        self,
        args: list[str],
        annotations: object | None,
        graph: StateFlowGraph,
        function: dict[str, object],
    ) -> list[list[str]]:
        if not args:
            return [[]]
        annotation_values = [str(annotations.get(arg) or "").lower() for arg in args] if isinstance(annotations, dict) else []
        if annotation_values and all(any(kind in value for kind in {"list", "tuple", "set"}) for value in annotation_values):
            partitions = [
                ["[]", "[1]"],
                ["[1]", "[]"],
                ["[1, 3]", "[2]"],
                ["[1, 2]", "[3, 4]"],
                ["[3, 4]", "[1, 2]"],
            ]
            return [item[: len(args)] for item in partitions if len(args) == 2]
        if annotation_values and all(value in {"int", "float"} for value in annotation_values):
            if len(args) == 1:
                return []
            partitions = [["0", "1"], ["1", "0"], ["-1", "1"], ["60", "90"], ["90", "60"]]
            return [item[: len(args)] for item in partitions if len(args) == 2]
        if annotation_values and all(value in {"str", "bool", "object", "any"} for value in annotation_values):
            return []
        context = self._argument_context(" ".join(args), graph, function)
        if any(token in context for token in ["word", "string", "substring", "text"]):
            values = ["'abc'", "'abcd'", "'ell'", "'hello'", "'baa'", "'abab'"]
            return [[values[index % len(values)] for index, _ in enumerate(args)]]
        if any(token in context for token in ["list", "array", "integer", "number"]):
            first = ["[1, 2, 3]", "[]", "[2, 4, -5, 3, 5, 7]", "[0]", "[-1, 1]"]
            return [[first[index % len(first)] if index == 0 else "1" for index, _ in enumerate(args)]]
        return []

    def _docstring_input_sets(self, graph: StateFlowGraph, function: dict[str, object]) -> list[list[str]]:
        source_path = Path(graph.source_path)
        try:
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            return []
        function_name = str(function["name"])
        examples: list[list[str]] = []
        node = self._find_function_ast(tree, function)
        if node:
            doc = ast.get_docstring(node) or ""
            examples.extend(self._calls_from_text(function_name, doc))
        return examples

    def _calls_from_text(self, function_name: str, text: str) -> list[list[str]]:
        calls: list[list[str]] = []
        pattern = re.compile(rf"(?:>>>\s*)?{re.escape(function_name)}\((.*?)\)")
        for match in pattern.finditer(text):
            expression = f"{function_name}({match.group(1)})"
            try:
                parsed = ast.parse(expression, mode="eval")
            except SyntaxError:
                continue
            if not isinstance(parsed.body, ast.Call):
                continue
            try:
                calls.append([repr(ast.literal_eval(arg)) for arg in parsed.body.args])
            except Exception:
                continue
        return calls

    def _argument_context(
        self,
        arg: str,
        graph: StateFlowGraph | None,
        function: dict[str, object] | None,
    ) -> str:
        if graph is None or function is None:
            return ""
        source_path = Path(graph.source_path)
        try:
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            return ""
        function_name = str(function["name"])
        node = self._find_function_ast(tree, function)
        if node:
            doc = ast.get_docstring(node) or ""
            names = " ".join(item.arg for item in node.args.args)
            code = ast.get_source_segment(source_path.read_text(encoding="utf-8"), node) or ""
            return f"{function_name} {arg} {names} {doc} {code}".lower()
        return ""

    def _find_function_ast(
        self,
        tree: ast.Module,
        function: dict[str, object],
    ) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        function_name = str(function["name"])
        class_name = function.get("class_name")
        if class_name:
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == function_name:
                            return item
            return None
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
                return node
        return None

    def _exception_candidate_args(self, args: list[str]) -> list[str]:
        if not args:
            return []
        return ["None" for _ in args]

    def _has_raise_node(self, graph: StateFlowGraph, function: dict[str, object]) -> bool:
        start = int(function["line_start"])
        end = int(function["line_end"])
        return any(node.node_type == "Raise" and start <= node.line_start <= end for node in graph.nodes.values())

    def _safe_name(self, name: str) -> str:
        cleaned = re.sub(r"\W+", "_", name).strip("_") or "function"
        if keyword.iskeyword(cleaned):
            cleaned += "_"
        return cleaned


def json_safe(value: object) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)
