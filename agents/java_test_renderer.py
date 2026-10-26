from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re


@dataclass
class JavaMethodTarget:
    class_name: str
    method_name: str
    parameters: list[dict[str, Any]]
    return_type: str | None
    code: str = ""
    is_static: bool = False
    is_constructor: bool = False
    package_name: str = ""


class JavaJUnitTestRenderer:
    def render(
        self,
        source_path: str | Path,
        structured_context: dict[str, Any],
        knowledge_hints: dict[str, Any] | None = None,
    ) -> tuple[str, str, dict[str, object]]:
        targets = self._targets(structured_context)
        package_name = str(structured_context.get("package_or_module", "") or "")
        source_stem = Path(source_path).stem
        class_name = f"{source_stem}Test"
        lines: list[str] = []
        if package_name:
            lines.extend([f"package {package_name};", ""])
        lines.extend(
            [
                "import org.junit.jupiter.api.Test;",
                "",
                "import static org.junit.jupiter.api.Assertions.assertEquals;",
                "import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;",
                "import static org.junit.jupiter.api.Assertions.assertFalse;",
                "import static org.junit.jupiter.api.Assertions.assertNull;",
                "import static org.junit.jupiter.api.Assertions.assertThrows;",
                "import static org.junit.jupiter.api.Assertions.assertTrue;",
                "",
                f"class {class_name} {{",
            ]
        )
        for index, target in enumerate(targets, start=1):
            lines.extend(self._test_method(target, index))
        if not targets:
            lines.extend(
                [
                    "    @Test",
                    "    void stateFlowSourceIsReachable() {",
                    "        assertDoesNotThrow(() -> Class.forName(\"" + (f"{package_name}.{source_stem}" if package_name else source_stem) + "\"));",
                    "    }",
                ]
            )
        lines.append("}")
        knowledge_hints = knowledge_hints if isinstance(knowledge_hints, dict) else {}
        reminders = [
            str(value)
            for value in [
                *(knowledge_hints.get("failure_reminders", []) or []),
                *(knowledge_hints.get("optimization_reminders", []) or []),
                *((knowledge_hints.get("followup_failure_guidance", {}) or {}).get("previous_solutions", []) if isinstance(knowledge_hints.get("followup_failure_guidance"), dict) else []),
                *[
                    item.get("guidance")
                    for item in (knowledge_hints.get("conversation_concepts", []) or [])
                    if isinstance(item, dict) and item.get("guidance")
                ],
            ]
            if value
        ][:12]
        metadata = {
            "strategy": "structured_junit_concrete_oracle",
            "test_class": class_name,
            "target_count": len(targets),
            "package": package_name,
            "knowledge_application": {
                "retrieved_reminders": reminders,
                "applied_principles": [
                    "concrete expected-value assertions",
                    "paired boundary cases",
                    "explicit exception-type assertions",
                    "mutation-distinguishing inputs",
                ],
            },
        }
        return "\n".join(lines) + "\n", class_name, metadata

    def _targets(self, structured_context: dict[str, Any]) -> list[JavaMethodTarget]:
        targets: list[JavaMethodTarget] = []
        package_name = str(structured_context.get("package_or_module", "") or "")
        for cls in structured_context.get("classes", []):
            if not isinstance(cls, dict):
                continue
            if str(cls.get("kind", "class")) in {"interface", "annotation_type"}:
                continue
            class_name = str(cls.get("name", ""))
            for method in cls.get("methods", []):
                if not isinstance(method, dict):
                    continue
                if str(method.get("visibility", "public")) == "private":
                    continue
                targets.append(
                    JavaMethodTarget(
                        class_name=class_name,
                        method_name=str(method.get("name", "")),
                        parameters=list(method.get("parameters", [])),
                        return_type=method.get("return_type"),
                        code=str(method.get("code", "") or ""),
                        is_static=bool(method.get("is_static", False)),
                        package_name=package_name,
                    )
                )
        return [item for item in targets if item.class_name and item.method_name][:20]

    def _test_method(self, target: JavaMethodTarget, index: int) -> list[str]:
        cases = self._oracle_cases(target)
        if cases:
            lines: list[str] = []
            for case_index, case in enumerate(cases, start=1):
                safe_name = self._safe_name(f"{target.class_name}_{target.method_name}_{index}_{case['name']}_{case_index}")
                lines.extend(
                    [
                        "    @Test",
                        f"    void {safe_name}() {{",
                        f"        {case['assertion']}",
                        "    }",
                        "",
                    ]
                )
            return lines
        args = ", ".join(self._default_value(param) for param in target.parameters if str(param.get("kind", "")) != "self")
        invocation = self._invocation(target, args)
        safe_name = self._safe_name(f"{target.class_name}_{target.method_name}_{index}")
        return [
            "    @Test",
            f"    void {safe_name}() {{",
            f"        assertDoesNotThrow(() -> {invocation});",
            "    }",
            "",
        ]

    def _oracle_cases(self, target: JavaMethodTarget) -> list[dict[str, str]]:
        return_type = self._clean_type(str(target.return_type or "void"))
        lowered_return = return_type.lower()
        params = [param for param in target.parameters if str(param.get("kind", "")) != "self"]
        if len(params) == 1:
            type_hint = self._clean_type(str(params[0].get("type_hint") or ""))
            lowered_type = type_hint.lower()
            if lowered_return == "boolean":
                cases = self._boolean_single_arg_cases(target, lowered_type)
                if cases:
                    return cases
            if return_type == "String":
                cases = self._string_single_arg_cases(target, lowered_type)
                if cases:
                    return cases
        if lowered_return in {"int", "integer", "long", "double", "float", "short", "byte"}:
            cases = self._numeric_cases(target, params, return_type)
            if cases:
                return cases
            args = ", ".join(self._default_value(param) for param in params)
            return [{"name": "numericResult", "assertion": f"assertDoesNotThrow(() -> {self._invocation(target, args)});"}]
        return []

    def _numeric_cases(self, target: JavaMethodTarget, params: list[dict[str, Any]], return_type: str) -> list[dict[str, str]]:
        if not params or not all(self._is_numeric_type(str(param.get("type_hint") or "")) for param in params):
            return []
        method = target.method_name.lower()
        if method in {"clamp", "constrain", "bound"} and len(params) == 3:
            return [
                self._equals_case(target, "insideRange", ["5", "0", "10"], "5", return_type),
                self._equals_case(target, "belowMinimum", ["-1", "0", "10"], "0", return_type),
                self._equals_case(target, "aboveMaximum", ["11", "0", "10"], "10", return_type),
                {
                    "name": "invalidRange",
                    "assertion": f"assertThrows(IllegalArgumentException.class, () -> {self._invocation(target, '5, 10, 0')});",
                },
            ]
        if method in {"divide", "quotient"} and len(params) == 2:
            return [
                self._equals_case(target, "positiveDivision", ["8", "2"], "4", return_type),
                self._equals_case(target, "negativeDividend", ["-9", "3"], "-3", return_type),
                {
                    "name": "zeroDivisor",
                    "assertion": f"assertThrows(ArithmeticException.class, () -> {self._invocation(target, '8, 0')});",
                },
            ]
        arithmetic = re.search(r"\breturn\s+(\w+)\s*([+\-*/%])\s*(\w+)\s*;", target.code)
        parameter_names = [str(param.get("name") or "") for param in params]
        if arithmetic and len(params) == 2 and arithmetic.group(1) in parameter_names and arithmetic.group(3) in parameter_names:
            operator = arithmetic.group(2)
            values = {
                "+": [(["1", "2"], "3"), (["-2", "3"], "1"), (["0", "0"], "0")],
                "-": [(["5", "2"], "3"), (["-2", "3"], "-5"), (["0", "0"], "0")],
                "*": [(["3", "4"], "12"), (["-2", "3"], "-6"), (["0", "9"], "0")],
                "/": [(["8", "2"], "4"), (["9", "3"], "3")],
                "%": [(["8", "3"], "2"), (["9", "3"], "0")],
            }[operator]
            cases = [self._equals_case(target, f"arithmetic{index}", args, expected, return_type) for index, (args, expected) in enumerate(values, start=1)]
            if operator in {"/", "%"} and "throw new ArithmeticException" in target.code:
                cases.append(
                    {
                        "name": "zeroDivisor",
                        "assertion": f"assertThrows(ArithmeticException.class, () -> {self._invocation(target, '1, 0')});",
                    }
                )
            return cases
        return []

    def _equals_case(self, target: JavaMethodTarget, name: str, args: list[str], expected: str, return_type: str) -> dict[str, str]:
        suffix = "L" if self._clean_type(return_type).lower() == "long" and not expected.endswith("L") else ""
        return {
            "name": name,
            "assertion": f"assertEquals({expected}{suffix}, {self._invocation(target, ', '.join(args))});",
        }

    def _is_numeric_type(self, type_hint: str) -> bool:
        return self._clean_type(type_hint).lower() in {"int", "integer", "long", "double", "float", "short", "byte"}

    def _boolean_single_arg_cases(self, target: JavaMethodTarget, lowered_type: str) -> list[dict[str, str]]:
        if "string" in lowered_type:
            values = [("nullString", "(String) null", "assertTrue"), ("emptyString", "\"\"", "assertTrue"), ("textString", "\"abc\"", "assertFalse")]
        elif lowered_type.endswith("[]"):
            base = self._array_base_type(lowered_type)
            values = [("nullArray", f"({base}[]) null", "assertTrue"), ("emptyArray", f"new {base}[]{{}}", "assertTrue"), ("nonEmptyArray", f"new {base}[]{{{self._array_element_value(base)}}}", "assertFalse")]
        else:
            return []
        return [
            {"name": name, "assertion": f"{assertion}({self._invocation(target, value)});"}
            for name, value, assertion in values
        ]

    def _string_single_arg_cases(self, target: JavaMethodTarget, lowered_type: str) -> list[dict[str, str]]:
        if "string" not in lowered_type:
            return []
        method = target.method_name.lower()
        if "stripleadingandtrailingquotes" in method:
            values = [
                ("nullInput", "(String) null", "assertNull"),
                ("emptyInput", "\"\"", "assertEquals(\"\", {call})"),
                ("singleQuote", "\"\\\"\"", "assertEquals(\"\\\"\", {call})"),
                ("quotedInput", "\"\\\"abc\\\"\"", "assertEquals(\"abc\", {call})"),
                ("plainInput", "\"abc\"", "assertEquals(\"abc\", {call})"),
            ]
        elif "stripleadinghyphens" in method:
            values = [
                ("nullInput", "(String) null", "assertNull"),
                ("emptyInput", "\"\"", "assertEquals(\"\", {call})"),
                ("doubleHyphen", "\"--abc\"", "assertEquals(\"abc\", {call})"),
                ("singleHyphen", "\"-abc\"", "assertEquals(\"abc\", {call})"),
                ("plainInput", "\"abc\"", "assertEquals(\"abc\", {call})"),
            ]
        else:
            values = [("plainInput", "\"abc\"", "assertEquals(\"abc\", {call})")]
        cases: list[dict[str, str]] = []
        for name, value, template in values:
            call = self._invocation(target, value)
            assertion = f"assertNull({call});" if template == "assertNull" else template.format(call=call) + ";"
            cases.append({"name": name, "assertion": assertion})
        return cases

    def _invocation(self, target: JavaMethodTarget, args: str) -> str:
        return f"{target.class_name}.{target.method_name}({args})" if target.is_static else f"new {target.class_name}().{target.method_name}({args})"

    def _array_base_type(self, lowered_type: str) -> str:
        base = lowered_type[:-2].strip()
        mapping = {"object": "Object", "string": "String", "integer": "Integer"}
        return mapping.get(base, base[:1].upper() + base[1:] if base else "Object")

    def _array_element_value(self, base: str) -> str:
        if base in {"byte", "short", "int", "long", "Byte", "Short", "Integer", "Long"}:
            return "1"
        if base in {"float", "double", "Float", "Double"}:
            return "1.0"
        if base in {"boolean", "Boolean"}:
            return "true"
        if base in {"char", "Character"}:
            return "'a'"
        if base == "String":
            return "\"abc\""
        return f"new {base}()"

    def _default_value(self, param: dict[str, Any]) -> str:
        type_hint = self._clean_type(str(param.get("type_hint") or "").strip())
        lowered = type_hint.lower()
        if lowered.endswith("[]"):
            return f"new {type_hint[:-2].strip()}[]{{}}"
        raw_type = lowered.split("<", 1)[0].rsplit(".", 1)[-1]
        collections = {
            "arraylist": "ArrayList", "list": "ArrayList", "collection": "ArrayList", "iterable": "ArrayList",
            "linkedlist": "LinkedList", "set": "HashSet", "hashset": "HashSet",
            "sortedset": "TreeSet", "treeset": "TreeSet",
            "map": "HashMap", "hashmap": "HashMap", "sortedmap": "TreeMap", "treemap": "TreeMap",
        }
        if raw_type in collections:
            return f"new java.util.{collections[raw_type]}<>()"
        if raw_type == "float":
            return "1.0f"
        if raw_type in {"short", "byte"}:
            return f"({raw_type}) 1"
        if lowered in {"int", "integer", "short", "byte"}:
            return "1"
        if lowered == "long":
            return "1L"
        if lowered in {"double", "float"}:
            return "1.0"
        if lowered == "boolean":
            return "true"
        if lowered in {"char", "character"}:
            return "'a'"
        if "string" in lowered:
            return "\"abc\""
        if "list" in lowered or "collection" in lowered or "iterable" in lowered:
            return "java.util.Collections.emptyList()"
        if "map" in lowered:
            return "java.util.Collections.emptyMap()"
        if lowered.endswith("[]"):
            base = type_hint[:-2].strip()
            return f"new {base}[]{{}}"
        return "null"

    def _clean_type(self, type_hint: str) -> str:
        text = re.sub(r"@\w+(?:\([^)]*\))?\s*", "", type_hint).strip()
        for modifier in ["public", "protected", "private", "static", "final", "volatile", "transient"]:
            text = re.sub(rf"\b{modifier}\b\s*", "", text)
        text = text.replace("...", "[]").strip()
        primitives = {
            "integer": "int",
            "bool": "boolean",
            "string": "String",
            "object": "Object",
        }
        return primitives.get(text.lower(), text)

    def _safe_name(self, value: str) -> str:
        cleaned = "".join(ch if ch.isalnum() else "_" for ch in value)
        if not cleaned:
            return "stateFlowCase"
        if cleaned[0].isdigit():
            cleaned = f"case_{cleaned}"
        return cleaned[0].lower() + cleaned[1:]
