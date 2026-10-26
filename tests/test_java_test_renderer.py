from __future__ import annotations

from agents.java_test_renderer import JavaJUnitTestRenderer


def test_java_numeric_methods_use_concrete_oracles_and_boundaries(tmp_path):
    source = tmp_path / "SimpleCalculator.java"
    source.write_text("class SimpleCalculator {}\n", encoding="utf-8")
    context = {
        "package_or_module": "sample",
        "classes": [
            {
                "name": "SimpleCalculator",
                "kind": "class",
                "methods": [
                    {
                        "name": "add",
                        "return_type": "int",
                        "parameters": [{"name": "left", "type_hint": "int"}, {"name": "right", "type_hint": "int"}],
                        "code": "public int add(int left, int right) { return left + right; }",
                    },
                    {
                        "name": "clamp",
                        "return_type": "int",
                        "parameters": [
                            {"name": "value", "type_hint": "int"},
                            {"name": "minimum", "type_hint": "int"},
                            {"name": "maximum", "type_hint": "int"},
                        ],
                        "code": "if (minimum > maximum) throw new IllegalArgumentException();",
                    },
                    {
                        "name": "divide",
                        "return_type": "int",
                        "parameters": [{"name": "dividend", "type_hint": "int"}, {"name": "divisor", "type_hint": "int"}],
                        "code": "if (divisor == 0) throw new ArithmeticException(); return dividend / divisor;",
                    },
                ],
            }
        ],
    }

    code, _, _ = JavaJUnitTestRenderer().render(source, context)

    assert "assertEquals(3, new SimpleCalculator().add(1, 2));" in code
    assert "assertEquals(0, new SimpleCalculator().clamp(-1, 0, 10));" in code
    assert "assertThrows(IllegalArgumentException.class" in code
    assert "assertThrows(ArithmeticException.class" in code
    assert "assertDoesNotThrow(()" not in code
