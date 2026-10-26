from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ParameterInfo:
    name: str
    type_hint: str | None = None
    default: str | None = None
    kind: str = "positional"


@dataclass
class DependencyInfo:
    name: str
    source: str | None = None
    alias: str | None = None
    kind: str = "import"


@dataclass
class ControlStructureInfo:
    kind: str
    expression: str
    line_start: int
    line_end: int
    qualified_owner: str | None = None
    true_condition: str | None = None
    false_condition: str | None = None


@dataclass
class CodeFragment:
    kind: str
    text: str
    line_start: int
    line_end: int
    qualified_owner: str | None = None


@dataclass
class FunctionInfo:
    name: str
    qualified_name: str
    visibility: str = "public"
    parameters: list[ParameterInfo] = field(default_factory=list)
    return_type: str | None = None
    thrown_exceptions: list[str] = field(default_factory=list)
    decorators_or_annotations: list[str] = field(default_factory=list)
    is_static: bool = False
    is_async: bool = False
    is_constructor: bool = False
    calls: list[str] = field(default_factory=list)
    branches: list[ControlStructureInfo] = field(default_factory=list)
    loops: list[ControlStructureInfo] = field(default_factory=list)
    returns: list[CodeFragment] = field(default_factory=list)
    raises_or_throws: list[CodeFragment] = field(default_factory=list)
    related_dependencies: list[str] = field(default_factory=list)
    line_start: int = 1
    line_end: int = 1
    code: str = ""
    class_name: str | None = None
    type_parameters: list[str] = field(default_factory=list)
    is_override: bool = False


@dataclass
class ClassInfo:
    name: str
    qualified_name: str
    visibility: str = "public"
    methods: list[FunctionInfo] = field(default_factory=list)
    constructors: list[FunctionInfo] = field(default_factory=list)
    decorators_or_annotations: list[str] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)
    line_start: int = 1
    line_end: int = 1
    code: str = ""
    kind: str = "class"
    type_parameters: list[str] = field(default_factory=list)


@dataclass
class StructuredContext:
    language: str
    project_root: str
    source_path: str
    package_or_module: str
    classes: list[ClassInfo] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)
    dependencies: list[DependencyInfo] = field(default_factory=list)
    control_structures: list[ControlStructureInfo] = field(default_factory=list)
    execution_fragments: list[CodeFragment] = field(default_factory=list)
    rewrite_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StructuredContext":
        def params(items: list[dict[str, Any]]) -> list[ParameterInfo]:
            return [ParameterInfo(**item) for item in items]

        def controls(items: list[dict[str, Any]]) -> list[ControlStructureInfo]:
            return [ControlStructureInfo(**item) for item in items]

        def fragments(items: list[dict[str, Any]]) -> list[CodeFragment]:
            return [CodeFragment(**item) for item in items]

        def function(item: dict[str, Any]) -> FunctionInfo:
            values = dict(item)
            values["parameters"] = params(values.get("parameters", []))
            values["branches"] = controls(values.get("branches", []))
            values["loops"] = controls(values.get("loops", []))
            values["returns"] = fragments(values.get("returns", []))
            values["raises_or_throws"] = fragments(values.get("raises_or_throws", []))
            return FunctionInfo(**values)

        classes = []
        for item in data.get("classes", []):
            values = dict(item)
            values["methods"] = [function(fn) for fn in values.get("methods", [])]
            values["constructors"] = [function(fn) for fn in values.get("constructors", [])]
            classes.append(ClassInfo(**values))
        return cls(
            language=str(data.get("language", "python")),
            project_root=str(data.get("project_root", "")),
            source_path=str(data.get("source_path", "")),
            package_or_module=str(data.get("package_or_module", "")),
            classes=classes,
            functions=[function(item) for item in data.get("functions", [])],
            dependencies=[DependencyInfo(**item) for item in data.get("dependencies", [])],
            control_structures=controls(data.get("control_structures", [])),
            execution_fragments=fragments(data.get("execution_fragments", [])),
            rewrite_summary=dict(data.get("rewrite_summary", {})),
        )
