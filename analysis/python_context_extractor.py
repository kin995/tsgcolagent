from __future__ import annotations

import ast
from pathlib import Path

from core.structured_context import (
    ClassInfo,
    CodeFragment,
    ControlStructureInfo,
    DependencyInfo,
    FunctionInfo,
    ParameterInfo,
    StructuredContext,
)
from analysis.context_extractor import StructuredContextExtractor


class PythonStructuredContextExtractor(StructuredContextExtractor):
    def extract(self, source_path: str | Path, project_root: str | Path | None = None) -> StructuredContext:
        path = Path(source_path).resolve()
        root = Path(project_root).resolve() if project_root else path.parent
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        module = path.stem
        dependencies = self._dependencies(tree)
        classes: list[ClassInfo] = []
        functions: list[FunctionInfo] = []
        controls: list[ControlStructureInfo] = []
        fragments: list[CodeFragment] = []
        for item in tree.body:
            if isinstance(item, ast.ClassDef):
                class_info = self._class_info(item, source, module, dependencies)
                classes.append(class_info)
                for method in class_info.methods + class_info.constructors:
                    controls.extend(method.branches + method.loops)
                    fragments.extend(method.returns + method.raises_or_throws)
            elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn = self._function_info(item, source, module, None, dependencies)
                functions.append(fn)
                controls.extend(fn.branches + fn.loops)
                fragments.extend(fn.returns + fn.raises_or_throws)
        return StructuredContext(
            language="python",
            project_root=str(root),
            source_path=str(path),
            package_or_module=module,
            classes=classes,
            functions=functions,
            dependencies=dependencies,
            control_structures=controls,
            execution_fragments=fragments,
            rewrite_summary={
                "extractor": self.__class__.__name__,
                "module_docstring_present": ast.get_docstring(tree) is not None,
                "raw_class_count": len(classes),
                "raw_function_count": len(functions),
            },
        )

    def _dependencies(self, tree: ast.Module) -> list[DependencyInfo]:
        dependencies: list[DependencyInfo] = []
        for item in tree.body:
            if isinstance(item, ast.Import):
                for alias in item.names:
                    dependencies.append(DependencyInfo(name=alias.name, alias=alias.asname, kind="import"))
            elif isinstance(item, ast.ImportFrom):
                module = item.module or ""
                for alias in item.names:
                    dependencies.append(DependencyInfo(name=alias.name, source=module, alias=alias.asname, kind="from_import"))
        return dependencies

    def _class_info(self, node: ast.ClassDef, source: str, module: str, dependencies: list[DependencyInfo]) -> ClassInfo:
        methods: list[FunctionInfo] = []
        constructors: list[FunctionInfo] = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method = self._function_info(item, source, module, node.name, dependencies)
                if method.is_constructor:
                    constructors.append(method)
                else:
                    methods.append(method)
        return ClassInfo(
            name=node.name,
            qualified_name=f"{module}.{node.name}",
            visibility="private" if node.name.startswith("_") else "public",
            methods=methods,
            constructors=constructors,
            decorators_or_annotations=[ast.unparse(item) for item in node.decorator_list],
            bases=[ast.unparse(item) for item in node.bases],
            line_start=node.lineno,
            line_end=getattr(node, "end_lineno", node.lineno),
            code=ast.get_source_segment(source, node) or "",
        )

    def _function_info(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        source: str,
        module: str,
        class_name: str | None,
        dependencies: list[DependencyInfo],
    ) -> FunctionInfo:
        owner = f"{module}.{class_name}" if class_name else module
        branches: list[ControlStructureInfo] = []
        loops: list[ControlStructureInfo] = []
        returns: list[CodeFragment] = []
        raises: list[CodeFragment] = []
        calls: list[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.If):
                branches.append(self._control("branch", ast.unparse(child.test), child, f"{owner}.{node.name}"))
            elif isinstance(child, ast.Match):
                branches.append(self._control("match", ast.unparse(child.subject), child, f"{owner}.{node.name}"))
            elif isinstance(child, ast.For):
                loops.append(self._control("loop", f"for {ast.unparse(child.target)} in {ast.unparse(child.iter)}", child, f"{owner}.{node.name}"))
            elif isinstance(child, ast.While):
                loops.append(self._control("loop", f"while {ast.unparse(child.test)}", child, f"{owner}.{node.name}"))
            elif isinstance(child, ast.Return):
                returns.append(self._fragment("return", ast.unparse(child.value) if child.value else "return", child, f"{owner}.{node.name}"))
            elif isinstance(child, ast.Raise):
                raises.append(self._fragment("raise", ast.unparse(child.exc) if child.exc else "raise", child, f"{owner}.{node.name}"))
            elif isinstance(child, ast.Call):
                calls.append(self._call_name(child))
        related = sorted({dep.name for dep in dependencies if any(call.startswith(dep.name + ".") or call == dep.name for call in calls)})
        return FunctionInfo(
            name=node.name,
            qualified_name=f"{owner}.{node.name}",
            visibility="private" if node.name.startswith("_") and node.name != "__init__" else "public",
            parameters=[
                ParameterInfo(
                    name=arg.arg,
                    type_hint=ast.unparse(arg.annotation) if arg.annotation else None,
                    kind="self" if arg.arg in {"self", "cls"} else "positional",
                )
                for arg in node.args.args
            ],
            return_type=ast.unparse(node.returns) if node.returns else None,
            thrown_exceptions=sorted({item.text.split("(", 1)[0] for item in raises if item.text}),
            decorators_or_annotations=[ast.unparse(item) for item in node.decorator_list],
            is_static=any(ast.unparse(item) == "staticmethod" for item in node.decorator_list),
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_constructor=node.name == "__init__",
            calls=[call for call in calls if call],
            branches=branches,
            loops=loops,
            returns=returns,
            raises_or_throws=raises,
            related_dependencies=related,
            line_start=node.lineno,
            line_end=getattr(node, "end_lineno", node.lineno),
            code=ast.get_source_segment(source, node) or "",
            class_name=class_name,
        )

    def _control(self, kind: str, expression: str, node: ast.AST, owner: str) -> ControlStructureInfo:
        return ControlStructureInfo(
            kind=kind,
            expression=expression,
            line_start=getattr(node, "lineno", 1),
            line_end=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
            qualified_owner=owner,
        )

    def _fragment(self, kind: str, text: str, node: ast.AST, owner: str) -> CodeFragment:
        return CodeFragment(
            kind=kind,
            text=text,
            line_start=getattr(node, "lineno", 1),
            line_end=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
            qualified_owner=owner,
        )

    def _call_name(self, node: ast.Call) -> str:
        try:
            return ast.unparse(node.func)
        except Exception:
            return ""

