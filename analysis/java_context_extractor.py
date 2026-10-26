from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from tree_sitter import Language, Node, Parser
import tree_sitter_java

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


class JavaStructuredContextExtractor(StructuredContextExtractor):
    CLASS_RE = re.compile(r"(?P<annotations>(?:@\w+(?:\([^)]*\))?\s*)*)(?P<visibility>public|protected|private)?\s*(?:final\s+|abstract\s+)?class\s+(?P<name>\w+)(?P<tail>[^{]*)\{", re.MULTILINE)
    METHOD_RE = re.compile(
        r"(?P<annotations>(?:@\w+(?:\([^)]*\))?\s*)*)"
        r"(?P<prefix>(?:[A-Za-z_][\w.<>\[\], ?]*\s+)+)?"
        r"(?P<name>\w+)\s*\((?P<params>[^)]*)\)\s*(?:throws\s+(?P<throws>[^{]+))?\{",
        re.MULTILINE,
    )
    MODIFIERS = {"public", "protected", "private", "static", "final", "synchronized", "abstract", "native", "strictfp", "default"}
    NON_METHOD_NAMES = {"if", "for", "while", "switch", "catch", "return", "throw", "new", "do", "else", "try", "finally"}

    def extract(self, source_path: str | Path, project_root: str | Path | None = None) -> StructuredContext:
        path = Path(source_path).resolve()
        root = Path(project_root).resolve() if project_root else path.parent
        source = path.read_text(encoding="utf-8", errors="ignore")
        try:
            return self._extract_tree_sitter(path, root, source)
        except Exception as exc:
            context = self._extract_regex(path, root, source)
            context.rewrite_summary["parser_error"] = f"{exc.__class__.__name__}: {exc}"
            return context

    def _extract_regex(self, path: Path, root: Path, source: str) -> StructuredContext:
        package = self._package(source)
        dependencies = self._dependencies(source)
        classes: list[ClassInfo] = []
        controls: list[ControlStructureInfo] = []
        fragments: list[CodeFragment] = []
        for match in self.CLASS_RE.finditer(source):
            class_name = match.group("name")
            body_start = match.end()
            body_end = self._matching_brace(source, body_start - 1)
            body = source[body_start:body_end]
            class_info = self._class_info(source, body, match, class_name, package, dependencies)
            classes.append(class_info)
            for method in class_info.methods + class_info.constructors:
                controls.extend(method.branches + method.loops)
                fragments.extend(method.returns + method.raises_or_throws)
        return StructuredContext(
            language="java",
            project_root=str(root),
            source_path=str(path),
            package_or_module=package,
            classes=classes,
            functions=[],
            dependencies=dependencies,
            control_structures=controls,
            execution_fragments=fragments,
            rewrite_summary={
                "extractor": self.__class__.__name__,
                "parser": "regex_structural_fallback",
                "raw_class_count": len(classes),
                "raw_function_count": sum(len(cls.methods) + len(cls.constructors) for cls in classes),
            },
        )

    def _extract_tree_sitter(self, path: Path, root: Path, source: str) -> StructuredContext:
        language = Language(tree_sitter_java.language())
        tree = Parser(language).parse(source.encode("utf-8"))
        if tree.root_node.has_error:
            raise ValueError("tree-sitter reported Java syntax errors")
        package = self._package(source)
        dependencies = self._dependencies(source)
        classes: list[ClassInfo] = []
        controls: list[ControlStructureInfo] = []
        fragments: list[CodeFragment] = []
        for node in self._walk(tree.root_node):
            if node.type not in {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration", "annotation_type_declaration"}:
                continue
            if self._has_type_parent(node):
                continue
            cls = self._ts_class_info(node, source, package, dependencies)
            classes.append(cls)
            for method in cls.methods + cls.constructors:
                controls.extend(method.branches + method.loops)
                fragments.extend(method.returns + method.raises_or_throws)
        return StructuredContext(
            language="java",
            project_root=str(root),
            source_path=str(path),
            package_or_module=package,
            classes=classes,
            functions=[],
            dependencies=dependencies,
            control_structures=controls,
            execution_fragments=fragments,
            rewrite_summary={
                "extractor": self.__class__.__name__,
                "parser": "tree_sitter_java",
                "raw_class_count": len(classes),
                "raw_function_count": sum(len(cls.methods) + len(cls.constructors) for cls in classes),
            },
        )

    def _ts_class_info(
        self,
        node: Node,
        source: str,
        package: str,
        dependencies: list[DependencyInfo],
    ) -> ClassInfo:
        name_node = node.child_by_field_name("name")
        name = self._text(source, name_node)
        body = node.child_by_field_name("body")
        methods: list[FunctionInfo] = []
        constructors: list[FunctionInfo] = []
        if body is not None:
            for child in body.named_children:
                if child.type not in {"method_declaration", "constructor_declaration", "compact_constructor_declaration"}:
                    continue
                method = self._ts_method_info(child, source, name, package, dependencies)
                (constructors if method.is_constructor else methods).append(method)
        kind = node.type.removesuffix("_declaration")
        modifiers = self._ts_modifiers(node, source)
        bases: list[str] = []
        for child in node.named_children:
            if child.type in {"superclass", "super_interfaces", "extends_interfaces", "type_list"}:
                bases.extend(self._type_names(self._text(source, child)))
        type_parameters = self._ts_type_parameters(node, source)
        return ClassInfo(
            name=name,
            qualified_name=f"{package}.{name}" if package else name,
            visibility=self._visibility(modifiers),
            methods=methods,
            constructors=constructors,
            decorators_or_annotations=[item for item in modifiers if item.startswith("@")],
            bases=list(dict.fromkeys(bases)),
            line_start=node.start_point.row + 1,
            line_end=node.end_point.row + 1,
            code=self._text(source, node),
            kind=kind,
            type_parameters=type_parameters,
        )

    def _ts_method_info(
        self,
        node: Node,
        source: str,
        class_name: str,
        package: str,
        dependencies: list[DependencyInfo],
    ) -> FunctionInfo:
        name = self._text(source, node.child_by_field_name("name")) or class_name
        is_constructor = node.type in {"constructor_declaration", "compact_constructor_declaration"}
        modifiers = self._ts_modifiers(node, source)
        parameters = self._ts_parameters(node.child_by_field_name("parameters"), source)
        return_type = None if is_constructor else self._text(source, node.child_by_field_name("type")) or None
        body = node.child_by_field_name("body")
        owner = f"{package}.{class_name}.{name}".strip(".")
        branches: list[ControlStructureInfo] = []
        loops: list[ControlStructureInfo] = []
        returns: list[CodeFragment] = []
        throws: list[CodeFragment] = []
        calls: list[str] = []
        for child in self._walk(body) if body is not None else []:
            if child.type in {"if_statement", "switch_expression", "switch_statement", "ternary_expression"}:
                condition = child.child_by_field_name("condition") or child.child_by_field_name("value")
                expression = self._text(source, condition) or self._text(source, child).split("{", 1)[0]
                branches.append(ControlStructureInfo("branch", expression, child.start_point.row + 1, child.end_point.row + 1, owner, expression, f"not ({expression})"))
            elif child.type in {"for_statement", "enhanced_for_statement", "while_statement", "do_statement"}:
                loops.append(ControlStructureInfo("loop", self._text(source, child).split("{", 1)[0], child.start_point.row + 1, child.end_point.row + 1, owner))
            elif child.type == "return_statement":
                returns.append(CodeFragment("return", self._text(source, child), child.start_point.row + 1, child.end_point.row + 1, owner))
            elif child.type == "throw_statement":
                throws.append(CodeFragment("throw", self._text(source, child), child.start_point.row + 1, child.end_point.row + 1, owner))
            elif child.type == "method_invocation":
                call_name = self._text(source, child.child_by_field_name("name"))
                if call_name:
                    calls.append(call_name)
            elif child.type == "object_creation_expression":
                created_type = self._text(source, child.child_by_field_name("type"))
                if created_type:
                    calls.append(created_type)
        thrown_exceptions: list[str] = []
        for child in node.named_children:
            if child.type == "throws":
                thrown_exceptions.extend(self._type_names(self._text(source, child)))
        annotations = [item for item in modifiers if item.startswith("@")]
        code = self._text(source, node)
        return FunctionInfo(
            name=name,
            qualified_name=owner,
            visibility=self._visibility(modifiers),
            parameters=parameters,
            return_type=return_type,
            thrown_exceptions=thrown_exceptions,
            decorators_or_annotations=annotations,
            is_static="static" in modifiers,
            is_constructor=is_constructor,
            calls=list(dict.fromkeys(calls)),
            branches=branches,
            loops=loops,
            returns=returns,
            raises_or_throws=throws,
            related_dependencies=[dep.name for dep in dependencies if dep.name.split(".")[-1] in code],
            line_start=node.start_point.row + 1,
            line_end=node.end_point.row + 1,
            code=code,
            class_name=class_name,
            type_parameters=self._ts_type_parameters(node, source),
            is_override=any(item.startswith("@Override") for item in annotations),
        )

    def _ts_parameters(self, node: Node | None, source: str) -> list[ParameterInfo]:
        if node is None:
            return []
        result: list[ParameterInfo] = []
        for child in self._walk(node):
            if child.type not in {"formal_parameter", "spread_parameter", "receiver_parameter"}:
                continue
            name = self._text(source, child.child_by_field_name("name"))
            type_hint = self._text(source, child.child_by_field_name("type"))
            if child.type == "spread_parameter" and type_hint:
                type_hint += "[]"
            if name:
                result.append(ParameterInfo(name=name, type_hint=type_hint or None))
        return result

    def _ts_modifiers(self, node: Node, source: str) -> list[str]:
        for child in node.named_children:
            if child.type == "modifiers":
                text = self._text(source, child)
                annotations = re.findall(r"@\w+(?:\([^)]*\))?", text)
                words = re.findall(r"\b(?:public|protected|private|static|final|abstract|synchronized|native|strictfp|default|sealed|non-sealed)\b", text)
                return [*annotations, *words]
        return []

    def _ts_type_parameters(self, node: Node, source: str) -> list[str]:
        target = node.child_by_field_name("type_parameters")
        if target is None:
            target = next((child for child in node.named_children if child.type == "type_parameters"), None)
        if target is None:
            return []
        return [self._text(source, child) for child in target.named_children if child.type == "type_parameter"]

    def _visibility(self, modifiers: list[str]) -> str:
        return next((item for item in ("public", "protected", "private") if item in modifiers), "package")

    def _type_names(self, text: str) -> list[str]:
        cleaned = re.sub(r"\b(?:extends|implements|throws)\b", "", text)
        return [item.strip() for item in cleaned.split(",") if item.strip()]

    def _text(self, source: str, node: Node | None) -> str:
        if node is None:
            return ""
        data = source.encode("utf-8")
        return data[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")

    def _walk(self, node: Node | None):
        if node is None:
            return
        stack = [node]
        while stack:
            current = stack.pop()
            yield current
            stack.extend(reversed(current.named_children))

    def _has_type_parent(self, node: Node) -> bool:
        parent = node.parent
        while parent is not None:
            if parent.type in {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration", "annotation_type_declaration"}:
                return True
            parent = parent.parent
        return False

    def _class_info(
        self,
        source: str,
        body: str,
        match: re.Match[str],
        class_name: str,
        package: str,
        dependencies: list[DependencyInfo],
    ) -> ClassInfo:
        methods: list[FunctionInfo] = []
        constructors: list[FunctionInfo] = []
        offset = match.end()
        for method_match in self.METHOD_RE.finditer(body):
            method = self._method_info(source, body, method_match, offset, class_name, package, dependencies)
            if method.name in self.NON_METHOD_NAMES:
                continue
            if method.is_constructor:
                constructors.append(method)
            else:
                methods.append(method)
        return ClassInfo(
            name=class_name,
            qualified_name=f"{package}.{class_name}" if package else class_name,
            visibility=match.group("visibility") or "package",
            methods=methods,
            constructors=constructors,
            decorators_or_annotations=self._annotations(match.group("annotations") or ""),
            bases=self._bases(match.group("tail") or ""),
            line_start=self._line(source, match.start()),
            line_end=self._line(source, self._matching_brace(source, match.end() - 1)),
            code=self._slice_lines(source, match.start(), self._matching_brace(source, match.end() - 1)),
        )

    def _method_info(
        self,
        source: str,
        class_body: str,
        match: re.Match[str],
        offset: int,
        class_name: str,
        package: str,
        dependencies: list[DependencyInfo],
    ) -> FunctionInfo:
        absolute_start = offset + match.start()
        absolute_body_start = offset + match.end()
        absolute_end = self._matching_brace(source, absolute_body_start - 1)
        body = source[absolute_body_start:absolute_end]
        name = match.group("name")
        signature = self._signature_parts(match.group("prefix") or "", name, class_name)
        calls = sorted(set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", body)))
        branches = [
            ControlStructureInfo("branch", expr.strip(), self._line(source, absolute_body_start + m.start()), self._line(source, absolute_body_start + m.end()), f"{package}.{class_name}.{name}".strip("."))
            for m in re.finditer(r"\bif\s*\(([^)]*)\)|\bswitch\s*\(([^)]*)\)", body)
            for expr in [m.group(1) or m.group(2) or ""]
        ]
        loops = [
            ControlStructureInfo("loop", m.group(0).strip(), self._line(source, absolute_body_start + m.start()), self._line(source, absolute_body_start + m.end()), f"{package}.{class_name}.{name}".strip("."))
            for m in re.finditer(r"\b(?:for|while)\s*\([^)]*\)", body)
        ]
        returns = [
            CodeFragment("return", m.group(0).strip(), self._line(source, absolute_body_start + m.start()), self._line(source, absolute_body_start + m.end()), f"{package}.{class_name}.{name}".strip("."))
            for m in re.finditer(r"\breturn\b[^;]*;", body)
        ]
        throws = [
            CodeFragment("throw", m.group(0).strip(), self._line(source, absolute_body_start + m.start()), self._line(source, absolute_body_start + m.end()), f"{package}.{class_name}.{name}".strip("."))
            for m in re.finditer(r"\bthrow\b[^;]*;", body)
        ]
        thrown_exceptions = [item.strip() for item in (match.group("throws") or "").split(",") if item.strip()]
        return FunctionInfo(
            name=name,
            qualified_name=f"{package}.{class_name}.{name}".strip("."),
            visibility=signature["visibility"],
            parameters=self._parameters(match.group("params") or ""),
            return_type=None if name == class_name else signature["return_type"],
            thrown_exceptions=thrown_exceptions,
            decorators_or_annotations=self._annotations(match.group("annotations") or ""),
            is_static=signature["is_static"],
            is_constructor=name == class_name,
            calls=[call for call in calls if call not in {"if", "for", "while", "switch", "catch", "return", "throw", "new"}],
            branches=branches,
            loops=loops,
            returns=returns,
            raises_or_throws=throws,
            related_dependencies=[dep.name for dep in dependencies if dep.name.split(".")[-1] in body],
            line_start=self._line(source, absolute_start),
            line_end=self._line(source, absolute_end),
            code=self._slice_lines(source, absolute_start, absolute_end),
            class_name=class_name,
        )

    def _signature_parts(self, prefix: str, name: str, class_name: str) -> dict[str, object]:
        tokens = [token for token in prefix.strip().split() if token]
        visibility = "package"
        modifiers: set[str] = set()
        type_tokens: list[str] = []
        for token in tokens:
            clean = token.strip()
            if clean in {"public", "protected", "private"}:
                visibility = clean
            elif clean in self.MODIFIERS:
                modifiers.add(clean)
            else:
                type_tokens.append(clean)
        if name == class_name:
            return {"visibility": visibility, "is_static": False, "return_type": None}
        return_type = " ".join(type_tokens).strip() or None
        return {"visibility": visibility, "is_static": "static" in modifiers, "return_type": return_type}

    def _package(self, source: str) -> str:
        match = re.search(r"\bpackage\s+([\w.]+)\s*;", source)
        return match.group(1) if match else ""

    def _dependencies(self, source: str) -> list[DependencyInfo]:
        return [DependencyInfo(name=m.group(1), kind="import") for m in re.finditer(r"\bimport\s+(?:static\s+)?([\w.*]+)\s*;", source)]

    def _annotations(self, text: str) -> list[str]:
        return [item.strip() for item in re.findall(r"@\w+(?:\([^)]*\))?", text)]

    def _bases(self, text: str) -> list[str]:
        bases: list[str] = []
        for keyword in ("extends", "implements"):
            match = re.search(rf"\b{keyword}\s+([^{keyword}{{]+)", text)
            if match:
                bases.extend(item.strip() for item in match.group(1).split(",") if item.strip())
        return bases

    def _parameters(self, text: str) -> list[ParameterInfo]:
        params: list[ParameterInfo] = []
        for raw in [item.strip() for item in text.split(",") if item.strip()]:
            raw = re.sub(r"@\w+(?:\([^)]*\))?\s*", "", raw).strip()
            parts = [part for part in raw.split() if part not in self.MODIFIERS]
            if len(parts) >= 2:
                name = parts[-1].replace("...", "").strip()
                type_hint = " ".join(parts[:-1]).replace("...", "[]").strip()
                params.append(ParameterInfo(name=name, type_hint=type_hint))
        return params

    def _matching_brace(self, source: str, open_index: int) -> int:
        depth = 0
        for index in range(open_index, len(source)):
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
                if depth == 0:
                    return index
        return len(source)

    def _line(self, source: str, index: int) -> int:
        return source.count("\n", 0, max(0, index)) + 1

    def _slice_lines(self, source: str, start: int, end: int) -> str:
        lines = source.splitlines()
        start_line = self._line(source, start)
        end_line = self._line(source, end)
        return "\n".join(lines[start_line - 1:end_line])
