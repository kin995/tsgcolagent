from __future__ import annotations

from dataclasses import replace

from core.structured_context import ClassInfo, FunctionInfo, StructuredContext


class ContextRewriter:
    """Normalize source-derived context without modifying the real source file."""

    def rewrite(self, context: StructuredContext) -> StructuredContext:
        classes = [self._rewrite_class(item) for item in context.classes]
        functions = [self._rewrite_function(item) for item in context.functions if self._is_test_relevant(item)]
        dependencies = sorted(context.dependencies, key=lambda item: (item.kind, item.name, item.alias or ""))
        summary = dict(context.rewrite_summary)
        summary.update(
            {
                "rewriter": self.__class__.__name__,
                "comments_and_format_noise_removed": True,
                "signatures_normalized": True,
                "call_and_control_flow_normalized": True,
                "test_relevant_classes": len(classes),
                "test_relevant_functions": len(functions) + sum(len(cls.methods) for cls in classes),
            }
        )
        return replace(context, classes=classes, functions=functions, dependencies=dependencies, rewrite_summary=summary)

    def _rewrite_class(self, item: ClassInfo) -> ClassInfo:
        methods = [self._rewrite_function(method) for method in item.methods if self._is_test_relevant(method)]
        constructors = [self._rewrite_function(method) for method in item.constructors]
        return replace(item, methods=methods, constructors=constructors, code=self._compact_code(item.code))

    def _rewrite_function(self, item: FunctionInfo) -> FunctionInfo:
        visibility = item.visibility or ("private" if item.name.startswith("_") and item.name != "__init__" else "public")
        calls = sorted(set(item.calls))
        related_dependencies = sorted(set(item.related_dependencies))
        return replace(
            item,
            visibility=visibility,
            calls=calls,
            related_dependencies=related_dependencies,
            code=self._compact_code(item.code),
        )

    def _is_test_relevant(self, item: FunctionInfo) -> bool:
        if item.is_constructor:
            return True
        return item.visibility != "private" and not item.name.startswith("_")

    def _compact_code(self, code: str) -> str:
        lines = [line.rstrip() for line in code.splitlines()]
        return "\n".join(line for line in lines if line.strip())

