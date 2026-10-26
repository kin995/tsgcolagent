from __future__ import annotations

from pathlib import Path

from analysis.context_extractor import extractor_for
from analysis.context_rewriter import ContextRewriter
from core.structured_context import (
    ClassInfo,
    CodeFragment,
    ControlStructureInfo,
    DependencyInfo,
    FunctionInfo,
    StructuredContext,
)
from core.state_flow_event import StateFlowEvent, append_state_flow_event
from graph.edge import StateEdge
from graph.node import StateNode
from graph.state_graph import StateFlowGraph


class StructuredGraphBuilder:
    """Build the initial state graph from normalized Java/Python StructuredContext."""

    def __init__(self, context: StructuredContext):
        self.context = context
        self.graph = StateFlowGraph(source_path=context.source_path)

    @classmethod
    def from_source(cls, source_path: str | Path, project_root: str | Path | None = None) -> "StructuredGraphBuilder":
        extractor = extractor_for(source_path)
        context = ContextRewriter().rewrite(extractor.extract(source_path, project_root=project_root))
        return cls(context)

    def build(self) -> StateFlowGraph:
        self.graph.metadata["graph_kind"] = "Test-State-Model"
        self.graph.metadata["language"] = self.context.language
        self.graph.metadata["structured_context"] = self.context.to_dict()
        self.graph.metadata["analysis_strategy"] = "structured_context"
        self.graph.metadata["method_core"] = {
            "structured_code_analysis_and_rewrite": True,
            "test_state_guidance": True,
            "test_state_model": True,
            "test_state_event_exchange": True,
            "dynamic_validation_in_test_state_agent": True,
            "state_positioning_repair_in_test_generation_agent": True,
            "state_target_optimization_in_test_generation_agent": True,
            "three_primary_agents": ["TestStateAgent", "TestGenerationAgent", "TestKnowledgeAgent"],
            "supported_languages": ["python", "java"],
        }
        append_state_flow_event(
            self.graph,
            StateFlowEvent(
                event_type="ContextExtracted",
                source_agent="测试状态智能体",
                target_agent="测试状态智能体",
                payload={
                    "language": self.context.language,
                    "source_path": self.context.source_path,
                    "class_count": len(self.context.classes),
                    "function_count": len(self.context.functions) + sum(len(cls.methods) for cls in self.context.classes),
                    "rewrite_summary": self.context.rewrite_summary,
                },
                iteration=self.graph.iteration,
            ),
        )
        module_id = self._module_node()
        for dependency in self.context.dependencies:
            dep_id = self._dependency_node(dependency)
            self._edge(module_id, dep_id, "depend")
        for cls in self.context.classes:
            self._class_nodes(module_id, cls)
        for fn in self.context.functions:
            self._function_nodes(module_id, fn)
        end_id = self._end_node()
        for node_id, node in list(self.graph.nodes.items()):
            if node.node_type in {"Function", "Method", "Constructor", "Class", "Interface", "Enum", "Record", "AnnotationType"}:
                self._edge(node_id, end_id, "method_exit" if node.node_type not in {"Class", "Interface", "Enum", "Record", "AnnotationType"} else "class_exit")
        append_state_flow_event(
            self.graph,
            StateFlowEvent(
                event_type="InitialGraphBuilt",
                source_agent="测试状态智能体",
                target_agent="测试状态智能体",
                related_nodes=list(self.graph.nodes.keys())[:80],
                payload={"node_count": len(self.graph.nodes), "edge_count": len(self.graph.edges)},
                iteration=self.graph.iteration,
            ),
        )
        self.graph.bump_version("structured context analysis completed")
        return self.graph

    def _module_node(self) -> str:
        node_type = "Package" if self.context.language == "java" else "Module"
        name = self.context.package_or_module or Path(self.context.source_path).stem
        node = StateNode(
            node_id=self._safe_id(f"program:{node_type.lower()}:{name}"),
            node_type=node_type,
            name=name,
            file_path=self.context.source_path,
            line_start=1,
            line_end=1,
            priority=0.7,
            metadata={"language": self.context.language, "structured": True},
        )
        self.graph.add_node(node)
        return node.node_id

    def _dependency_node(self, dependency: DependencyInfo) -> str:
        node = StateNode(
            node_id=self._safe_id(f"context:dependency:{dependency.kind}:{dependency.source or ''}:{dependency.name}"),
            node_type="Dependency",
            name=dependency.name,
            file_path=self.context.source_path,
            line_start=1,
            line_end=1,
            priority=0.35,
            metadata={"kind": dependency.kind, "source": dependency.source, "alias": dependency.alias},
        )
        self.graph.add_node(node)
        return node.node_id

    def _class_nodes(self, parent_id: str, cls: ClassInfo) -> None:
        class_node_type = {"interface": "Interface", "enum": "Enum", "record": "Record", "annotation_type": "AnnotationType"}.get(cls.kind, "Class")
        class_id = self._state_node(class_node_type, cls.name, cls.line_start, cls.line_end, cls.code, 0.75, {"qualified_name": cls.qualified_name, "visibility": cls.visibility, "bases": cls.bases, "kind": cls.kind, "type_parameters": cls.type_parameters})
        self._edge(parent_id, class_id, "defines")
        for annotation in cls.decorators_or_annotations:
            ann_id = self._context_node("Annotation", annotation, cls.line_start, {"owner": cls.qualified_name})
            self._edge(class_id, ann_id, "depend")
        for constructor in cls.constructors:
            self._function_nodes(class_id, constructor)
        for method in cls.methods:
            self._function_nodes(class_id, method)

    def _function_nodes(self, parent_id: str, fn: FunctionInfo) -> None:
        node_type = "Constructor" if fn.is_constructor else "Async" if fn.is_async else "Method" if fn.class_name else "Function"
        metadata = {
            "qualified_name": fn.qualified_name,
            "visibility": fn.visibility,
            "scope": fn.class_name or "",
            "args": [param.name for param in fn.parameters],
            "arg_annotations": {param.name: param.type_hint for param in fn.parameters if param.type_hint},
            "returns": fn.return_type,
            "is_static": fn.is_static,
            "is_async": fn.is_async,
            "is_generator": "yield" in fn.code,
            "is_constructor": fn.is_constructor,
            "language": self.context.language,
            "type_parameters": fn.type_parameters,
            "is_override": fn.is_override,
        }
        fn_id = self._state_node(node_type, fn.name, fn.line_start, fn.line_end, fn.code, 0.85, metadata)
        self._edge(parent_id, fn_id, "defines")
        if fn.is_override:
            override_id = self._context_node("Override", fn.qualified_name, fn.line_start, {"owner": fn.qualified_name})
            self._edge(override_id, fn_id, "depend")
        for type_parameter in fn.type_parameters:
            generic_id = self._context_node("GenericParameter", type_parameter, fn.line_start, {"owner": fn.qualified_name})
            self._edge(generic_id, fn_id, "type_flow")
        for param in fn.parameters:
            param_id = self._context_node("Parameter", param.name, fn.line_start, {"type": param.type_hint, "owner": fn.qualified_name, "kind": param.kind})
            self._edge(param_id, fn_id, "type_flow")
        if fn.return_type:
            return_type_id = self._context_node("ReturnType", fn.return_type, fn.line_start, {"owner": fn.qualified_name})
            self._edge(fn_id, return_type_id, "type_flow")
        previous = fn_id
        for item in [*fn.branches, *fn.loops]:
            current = self._control_node(item)
            self._edge(previous, current, "flow")
            if item.kind == "branch":
                self._edge(current, current, "true", condition=item.true_condition or item.expression)
                self._edge(current, current, "false", condition=item.false_condition or f"not ({item.expression})")
            elif item.kind == "loop":
                self._edge(current, current, "loop_body")
                self._edge(current, current, "loop_back")
            previous = current
        for call in fn.calls:
            call_id = self._state_node("Call", call, fn.line_start, fn.line_start, "", 0.55, {"owner": fn.qualified_name})
            self._edge(previous, call_id, "call")
            previous = call_id
        for fragment in fn.returns:
            return_id = self._fragment_node(fragment)
            self._edge(previous, return_id, "return")
        for fragment in fn.raises_or_throws:
            raise_id = self._fragment_node(fragment)
            self._edge(previous, raise_id, "throw" if self.context.language == "java" else "exception_exit")
        for exception in fn.thrown_exceptions:
            exc_id = self._context_node("Exception", exception, fn.line_start, {"owner": fn.qualified_name})
            self._edge(fn_id, exc_id, "throw")

    def _control_node(self, item: ControlStructureInfo) -> str:
        node_type = "Loop" if item.kind == "loop" else "Branch" if item.kind == "branch" else "Match"
        return self._state_node(node_type, item.expression, item.line_start, item.line_end, "", 0.9, {"owner": item.qualified_owner, "kind": item.kind})

    def _fragment_node(self, item: CodeFragment) -> str:
        node_type = "Return" if item.kind == "return" else "Raise" if item.kind == "raise" else "Throw"
        return self._state_node(node_type, item.text, item.line_start, item.line_end, item.text, 0.75, {"owner": item.qualified_owner, "kind": item.kind})

    def _context_node(self, node_type: str, name: str, line: int, metadata: dict[str, object]) -> str:
        node = StateNode(
            node_id=self._safe_id(f"context:{node_type.lower()}:{metadata.get('owner', '')}:{name}:{line}"),
            node_type=node_type,
            name=name,
            file_path=self.context.source_path,
            line_start=line,
            line_end=line,
            priority=0.4,
            metadata=metadata,
        )
        self.graph.add_node(node)
        return node.node_id

    def _state_node(self, node_type: str, name: str, line_start: int, line_end: int, code: str, priority: float, metadata: dict[str, object]) -> str:
        qualified = metadata.get("qualified_name") or metadata.get("owner") or name
        node = StateNode(
            node_id=self._safe_id(f"program:{node_type.lower()}:{qualified}:{line_start}"),
            node_type="Function" if node_type == "Method" else node_type,
            name=name,
            file_path=self.context.source_path,
            line_start=max(1, line_start),
            line_end=max(line_start, line_end),
            code=code,
            priority=priority,
            metadata=metadata,
        )
        self.graph.add_node(node)
        return node.node_id

    def _end_node(self) -> str:
        node = StateNode(
            node_id="program:end:module",
            node_type="End",
            name="ModuleEnd" if self.context.language == "python" else "PackageEnd",
            file_path=self.context.source_path,
            line_start=1,
            line_end=1,
            priority=0.2,
            metadata={"language": self.context.language},
        )
        self.graph.add_node(node)
        return node.node_id

    def _edge(self, source: str | None, target: str | None, edge_type: str, **metadata: object) -> None:
        if source and target and source != target:
            self.graph.add_edge(StateEdge(source=source, target=target, edge_type=edge_type, metadata=metadata))

    def _safe_id(self, raw: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in {":", "_", "-", "."} else "_" for ch in raw)
        if safe not in self.graph.nodes:
            return safe
        index = 2
        while f"{safe}:{index}" in self.graph.nodes:
            index += 1
        return f"{safe}:{index}"
