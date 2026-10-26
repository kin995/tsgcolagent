from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .edge import StateEdge
from .node import StateNode
from .state_graph import StateFlowGraph


PROGRAM_NODE_TYPES = {
    "Function",
    "Method",
    "Constructor",
    "Class",
    "Interface",
    "Enum",
    "Record",
    "AnnotationType",
    "Branch",
    "Loop",
    "Return",
    "Call",
    "Exception",
    "Raise",
    "Throw",
    "End",
    "Async",
    "With",
    "Match",
    "Lambda",
    "Generator",
}

EVIDENCE_NODE_TYPES = {
    "TestCase",
    "Assertion",
    "Mutation",
    "RuntimeState",
    "RuntimeType",
    "Execution",
}


@dataclass
class GraphGaps:
    coverage: float
    mutation: float
    assertion: float
    exception: float
    runtime_type: float

    def as_dict(self) -> dict[str, float]:
        return {
            "coverage": self.coverage,
            "mutation": self.mutation,
            "assertion": self.assertion,
            "exception": self.exception,
            "runtime_type": self.runtime_type,
        }

    def largest(self) -> tuple[str, float]:
        items = self.as_dict()
        return max(items.items(), key=lambda item: item[1])


class GraphQuery:
    def __init__(self, graph: StateFlowGraph):
        self.graph = graph

    def program_nodes(self) -> list[StateNode]:
        return [node for node in self.graph.nodes.values() if node.node_type in PROGRAM_NODE_TYPES]

    def evidence_nodes(self) -> list[StateNode]:
        return [node for node in self.graph.nodes.values() if node.node_type in EVIDENCE_NODE_TYPES]

    def get_highest_defect(self, program_only: bool = True) -> StateNode | None:
        nodes = self.program_nodes() if program_only else list(self.graph.nodes.values())
        if not nodes:
            return None
        return max(nodes, key=lambda node: node.defect_score)

    def get_root_cause(self, target_id: str | None = None) -> dict[str, object] | None:
        target = self.graph.get_node(target_id) if target_id else self.get_highest_defect(program_only=False)
        if not target:
            return None
        path = self._reverse_program_path(target.node_id)
        root = None
        for node_id in path:
            node = self.graph.get_node(node_id)
            if node and node.node_type in {"Function", "Async", "Method", "Constructor", "Class", "Branch", "Loop"}:
                if root is None or node.defect_score > root.defect_score:
                    root = node
        if root is None:
            root = self.get_highest_defect(program_only=True)
        return {
            "target": target.to_dict(),
            "root": root.to_dict() if root else None,
            "path": path,
        }

    def get_uncovered_paths(self) -> list[dict[str, object]]:
        paths: list[dict[str, object]] = []
        for edge in self.graph.uncovered_edges():
            source = self.graph.get_node(edge.source)
            target = self.graph.get_node(edge.target)
            if source and target and self._traceable_program_edge(source, target, edge):
                paths.append({"edge": edge.to_dict(), "source": source.to_dict(), "target": target.to_dict()})
        return paths

    def _traceable_program_edge(self, source: StateNode, target: StateNode, edge: object) -> bool:
        if source.node_type not in PROGRAM_NODE_TYPES or target.node_type not in PROGRAM_NODE_TYPES:
            return False
        metadata = getattr(edge, "metadata", {})
        inferred = bool(source.line_start != target.line_start)
        return bool(metadata.get("traceable", inferred))

    def get_survived_mutations(self) -> list[StateNode]:
        execution = self._latest_execution()
        execution_id = execution.node_id if execution else None
        mutation_nodes = [
            node for node in self.graph.nodes.values()
            if node.node_type == "Mutation" and node.metadata.get("status") == "survived"
            and (execution_id is None or node.metadata.get("execution_id") == execution_id)
        ]
        return mutation_nodes

    def get_runtime_types(self) -> list[StateNode]:
        return [node for node in self.graph.nodes.values() if node.node_type == "RuntimeType"]

    def get_related_assertions(self, node_id: str) -> list[StateNode]:
        related: list[StateNode] = []
        runtime_states = {edge.source for edge in self.graph.edges if edge.target == node_id and edge.edge_type in {"Depend", "Trigger"}}
        runtime_states.update(edge.target for edge in self.graph.edges if edge.source == node_id and edge.edge_type == "Produce")
        for edge in self.graph.edges:
            if edge.edge_type == "Observe" and edge.target in runtime_states:
                node = self.graph.get_node(edge.source)
                if node and node.node_type == "Assertion":
                    related.append(node)
        return related

    def get_execution_trace(self, execution_id: str | None = None) -> dict[str, object]:
        executions = [node for node in self.graph.nodes.values() if node.node_type == "Execution"]
        if not executions:
            return {"execution": None, "nodes": [], "edges": []}
        execution = self.graph.get_node(execution_id) if execution_id else executions[-1]
        if not execution:
            return {"execution": None, "nodes": [], "edges": []}
        trace_edges = [edge for edge in self.graph.edges if edge.source == execution.node_id or edge.target == execution.node_id]
        trace_nodes = []
        for edge in trace_edges:
            for node_id in [edge.source, edge.target]:
                node = self.graph.get_node(node_id)
                if node and node.node_id not in {item.node_id for item in trace_nodes}:
                    trace_nodes.append(node)
        return {
            "execution": execution.to_dict(),
            "nodes": [node.to_dict() for node in trace_nodes],
            "edges": [edge.to_dict() for edge in trace_edges],
        }

    def get_candidate_repairs(self) -> list[dict[str, object]]:
        candidates = sorted(
            [node for node in self.program_nodes() if node.defect_score > 0],
            key=lambda node: node.defect_score,
            reverse=True,
        )
        result: list[dict[str, object]] = []
        for node in candidates[:20]:
            related_tests = self._related_tests(node.node_id)
            result.append(
                {
                    "node": node.to_dict(),
                    "root_cause": self.get_root_cause(node.node_id),
                    "related_tests": [test.to_dict() for test in related_tests],
                    "actions": self._actions_for_node(node),
                }
            )
        return result

    def get_gaps(self) -> GraphGaps:
        program = self.program_nodes()
        if not program:
            return GraphGaps(0.0, 0.0, 0.0, 0.0, 0.0)
        uncovered = sum(1 for node in program if not node.visited)
        coverage_gap = self._clamp(uncovered / len(program))
        survived = len(self.get_survived_mutations())
        current_mutations = self._current_mutations()
        mutation_denominator = len([node for node in current_mutations if node.metadata.get("status") in {"killed", "survived"}])
        mutation_gap = self._clamp(survived / mutation_denominator) if mutation_denominator else 0.0
        observable = [node for node in program if node.visited and node.node_type in {"Function", "Async", "Method", "Constructor"}]
        assertion_gap = self._clamp(sum(1 for node in observable if node.assert_count <= 0) / len(observable)) if observable else 1.0
        exception_nodes = [node for node in program if node.node_type in {"Exception", "Raise"}]
        exception_gap = self._clamp(sum(1 for node in exception_nodes if not node.visited) / len(exception_nodes)) if exception_nodes else 0.0
        functions = [node for node in program if node.node_type in {"Function", "Async", "Method", "Constructor"}]
        typed_functions = {edge.source for edge in self.graph.edges if edge.edge_type == "TypeFlow"}
        type_gap = self._clamp(1.0 - (len(typed_functions) / len(functions))) if functions else 0.0
        return GraphGaps(
            coverage=round(coverage_gap, 4),
            mutation=round(mutation_gap, 4),
            assertion=round(assertion_gap, 4),
            exception=round(exception_gap, 4),
            runtime_type=round(max(0.0, type_gap), 4),
        )

    def _latest_execution(self) -> StateNode | None:
        executions = [node for node in self.graph.nodes.values() if node.node_type == "Execution"]
        return executions[-1] if executions else None

    def _current_mutations(self) -> list[StateNode]:
        execution = self._latest_execution()
        if not execution:
            return []
        return [
            node for node in self.graph.nodes.values()
            if node.node_type == "Mutation" and node.metadata.get("execution_id") == execution.node_id
        ]

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    def _related_tests(self, node_id: str) -> list[StateNode]:
        tests: list[StateNode] = []
        for edge in self.graph.edges:
            if edge.target == node_id and edge.edge_type == "Execute":
                node = self.graph.get_node(edge.source)
                if node and node.node_type == "TestCase":
                    tests.append(node)
        return tests

    def _actions_for_node(self, node: StateNode) -> list[str]:
        actions: list[str] = []
        if node.coverage_defect > 0:
            actions.append("add_boundary_or_path_case")
        if node.assertion_defect > 0:
            actions.append("strengthen_assertion")
        if node.mutation_defect > 0:
            actions.append("add_mutation_killing_assertion")
        if node.exception_defect > 0:
            actions.append("add_exception_case")
        if node.type_defect > 0:
            actions.append("add_runtime_type_case")
        return actions or ["inspect"]

    def _reverse_program_path(self, node_id: str, limit: int = 20) -> list[str]:
        seen = {node_id}
        queue: deque[tuple[str, list[str]]] = deque([(node_id, [node_id])])
        best = [node_id]
        while queue:
            current, path = queue.popleft()
            if len(path) > len(best):
                best = path
            if len(path) >= limit:
                continue
            for edge in self._incoming_edges(current):
                source = self.graph.get_node(edge.source)
                if not source or edge.source in seen:
                    continue
                if source.node_type in PROGRAM_NODE_TYPES:
                    seen.add(edge.source)
                    queue.append((edge.source, path + [edge.source]))
        return best

    def _incoming_edges(self, node_id: str) -> list[StateEdge]:
        return [edge for edge in self.graph.edges if edge.target == node_id]
