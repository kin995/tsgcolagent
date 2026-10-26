from __future__ import annotations

from graph.state_graph import StateFlowGraph


PROGRAM_NODE_TYPES = {"Function", "Class", "Interface", "Enum", "Record", "AnnotationType", "Branch", "Loop", "Return", "Call", "Exception", "Raise", "Throw", "End", "Async", "With", "Match", "Lambda", "Generator"}


class StateFlowCoverageMetric:
    def calculate(self, graph: StateFlowGraph) -> float:
        components = self.components(graph)
        node_score = components["node_coverage"]
        edge_score = components["edge_coverage"]
        path_score = components["path_coverage"]
        return round((node_score * 0.45) + (edge_score * 0.35) + (path_score * 0.20), 4)

    def components(self, graph: StateFlowGraph) -> dict[str, float | int]:
        nodes = self._program_nodes(graph)
        program_node_ids = {node.node_id for node in nodes}
        edges = self._traceable_program_edges(graph, program_node_ids)
        branch_like = [node for node in nodes if node.node_type in {"Branch", "Loop", "Exception", "Match"}]
        return {
            "node_coverage": self.node_coverage(graph),
            "edge_coverage": self.edge_coverage(graph),
            "path_coverage": self.path_coverage(graph),
            "program_nodes": len(nodes),
            "visited_program_nodes": sum(1 for node in nodes if node.visited),
            "program_edges": len(edges),
            "visited_program_edges": sum(1 for edge in edges if edge.visited),
            "branch_like_nodes": len(branch_like),
            "visited_branch_like_nodes": sum(1 for node in branch_like if node.visited),
        }

    def node_coverage(self, graph: StateFlowGraph) -> float:
        nodes = self._program_nodes(graph)
        if not nodes:
            return 1.0
        return sum(1 for node in nodes if node.visited) / len(nodes)

    def edge_coverage(self, graph: StateFlowGraph) -> float:
        program_node_ids = {node.node_id for node in self._program_nodes(graph)}
        edges = self._traceable_program_edges(graph, program_node_ids)
        if not edges:
            return 1.0
        return sum(1 for edge in edges if edge.visited) / len(edges)

    def path_coverage(self, graph: StateFlowGraph) -> float:
        branch_like = [node for node in self._program_nodes(graph) if node.node_type in {"Branch", "Loop", "Exception", "Match"}]
        if not branch_like:
            return self.node_coverage(graph)
        covered = sum(1 for node in branch_like if node.visited)
        return covered / len(branch_like)

    def _program_nodes(self, graph: StateFlowGraph):
        return [node for node in graph.nodes.values() if node.node_type in PROGRAM_NODE_TYPES]

    def _traceable_program_edges(self, graph: StateFlowGraph, program_node_ids: set[str]):
        edges = []
        for edge in graph.edges:
            if edge.source not in program_node_ids or edge.target not in program_node_ids:
                continue
            source = graph.get_node(edge.source)
            target = graph.get_node(edge.target)
            inferred_traceable = bool(source and target and source.line_start != target.line_start)
            traceable = bool(edge.metadata.get("traceable", inferred_traceable))
            edge.metadata["traceable"] = traceable
            if traceable:
                edges.append(edge)
        return edges
