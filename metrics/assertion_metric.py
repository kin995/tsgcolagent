from __future__ import annotations

from graph.state_graph import StateFlowGraph


PROGRAM_NODE_TYPES = {"Function", "Class", "Interface", "Enum", "Record", "AnnotationType", "Branch", "Loop", "Return", "Call", "Exception", "Raise", "Throw", "End", "Async", "With", "Match", "Lambda", "Generator"}


class AssertionEffectivenessMetric:
    def calculate(self, graph: StateFlowGraph, total_asserts: int = 0, mutation_score: float = 0.0) -> float:
        assertions = [node for node in graph.nodes.values() if node.node_type == "Assertion"]
        effective_assertions = [node for node in assertions if node.metadata.get("effective", True)]
        if not effective_assertions:
            return 0.0

        # Exception assertions used to receive the same perfect score as exact
        # value checks.  That made exception-only suites look excellent even
        # though they did not describe the function's normal contract.
        oracle_scores = [self._oracle_score(node) for node in effective_assertions]
        oracle_quality = sum(oracle_scores) / len(oracle_scores)
        normal_assertions = [node for node in effective_assertions if "pytest.raises" not in (node.code or "")]
        normal_ratio = len(normal_assertions) / len(effective_assertions)
        mapped_assertions = {
            edge.source for edge in graph.edges
            if edge.edge_type == "Observe"
            and graph.get_node(edge.source)
            and graph.get_node(edge.source).node_type == "Assertion"
        }
        assertion_target_mapping = len(mapped_assertions) / len(effective_assertions)
        oracle_kinds = {self._oracle_kind(node) for node in effective_assertions}
        diversity = min(1.0, len(oracle_kinds) / 3.0)

        # Mutation is intentionally not included: SFQ already accounts for it.
        score = (
            (oracle_quality * 0.45)
            + (normal_ratio * 0.25)
            + (assertion_target_mapping * 0.15)
            + (diversity * 0.15)
        )
        if not normal_assertions:
            score = min(score, 0.30)
        return round(max(0.0, min(1.0, score)), 4)

    def _oracle_score(self, node) -> float:
        code = node.code or ""
        if "pytest.raises" not in code:
            return float(node.metadata.get("specificity", 0.3))
        broad_or_usage_errors = ("Exception", "BaseException", "TypeError", "AttributeError")
        if any(name in code for name in broad_or_usage_errors):
            return 0.15
        return 0.60

    def _oracle_kind(self, node) -> str:
        code = node.code or ""
        if "pytest.raises" in code:
            return "exception"
        if "pytest.approx" in code:
            return "approximate"
        if any(operator in code for operator in ("==", "!=", "<=", ">=", "<", ">")):
            return "comparison"
        return "property"
