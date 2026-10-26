from __future__ import annotations

from .graph_query import EVIDENCE_NODE_TYPES, PROGRAM_NODE_TYPES
from .state_graph import StateFlowGraph


EDGE_DEFECT_WEIGHTS = {
    "flow": 0.8,
    "true": 0.8,
    "false": 0.8,
    "loop_body": 0.8,
    "loop_back": 0.8,
    "return": 0.8,
    "raise": 0.85,
    "except": 0.85,
    "call": 0.7,
    "defines": 0.6,
    "Depend": 0.9,
    "Observe": 0.95,
    "Trigger": 0.9,
    "RepairTarget": 1.0,
    "Execute": 0.5,
    "Produce": 0.6,
    "TypeFlow": 0.75,
    "Kill": 0.2,
    "Survive": 1.0,
}


class GraphDefectPropagation:
    def __init__(self, graph: StateFlowGraph):
        self.graph = graph

    def run(self, steps: int = 8, decay: float = 0.85) -> dict[str, object]:
        self._initialize_defects()
        for _ in range(steps):
            additions: dict[str, float] = {node_id: 0.0 for node_id in self.graph.nodes}
            propagated_reasons: dict[str, list[dict[str, object]]] = {node_id: [] for node_id in self.graph.nodes}
            for edge in self.graph.edges:
                child = self.graph.get_node(edge.target)
                parent = self.graph.get_node(edge.source)
                if not child or not parent:
                    continue
                weight = EDGE_DEFECT_WEIGHTS.get(edge.edge_type, min(1.0, edge.weight))
                if edge.edge_type in {"Trigger", "RepairTarget"}:
                    value = parent.defect_score * weight * decay
                    additions[child.node_id] += value
                    propagated_reasons[child.node_id].extend(self._propagate_reasons(parent, child, edge.edge_type, value))
                else:
                    value = child.defect_score * weight * decay
                    additions[parent.node_id] += value
                    propagated_reasons[parent.node_id].extend(self._propagate_reasons(child, parent, edge.edge_type, value))
            for node_id, value in additions.items():
                node = self.graph.get_node(node_id)
                if node:
                    node.defect_score = round(min(1.0, max(node.defect_score, value)), 4)
                    if propagated_reasons[node_id]:
                        self._merge_reasons(node, propagated_reasons[node_id])
        root = self._highest_program_defect()
        result = {
            "root_cause_node": root.to_dict() if root else None,
            "max_defect_score": root.defect_score if root else 0.0,
            "root_cause_reasons": self._root_reasons(root) if root else [],
            "root_cause_explanation": self._explain_root(root) if root else "",
        }
        self.graph.metadata["last_defect_propagation"] = result
        self.graph.history.append(
            {
                "version": self.graph.version,
                "iteration": self.graph.iteration,
                "reason": "graph defect propagation",
                "result": result,
            }
        )
        return result

    def _initialize_defects(self) -> None:
        for node in self.graph.nodes.values():
            node.coverage_defect = 1.0 if node.node_type in PROGRAM_NODE_TYPES and not node.visited else 0.0
            # Assertions are currently mapped to callable-level runtime states.
            # Marking every visited branch/return/end node as assertion-missing
            # makes fully tested functions look like universal 100% risk.
            node.assertion_defect = 1.0 if node.node_type in {"Function", "Async", "Method", "Constructor"} and node.visited and node.assert_count <= 0 else 0.0
            node.mutation_defect = 1.0 if node.mutation_status == "survived" else 0.0
            node.exception_defect = 1.0 if node.node_type in {"Exception", "Raise"} and not node.visited else 0.0
            node.type_defect = 0.0
            if node.node_type == "Mutation" and node.metadata.get("status") == "survived":
                node.mutation_defect = 1.0
            if node.node_type == "RuntimeType" and node.metadata.get("type") in {None, "unknown"}:
                node.type_defect = 1.0
            node.defect_score = round(
                min(
                    1.0,
                    max(
                        node.coverage_defect,
                        node.assertion_defect,
                        node.mutation_defect,
                        node.exception_defect,
                        node.type_defect,
                    ),
                ),
                4,
            )
            node.repair_flag = node.defect_score > 0
            node.metadata["defect_reasons"] = self._initial_reasons(node)

    def _highest_program_defect(self):
        program = [node for node in self.graph.nodes.values() if node.node_type in PROGRAM_NODE_TYPES]
        if not program:
            return None
        return max(program, key=lambda node: node.defect_score)

    def _initial_reasons(self, node) -> list[dict[str, object]]:
        reasons: list[dict[str, object]] = []
        if node.coverage_defect > 0:
            reasons.append({"kind": "coverage_missing", "node": node.node_id, "message": f"{node.node_type} {node.name} was not covered"})
        if node.assertion_defect > 0:
            reasons.append({"kind": "assertion_missing", "node": node.node_id, "message": f"{node.node_type} {node.name} was covered without mapped assertions"})
        if node.mutation_defect > 0:
            reasons.append({"kind": "mutation_survived", "node": node.node_id, "message": f"Mutation evidence survived near {node.name}"})
        if node.exception_defect > 0:
            reasons.append({"kind": "exception_uncovered", "node": node.node_id, "message": f"Exception path {node.name} was not exercised"})
        if node.type_defect > 0:
            reasons.append({"kind": "runtime_type_unknown", "node": node.node_id, "message": f"Runtime type evidence is unknown for {node.name}"})
        return reasons

    def _propagate_reasons(self, source, target, edge_type: str, propagated_score: float) -> list[dict[str, object]]:
        result = []
        for reason in source.metadata.get("defect_reasons", [])[:5]:
            chain = list(reason.get("chain", []))
            chain.append({"from": source.node_id, "to": target.node_id, "edge_type": edge_type})
            result.append(
                {
                    "kind": reason.get("kind", "propagated_defect"),
                    "node": target.node_id,
                    "message": reason.get("message", ""),
                    "propagated_from": source.node_id,
                    "via_edge": edge_type,
                    "propagated_score": round(propagated_score, 4),
                    "chain": chain[-6:],
                }
            )
        return result

    def _merge_reasons(self, node, reasons: list[dict[str, object]]) -> None:
        existing = node.metadata.get("defect_reasons", [])
        seen = {(item.get("kind"), item.get("propagated_from"), item.get("via_edge")) for item in existing}
        for reason in sorted(reasons, key=lambda item: float(item.get("propagated_score", 0.0)), reverse=True):
            key = (reason.get("kind"), reason.get("propagated_from"), reason.get("via_edge"))
            if key not in seen:
                existing.append(reason)
                seen.add(key)
        node.metadata["defect_reasons"] = existing[:12]

    def _explain_root(self, root) -> str:
        reasons = root.metadata.get("defect_reasons", [])
        if not reasons:
            return f"{root.node_type} {root.name} has no propagated residual defect; it is reported as the highest-ranked program node for traceability."
        kinds = sorted({str(item.get("kind")) for item in reasons})
        return f"{root.node_type} {root.name} is selected because these defect causes propagate to it: {', '.join(kinds)}."

    def _root_reasons(self, root) -> list[dict[str, object]]:
        reasons = root.metadata.get("defect_reasons", [])
        if reasons:
            return reasons[:8]
        return [
            {
                "kind": "no_residual_defect",
                "node": root.node_id,
                "message": "No coverage, assertion, mutation, exception or type defect remains on the selected root node.",
            }
        ]
