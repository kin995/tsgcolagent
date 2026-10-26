from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .edge import StateEdge
from .node import StateNode


MAX_ROLLBACK_SNAPSHOTS = 3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StateFlowGraph:
    source_path: str
    version: int = 1
    iteration: int = 0
    timestamp: str = field(default_factory=utc_now)
    nodes: dict[str, StateNode] = field(default_factory=dict)
    edges: list[StateEdge] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    flow_history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    _snapshots: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def add_node(self, node: StateNode) -> None:
        self.nodes[node.node_id] = node

    def upsert_node(self, node: StateNode) -> StateNode:
        existing = self.nodes.get(node.node_id)
        if existing:
            existing.name = node.name
            existing.line_start = node.line_start
            existing.line_end = node.line_end
            existing.code = node.code
            existing.priority = max(existing.priority, node.priority)
            existing.metadata.update(node.metadata)
            return existing
        self.add_node(node)
        return node

    def add_edge(self, edge: StateEdge) -> None:
        if not any(existing.edge_id == edge.edge_id for existing in self.edges):
            self.edges.append(edge)

    def next_node_id(self, prefix: str) -> str:
        existing = [node_id for node_id in self.nodes if node_id.startswith(f"{prefix}_")]
        return f"{prefix}_{len(existing) + 1:03d}"

    def add_entity(
        self,
        node_type: str,
        name: str,
        line_start: int = 1,
        line_end: int | None = None,
        code: str = "",
        priority: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> StateNode:
        metadata = metadata or {}
        node_id = str(metadata.get("stable_id") or self.next_node_id(node_type.lower()))
        node = StateNode(
            node_id=node_id,
            node_type=node_type,
            name=name,
            file_path=self.source_path,
            line_start=line_start,
            line_end=line_end or line_start,
            code=code,
            priority=priority,
            metadata=metadata,
        )
        return self.upsert_node(node) if metadata.get("stable_id") else self._add_unique_entity(node)

    def _add_unique_entity(self, node: StateNode) -> StateNode:
        if node.node_id not in self.nodes:
            self.add_node(node)
            return node
        prefix = node.node_id
        index = 2
        while f"{prefix}_{index}" in self.nodes:
            index += 1
        node.node_id = f"{prefix}_{index}"
        self.add_node(node)
        return node

    def add_relation(
        self,
        source: str | None,
        target: str | None,
        edge_type: str,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if source and target and source != target:
            self.add_edge(StateEdge(source=source, target=target, edge_type=edge_type, weight=weight, metadata=metadata or {}))

    def get_node(self, node_id: str) -> StateNode | None:
        return self.nodes.get(node_id)

    def mark_node_visited(self, node_id: str, count: int = 1) -> None:
        node = self.get_node(node_id)
        if node:
            node.mark_visited(count)

    def mark_edge_visited(self, source: str, target: str, edge_type: str | None = None) -> None:
        for edge in self.edges:
            if edge.source == source and edge.target == target and (edge_type is None or edge.edge_type == edge_type):
                edge.mark_visited()
                return

    def record_flow(self, name: str, node_ids: list[str], edge_ids: list[str] | None = None) -> None:
        self.flow_history.append(
            {
                "event_type": "ExecutionPathRecorded",
                "source_agent": "测试生成智能体",
                "target_agent": "测试状态智能体",
                "name": name,
                "timestamp": utc_now(),
                "nodes": list(node_ids),
                "edges": list(edge_ids or []),
                "iteration": self.iteration,
            }
        )

    def snapshot(self, reason: str) -> None:
        self._snapshots.append(self.to_dict(include_snapshots=False))
        if len(self._snapshots) > MAX_ROLLBACK_SNAPSHOTS:
            del self._snapshots[:-MAX_ROLLBACK_SNAPSHOTS]
        self.history.append({"version": self.version, "timestamp": utc_now(), "reason": reason})

    def rollback(self, steps: int = 1) -> bool:
        if steps <= 0 or len(self._snapshots) < steps:
            return False
        data = self._snapshots[-steps]
        restored = self.from_dict(data)
        self.version = restored.version + 1
        self.timestamp = utc_now()
        self.nodes = restored.nodes
        self.edges = restored.edges
        self.history = restored.history
        self.flow_history = restored.flow_history
        self.metadata = restored.metadata
        self.history.append({"version": self.version, "timestamp": self.timestamp, "reason": "rollback"})
        return True

    def bump_version(self, reason: str) -> None:
        self.snapshot(reason)
        self.version += 1
        self.iteration += 1
        self.timestamp = utc_now()

    def uncovered_nodes(self) -> list[StateNode]:
        return [node for node in self.nodes.values() if not node.visited]

    def uncovered_edges(self) -> list[StateEdge]:
        return [edge for edge in self.edges if not edge.visited]

    def nodes_by_type(self, node_type: str) -> list[StateNode]:
        return [node for node in self.nodes.values() if node.node_type == node_type]

    def to_dict(self, include_snapshots: bool = True) -> dict[str, Any]:
        data = {
            "source_path": self.source_path,
            "version": self.version,
            "graph_version": self.version,
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "edges": [edge.to_dict() for edge in self.edges],
            "history": self.history,
            "flow_history": self.flow_history,
            "metadata": self.metadata,
        }
        if include_snapshots:
            data["snapshots"] = self._snapshots
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateFlowGraph":
        graph = cls(
            source_path=data["source_path"],
            version=data.get("version", 1),
            iteration=data.get("iteration", 0),
            timestamp=data.get("timestamp", utc_now()),
            history=data.get("history", []),
            flow_history=data.get("flow_history", []),
            metadata=data.get("metadata", {}),
        )
        graph.nodes = {node_id: StateNode.from_dict(node_data) for node_id, node_data in data.get("nodes", {}).items()}
        graph.edges = [StateEdge.from_dict(edge_data) for edge_data in data.get("edges", [])]
        graph._snapshots = data.get("snapshots", [])
        return graph

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(self.to_dict(include_snapshots=False), indent=2, ensure_ascii=False)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        try:
            temporary.replace(target)
        except OSError:
            target.write_text(content, encoding="utf-8")
            try:
                temporary.unlink()
            except OSError:
                pass

    @classmethod
    def load(cls, path: str | Path) -> "StateFlowGraph":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def clone(self) -> "StateFlowGraph":
        return self.from_dict(copy.deepcopy(self.to_dict(include_snapshots=False)))
