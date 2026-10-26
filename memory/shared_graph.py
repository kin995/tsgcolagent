from __future__ import annotations

from pathlib import Path

from graph.state_graph import StateFlowGraph


class SharedGraphMemory:
    def __init__(self, storage_path: str | Path):
        self.storage_path = Path(storage_path)
        self.graph: StateFlowGraph | None = None

    def load(self) -> StateFlowGraph:
        if self.graph is not None:
            return self.graph
        if not self.storage_path.exists():
            raise FileNotFoundError(f"State graph not found: {self.storage_path}")
        self.graph = StateFlowGraph.load(self.storage_path)
        return self.graph

    def save(self, graph: StateFlowGraph) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        graph.save(self.storage_path)
        self.graph = graph

    def set(self, graph: StateFlowGraph) -> None:
        self.save(graph)

    def snapshot(self, reason: str) -> None:
        graph = self.load()
        graph.bump_version(reason)
        self.save(graph)
