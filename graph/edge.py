from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StateEdge:
    source: str
    target: str
    edge_type: str = "flow"
    visited: bool = False
    weight: float = 1.0
    flow_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def edge_id(self) -> str:
        return f"{self.source}->{self.target}:{self.edge_type}"

    def mark_visited(self, count: int = 1) -> None:
        self.visited = True
        self.flow_count += count
        self.weight += count * 0.1

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type,
            "visited": self.visited,
            "weight": self.weight,
            "flow_count": self.flow_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateEdge":
        return cls(**data)
