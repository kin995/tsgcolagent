from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StateNode:
    node_id: str
    node_type: str
    name: str
    file_path: str
    line_start: int
    line_end: int
    code: str = ""
    visited: bool = False
    visit_count: int = 0
    assert_count: int = 0
    mutation_status: str = "unknown"
    coverage_status: str = "uncovered"
    repair_flag: bool = False
    priority: float = 0.5
    coverage_defect: float = 0.0
    assertion_defect: float = 0.0
    mutation_defect: float = 0.0
    exception_defect: float = 0.0
    type_defect: float = 0.0
    defect_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def mark_visited(self, count: int = 1) -> None:
        self.visited = True
        self.visit_count += count
        self.coverage_status = "covered"

    def add_asserts(self, count: int = 1) -> None:
        self.assert_count += count

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "name": self.name,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "code": self.code,
            "visited": self.visited,
            "visit_count": self.visit_count,
            "assert_count": self.assert_count,
            "mutation_status": self.mutation_status,
            "coverage_status": self.coverage_status,
            "repair_flag": self.repair_flag,
            "priority": self.priority,
            "coverage_defect": self.coverage_defect,
            "assertion_defect": self.assertion_defect,
            "mutation_defect": self.mutation_defect,
            "exception_defect": self.exception_defect,
            "type_defect": self.type_defect,
            "defect_score": self.defect_score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateNode":
        return cls(**data)
