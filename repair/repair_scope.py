from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class RepairScope:
    level: str
    target_id: str
    reason: str
    related_nodes: list[str] = field(default_factory=list)
    related_errors: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

