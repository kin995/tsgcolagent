from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ValidationError:
    code: str
    message: str
    severity: str = "error"
    related_nodes: list[str] = field(default_factory=list)
    related_edges: list[str] = field(default_factory=list)
    repair_hint: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ValidationReport:
    passed: bool
    score: float
    rule_score: float
    logic_score: float
    feasibility_score: float
    coverage_score: float
    repair_scope_score: float
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["errors"] = [item.to_dict() for item in self.errors]
        data["warnings"] = [item.to_dict() for item in self.warnings]
        return data

