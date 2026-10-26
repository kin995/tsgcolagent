from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnalysisArtifact:
    behavior_summary: list[str] = field(default_factory=list)
    important_branches: list[dict[str, Any]] = field(default_factory=list)
    exception_contracts: list[dict[str, Any]] = field(default_factory=list)
    boundary_values: list[dict[str, Any]] = field(default_factory=list)
    test_generation_risks: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> "AnalysisArtifact":
        if not isinstance(data, dict):
            return cls()
        return cls(
            behavior_summary=_list_of_str(data.get("behavior_summary")),
            important_branches=_validate_branches(_list_of_dict(data.get("important_branches") or data.get("branches"))),
            exception_contracts=_list_of_dict(data.get("exception_contracts") or data.get("exceptions")),
            boundary_values=_list_of_dict(data.get("boundary_values") or data.get("boundaries")),
            test_generation_risks=_list_of_dict(data.get("test_generation_risks") or data.get("risks")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "behavior_summary": self.behavior_summary,
            "important_branches": self.important_branches,
            "exception_contracts": self.exception_contracts,
            "boundary_values": self.boundary_values,
            "test_generation_risks": self.test_generation_risks,
        }


@dataclass
class ExecutionDiagnosis:
    failure_category: str = "unknown"
    root_candidates: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    recommended_focus: str = "general"
    confidence: float = 0.0

    @classmethod
    def from_dict(cls, data: Any) -> "ExecutionDiagnosis":
        if not isinstance(data, dict):
            return cls()
        return cls(
            failure_category=str(data.get("failure_category") or data.get("status") or "unknown"),
            root_candidates=_list_of_str(data.get("root_candidates") or data.get("likely_causes")),
            evidence_ids=_list_of_str(data.get("evidence_ids")),
            recommended_focus=str(data.get("recommended_focus") or data.get("next_agent_focus") or "general"),
            confidence=_clamp(float(data.get("confidence", 0.0) or 0.0)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "failure_category": self.failure_category,
            "root_candidates": self.root_candidates,
            "evidence_ids": self.evidence_ids,
            "recommended_focus": self.recommended_focus,
            "confidence": self.confidence,
        }


def _list_of_str(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _list_of_dict(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _validate_branches(value: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in value:
        result.append(
            {
                "evidence_id": str(item.get("evidence_id", "")),
                "expression": str(item.get("expression", item.get("name", ""))),
                "true_condition": str(item.get("true_condition", "")),
                "false_condition": str(item.get("false_condition", "")),
            }
        )
    return result


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
