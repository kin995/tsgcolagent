from __future__ import annotations

import ast
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from executor.failure_classifier import FailureDiagnosis, PytestFailureClassifier
from executor.pytest_runner import PytestRunner
from executor.test_quality import analyze_test_usability


@dataclass
class CandidateValidationResult:
    accepted: bool
    stage: str
    failure_category: str | None = None
    pytest_output: str = ""
    collected_count: int = 0
    candidate_path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "stage": self.stage,
            "failure_category": self.failure_category,
            "pytest_output": self.pytest_output[-3000:],
            "collected_count": self.collected_count,
            "candidate_path": self.candidate_path,
            "details": self.details,
        }


class CandidateTestValidator:
    def __init__(self, artifacts_dir: str | Path):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.pytest_runner = PytestRunner()
        self.classifier = PytestFailureClassifier()

    def validate_and_write(
        self,
        candidate_code: str,
        final_path: str | Path,
        cwd: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CandidateValidationResult:
        final_path = Path(final_path).resolve()
        metadata = metadata or {}
        try:
            ast.parse(candidate_code)
            compile(candidate_code, str(final_path), "exec")
        except SyntaxError as exc:
            return self._reject(candidate_code, "ast_compile", "syntax_error", str(exc), metadata)

        with tempfile.TemporaryDirectory(prefix="stateflow_candidate_") as temp:
            temp_path = Path(temp) / final_path.name
            temp_path.write_text(candidate_code, encoding="utf-8")
            collect = self.pytest_runner.collect_only(temp_path, cwd=Path(cwd).resolve() if cwd else temp_path.parent)
            if collect.timed_out:
                return self._reject(candidate_code, "collect_only", "timeout", collect.stderr, metadata)
            diagnosis = self.classifier.classify(collect, temp_path, timed_out=collect.timed_out)
            if not collect.passed:
                return self._reject(candidate_code, "collect_only", diagnosis.failure_category, collect.stdout + collect.stderr, metadata)
            if not collect.collected_count:
                return self._reject(candidate_code, "collect_only", "no_tests_collected", collect.stdout + collect.stderr, metadata)

            pytest_result = self.pytest_runner.run(temp_path, cwd=Path(cwd).resolve() if cwd else temp_path.parent)
            diagnosis = self.classifier.classify(pytest_result, temp_path, timed_out=pytest_result.timed_out)
            if not pytest_result.passed:
                return self._reject(candidate_code, "baseline_pytest", diagnosis.failure_category, pytest_result.stdout + pytest_result.stderr, metadata)
            usability = analyze_test_usability(temp_path)
            if usability.exception_only or usability.normal_behavior_tests == 0:
                category = "exception_only" if usability.exception_only else "no_normal_behavior"
                return self._reject(
                    candidate_code,
                    "semantic_usability",
                    category,
                    "candidate has no normal-behavior test with a target call and a result assertion",
                    {**metadata, "test_usability": usability.to_dict()},
                )
            if usability.weak_assertions_only:
                return self._reject(
                    candidate_code,
                    "semantic_usability",
                    "weak_assertions_only",
                    "candidate contains only weak assertions",
                    {**metadata, "test_usability": usability.to_dict()},
                )

        final_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path = final_path.with_suffix(final_path.suffix + ".bak")
        if final_path.exists():
            shutil.copy2(final_path, backup_path)
        temp_final = final_path.with_suffix(final_path.suffix + ".tmp")
        temp_final.write_text(candidate_code, encoding="utf-8")
        try:
            os.replace(temp_final, final_path)
        except OSError:
            final_path.write_text(candidate_code, encoding="utf-8")
            try:
                temp_final.unlink()
            except OSError:
                pass
        return CandidateValidationResult(
            accepted=True,
            stage="accepted",
            collected_count=collect.collected_count or 0,
            details={
                "metadata": metadata,
                "test_usability": usability.to_dict(),
                "backup_path": str(backup_path) if backup_path.exists() else None,
            },
        )

    def _reject(self, candidate_code: str, stage: str, failure_category: str, output: str, metadata: dict[str, Any]) -> CandidateValidationResult:
        index = len(list(self.artifacts_dir.glob("rejected_candidate_*.py"))) + 1
        candidate_path = self.artifacts_dir / f"rejected_candidate_{index:03d}.py"
        meta_path = self.artifacts_dir / f"rejected_candidate_{index:03d}.json"
        candidate_path.write_text(candidate_code, encoding="utf-8")
        record = {
            "stage": stage,
            "failure_category": failure_category,
            "pytest_output": output[-3000:],
            "metadata": metadata,
        }
        meta_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return CandidateValidationResult(
            accepted=False,
            stage=stage,
            failure_category=failure_category,
            pytest_output=output[-3000:],
            candidate_path=str(candidate_path),
            details={"metadata_path": str(meta_path), "metadata": metadata},
        )
