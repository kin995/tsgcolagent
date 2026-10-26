from __future__ import annotations

import json
import os
import subprocess
from metrics.measurement import measured_python_run
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CoverageResult:
    covered_lines: set[int]
    missing_lines: set[int]
    covered_arcs: set[tuple[int, int]]
    line_coverage: float
    branch_coverage: float | None
    combined_coverage: float
    stdout: str
    stderr: str
    valid: bool = True
    covered_line_count: int = 0
    statement_count: int = 0
    covered_branch_count: int = 0
    branch_count: int = 0

    @property
    def percent(self) -> float:
        """Backward-compatible alias for the old combined percentage."""
        return self.combined_coverage


class CoverageRunner:
    def run(
        self,
        source_path: str | Path,
        test_path: str | Path,
        cwd: str | Path | None = None,
        timeout: int = 40,
    ) -> CoverageResult:
        cwd_path = Path(cwd or Path.cwd()).resolve()
        source_path = Path(source_path).resolve()
        test_path = Path(test_path).resolve()
        try:
            test_arg = str(test_path.relative_to(cwd_path.resolve()))
        except ValueError:
            cwd_path = test_path.parent
            test_arg = test_path.name
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        try:
            with tempfile.TemporaryDirectory(prefix="stateflow_cov_") as temp:
                temp_root = Path(temp)
                data_path = temp_root / ".coverage"
                json_path = temp_root / "coverage.json"
                env = os.environ.copy()
                env["COVERAGE_FILE"] = str(data_path)
                commands = [
                    [
                        sys.executable,
                        "-m",
                        "coverage",
                        "run",
                        "--branch",
                        "--source",
                        str(source_path.parent),
                        "-m",
                        "pytest",
                        test_arg,
                        "-q",
                    ],
                    [sys.executable, "-m", "coverage", "json", "--data-file", str(data_path), "-o", str(json_path)],
                ]
                for command in commands:
                    completed = measured_python_run(
                        command,
                        kind="coverage",
                        cwd=str(cwd_path),
                        text=True,
                        capture_output=True,
                        timeout=timeout,
                        env=env,
                    )
                    stdout_parts.append(completed.stdout)
                    stderr_parts.append(completed.stderr)
                    if completed.returncode != 0:
                        return self._invalid("\n".join(stdout_parts), "\n".join(stderr_parts))
                data = json.loads(json_path.read_text(encoding="utf-8"))
            files = data.get("files", {})
            entry = self._find_source_entry(files, source_path, cwd_path)
            if not entry:
                stderr_parts.append(f"coverage entry not found for exact source: {source_path}")
                return self._invalid("\n".join(stdout_parts), "\n".join(stderr_parts))
            covered = set(entry.get("executed_lines", []))
            missing = set(entry.get("missing_lines", []))
            covered_arcs = {
                (int(item[0]), int(item[1]))
                for item in entry.get("executed_branches", [])
                if isinstance(item, list) and len(item) == 2
            }
            summary = entry.get("summary", {})
            statements = int(summary.get("num_statements", 0) or 0)
            missing_lines = int(summary.get("missing_lines", 0) or 0)
            covered_line_count = max(0, statements - missing_lines)
            branch_count = int(summary.get("num_branches", 0) or 0)
            covered_branch_count = int(summary.get("covered_branches", 0) or 0)
            line_coverage = covered_line_count / statements if statements else 1.0
            branch_coverage = covered_branch_count / branch_count if branch_count else None
            combined = float(summary.get("percent_covered", 0.0)) / 100.0
            return CoverageResult(
                covered_lines=covered,
                missing_lines=missing,
                covered_arcs=covered_arcs,
                line_coverage=round(line_coverage, 4),
                branch_coverage=round(branch_coverage, 4) if branch_coverage is not None else None,
                combined_coverage=round(combined, 4),
                stdout="\n".join(stdout_parts),
                stderr="\n".join(stderr_parts),
                valid=True,
                covered_line_count=covered_line_count,
                statement_count=statements,
                covered_branch_count=covered_branch_count,
                branch_count=branch_count,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as exc:
            return self._invalid("\n".join(stdout_parts), str(exc))

    def _invalid(self, stdout: str, stderr: str) -> CoverageResult:
        return CoverageResult(
            covered_lines=set(),
            missing_lines=set(),
            covered_arcs=set(),
            line_coverage=0.0,
            branch_coverage=None,
            combined_coverage=0.0,
            stdout=stdout,
            stderr=stderr,
            valid=False,
        )

    def _find_source_entry(self, files: dict[str, object], source_path: Path, cwd_path: Path) -> dict[str, object] | None:
        source_resolved = source_path.resolve()
        fallback: dict[str, object] | None = None
        for key, value in files.items():
            if not isinstance(value, dict):
                continue
            key_path = Path(key)
            candidates = [key_path]
            if not key_path.is_absolute():
                candidates.append(cwd_path / key_path)
                candidates.append(source_path.parent / key_path.name)
            if any(candidate.resolve() == source_resolved for candidate in candidates):
                return value
            if key_path.name == source_path.name:
                fallback = value
        return fallback
