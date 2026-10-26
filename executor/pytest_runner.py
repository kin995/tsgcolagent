from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from metrics.measurement import measured_python_run


@dataclass
class PytestResult:
    returncode: int
    stdout: str
    stderr: str
    passed: bool
    timed_out: bool = False
    collected_count: int | None = None


class PytestRunner:
    def run(self, test_path: str | Path, cwd: str | Path | None = None, timeout: int = 30) -> PytestResult:
        return self._run(test_path, cwd=cwd, timeout=timeout, collect_only=False)

    def collect_only(self, test_path: str | Path, cwd: str | Path | None = None, timeout: int = 20) -> PytestResult:
        return self._run(test_path, cwd=cwd, timeout=timeout, collect_only=True)

    def _run(self, test_path: str | Path, cwd: str | Path | None = None, timeout: int = 30, collect_only: bool = False) -> PytestResult:
        test_path = Path(test_path).resolve()
        run_cwd = Path(cwd).resolve() if cwd else test_path.parent
        try:
            test_arg = str(test_path.relative_to(run_cwd))
        except ValueError:
            run_cwd = test_path.parent
            test_arg = test_path.name
        command = [sys.executable, "-m", "pytest", test_arg, "-q"]
        if collect_only:
            command.append("--collect-only")
        try:
            completed = measured_python_run(
                command,
                kind="pytest",
                cwd=str(run_cwd),
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return PytestResult(
                returncode=124,
                stdout=exc.stdout if isinstance(exc.stdout, str) else "",
                stderr=exc.stderr if isinstance(exc.stderr, str) else str(exc),
                passed=False,
                timed_out=True,
                collected_count=0 if collect_only else None,
            )
        stdout = completed.stdout
        collected = _parse_collected_count(stdout) if collect_only else _parse_executed_count(stdout)
        return PytestResult(
            returncode=completed.returncode,
            stdout=stdout,
            stderr=completed.stderr,
            passed=completed.returncode == 0,
            collected_count=collected,
        )


def _parse_collected_count(output: str) -> int:
    import re

    match = re.search(r"(\d+)\s+tests?\s+collected", output)
    if match:
        return int(match.group(1))
    node_lines = [line for line in output.splitlines() if "::test_" in line]
    return len(node_lines)


def _parse_executed_count(output: str) -> int | None:
    import re

    counts = [int(value) for value in re.findall(r"(\d+)\s+(?:passed|failed|skipped|xfailed|xpassed|errors?)", output)]
    return sum(counts) if counts else None
