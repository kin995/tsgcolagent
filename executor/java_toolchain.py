from __future__ import annotations

from pathlib import Path
from typing import Any
import shutil
import threading

from executor.gradle_runner import GradleRunner
from executor.jacoco_runner import JacocoRunner
from executor.junit_runner import JUnitRunner
from executor.pit_runner import PitRunner


_JAVA_EVALUATION_LOCK = threading.Lock()
_JAVA_WORKSPACE_LOCKS: dict[str, threading.Lock] = {}
_JAVA_WORKSPACE_LOCKS_GUARD = threading.Lock()


def _java_evaluation_lock(project_root: Path) -> threading.Lock:
    if not (project_root / ".stateflow_humanevaljava_workspace").is_file():
        return _JAVA_EVALUATION_LOCK
    key = str(project_root.resolve()).casefold()
    with _JAVA_WORKSPACE_LOCKS_GUARD:
        return _JAVA_WORKSPACE_LOCKS.setdefault(key, threading.Lock())


class JavaToolchain:
    language = "java"

    def __init__(self):
        self.junit = JUnitRunner()
        self.gradle = GradleRunner()
        self.jacoco = JacocoRunner()
        self.pit = PitRunner()

    def run(
        self,
        project_root: str | Path,
        test_selector: str | None = None,
        target_class: str | None = None,
        timeout: int = 120,
        evaluation_profile: str = "full",
    ) -> dict[str, Any]:
        profile = "screening" if evaluation_profile == "screening" else "full"
        project_root = Path(project_root)
        with _java_evaluation_lock(project_root):
            self._clean_transient_outputs(project_root)
            build_system = self._build_system(project_root)
            if build_system == "gradle":
                combined = self.gradle.run_tasks(
                    project_root,
                    ["test", "jacocoTestReport"],
                    timeout=timeout,
                    test_selector=test_selector,
                )
                junit = dict(combined)
                jacoco = dict(combined)
                jacoco["valid"] = False
                if junit.get("passed"):
                    jacoco_xml = self._latest_report(project_root / "build" / "reports" / "jacoco", "*.xml")
                    if jacoco_xml:
                        jacoco.update(self.jacoco.parse_report(jacoco_xml, target_class=target_class))
            else:
                jacoco = self.jacoco.run(project_root, test_selector=test_selector, target_class=target_class, timeout=timeout)
                junit = self._junit_from_combined(jacoco)

            if junit.get("passed") and profile == "full":
                pit = (
                    self.gradle.run(project_root, "pitest", timeout=timeout)
                    if build_system == "gradle"
                    else self.pit.run(project_root, test_selector=test_selector, target_class=target_class, timeout=max(180, timeout))
                )
                if build_system == "gradle":
                    pit_xml = self._latest_report(project_root / "build" / "reports" / "pitest", "mutations.xml")
                    if pit_xml:
                        pit_metrics = self.pit.parse_report(pit_xml, target_class=target_class)
                        pit.update(pit_metrics)
                        pit["mutation_score"] = pit_metrics.get("score")
                        pit["valid"] = pit_metrics.get("score") is not None and int(pit.get("returncode", 0) or 0) == 0
            else:
                pit = {
                    "valid": False,
                    "mutation_score": None,
                    "build_system": build_system,
                    "skipped": True,
                    "skipped_reason": "deferred_until_final_candidate" if junit.get("passed") else "baseline_not_runnable",
                    "evaluation_profile": profile,
                }
            return {
                "language": self.language,
                "build_system": build_system,
                "passed": bool(junit.get("passed")),
                "junit": junit,
                "line_coverage": jacoco.get("line_coverage"),
                "branch_coverage": jacoco.get("branch_coverage"),
                "method_coverage": jacoco.get("method_coverage"),
                "mutation_score": pit.get("mutation_score"),
                "jacoco": jacoco,
                "pit": pit,
                "evaluation_profile": profile,
                "combined_test_coverage": True,
            }

    def _junit_from_combined(self, result: dict[str, object]) -> dict[str, object]:
        return {
            "command": result.get("command", []),
            "passed": result.get("returncode") == 0 and not result.get("timed_out") and not result.get("tool_missing"),
            "returncode": result.get("returncode"),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "timed_out": result.get("timed_out", False),
            "tool_missing": result.get("tool_missing", False),
            "tool_resolution": result.get("tool_resolution"),
            "combined_with_jacoco": True,
        }

    def run_pit_only(
        self,
        project_root: str | Path,
        test_selector: str | None = None,
        target_class: str | None = None,
        timeout: int = 180,
    ) -> dict[str, object]:
        project_root = Path(project_root)
        with _java_evaluation_lock(project_root):
            build_system = self._build_system(project_root)
            pit_reports = (
                project_root / "build" / "reports" / "pitest"
                if build_system == "gradle"
                else project_root / "target" / "pit-reports"
            )
            if pit_reports.is_dir():
                shutil.rmtree(pit_reports, ignore_errors=True)
            if build_system == "gradle":
                pit = self.gradle.run(project_root, "pitest", timeout=timeout)
                pit_xml = self._latest_report(project_root / "build" / "reports" / "pitest", "mutations.xml")
                if pit_xml:
                    metrics = self.pit.parse_report(pit_xml, target_class=target_class)
                    pit.update(metrics)
                    pit["mutation_score"] = metrics.get("score")
                    pit["valid"] = metrics.get("score") is not None and int(pit.get("returncode", 0) or 0) == 0
                return pit
            return self.pit.run(
                project_root,
                test_selector=test_selector,
                target_class=target_class,
                timeout=timeout,
            )

    def _build_system(self, project_root: Path) -> str:
        if (project_root / "build.gradle").exists() or (project_root / "build.gradle.kts").exists() or (project_root / "gradlew").exists() or (project_root / "gradlew.bat").exists():
            return "gradle"
        return "maven"

    def _clean_transient_outputs(self, project_root: Path) -> None:
        for path in [
            project_root / "target" / "surefire-reports",
            project_root / "target" / "pit-reports",
            project_root / "target" / "site" / "jacoco",
            project_root / "target" / "jacoco.exec",
            project_root / "build" / "test-results",
            project_root / "build" / "reports" / "tests",
            project_root / "build" / "reports" / "jacoco",
            project_root / "build" / "reports" / "pitest",
            project_root / "build" / "jacoco",
        ]:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.is_file():
                    path.unlink()
            except OSError:
                pass

    def _latest_report(self, root: Path, pattern: str) -> Path | None:
        if not root.exists():
            return None
        matches = sorted(root.rglob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
        return matches[0] if matches else None
