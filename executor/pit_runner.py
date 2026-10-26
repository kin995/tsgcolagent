from __future__ import annotations

import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from executor.build_tools import maven_local_repo_arg, maven_project_settings_args, resolve_maven_command
from executor.java_process import run_java_process
from metrics.measurement import measured_java_run


class PitRunner:
    def run(
        self,
        project_root: str | Path,
        test_selector: str | None = None,
        target_class: str | None = None,
        timeout: int = 180,
    ) -> dict[str, object]:
        project_root = Path(project_root)
        resolution = resolve_maven_command(project_root)
        if not resolution.available or not resolution.command:
            return {
                "valid": False,
                "mutation_score": None,
                "stderr": resolution.message,
                "tool_missing": True,
                "tool_resolution": resolution.to_dict(),
            }
        try:
            command = [resolution.command, "-q", maven_local_repo_arg(project_root), *maven_project_settings_args(project_root)]
            if target_class:
                command.append(f"-DtargetClasses={target_class}")
            if test_selector:
                command.append(f"-DtargetTests={self._target_test_pattern(target_class, test_selector)}")
            command.append("org.pitest:pitest-maven:mutationCoverage")
            result = measured_java_run(command, runner=run_java_process, cwd=project_root, timeout=timeout)
            output = result.stdout + "\n" + result.stderr
            score = self._score(output)
            xml_results = self._results_from_xml(project_root, target_class=target_class)
            xml_score = xml_results.get("score")
            # XML can be filtered to the requested class; Maven stdout is often project-wide.
            if xml_score is not None:
                score = xml_score
            return {
                "valid": score is not None and result.returncode == 0,
                "mutation_score": score,
                "xml_score": xml_score,
                **xml_results,
                "command": command,
                "target_class": target_class,
                "test_selector": test_selector,
                "returncode": result.returncode,
                "stdout": result.stdout[-8000:],
                "stderr": result.stderr[-8000:],
            }
        except FileNotFoundError as exc:
            return {"valid": False, "mutation_score": None, "stderr": str(exc), "tool_missing": True}
        except subprocess.TimeoutExpired as exc:
            return {"valid": False, "mutation_score": None, "stdout": (exc.stdout or "")[-8000:], "stderr": (exc.stderr or "")[-8000:], "timed_out": True}

    def _target_test_pattern(self, target_class: str | None, test_selector: str) -> str:
        if not target_class or "." not in target_class:
            return test_selector
        return f"{'.'.join(target_class.split('.')[:-1])}.{test_selector}"

    def _score(self, output: str) -> float | None:
        match = re.search(r"Mutation Coverage\s*:\s*(\d+)%", output)
        if match:
            return round(int(match.group(1)) / 100, 4)
        return None

    def _score_from_xml(self, project_root: Path, target_class: str | None = None) -> float | None:
        return self._results_from_xml(project_root, target_class=target_class).get("score")

    def _results_from_xml(self, project_root: Path, target_class: str | None = None) -> dict[str, object]:
        reports_root = project_root / "target" / "pit-reports"
        if not reports_root.exists():
            return {}
        xml_files = sorted(reports_root.rglob("mutations.xml"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not xml_files:
            return {}
        return self.parse_report(xml_files[0], target_class=target_class)

    def parse_report(self, path: str | Path, target_class: str | None = None) -> dict[str, object]:
        path = Path(path)
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError):
            return {}
        counts = {
            "killed": 0,
            "survived": 0,
            "timeout": 0,
            "invalid": 0,
            "infra_error": 0,
            "no_coverage": 0,
        }
        lines: dict[str, list[int]] = {name: [] for name in counts}
        details: list[dict[str, object]] = []
        detected_count = 0
        for mutation in root.findall("mutation"):
            status = str(mutation.attrib.get("status", "")).upper()
            detected = str(mutation.attrib.get("detected", "")).lower() == "true"
            mutated_class = mutation.findtext("mutatedClass", default="")
            if target_class and mutated_class != target_class:
                continue
            line_text = mutation.findtext("lineNumber", default="0")
            try:
                line = int(line_text)
            except ValueError:
                line = 0
            category = self._status_category(status)
            counts[category] += 1
            if line > 0:
                lines[category].append(line)
            if detected or status in {"KILLED", "TIMED_OUT", "MEMORY_ERROR"}:
                detected_count += 1
            details.append(
                {
                    "status": status.lower(),
                    "category": category,
                    "detected": detected,
                    "line": line,
                    "class": mutated_class,
                    "method": mutation.findtext("mutatedMethod", default=""),
                    "mutator": mutation.findtext("mutator", default="").rsplit(".", 1)[-1],
                    "description": mutation.findtext("description", default=""),
                    "killing_test": mutation.findtext("killingTest", default=""),
                }
            )
        total = len(details)
        score_denominator = total - counts["invalid"]
        effective_mutants = counts["killed"] + counts["survived"]
        score = round(detected_count / score_denominator, 4) if score_denominator > 0 else None
        low_sample_reason = "" if effective_mutants >= 3 else f"effective mutants fewer than 3 ({effective_mutants})"
        return {
            **counts,
            "total": total,
            "effective_mutants": effective_mutants,
            "score_reliability": "unavailable" if effective_mutants == 0 else "low" if effective_mutants < 3 else "normal",
            "score": score,
            "detected_count": detected_count,
            "score_denominator": score_denominator,
            "killed_lines": lines["killed"],
            "survived_lines": lines["survived"],
            "timeout_lines": lines["timeout"],
            "invalid_lines": lines["invalid"],
            "infra_error_lines": lines["infra_error"],
            "no_coverage_lines": lines["no_coverage"],
            "details": details,
            "candidate_count": total,
            "attempted_mutants": total - counts["invalid"] - counts["no_coverage"],
            "low_sample_reason": low_sample_reason,
            "autonomy_action": "" if not low_sample_reason else "plan boundary and branch-distinguishing assertions",
            "xml_path": str(path.resolve()),
        }

    def _status_category(self, status: str) -> str:
        if status == "KILLED":
            return "killed"
        if status == "SURVIVED":
            return "survived"
        if status == "TIMED_OUT":
            return "timeout"
        if status == "NON_VIABLE":
            return "invalid"
        if status == "NO_COVERAGE":
            return "no_coverage"
        return "infra_error"
