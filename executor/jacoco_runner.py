from __future__ import annotations

import csv
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from executor.build_tools import maven_local_repo_arg, maven_project_settings_args, resolve_maven_command
from executor.java_process import run_java_process
from metrics.measurement import measured_java_run


class JacocoRunner:
    def run(
        self,
        project_root: str | Path,
        test_selector: str | None = None,
        target_class: str | None = None,
        timeout: int = 120,
    ) -> dict[str, object]:
        project_root = Path(project_root)
        self._clean_previous_report(project_root)
        resolution = resolve_maven_command(project_root)
        if not resolution.available or not resolution.command:
            return {
                "command": [],
                "returncode": None,
                "stdout": "",
                "stderr": resolution.message,
                "tool_missing": True,
                "tool_resolution": resolution.to_dict(),
                "valid": False,
                "csv_path": None,
            }
        command = [resolution.command, "-q", maven_local_repo_arg(project_root), *maven_project_settings_args(project_root)]
        if test_selector:
            command.append(f"-Dtest={test_selector}")
        command.extend(["test", "jacoco:report"])
        result = self._run(command, project_root, timeout)
        csv_path = project_root / "target" / "site" / "jacoco" / "jacoco.csv"
        metrics = self._parse_csv(csv_path, target_class=target_class) if csv_path.exists() else {}
        result.update(metrics)
        xml_path = project_root / "target" / "site" / "jacoco" / "jacoco.xml"
        xml_metrics = self.parse_report(xml_path, target_class=target_class) if xml_path.exists() else {}
        result.update(xml_metrics)
        result["valid"] = bool(metrics or xml_metrics)
        result["csv_path"] = str(csv_path.resolve()) if csv_path.exists() else None
        result["xml_path"] = str(xml_path.resolve()) if xml_path.exists() else None
        result["target_class"] = target_class
        return result

    def parse_report(self, path: str | Path, target_class: str | None = None) -> dict[str, object]:
        path = Path(path)
        line_data = self._parse_xml(path, target_class=target_class)
        counter_data = self._parse_xml_counters(path, target_class=target_class)
        if not line_data and not counter_data:
            return {}
        return {**counter_data, **line_data, "valid": True, "xml_path": str(path.resolve()), "target_class": target_class}

    def _clean_previous_report(self, project_root: Path) -> None:
        for path in [
            project_root / "target" / "jacoco.exec",
            project_root / "target" / "site" / "jacoco",
        ]:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
            except OSError:
                pass

    def _run(self, command: list[str], cwd: Path, timeout: int) -> dict[str, object]:
        try:
            result = measured_java_run(command, runner=run_java_process, cwd=cwd, timeout=timeout)
            return {"command": command, "returncode": result.returncode, "stdout": result.stdout[-8000:], "stderr": result.stderr[-8000:]}
        except FileNotFoundError as exc:
            return {"command": command, "returncode": None, "stdout": "", "stderr": str(exc), "tool_missing": True}
        except subprocess.TimeoutExpired as exc:
            return {"command": command, "returncode": None, "stdout": (exc.stdout or "")[-8000:], "stderr": (exc.stderr or "")[-8000:], "timed_out": True}

    def _parse_csv(self, path: Path, target_class: str | None = None) -> dict[str, float]:
        totals = {"LINE_MISSED": 0, "LINE_COVERED": 0, "BRANCH_MISSED": 0, "BRANCH_COVERED": 0, "METHOD_MISSED": 0, "METHOD_COVERED": 0}
        matched_rows = 0
        with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
            for row in csv.DictReader(handle):
                if target_class and not self._matches_target_class(row, target_class):
                    continue
                matched_rows += 1
                for key in totals:
                    totals[key] += int(row.get(key, 0) or 0)

        def ratio(covered: str, missed: str) -> float:
            total = totals[covered] + totals[missed]
            return round(totals[covered] / total, 4) if total else 0.0

        if target_class and matched_rows == 0:
            return {}
        return {
            "line_coverage": ratio("LINE_COVERED", "LINE_MISSED"),
            "branch_coverage": ratio("BRANCH_COVERED", "BRANCH_MISSED"),
            "method_coverage": ratio("METHOD_COVERED", "METHOD_MISSED"),
            "matched_class_rows": matched_rows,
        }

    def _parse_xml(self, path: Path, target_class: str | None = None) -> dict[str, object]:
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError):
            return {}
        covered_lines: set[int] = set()
        missing_lines: set[int] = set()
        target_source = self._target_source_file(target_class)
        for package in root.findall("package"):
            package_name = str(package.attrib.get("name", "")).replace("/", ".")
            for source_file in package.findall("sourcefile"):
                source_name = str(source_file.attrib.get("name", ""))
                if target_class and not self._matches_target_source(package_name, source_name, target_class, target_source):
                    continue
                for line in source_file.findall("line"):
                    number = int(line.attrib.get("nr", "0") or 0)
                    covered = int(line.attrib.get("ci", "0") or 0) > 0 or int(line.attrib.get("cb", "0") or 0) > 0
                    if covered:
                        covered_lines.add(number)
                    else:
                        missing_lines.add(number)
        if target_class and not covered_lines and not missing_lines:
            return {}
        return {
            "covered_lines": sorted(covered_lines),
            "missing_lines": sorted(missing_lines),
        }

    def _parse_xml_counters(self, path: Path, target_class: str | None = None) -> dict[str, float]:
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError):
            return {}
        counters = {"LINE": [0, 0], "BRANCH": [0, 0], "METHOD": [0, 0]}
        matched = 0
        for package in root.findall("package"):
            for cls in package.findall("class"):
                qualified = str(cls.attrib.get("name", "")).replace("/", ".")
                if target_class and qualified != target_class:
                    continue
                matched += 1
                for counter in cls.findall("counter"):
                    kind = str(counter.attrib.get("type", ""))
                    if kind in counters:
                        counters[kind][0] += int(counter.attrib.get("missed", "0") or 0)
                        counters[kind][1] += int(counter.attrib.get("covered", "0") or 0)
        if target_class and matched == 0:
            return {}

        def ratio(kind: str) -> float:
            missed, covered = counters[kind]
            return round(covered / (missed + covered), 4) if missed + covered else 0.0

        return {
            "line_coverage": ratio("LINE"),
            "branch_coverage": ratio("BRANCH"),
            "method_coverage": ratio("METHOD"),
            "matched_class_rows": matched,
        }

    def _matches_target_class(self, row: dict[str, str], target_class: str) -> bool:
        package_name = str(row.get("PACKAGE") or "")
        class_name = str(row.get("CLASS") or "")
        qualified = f"{package_name}.{class_name}" if package_name else class_name
        return qualified == target_class or class_name == target_class

    def _target_source_file(self, target_class: str | None) -> str | None:
        if not target_class:
            return None
        class_name = target_class.split(".")[-1].split("$")[0]
        return f"{class_name}.java"

    def _matches_target_source(self, package_name: str, source_name: str, target_class: str, target_source: str | None) -> bool:
        if target_source and source_name == target_source:
            target_package = ".".join(target_class.split(".")[:-1])
            return not target_package or package_name == target_package
        return False
