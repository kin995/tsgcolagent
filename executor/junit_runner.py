from __future__ import annotations

import subprocess
from pathlib import Path

from executor.build_tools import maven_local_repo_arg, maven_project_settings_args, resolve_maven_command
from executor.java_process import run_java_process
from metrics.measurement import measured_java_run


class JUnitRunner:
    def run(self, project_root: str | Path, test_selector: str | None = None, timeout: int = 120) -> dict[str, object]:
        project_root = Path(project_root)
        resolution = resolve_maven_command(project_root)
        if not resolution.available or not resolution.command:
            return {
                "command": [],
                "passed": False,
                "returncode": None,
                "stdout": "",
                "stderr": resolution.message,
                "tool_missing": True,
                "tool_resolution": resolution.to_dict(),
            }
        command = [resolution.command, "-q", maven_local_repo_arg(project_root), *maven_project_settings_args(project_root)]
        if test_selector:
            command.append(f"-Dtest={test_selector}")
        command.append("test")
        return self._run(command, project_root, timeout)

    def _run(self, command: list[str], cwd: Path, timeout: int) -> dict[str, object]:
        try:
            result = measured_java_run(command, runner=run_java_process, cwd=cwd, timeout=timeout)
            return {
                "command": command,
                "passed": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout[-8000:],
                "stderr": result.stderr[-8000:],
            }
        except FileNotFoundError as exc:
            return {"command": command, "passed": False, "returncode": None, "stdout": "", "stderr": str(exc), "tool_missing": True}
        except subprocess.TimeoutExpired as exc:
            return {"command": command, "passed": False, "returncode": None, "stdout": (exc.stdout or "")[-8000:], "stderr": (exc.stderr or "")[-8000:], "timed_out": True}
