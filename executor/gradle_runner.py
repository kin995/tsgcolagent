from __future__ import annotations

import subprocess
from pathlib import Path
from executor.java_process import run_java_process
from metrics.measurement import measured_java_run



class GradleRunner:
    def command(self, project_root: str | Path, task: str) -> list[str]:
        root = Path(project_root)
        wrapper = root / ("gradlew.bat" if _is_windows() else "gradlew")
        executable = str(wrapper) if wrapper.exists() else "gradle"
        return [executable, task, "--quiet"]

    def run(self, project_root: str | Path, task: str = "test", timeout: int = 120) -> dict[str, object]:
        return self.run_tasks(project_root, [task], timeout=timeout)

    def run_tasks(
        self,
        project_root: str | Path,
        tasks: list[str],
        timeout: int = 120,
        test_selector: str | None = None,
    ) -> dict[str, object]:
        root = Path(project_root)
        base = self.command(root, tasks[0] if tasks else "test")
        command = [base[0]]
        for task in tasks:
            command.append(task)
            if task == "test" and test_selector:
                command.extend(["--tests", test_selector])
        command.append("--quiet")
        try:
            result = measured_java_run(command, runner=run_java_process, cwd=root, timeout=timeout)
            return {
                "build_system": "gradle",
                "command": command,
                "passed": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout[-8000:],
                "stderr": result.stderr[-8000:],
            }
        except FileNotFoundError as exc:
            return {"build_system": "gradle", "command": command, "passed": False, "returncode": None, "stdout": "", "stderr": str(exc), "tool_missing": True}
        except subprocess.TimeoutExpired as exc:
            return {"build_system": "gradle", "command": command, "passed": False, "returncode": None, "stdout": (exc.stdout or "")[-8000:], "stderr": (exc.stderr or "")[-8000:], "timed_out": True}


def _is_windows() -> bool:
    return "\\" in str(Path.cwd())
