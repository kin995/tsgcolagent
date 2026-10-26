from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path


def run_java_process(command: list[str], *, cwd: Path, timeout: float) -> subprocess.CompletedProcess[str]:
    """Bound the whole Java tool process tree, with periodic stage progress."""
    flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) if os.name == "nt" else 0
    started = time.monotonic()
    process = subprocess.Popen(
        command, cwd=cwd, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace",
        creationflags=flags, start_new_session=os.name != "nt",
    )
    label = command[-1] if command else "java"
    print(f"[java-tool] start {label} pid={process.pid} timeout={timeout}s workspace={cwd}", flush=True)
    try:
        while True:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                stdout, stderr = process.communicate(timeout=min(15.0, remaining))
                print(f"[java-tool] done {label} rc={process.returncode} elapsed={time.monotonic()-started:.1f}s", flush=True)
                return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                if time.monotonic() - started >= timeout:
                    raise
                print(f"[java-tool] waiting {label} elapsed={time.monotonic()-started:.1f}s", flush=True)
    except BaseException as exc:
        _stop_tree(process)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", "Process-tree cleanup did not close output pipes within 5s"
        if isinstance(exc, subprocess.TimeoutExpired):
            print(f"[java-tool] timeout {label}; process-tree cleanup requested", flush=True)
            raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr) from exc
        raise


def _stop_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if result.returncode == 0:
                return
        else:
            os.killpg(process.pid, signal.SIGKILL)
            return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        process.kill()
    except OSError:
        pass
