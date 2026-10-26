"""Sidecar measurements; no metric is fed back into agent decision state.

ContextVars follow LangGraph node contexts and isolate concurrent tasks. The
ledger is append-only across workflow rounds and retries; cached evaluations
do not pass through the subprocess probes and therefore do not add executions.
"""
from __future__ import annotations

import contextvars
import copy
import hashlib
import json
import os
import re
import subprocess
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from metrics.paper_metrics import direct_metrics, tir_statistics

CURRENT = contextvars.ContextVar("tsqcol_measurement", default=None)


def read_json(path: Path, fallback=None):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return fallback


def digest(path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except (OSError, TypeError):
        return None


def write_summary(path: Path, values: dict):
    text = json.dumps(values, ensure_ascii=False, indent=2)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    try:
        os.replace(temporary, path)
    except OSError:
        # Same fallback as the project's experiment registry: Windows can
        # reject replace on Unicode user-profile paths. The append-only ledger
        # remains the recoverable source if a reader catches a partial summary.
        path.write_text(text, encoding="utf-8")
        temporary.unlink(missing_ok=True)


class Measurement:
    def __init__(self, output_dir, source, round_number):
        self.root = Path(output_dir) / "paper_metrics"
        self.root.mkdir(parents=True, exist_ok=True)
        self.source_hash = digest(source)
        self.session_id = uuid.uuid4().hex
        self.task_id = str(Path(source).resolve())
        self.round = round_number
        self.started = time.perf_counter()
        self.lock = threading.RLock()
        self.pending = None
        self.step = 0
        self.last_evidence = {}
        self.append({"type": "session_start", "source_hash": self.source_hash,
                     "legacy_prefix": (Path(output_dir) / "stateflow_summary.json").exists() and not (self.root / "summary.json").exists()})
        self.summary({}, 0)

    def append(self, record):
        with self.lock:
            row = {"session_id": self.session_id, "task_id": self.task_id, "round": self.round,
                   "source_hash": self.source_hash, **record}
            with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def begin_action(self, plan, graph, test_path=None, initial=False):
        if self.pending:
            return
        self.step += 1
        goals = copy.deepcopy(plan.get("paper_targets") or [])
        if not goals:
            if initial or not self.last_evidence.get("report", {}).get("pytest_passed"):
                goals.append({"id": "execution:executable", "predicate": "executable", "category": "recovery"})
            selected = str(plan.get("selected_agent") or "")
            fields = {"CoverageAgent": [("line_coverage", "coverage"), ("branch_coverage", "coverage")],
                      "BranchAgent": [("branch_coverage", "coverage"), ("line_coverage", "coverage")],
                      "AssertionAgent": [("ae", "observation")], "OracleRepairAgent": [("ae", "observation")],
                      "MutationAgent": [("mutation_score", "mutation")]}.get(selected, [])
            baseline = direct_metrics(self.last_evidence.get("report") or {})
            for field, category in fields:
                goals.append({"id": f"metric:{field}", "category": category, "predicate": "metric_increase",
                              "field": field, "baseline": baseline.get(field),
                              "baseline_profile": (self.last_evidence.get("report") or {}).get("evaluation_profile")})
            if selected == "BoundaryAgent" and "branch" in str(plan.get("focus") or ""):
                goals.append({"id": "metric:branch_coverage", "category": "coverage", "predicate": "metric_increase",
                              "field": "branch_coverage", "baseline": baseline.get("branch_coverage")})
            elif selected in {"BoundaryAgent", "TypeAnalysisAgent"}:
                category = "exception" if selected == "BoundaryAgent" else "type"
                goals.append({"id": f"runtime:{category}", "category": category, "predicate": "runtime_growth",
                              "baseline_observations": copy.deepcopy(self.last_evidence.get("runtime_observations"))})
            if not goals:
                goals.append({"id": f"unmapped:{selected or 'plan'}", "category": "unmapped",
                              "predicate": "explicit_fact", "reason": "no_comparable_tool_predicate"})
        self.pending = {"task_id": self.task_id, "session_id": self.session_id, "round": self.round,
                        "step": self.step, "goals": goals, "preregistered": True,
                        "before": copy.deepcopy(self.last_evidence) or {"no_test": bool(initial and not digest(test_path))},
                        "before_hash": digest(test_path), "acted": True, "completed": False}
        context = graph.metadata.get("last_knowledge_context") or {} if hasattr(graph, "metadata") else {}
        self.pending["knowledge_ids"] = list(dict.fromkeys(context.get("selected_ids") or []))
        self.append({"type": "action_start", "action": self.pending})

    def finish_action(self, evidence=None, unchanged=False):
        if not self.pending:
            return
        self.pending["retained_after"] = copy.deepcopy(self.pending["before"] if unchanged else evidence or {})
        self.pending["completed"] = True
        self.append({"type": "action_end", "action": self.pending})
        self.pending = None

    def observe(self, report, graph, test_path):
        self.last_evidence = {"report": copy.deepcopy(report), "test_hash": digest(test_path),
                              "coverage": copy.deepcopy(graph.metadata.get("last_coverage") or {}),
                              "mutation": copy.deepcopy(graph.metadata.get("last_mutation") or {}),
                              "runtime_observations": copy.deepcopy(graph.metadata.get("last_runtime_observations"))}
        self.append({"type": "retained_evidence", "evidence": self.last_evidence})
        self.finish_action(self.last_evidence)
        self.summary(report, time.perf_counter() - self.started)

    def summary(self, report, duration=None, completed=False):
        if completed:
            self.append({"type": "session_end", "duration_seconds": time.perf_counter() - self.started})
        rows, corrupt = [], False
        with self.lock:
            for line in (self.root / "events.jsonl").read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                    if item.get("source_hash") == self.source_hash:
                        rows.append(item)
                except ValueError:
                    corrupt = True
        calls = {r["call_id"]: r for r in rows if r.get("type") == "execution_end"}
        starts = {r["call_id"] for r in rows if r.get("type") == "execution_start"}
        sessions = {r["session_id"] for r in rows if r.get("type") == "session_start"}
        ended = {r["session_id"] for r in rows if r.get("type") == "session_end"}
        complete = not corrupt and starts == calls.keys() and sessions <= ended and not any(r.get("legacy_prefix") for r in rows) and all(r.get("complete") for r in calls.values())
        actions = {}
        for row in rows:
            if row.get("type") in {"action_start", "action_end"}:
                action = row["action"]
                actions[(action["session_id"], action["step"])] = action
        times = {r["session_id"]: float(r.get("duration_seconds") or 0) for r in rows if r.get("type") == "session_end"}
        elapsed = sum(times.values()) + (time.perf_counter() - self.started if self.session_id not in times else 0)
        values = direct_metrics(report, max(float(duration or 0), elapsed))
        values.update(tir_statistics(list(actions.values())))
        count = sum(int(r.get("count") or 0) for r in calls.values())
        values.update({"exec_count": count, "avg_exec": count if complete else None,
                       "exec_count_status": "complete" if complete else "incomplete_evidence",
                       "exec_unit": "test_tool_session", "execution_calls": len(calls),
                       "execution_unknown_calls": sum(not r.get("complete") for r in calls.values()),
                       "measurement_version": 1, "source_hash": self.source_hash,
                       "legacy_prefix": any(r.get("legacy_prefix") for r in rows),
                       "knowledge_occurrence": len({r["record_id"] for r in rows if r.get("type") == "knowledge_record"}),
                       "knowledge_record_ids": sorted({r["record_id"] for r in rows if r.get("type") == "knowledge_record"}),
                       "knowledge_use_frequency": sum(len(a.get("knowledge_ids") or []) for a in actions.values() if a.get("acted")),
                       "ledger_path": str((self.root / "events.jsonl").resolve())})
        write_summary(self.root / "summary.json", values)
        return values


def measured_python_run(command, *, kind, **kwargs):
    measurement = CURRENT.get()
    if measurement is None:
        return subprocess.run(command, **kwargs)
    if "pytest" not in command or "--collect-only" in command:
        return subprocess.run(command, **kwargs)
    call_id = uuid.uuid4().hex
    path = measurement.root / f"pytest-{call_id}.jsonl"
    env = dict(kwargs.get("env") or os.environ)
    # Only the child environment is changed. The parent and other tasks stay isolated.
    module_root = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [module_root, env.get("PYTHONPATH")]))
    env["TSQCOL_TEST_EVENTS"] = str(path.resolve())
    command = [*command, "-p", "pytest_counter"]
    measurement.append({"type": "execution_start", "call_id": call_id, "kind": kind})
    completed = None
    try:
        completed = subprocess.run(command, **{**kwargs, "env": env})
        return completed
    finally:
        records = []
        corrupt = False
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    records.append(json.loads(line))
                except ValueError:
                    corrupt = True
        # One observed test start proves one session, even if it later times out.
        exact = not corrupt and (any(r.get("event") == "test_start" for r in records) or (completed is not None and any(r.get("event") == "session_finish" for r in records)))
        measurement.append({"type": "execution_end", "call_id": call_id, "kind": kind,
                            "count": int(any(r.get("event") == "test_start" for r in records)),
                            "test_case_starts": sum(r.get("event") == "test_start" for r in records),
                            "complete": exact, "raw_path": str(path.resolve())})


def java_report_fingerprints(root: Path):
    # Avoid traversing source trees, dependency caches and .git on every call.
    paths = [p for prefix in ("", "*/", "*/*/")
             for suffix in ("target/surefire-reports/TEST-*.xml", "build/test-results/test/TEST-*.xml")
             for p in root.glob(prefix + suffix)]
    return {str(p): (p.stat().st_mtime_ns, p.stat().st_size) for p in paths}


def java_execution_count(root: Path, before: dict, stdout: str, stderr: str, returncode, mutation=False):
    output = stdout + "\n" + stderr
    if mutation:
        # A PIT campaign is one tool session, not one call per mutant or case.
        # Its internal test count remains supplementary evidence only.
        mutant = re.findall(r"Ran\s+(\d+)\s+tests\s*\(", output)
        if mutant and returncode == 0:
            return 1, True
        return 0, False
    after = java_report_fingerprints(root)
    count = 0
    changed = [Path(p) for p, stamp in after.items() if before.get(p) != stamp]
    try:
        for path in changed:
            count += len(ET.parse(path).getroot().findall(".//testcase"))
    except (ET.ParseError, OSError):
        return int(count > 0), False
    if count:
        return 1, True
    if re.search(r"Tests run:\s*[1-9]\d*", output):
        return 1, True
    if re.search(r"COMPILATION ERROR|Compilation failure|No tests to run|No tests were executed", output, re.I):
        return 0, True
    return int(count > 0), bool(changed) and returncode in {0, 1}


def measured_java_run(command, *, runner, cwd, timeout):
    measurement = CURRENT.get()
    test_command = any(str(arg) in {"test", "pitest"} or "mutationCoverage" in str(arg) for arg in command)
    if measurement is None or not test_command:
        return runner(command, cwd=cwd, timeout=timeout)
    root = Path(cwd)
    call_id = uuid.uuid4().hex
    before = java_report_fingerprints(root)
    mutation = any("pitest" in str(arg) or "mutationCoverage" in str(arg) for arg in command)
    measurement.append({"type": "execution_start", "call_id": call_id, "kind": "java_mutation" if mutation else "java_test"})
    result = None
    try:
        result = runner(command, cwd=cwd, timeout=timeout)
        return result
    finally:
        count, complete = java_execution_count(root, before, result.stdout if result else "", result.stderr if result else "", result.returncode if result else None, mutation)
        measurement.append({"type": "execution_end", "call_id": call_id,
                            "count": count, "complete": complete,
                            "reason": None if complete else "java_internal_execution_count_unavailable"})


def reconcile_round(output_dir, retained_report, round_number, accepted, duration=None):
    """Outer R2/R3 rollback preserves costs but replaces rejected outcome facts."""
    root = Path(output_dir) / "paper_metrics"
    values = read_json(root / "summary.json", {})
    if not values or not (root / "events.jsonl").exists():
        return
    rows = []
    for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            if row.get("source_hash") == values.get("source_hash"):
                rows.append(row)
        except ValueError:
            return
    latest_session = next((r["session_id"] for r in reversed(rows) if r.get("type") == "session_start" and r.get("round") == round_number), None)
    actions = {(r["action"]["session_id"], r["action"]["step"]): r["action"] for r in rows if r.get("type") in {"action_start", "action_end"}}
    if not accepted and latest_session:
        graph = read_json(Path(output_dir) / "state_flow_graph.json", {})
        meta = graph.get("metadata") or {}
        evidence = {"report": retained_report or {}, "coverage": meta.get("last_coverage") or {}, "mutation": meta.get("last_mutation") or {}, "runtime_observations": meta.get("last_runtime_observations")}
        for (session, _), action in actions.items():
            if session == latest_session:
                action["retained_after"] = evidence
                action["outer_round_rejected"] = True
                with (root / "events.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"type": "action_end", "source_hash": values.get("source_hash"), "action": action}) + "\n")
    values.update(direct_metrics(retained_report or {}, duration))
    values.update(tir_statistics(list(actions.values())))
    write_summary(root / "summary.json", values)


def batch_paper_metrics(results, selected_tasks=None):
    from metrics.paper_metrics import aggregate_metrics
    tasks = []
    for item in results:
        root = Path(item.output_dir)
        values = read_json(root / "paper_metrics" / "summary.json", {})
        values.update(direct_metrics(item.final_report or {}, item.duration_seconds))
        tasks.append({"metrics": values, "grade": item.final_grade})
    return aggregate_metrics(tasks, selected_tasks)
