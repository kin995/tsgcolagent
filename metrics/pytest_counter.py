"""Opt-in subprocess probe; imported by pytest only for instrumented runs."""
import json
import os


def _record(event, **fields):
    path = os.environ.get("TSQCOL_TEST_EVENTS")
    if path:
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": event, **fields}) + "\n")


def pytest_sessionstart(session):
    _record("session_start")


def pytest_runtest_logstart(nodeid, location):
    _record("test_start", nodeid=nodeid)


def pytest_sessionfinish(session, exitstatus):
    _record("session_finish", exitstatus=int(exitstatus))
