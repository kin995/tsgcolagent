from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_test_knowledge_store(tmp_path, monkeypatch):
    monkeypatch.setenv("TSG_KNOWLEDGE_STORE", str(tmp_path / "knowledge_store" / "test_knowledge_guidance.json"))
