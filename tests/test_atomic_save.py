"""Concurrency and atomicity tests for audit persistence."""
import asyncio
import concurrent.futures
import json
import os
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.routers.forgetting_audits import save, read_audit, path_for
from app.models.production import User


def test_atomic_save_integrity(tmp_path, monkeypatch):
    test_id = "a" * 32
    monkeypatch.setattr("app.routers.forgetting_audits.STORE", tmp_path)
    monkeypatch.setattr("app.routers.forgetting_audits.ALT_STORE", tmp_path)

    sample_data = {"id": test_id, "name": "Atomic Test", "owner_id": "test_user"}
    save(sample_data)

    target_file = tmp_path / f"{test_id}.json"
    assert target_file.exists()
    loaded = json.loads(target_file.read_text())
    assert loaded["name"] == "Atomic Test"


def test_concurrent_atomic_saves(tmp_path, monkeypatch):
    test_id = "b" * 32
    monkeypatch.setattr("app.routers.forgetting_audits.STORE", tmp_path)
    monkeypatch.setattr("app.routers.forgetting_audits.ALT_STORE", tmp_path)

    def write_worker(idx: int):
        data = {
            "id": test_id,
            "name": f"Worker {idx}",
            "iteration": idx,
            "owner_id": "test_user",
        }
        save(data)
        return idx

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(write_worker, i) for i in range(25)]
        results = [f.result() for f in futures]

    assert len(results) == 25
    final_file = tmp_path / f"{test_id}.json"
    assert final_file.exists()
    final_data = json.loads(final_file.read_text())
    assert "iteration" in final_data
    assert 0 <= final_data["iteration"] < 25
