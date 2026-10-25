"""Local LLM experiment job API. Jobs run the repository's real run_llm_suite.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.auth import get_current_user
from app.models.production import User


ROOT = Path(__file__).resolve().parents[3]

JOB_ROOT = ROOT / "results" / "llm" / "app_jobs"

router = APIRouter(prefix="/llm/jobs", tags=["llm-experiments"])

_LOCK = threading.Lock()

_PROCESSES: dict[str, subprocess.Popen] = {}

_METHODS = {
    "retain_only_retraining",
    "retain_finetuning",
    "gradient_ascent",
    "gradient_difference",
    "npo",
}


class JobCreate(BaseModel):
    model: str = Field(min_length=1, max_length=300)
    methods: list[str] = Field(min_length=1, max_length=5)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    epochs: int = Field(default=1, ge=1, le=5)
    warmup_epochs: int = Field(default=1, ge=0, le=5)
    learning_rate: float = Field(default=2e-5, gt=0, le=0.01)
    max_length: int = Field(default=256, ge=32, le=2048)
    max_new_tokens: int = Field(default=32, ge=1, le=256)
    context_sizes: list[int] = Field(default=[8, 32], min_length=1, max_length=5)
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    full_niah: bool = False


def _read(job_id: str, user: User) -> dict:
    path = JOB_ROOT / job_id / "job.json"

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="LLM experiment not found",
        )

    data = json.loads(path.read_text(encoding="utf-8"))

    # Every LLM job must belong to an authenticated user.
    # Old jobs without owner_id fail closed instead of being exposed.
    owner_id = data.get("owner_id")
    if not owner_id or owner_id != user.id:
        raise HTTPException(
            status_code=403,
            detail="Not authorized for this LLM experiment",
        )

    with _LOCK:
        proc = _PROCESSES.get(job_id)

    if data["status"] in {"queued", "running"}:
        if proc is not None and proc.poll() is not None:
            data["status"] = (
                "complete" if proc.returncode == 0 else "failed"
            )
            data["return_code"] = proc.returncode
            data["completed_at"] = datetime.now(timezone.utc).isoformat()
            path.write_text(
                json.dumps(data, indent=2),
                encoding="utf-8",
            )

        elif proc is None:
            data["status"] = "interrupted"
            data["message"] = (
                "Backend restarted before this job's process could be "
                "reattached. Inspect the output directory for partial artifacts."
            )
            path.write_text(
                json.dumps(data, indent=2),
                encoding="utf-8",
            )

    log = JOB_ROOT / job_id / "runner.log"

    data["log_tail"] = (
        log.read_text(encoding="utf-8", errors="replace")[-6000:]
        if log.exists()
        else ""
    )

    return data


@router.post("", status_code=202)
def create_job(
    req: JobCreate,
    user: User = Depends(get_current_user),
):
    if (
        len(set(req.methods)) != len(req.methods)
        or any(m not in _METHODS for m in req.methods)
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "methods must be unique values from: "
                + ", ".join(sorted(_METHODS))
            ),
        )

    if any(n < 0 or n > 512 for n in req.context_sizes):
        raise HTTPException(
            status_code=422,
            detail="context sizes must be between 0 and 512",
        )

    job_id = uuid.uuid4().hex
    job_dir = JOB_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=False)

    output_dir = job_dir / "results"

    payload = req.model_dump()

    record = {
        "id": job_id,
        "owner_id": user.id,
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": payload,
        "output_dir": str(output_dir.relative_to(ROOT)),
        "command": None,
    }

    job_file = job_dir / "job.json"

    job_file.write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        str(ROOT / "run_llm_suite.py"),
        "--model",
        req.model,
        "--output",
        str(output_dir),
        "--methods",
        *req.methods,
        "--seed",
        str(req.seed),
        "--epochs",
        str(req.epochs),
        "--warmup-epochs",
        str(req.warmup_epochs),
        "--learning-rate",
        str(req.learning_rate),
        "--max-length",
        str(req.max_length),
        "--max-new-tokens",
        str(req.max_new_tokens),
        "--context-sizes",
        *map(str, req.context_sizes),
        "--device",
        req.device,
    ]

    if req.full_niah:
        cmd.append("--full-niah")

    try:
        log = (job_dir / "runner.log").open(
            "w",
            encoding="utf-8",
        )

        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
            },
        )

        with _LOCK:
            _PROCESSES[job_id] = proc

        record.update(
            {
                "status": "running",
                "pid": proc.pid,
                "command": cmd,
            }
        )

        job_file.write_text(
            json.dumps(record, indent=2),
            encoding="utf-8",
        )

    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "message": str(exc),
            }
        )

        job_file.write_text(
            json.dumps(record, indent=2),
            encoding="utf-8",
        )

        raise HTTPException(
            status_code=500,
            detail=f"Could not start LLM runner: {exc}",
        )

    return _read(job_id, user)


@router.get("")
def list_jobs(
    user: User = Depends(get_current_user),
):
    JOB_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    items = []

    for path in JOB_ROOT.iterdir():
        if path.is_dir() and (path / "job.json").exists():
            try:
                items.append(_read(path.name, user))
            except HTTPException as exc:
                # Jobs belonging to other users, and legacy jobs without
                # ownership information, are simply excluded from the list.
                if exc.status_code in {403, 404}:
                    continue
                raise
            except (ValueError, OSError):
                continue

    return sorted(
        items,
        key=lambda x: x.get("created_at", ""),
        reverse=True,
    )[:50]


@router.get("/{job_id}")
def get_job(
    job_id: str,
    user: User = Depends(get_current_user),
):
    if not all(c in "0123456789abcdef" for c in job_id) or len(job_id) != 32:
        raise HTTPException(
            status_code=404,
            detail="LLM experiment not found",
        )

    return _read(job_id, user)


@router.get("/{job_id}/results")
def get_job_results(
    job_id: str,
    user: User = Depends(get_current_user),
):
    job = _read(job_id, user)

    folder = JOB_ROOT / job_id / "results"

    files = []

    if folder.exists():
        for p in folder.rglob("*.json"):
            if p.is_file():
                try:
                    data = json.loads(
                        p.read_text(encoding="utf-8")
                    )

                    files.append(
                        {
                            "path": str(p.relative_to(folder)),
                            "data": data,
                        }
                    )

                except (ValueError, OSError):
                    files.append(
                        {
                            "path": str(p.relative_to(folder)),
                            "unavailable": True,
                        }
                    )

    return {
        "job": job,
        "files": files,
    }
