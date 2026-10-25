"""
RENA backend — /audits router. Calls the REAL core pipeline (core.pipeline),
the same one proven to work via the test suite in /tests. No mocked results.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # reach /core

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Header
from sqlalchemy.orm import Session
import numpy as np

from app.database import get_db
from app.models.audit import Audit
from app.schemas.audit import AuditCreateRequest, AuditResponse
from app.auth import current_user, require_audit_access, require_project_access

from core.dataset import build_split
from core.model import train_base_model
from core.pipeline import run_audit

router = APIRouter(prefix="/audits", tags=["audits"])

_original_model_cache = {}  # keyed by split configuration


def _get_or_train_base(forget_fraction: float, split_seed: int):
    cache_key = (float(forget_fraction), int(split_seed))
    if cache_key not in _original_model_cache:
        split = build_split(forget_fraction=forget_fraction, random_state=split_seed)
        X_train = np.vstack([split.X_retain, split.X_forget])
        y_train = np.concatenate([split.y_retain, split.y_forget])
        model = train_base_model(X_train, y_train)
        _original_model_cache[cache_key] = (split, model)
    return _original_model_cache[cache_key]


@router.post("", response_model=AuditResponse, status_code=201)
def create_audit(req: AuditCreateRequest, db: Session = Depends(get_db), authorization: str|None=Header(default=None)):
    user=current_user(authorization,db)
    if req.project_id:
        require_project_access(req.project_id, db, user)
    split, original_model = _get_or_train_base(req.forget_fraction, req.split_seed)
    sample_ids = None
    if req.algorithm == "sisa":
        n_train = len(split.X_retain) + len(split.X_forget)
        sample_ids = np.arange(n_train)

    try:
        result = run_audit(req.algorithm, split, original_model, sample_ids=sample_ids, reference_count=req.reference_retrains, reference_seed=req.reference_seed, split_seed=req.split_seed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audit pipeline failed: {e}")

    dataset_summary = {**split.summary(), "dataset": "sklearn_digits", "split_seed": req.split_seed,
                       "forget_fraction": req.forget_fraction, "base_model_seed": 42,
                       "reference_retrains": req.reference_retrains, "reference_seed": req.reference_seed,
                       "split_strategy": "stratified_train_test_then_retain_forget",
                       "note": "RENA-X uses independent retain-only retrains as an empirical reference distribution. The reference ensemble is held fixed for this audit."}
    dataset_summary["provenance"] = result.get("provenance")
    audit = Audit().from_pipeline_result(result, dataset_summary)
    audit.owner_id = user.id
    audit.project_id = req.project_id
    db.add(audit)
    db.commit()
    db.refresh(audit)
    return audit


@router.get("", response_model=list[AuditResponse])
def list_audits(db: Session = Depends(get_db), limit: int = 50, authorization: str|None=Header(default=None)):
    user=current_user(authorization,db)
    q=db.query(Audit).filter(Audit.owner_id==user.id)
    return q.order_by(Audit.created_at.desc()).limit(limit).all()


@router.get("/{audit_id}", response_model=AuditResponse)
def get_audit(audit_id: str, db: Session = Depends(get_db), authorization: str|None=Header(default=None)):
    user=current_user(authorization,db)
    audit = require_audit_access(audit_id, db, user)
    return audit


@router.get("/{audit_id}/report.pdf")
def get_audit_report(audit_id: str, db: Session = Depends(get_db), authorization: str|None=Header(default=None)):
    from fastapi.responses import FileResponse
    from reports.generate_pdf import generate_report

    user=current_user(authorization,db)
    audit=require_audit_access(audit_id,db,user)

    result = {
        "algorithm": audit.algorithm,
        "runtime_seconds": audit.runtime_seconds,
        "sisa_shards_retrained": audit.sisa_shards_retrained,
        "accuracy": {
            "retain": audit.accuracy_retain, "test": audit.accuracy_test,
            "forget": audit.accuracy_forget, "original_model_test": audit.accuracy_original_test,
        },
        "module_results": audit.module_results,
        "consensus": audit.consensus_detail,
        "diagnosis": audit.diagnosis,
        "method_trace": audit.method_trace,
        "dataset_summary": audit.dataset_summary,
    }
    path = f"/tmp/rena_report_{audit_id}.pdf"
    generate_report(result, audit.dataset_summary, path, audit_id=audit.id)
    return FileResponse(path, media_type="application/pdf", filename=f"rena_audit_{audit_id}.pdf")


@router.get("/{audit_id}/report.json")
def get_audit_json_report(audit_id: str, db: Session = Depends(get_db), authorization: str|None=Header(default=None)):
    """Return a portable, machine-readable audit record with evidence and provenance."""
    user=current_user(authorization,db)
    audit=require_audit_access(audit_id,db,user)
    return {
        "id": audit.id, "algorithm": audit.algorithm, "status": audit.status,
        "created_at": audit.created_at.isoformat() if audit.created_at else None,
        "completed_at": audit.completed_at.isoformat() if audit.completed_at else None,
        "runtime_seconds": audit.runtime_seconds,
        "sisa_shards_retrained": audit.sisa_shards_retrained,
        "accuracy": {"retain": audit.accuracy_retain, "test": audit.accuracy_test,
                     "forget": audit.accuracy_forget, "original_model_test": audit.accuracy_original_test},
        "consensus": audit.consensus_detail, "diagnosis": audit.diagnosis,
        "module_results": audit.module_results, "method_trace": audit.method_trace,
        "dataset_summary": audit.dataset_summary,
        "interpretation_note": "Audit metrics are behavioral/statistical evidence and do not prove physical parameter erasure."
    }
