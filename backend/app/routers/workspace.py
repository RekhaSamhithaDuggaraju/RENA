import os, uuid, datetime, json, hashlib, threading
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks, Header
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.workspace import Project, Asset, Job
from app.schemas.workspace import ProjectCreate, JobCreate
from app.auth import current_user, require_project_access, require_audit_access
from app.models.production import AuditEvent

router = APIRouter(prefix='/workspace', tags=['workspace'])
DATA_ROOT = Path(os.environ.get('RENA_DATA_ROOT', '/tmp/rena_data')).resolve()
MAX_UPLOAD_MB = int(os.environ.get('RENA_MAX_UPLOAD_MB', '256'))
ALLOWED_EXT = {'.pt','.pth','.onnx','.safetensors','.npz','.npy','.csv','.json','.parquet','.zip'}
ALLOWED_KINDS = {'model','dataset','forget_set','retain_set','test_set','reference'}


def _safe_project(pid: str):
    p = DATA_ROOT / 'projects' / pid
    p.mkdir(parents=True, exist_ok=True)
    return p

def _asset_meta(path: Path, kind: str):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return {'sha256': h.hexdigest(), 'size_bytes': path.stat().st_size, 'extension': path.suffix.lower(), 'kind': kind}

@router.post('/projects', status_code=201)
def create_project(req: ProjectCreate, db: Session=Depends(get_db), authorization: str|None=Header(default=None)):
    user=current_user(authorization,db)
    p=Project(name=req.name.strip(), description=req.description, settings=req.settings, owner_id=user.id if user else None)
    db.add(p); db.commit(); db.refresh(p)
    return {'id':p.id,'name':p.name,'description':p.description,'settings':p.settings,'created_at':p.created_at.isoformat()}

@router.get('/projects')
def list_projects(db: Session=Depends(get_db), authorization: str|None=Header(default=None)):
    user=current_user(authorization,db)
    q=db.query(Project).filter(Project.owner_id==user.id)
    return [{'id':p.id,'name':p.name,'description':p.description,'settings':p.settings,'created_at':p.created_at.isoformat()} for p in q.order_by(Project.created_at.desc()).all()]

@router.get('/projects/{project_id}')
def get_project(project_id:str, db:Session=Depends(get_db), authorization: str|None=Header(default=None)):
    user=current_user(authorization,db)
    p=require_project_access(project_id,db,user)
    assets=db.query(Asset).filter(Asset.project_id==project_id).order_by(Asset.created_at.desc()).all()
    return {'id':p.id,'name':p.name,'description':p.description,'settings':p.settings,'created_at':p.created_at.isoformat(),'assets':[{'id':a.id,'kind':a.kind,'name':a.name,'version':a.version,'metadata':a.metadata_json,'created_at':a.created_at.isoformat()} for a in assets]}

@router.post('/projects/{project_id}/assets', status_code=201)
async def upload_asset(project_id:str, kind:str=Form(...), name:str=Form(...), version:str=Form('v1'), file:UploadFile=File(...), db:Session=Depends(get_db), authorization: str|None=Header(default=None)):
    user=current_user(authorization,db)
    p=require_project_access(project_id,db,user)
    if kind not in ALLOWED_KINDS: raise HTTPException(422, f'Unsupported asset kind: {kind}')
    ext=Path(file.filename or '').suffix.lower()
    if ext not in ALLOWED_EXT: raise HTTPException(415,f'Unsupported file type: {ext or "none"}')
    root=(_safe_project(project_id)/kind).resolve()
    project_root=_safe_project(project_id).resolve()
    if not root.is_relative_to(project_root): raise HTTPException(422,'Invalid asset path')
    root.mkdir(parents=True, exist_ok=True)
    target=root/(f'{uuid.uuid4().hex}_{Path(file.filename).name}')
    size=0
    with target.open('wb') as out:
        while True:
            chunk=await file.read(1024*1024)
            if not chunk: break
            size += len(chunk)
            if size > MAX_UPLOAD_MB*1024*1024:
                target.unlink(missing_ok=True); raise HTTPException(413,'Upload exceeds configured size limit')
            out.write(chunk)
    meta=_asset_meta(target,kind); meta['original_filename']=Path(file.filename).name
    a=Asset(project_id=project_id,kind=kind,name=name.strip() or Path(file.filename).stem,version=version,path=str(target),metadata_json=meta)
    db.add(a); db.commit(); db.refresh(a)
    return {'id':a.id,'project_id':project_id,'kind':kind,'name':a.name,'version':a.version,'metadata':meta,'created_at':a.created_at.isoformat()}

@router.get('/projects/{project_id}/assets')
def list_assets(project_id:str, db:Session=Depends(get_db), authorization: str|None=Header(default=None)):
    user=current_user(authorization,db)
    require_project_access(project_id,db,user)
    return [{'id':a.id,'kind':a.kind,'name':a.name,'version':a.version,'metadata':a.metadata_json,'created_at':a.created_at.isoformat()} for a in db.query(Asset).filter(Asset.project_id==project_id).order_by(Asset.created_at.desc()).all()]

@router.post('/jobs', status_code=202)
def create_job(req:JobCreate, background:BackgroundTasks, db:Session=Depends(get_db), authorization: str|None=Header(default=None)):
    user=current_user(authorization,db)
    if req.project_id:
        require_project_access(req.project_id,db,user)
    j=Job(project_id=req.project_id,kind=req.kind,payload=req.payload,status='pending',progress={'stage':'queued','percent':0})
    db.add(j); db.commit(); db.refresh(j)
    # Lightweight job lifecycle; domain execution can be attached by kind-specific runners.
    jid=j.id
    background.add_task(_run_job, jid)
    return {'id':jid,'status':'pending','progress':j.progress}

def _run_job(job_id):
    from app.database import SessionLocal
    db=SessionLocal(); j=db.query(Job).filter(Job.id==job_id).first()
    if not j: db.close(); return
    try:
        if j.status == 'cancelled': db.close(); return
        j.status='running'; j.started_at=datetime.datetime.now(datetime.timezone.utc); j.progress={'stage':'preparing','percent':10}; db.add(AuditEvent(project_id=j.project_id,job_id=j.id,event_type='job_started',message='Background job started',details={'kind':j.kind})); db.commit()
        payload=j.payload or {}
        if j.kind=='certificate':
            from app.models.audit import Audit
            from app.routers.production import _issue_certificate_record
            audit_id=payload.get('audit_id')
            if not audit_id: raise ValueError('certificate job requires audit_id')
            a=db.query(Audit).filter(Audit.id==audit_id).first()
            if not a or (j.project_id and a.project_id!=j.project_id): raise ValueError('certificate audit is not in this project')
            j.progress={'stage':'building signed certificate','percent':60}; db.commit()
            c=_issue_certificate_record(a,db); j.result={'certificate_id':c.id,'certificate_hash':c.certificate_hash,'verdict':'PASS','decision_source':'RENA-X'}
        elif j.kind=='comparison':
            from app.models.audit import Audit
            from app.verdicts import authoritative_verdict, authoritative_score
            ids=payload.get('audit_ids',[])
            rows=db.query(Audit).filter(Audit.id.in_(ids)).all()
            if len(rows)!=len(ids): raise ValueError('comparison contains an unknown audit')
            if j.project_id and any(a.project_id!=j.project_id for a in rows): raise ValueError('comparison contains an audit outside this project')
            j.progress={'stage':'comparing RENA-X evidence','percent':60}; db.commit()
            out=[{'id':a.id,'algorithm':a.algorithm,'verdict':authoritative_verdict(a),'score':authoritative_score(a),'retain':a.accuracy_retain,'forget':a.accuracy_forget,'runtime':a.runtime_seconds} for a in rows]
            j.result={'audit_ids':ids,'audits':out,'ranking':sorted(out,key=lambda x:x['score'] if x['score'] is not None else -1,reverse=True),'decision_source':'RENA-X'}
        elif j.kind=='attack':
            j.progress={'stage':'running model-level adversarial suite','percent':25}; db.commit()
            from core.dataset import build_split
            from core.model import train_base_model
            from core.research_suite import adversarial_model_suite
            frac=float(payload.get('forget_fraction',0.1)); seed=int(payload.get('split_seed',42)); refs=int(payload.get('reference_retrains',10))
            result=adversarial_model_suite(seed, frac, refs)
            db.refresh(j)
            if j.status == 'cancelled': db.commit(); db.close(); return
            j.progress={'stage':'persisting attack evidence','percent':90}; db.commit()
            j.result={'profile':payload.get('profile','default'),'status':'complete','evidence':result,'decision_source':'RENA-X'}
        else:
            from core.dataset import build_split
            from core.model import train_base_model
            from core.pipeline import run_audit
            import numpy as np
            algo=payload.get('algorithm','ground_truth_retraining')
            frac=float(payload.get('forget_fraction',0.1)); seed=int(payload.get('split_seed',42))
            refs=int(payload.get('reference_retrains',10)); refseed=int(payload.get('reference_seed',50000))
            j.progress={'stage':'building dataset and original model','percent':20}; db.commit()
            split=build_split(forget_fraction=frac, random_state=seed)
            X=np.vstack([split.X_retain,split.X_forget]); y=np.concatenate([split.y_retain,split.y_forget])
            original=train_base_model(X,y)
            if db.query(Job).filter(Job.id==job_id).first().status == 'cancelled': db.close(); return
            j.progress={'stage':'running RENA audit','percent':45}; db.commit()
            sample_ids=np.arange(len(X)) if algo=='sisa' else None
            result=run_audit(algo,split,original,sample_ids=sample_ids,reference_count=refs,reference_seed=refseed,split_seed=seed)
            db.refresh(j)
            if j.status == 'cancelled': db.commit(); db.close(); return
            j.progress={'stage':'persisting audit','percent':85}; db.commit()
            from app.models.audit import Audit
            summary={**split.summary(),'dataset':'sklearn_digits','split_seed':seed,'forget_fraction':frac,'reference_retrains':refs,'reference_seed':refseed,'job_id':j.id}
            a=Audit().from_pipeline_result(result,summary); a.project_id=j.project_id; a.owner_id=(db.query(Project).filter(Project.id==j.project_id).first().owner_id if j.project_id else None); db.add(a); db.flush()
            from app.verdicts import authoritative_verdict, authoritative_score
            j.result={'audit_id':a.id,'verdict':authoritative_verdict(a),'score':authoritative_score(a),'classical_verdict':a.consensus_verdict,'decision_source':'RENA-X'}
        if j.status == 'cancelled': db.commit(); return
        j.progress={'stage':'complete','percent':100}; j.status='complete'; j.completed_at=datetime.datetime.now(datetime.timezone.utc); db.add(AuditEvent(project_id=j.project_id,job_id=j.id,event_type='job_completed',message='Background job completed',details=j.result or {})); db.commit()
    except Exception as e:
        j.status='failed'; j.error=str(e); j.completed_at=datetime.datetime.now(datetime.timezone.utc); db.add(AuditEvent(project_id=j.project_id,job_id=j.id,event_type='job_failed',message='Background job failed',details={'error':str(e)})); db.commit()
    finally: db.close()

@router.get('/jobs')
def list_jobs(db:Session=Depends(get_db), authorization: str|None=Header(default=None)):
    user=current_user(authorization,db)
    q=db.query(Job).join(Project, Project.id==Job.project_id).filter(Project.owner_id==user.id)
    return [{'id':j.id,'project_id':j.project_id,'kind':j.kind,'status':j.status,'progress':j.progress,'result':j.result,'error':j.error,'created_at':j.created_at.isoformat()} for j in q.order_by(Job.created_at.desc()).limit(100).all()]

@router.get('/jobs/{job_id}')
def get_job(job_id:str, db:Session=Depends(get_db), authorization: str|None=Header(default=None)):
    user=current_user(authorization,db)
    j=db.query(Job).filter(Job.id==job_id).first()
    if not j: raise HTTPException(404,'Job not found')
    if j.project_id: require_project_access(j.project_id,db,user)
    return {'id':j.id,'project_id':j.project_id,'kind':j.kind,'status':j.status,'progress':j.progress,'result':j.result,'error':j.error,'created_at':j.created_at.isoformat(),'completed_at':j.completed_at.isoformat() if j.completed_at else None}

@router.post('/jobs/{job_id}/cancel')
def cancel_job(job_id:str, db:Session=Depends(get_db), authorization: str|None=Header(default=None)):
    user=current_user(authorization,db)
    j=db.query(Job).filter(Job.id==job_id).first()
    if not j: raise HTTPException(404,'Job not found')
    if j.project_id: require_project_access(j.project_id,db,user)
    if j.status in ('pending','running'): j.status='cancelled'; j.progress={'stage':'cancelled','percent':j.progress.get('percent',0)}; db.commit()
    return {'id':j.id,'status':j.status}

@router.get('/audit/{audit_id}/certificate')
def certificate_legacy(audit_id:str):
    raise HTTPException(status_code=410, detail='LEGACY/DEPRECATED certificate endpoint. Use POST /production/audits/{audit_id}/certificate for the canonical signed RENA-X certificate.')


@router.post('/jobs/{job_id}/retry', status_code=202)
def retry_job(job_id:str, background:BackgroundTasks, db:Session=Depends(get_db), authorization: str|None=Header(default=None)):
    user=current_user(authorization,db)
    old=db.query(Job).filter(Job.id==job_id).first()
    if old and old.project_id: require_project_access(old.project_id,db,user)
    if not old: raise HTTPException(404,'Job not found')
    if old.status not in ('failed','cancelled'): raise HTTPException(409,'Only failed or cancelled jobs can be retried')
    j=Job(project_id=old.project_id,kind=old.kind,payload=old.payload,status='pending',progress={'stage':'queued','percent':0})
    db.add(j); db.commit(); db.refresh(j); background.add_task(_run_job,j.id)
    db.add(AuditEvent(project_id=old.project_id,job_id=j.id,event_type='job_retry',message='Job retried from previous execution',details={'previous_job_id':old.id}));db.commit()
    return {'id':j.id,'status':j.status,'previous_job_id':old.id}

@router.get('/projects/{project_id}/events')
def project_events(project_id:str,db:Session=Depends(get_db), authorization: str|None=Header(default=None)):
    user=current_user(authorization,db); require_project_access(project_id,db,user)
    return [{'id':e.id,'event_type':e.event_type,'message':e.message,'details':e.details,'created_at':e.created_at.isoformat()} for e in db.query(AuditEvent).filter(AuditEvent.project_id==project_id).order_by(AuditEvent.created_at.desc()).limit(250).all()]

@router.get('/resources')
def resources(authorization: str|None=Header(default=None), db:Session=Depends(get_db)):
    current_user(authorization,db)
    import shutil
    total,used,free=shutil.disk_usage(DATA_ROOT if DATA_ROOT.exists() else '/')
    return {'upload_limit_mb':MAX_UPLOAD_MB,'disk_free_bytes':free,'disk_used_bytes':used,'disk_total_bytes':total,'max_concurrent_jobs':int(os.environ.get('RENA_MAX_CONCURRENT_JOBS','2')),'job_timeout_seconds':int(os.environ.get('RENA_JOB_TIMEOUT_SECONDS','1800')),'execution_mode':'FastAPI background fallback; use an external worker for strict process isolation in production'}
