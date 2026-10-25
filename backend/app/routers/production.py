import os, json, hashlib, datetime, threading, hmac

from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.production import User, AuditEvent, AuditCertificate, AuditTemplate, Schedule
from app.models.workspace import Project, Job
from app.models.audit import Audit
from app.auth import hash_password, verify_password, token_for, current_user, require_project_access, require_audit_access
from app.verdicts import authoritative_verdict, authoritative_score, confidence_and_quality

router=APIRouter(prefix='/production',tags=['production'])
CERT_SECRET=os.environ.get('RENA_CERTIFICATE_SECRET')
if os.environ.get('RENA_PRODUCTION','0').lower() in {'1','true','yes'} and not CERT_SECRET:
    raise RuntimeError('RENA_CERTIFICATE_SECRET is required in production')
CERT_SECRET=CERT_SECRET or 'local-development-only-certificate-secret'

def _sign(payload):
    if CERT_SECRET == 'change-me-in-production': raise HTTPException(503,'Certificate signing secret is not configured')
    body=json.dumps(payload,sort_keys=True,default=str).encode()
    return hmac.new(CERT_SECRET.encode(),body,hashlib.sha256).hexdigest()


class AuthIn(BaseModel): email: str; password: str=Field(min_length=8,max_length=256)
class TemplateIn(BaseModel): project_id:str; name:str; config:dict={}
class ScheduleIn(BaseModel): project_id:str; name:str; cron:str; payload:dict={}; enabled:bool=True
class CompareIn(BaseModel): audit_ids:list[str]

@router.post('/auth/register')
def register(req:AuthIn,db:Session=Depends(get_db)):
    if db.query(User).filter(User.email==req.email.lower()).first(): raise HTTPException(409,'Account already exists')
    u=User(email=req.email.lower(),password_hash=hash_password(req.password)); db.add(u); db.commit(); db.refresh(u)
    return {'token':token_for(u),'user':{'id':u.id,'email':u.email,'role':u.role}}

@router.post('/auth/login')
def login(req:AuthIn,db:Session=Depends(get_db)):
    u=db.query(User).filter(User.email==req.email.lower()).first()
    if not u or not verify_password(req.password,u.password_hash): raise HTTPException(401,'Invalid email or password')
    return {'token':token_for(u),'user':{'id':u.id,'email':u.email,'role':u.role}}

@router.get('/auth/me')
def me(authorization:str|None=Header(default=None),db:Session=Depends(get_db)):
    u=current_user(authorization,db)
    return {'authenticated':True,'user':{'id':u.id,'email':u.email,'role':u.role}}

@router.post('/projects/{project_id}/templates',status_code=201)
def create_template(project_id:str,req:TemplateIn,db:Session=Depends(get_db),authorization:str|None=Header(default=None)):
    user=current_user(authorization,db); require_project_access(project_id,db,user)
    p=db.query(Project).filter(Project.id==project_id).first()
    if not p: raise HTTPException(404,'Project not found')
    t=AuditTemplate(project_id=project_id,name=req.name,config=req.config);db.add(t);db.commit();db.refresh(t);return {'id':t.id,'name':t.name,'config':t.config}

@router.get('/projects/{project_id}/templates')
def templates(project_id:str,db:Session=Depends(get_db),authorization:str|None=Header(default=None)):
    user=current_user(authorization,db); require_project_access(project_id,db,user)
    return [{'id':t.id,'name':t.name,'config':t.config,'created_at':t.created_at.isoformat()} for t in db.query(AuditTemplate).filter(AuditTemplate.project_id==project_id).order_by(AuditTemplate.created_at.desc()).all()]

@router.post('/projects/{project_id}/templates/{template_id}/duplicate')
def duplicate_template(project_id:str,template_id:str,db:Session=Depends(get_db),authorization:str|None=Header(default=None)):
    user=current_user(authorization,db); require_project_access(project_id,db,user)
    t=db.query(AuditTemplate).filter(AuditTemplate.id==template_id,AuditTemplate.project_id==project_id).first()
    if not t: raise HTTPException(404,'Template not found')
    n=AuditTemplate(project_id=project_id,name=t.name+' copy',config=dict(t.config or {}));db.add(n);db.commit();db.refresh(n);return {'id':n.id,'name':n.name,'config':n.config}

@router.post('/projects/{project_id}/schedules',status_code=201)
def create_schedule(project_id:str,req:ScheduleIn,db:Session=Depends(get_db),authorization:str|None=Header(default=None)):
    user=current_user(authorization,db); require_project_access(project_id,db,user)
    if not db.query(Project).filter(Project.id==project_id).first(): raise HTTPException(404,'Project not found')
    s=Schedule(project_id=project_id,name=req.name,cron=req.cron,payload=req.payload,enabled=req.enabled);db.add(s);db.commit();db.refresh(s);return {'id':s.id,'name':s.name,'cron':s.cron,'enabled':s.enabled,'payload':s.payload}

@router.get('/projects/{project_id}/schedules')
def schedules(project_id:str,db:Session=Depends(get_db),authorization:str|None=Header(default=None)):
    user=current_user(authorization,db); require_project_access(project_id,db,user)
    return [{'id':s.id,'name':s.name,'cron':s.cron,'enabled':s.enabled,'payload':s.payload,'next_run_at':s.next_run_at.isoformat() if s.next_run_at else None,'last_run_at':s.last_run_at.isoformat() if s.last_run_at else None} for s in db.query(Schedule).filter(Schedule.project_id==project_id).all()]

@router.post('/events')
def add_event(event_type:str,message:str,details:dict|None=None,project_id:str|None=None,audit_id:str|None=None,job_id:str|None=None,db:Session=Depends(get_db),authorization:str|None=Header(default=None)):
    user=current_user(authorization,db)
    if project_id: require_project_access(project_id,db,user)
    if audit_id: require_audit_access(audit_id,db,user)
    e=AuditEvent(event_type=event_type,message=message,details=details or {},project_id=project_id,audit_id=audit_id,job_id=job_id);db.add(e);db.commit();db.refresh(e);return {'id':e.id,'created_at':e.created_at.isoformat()}

@router.get('/audits/{audit_id}/timeline')
def timeline(audit_id:str,db:Session=Depends(get_db),authorization:str|None=Header(default=None)):
    user=current_user(authorization,db); require_audit_access(audit_id,db,user)
    return [{'id':e.id,'event_type':e.event_type,'message':e.message,'details':e.details,'created_at':e.created_at.isoformat()} for e in db.query(AuditEvent).filter(AuditEvent.audit_id==audit_id).order_by(AuditEvent.created_at.asc()).all()]

def _issue_certificate_record(a, db):
    existing=db.query(AuditCertificate).filter(AuditCertificate.audit_id==a.id).first()
    if existing:
        version=str((existing.payload or {}).get('certificate_version',''))
        if version == '3.0' and existing.signature:
            return existing
        db.delete(existing); db.flush()
    verdict=authoritative_verdict(a)
    if verdict != 'PASS':
        raise HTTPException(409,'Canonical certificates are only issued for a RENA-X PASS. FAIL and INCONCLUSIVE audits require remediation or explicit review outside the certificate flow.')
    cq=confidence_and_quality(a)
    import platform
    payload={'certificate_version':'4.0','audit_id':a.id,'algorithm':a.algorithm,'verdict':verdict,'rena_x_score':authoritative_score(a),'classical_verdict':a.consensus_verdict,'score':authoritative_score(a),'agreement':a.consensus_agreement,'accuracy':{'retain':a.accuracy_retain,'forget':a.accuracy_forget,'test':a.accuracy_test},'dataset':a.dataset_summary,'module_results':a.module_results,'diagnosis':a.diagnosis,'created_at':a.created_at.isoformat() if a.created_at else None,'software':'RENA V22','confidence':cq['confidence'],'confidence_label':cq['confidence_label'],'evidence_quality':cq['evidence_quality'],'reproducibility':{'rena_version':'V22','git_commit':os.environ.get('RENA_GIT_COMMIT','unknown'),'python_version':platform.python_version(),'os':platform.platform(),'cpu':platform.processor() or 'unknown','gpu':os.environ.get('RENA_GPU','unknown'),'cuda_version':os.environ.get('RENA_CUDA_VERSION','unknown'),'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dataset_summary':a.dataset_summary,'model_hash':'not_recorded','configuration_hash':hashlib.sha256(json.dumps(a.dataset_summary or {},sort_keys=True,default=str).encode()).hexdigest(),'dependency_lock_hash':hashlib.sha256((open(os.path.join(os.path.dirname(__file__),'../../requirements.txt'),encoding='utf-8').read() if os.path.exists(os.path.join(os.path.dirname(__file__),'../../requirements.txt')) else '').encode()).hexdigest()},'limitations':['Evidence-based audit; not proof of physical parameter erasure.','This certificate validates behavioral evidence under the recorded evaluation protocol; it does not establish physical deletion of model parameters or universal unlearning across unseen architectures.']}
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest()
    signature=_sign(payload)
    payload['signature']=signature
    c=AuditCertificate(audit_id=a.id,certificate_hash=digest,payload=payload,signature=signature)
    db.add(c); db.add(AuditEvent(audit_id=a.id,event_type='certificate_issued',message='Canonical signed RENA-X certificate issued',details={'hash':digest,'verdict':verdict})); db.flush()
    return c

@router.post('/audits/{audit_id}/certificate')
def issue_certificate(audit_id:str,db:Session=Depends(get_db),authorization:str|None=Header(default=None)):
    user=current_user(authorization,db); a=require_audit_access(audit_id,db,user)
    c=_issue_certificate_record(a,db); db.commit(); db.refresh(c)
    return {'id':c.id,'certificate_hash':c.certificate_hash,'payload':c.payload}

@router.get('/certificates/{certificate_id}/pdf')
def certificate_pdf(certificate_id: str, db: Session = Depends(get_db), authorization: str | None = Header(default=None)):
    from fastapi.responses import FileResponse
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.barcode.qr import QrCodeWidget
    user=current_user(authorization,db)
    c=db.query(AuditCertificate).filter(AuditCertificate.id==certificate_id).first()
    if not c: raise HTTPException(404,'Certificate not found')
    require_audit_access(c.audit_id,db,user)
    payload=c.payload or {}
    out=f'/tmp/rena_certificate_{certificate_id}.pdf'
    styles=getSampleStyleSheet()
    title=ParagraphStyle('certTitle',parent=styles['Title'],fontSize=24,leading=28,textColor=colors.HexColor('#3A2E6E'))
    h=ParagraphStyle('certH',parent=styles['Heading2'],textColor=colors.HexColor('#3A2E6E'),spaceBefore=14,spaceAfter=7)
    body=styles['BodyText']
    doc=SimpleDocTemplate(out,pagesize=letter,leftMargin=48,rightMargin=48,topMargin=42,bottomMargin=42)
    story=[Paragraph('RENA-X',title),Paragraph('VERIFIED AUDIT CERTIFICATE',styles['Heading2']),Spacer(1,8)]
    story += [Paragraph(f"<b>{payload.get('verdict','INCONCLUSIVE')}</b> — {payload.get('confidence_label','UNKNOWN')} CONFIDENCE", styles['Heading1']),Spacer(1,10)]
    rows=[['Certificate ID',c.id],['Certificate Hash',c.certificate_hash],['Audit ID',c.audit_id],['RENA Version',payload.get('software','RENA V22')],['Evidence Quality',payload.get('evidence_quality','UNKNOWN')],['Score',str(payload.get('rena_x_score','—'))],['Timestamp',str(payload.get('created_at','—'))]]
    t=Table(rows,colWidths=[1.6*72,5.0*72]); t.setStyle(TableStyle([('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'),('TEXTCOLOR',(0,0),(0,-1),colors.HexColor('#6B7280')),('GRID',(0,0),(-1,-1),0.3,colors.HexColor('#DDDDDD')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7)])); story += [t]
    story += [Paragraph('MODEL / AUDIT SCOPE',h),Paragraph(str(payload.get('dataset',{})),body),Paragraph('VERIFICATION',h),Paragraph(f"RENA-X score: {payload.get('rena_x_score','—')}<br/>Confidence: {payload.get('confidence_label','UNKNOWN')} ({payload.get('confidence','—')})<br/>Evidence quality: {payload.get('evidence_quality','UNKNOWN')}",body)]
    story += [Paragraph('LIMITATIONS',h),Paragraph('This certificate validates behavioral evidence under the recorded evaluation protocol. It does not establish physical deletion of model parameters, perfect information deletion, or universal unlearning across unseen datasets or architectures.',body)]
    qr=QrCodeWidget(f"{os.environ.get('RENA_PUBLIC_BASE_URL','http://localhost:3000')}/verify/{c.certificate_hash}"); qr.barWidth=100; qr.barHeight=100; d=Drawing(110,110); d.add(qr); story += [Spacer(1,10),d,Paragraph('Scan to verify certificate integrity.',styles['BodyText'])]
    doc.build(story)
    return FileResponse(out,media_type='application/pdf',filename=f'rena_certificate_{certificate_id}.pdf')


@router.get('/certificates/verify/{certificate_hash}')
def verify_certificate(certificate_hash:str,db:Session=Depends(get_db)):
    c=db.query(AuditCertificate).filter(AuditCertificate.certificate_hash==certificate_hash).first()
    if not c:return {'valid':False,'reason':'Certificate not found'}
    stored=dict(c.payload or {}); signature=stored.pop('signature',None)
    actual=hashlib.sha256(json.dumps(stored,sort_keys=True,default=str).encode()).hexdigest()
    signed=_sign(stored) if signature else None
    valid_hash = actual == certificate_hash
    valid_signature = bool(signature) and hmac.compare_digest(signature, signed or '')
    valid = valid_hash and valid_signature
    payload = c.payload or {}
    public = {
        'certificate_version': payload.get('certificate_version'),
        'certificate_id': c.id,
        'certificate_hash': certificate_hash,
        'audit_id': c.audit_id,
        'verdict': payload.get('verdict'),
        'rena_x_score': payload.get('rena_x_score'),
        'confidence': payload.get('confidence'),
        'confidence_label': payload.get('confidence_label'),
        'evidence_quality': payload.get('evidence_quality'),
        'created_at': payload.get('created_at'),
        'software': payload.get('software'),
        'limitations': payload.get('limitations', []),
    }
    return {'valid':valid,'signature_valid':valid_signature,'hash_valid':valid_hash,'certificate_hash':certificate_hash,'audit_id':c.audit_id,'public':public,'integrity':'match' if valid_hash else 'mismatch'}

@router.get('/verify/{certificate_hash}')
def public_verify_certificate(certificate_hash: str, db: Session = Depends(get_db)):
    return verify_certificate(certificate_hash, db)


@router.post('/ci/gate/{audit_id}')
def ci_gate(audit_id:str,policy:str='strict',db:Session=Depends(get_db),authorization:str|None=Header(default=None)):
    user=current_user(authorization,db); require_audit_access(audit_id,db,user)
    a=db.query(Audit).filter(Audit.id==audit_id).first()
    if not a:raise HTTPException(404,'Audit not found')
    verdict=authoritative_verdict(a).upper(); passed=verdict == 'PASS'
    exit_code=0 if passed else 1
    return {'audit_id':audit_id,'policy':policy,'verdict':authoritative_verdict(a),'classical_verdict':a.consensus_verdict,'passed':passed,'exit_code':exit_code,'reason':'Deployment gate passed' if passed else 'Manual review or remediation required'}

@router.get('/compare')
def compare(audit_ids:str,db:Session=Depends(get_db),authorization:str|None=Header(default=None)):
    user=current_user(authorization,db)
    ids=[x.strip() for x in audit_ids.split(',') if x.strip()]; rows=db.query(Audit).filter(Audit.id.in_(ids)).all()
    for _audit in rows: require_audit_access(_audit.id,db,user)
    if not rows:return {'audits':[],'ranking':[]}
    out=[{'id':a.id,'algorithm':a.algorithm,'verdict':authoritative_verdict(a),'score':authoritative_score(a),'agreement':a.consensus_agreement,'retain':a.accuracy_retain,'forget':a.accuracy_forget,'runtime':a.runtime_seconds} for a in rows]
    return {'audits':out,'ranking':sorted(out,key=lambda x:(x['score'] if x['score'] is not None else -1),reverse=True)}

@router.get('/notifications/health')
def notification_health(authorization:str|None=Header(default=None),db:Session=Depends(get_db)):
    current_user(authorization,db)
    return {'webhook_configured':bool(os.environ.get('RENA_WEBHOOK_URL')),'email_configured':bool(os.environ.get('RENA_SMTP_HOST')),'message':'Notifications are optional; configure webhook/SMTP environment variables to enable delivery.'}
