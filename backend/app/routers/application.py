import datetime, hashlib, json, secrets
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.audit import Audit
from app.models.production import AuditShareLink, AuditEvent
from app.models.workspace import Asset
from app.auth import current_user, require_audit_access
from app.verdicts import authoritative_verdict, authoritative_score, confidence_and_quality

router = APIRouter(prefix='/application', tags=['application'])

@router.get('/audits/{audit_id}/recommendations')
def recommendations(audit_id: str, db: Session = Depends(get_db), authorization: str|None=Header(default=None)):
    user=current_user(authorization,db)
    a = require_audit_access(audit_id,db,user)
    if not a: raise HTTPException(404, 'Audit not found')
    rec=[]
    retain = a.accuracy_retain
    verdict=authoritative_verdict(a).upper()
    ax=((a.diagnosis or {}).get('advanced_audit') or {})
    forget_progress=float(((ax.get('axes') or {}).get('forget_progress', ax.get('calibration',{}).get('forget_progress',0.0))) or 0.0)
    retain_ratio=float(((ax.get('axes') or {}).get('retain_utility_ratio', ax.get('calibration',{}).get('retain_utility_ratio',1.0))) or 0.0)
    if forget_progress < 0.40:
        rec.append({'priority':'high','title':'Inspect residual forgetting','reason':f'RENA-X forget progress is {forget_progress:.3f}, below the configured pass threshold of 0.400.','action':'Inspect the highest-risk forget samples and consider stronger unlearning.'})
    if retain is not None and retain < 0.9:
        rec.append({'priority':'high','title':'Protect retained utility','reason':f'Retain utility is {retain:.3f}.','action':'Reduce unlearning strength or use a retain-aware recovery step.'})
    if verdict in {'FAIL','REVIEW','INCONCLUSIVE'}:
        rec.append({'priority':'medium','title':'Re-run with a stronger audit profile','reason':f'Current verdict is {verdict}.','action':'Use the Thorough profile and inspect adversarial and held-out-relearning results.'})
    if not rec:
        rec.append({'priority':'low','title':'Audit looks healthy','reason':'No deterministic warning crossed the configured recommendation thresholds.','action':'Keep the certificate and consider a scheduled re-audit after model changes.'})
    return {'audit_id':audit_id,'generated_by':'RENA deterministic rules','recommendations':rec,'confidence_and_quality':confidence_and_quality(a)}

@router.post('/audits/{audit_id}/share')
def create_share(audit_id: str, db: Session = Depends(get_db), authorization: str|None=Header(default=None)):
    user=current_user(authorization,db)
    a=require_audit_access(audit_id,db,user)
    token=secrets.token_urlsafe(24)
    link=AuditShareLink(audit_id=audit_id,token=token)
    db.add(link); db.add(AuditEvent(audit_id=audit_id,event_type='share_created',message='Read-only audit share link created',details={'token_id':link.id})); db.commit(); db.refresh(link)
    return {'id':link.id,'token':token,'created_at':link.created_at.isoformat(),'revoked':False}

@router.post('/shares/{token}/revoke')
def revoke_share(token: str, db: Session = Depends(get_db), authorization: str|None=Header(default=None)):
    user=current_user(authorization,db)
    link=db.query(AuditShareLink).filter(AuditShareLink.token==token).first()
    if not link: raise HTTPException(404,'Share link not found')
    a=require_audit_access(link.audit_id,db,user)
    link.revoked=True; db.commit(); return {'revoked':True,'id':link.id}

@router.get('/shares/{token}')
def read_share(token: str, db: Session = Depends(get_db)):
    link=db.query(AuditShareLink).filter(AuditShareLink.token==token).first()
    if not link or link.revoked or (link.expires_at and link.expires_at < datetime.datetime.now(datetime.timezone.utc)): raise HTTPException(404,'Share link is unavailable')
    a=db.query(Audit).filter(Audit.id==link.audit_id).first()
    if not a: raise HTTPException(404,'Audit not found')
    return {'audit_id':a.id,'algorithm':a.algorithm,'verdict':authoritative_verdict(a),'score':authoritative_score(a),'classical_verdict':a.consensus_verdict,'agreement':a.consensus_agreement,'accuracy':{'retain':a.accuracy_retain,'forget':a.accuracy_forget,'test':a.accuracy_test},'dataset':a.dataset_summary,'module_results':a.module_results,'diagnosis':a.diagnosis,'shared_read_only':True}

@router.get('/audits/{audit_id}/artifacts')
def artifacts(audit_id: str, db: Session = Depends(get_db), authorization: str|None=Header(default=None)):
    user=current_user(authorization,db); a=require_audit_access(audit_id,db,user)
    paths=[]
    if isinstance(a.module_results,dict):
        for key,val in a.module_results.items():
            if isinstance(val,dict):
                for k,v in val.items():
                    if isinstance(v,str) and ('/' in v or v.endswith(('.json','.csv','.png','.pdf'))): paths.append({'name':f'{key}:{k}','path':v})
    return {'audit_id':audit_id,'artifacts':paths,'note':'Only artifacts explicitly recorded by the audit are exposed.'}

@router.get('/audits/{audit_id}/export-bundle')
def export_bundle_manifest(audit_id: str, db: Session = Depends(get_db), authorization: str|None=Header(default=None)):
    user=current_user(authorization,db); a=require_audit_access(audit_id,db,user)
    cq=confidence_and_quality(a)
    payload={'audit_id':a.id,'algorithm':a.algorithm,'verdict':authoritative_verdict(a),'score':authoritative_score(a),'confidence':cq['confidence'],'confidence_label':cq['confidence_label'],'evidence_quality':cq['evidence_quality'],'classical_verdict':a.consensus_verdict,'agreement':a.consensus_agreement,'dataset':a.dataset_summary,'module_results':a.module_results,'diagnosis':a.diagnosis}
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest()
    return {'audit_id':audit_id,'bundle_hash':digest,'manifest':payload,'formats':['json','pdf','certificate']}

@router.get('/presets')
def presets():
    return [
        {'id':'quick','name':'Quick','description':'Fast screening with reduced reference count.','config':{'reference_retrains':5,'profile':'quick'}},
        {'id':'standard','name':'Standard','description':'Balanced audit for routine model checks.','config':{'reference_retrains':10,'profile':'standard'}},
        {'id':'thorough','name':'Thorough','description':'Full verification with adversarial and recovery checks.','config':{'reference_retrains':30,'profile':'thorough'}},
        {'id':'custom','name':'Custom','description':'Choose every audit control explicitly.','config':{'profile':'custom'}}
    ]
