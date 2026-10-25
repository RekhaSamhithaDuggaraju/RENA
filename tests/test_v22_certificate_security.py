import os, sys, json, hmac, hashlib
from pathlib import Path
os.environ['RENA_AUTH_SECRET']='v22-test-auth'
os.environ['RENA_CERTIFICATE_SECRET']='v22-test-cert'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from fastapi.testclient import TestClient
from app.main import app
from app.database import SessionLocal
from app.models.audit import Audit
from app.models.production import AuditCertificate
from app.verdicts import authoritative_verdict

def auth(c,e):
    return c.post('/production/auth/register',json={'email':e,'password':'password123'}).json()['token']

def test_inconclusive_cannot_generate_certificate():
    with TestClient(app) as c:
        t=auth(c,'inc@example.com'); h={'Authorization':f'Bearer {t}'}
        pid=c.post('/workspace/projects',headers=h,json={'name':'inc'}).json()['id']
        db=SessionLocal(); a=Audit(algorithm='ground_truth_retraining',status='complete',project_id=pid,rena_x_verdict='INCONCLUSIVE',rena_x_score=.52,authoritative_verdict='INCONCLUSIVE',diagnosis={'advanced_audit':{'verdict':'REVIEW'}},dataset_summary={}); db.add(a); db.commit(); db.refresh(a); aid=a.id; db.close()
        r=c.post(f'/production/audits/{aid}/certificate',headers=h); assert r.status_code==409

def test_certificate_tampering_is_detected():
    with TestClient(app) as c:
        t=auth(c,'tamper@example.com'); h={'Authorization':f'Bearer {t}'}
        pid=c.post('/workspace/projects',headers=h,json={'name':'tamper'}).json()['id']
        db=SessionLocal(); a=Audit(algorithm='ground_truth_retraining',status='complete',project_id=pid,rena_x_verdict='PASS',rena_x_score=.96,authoritative_verdict='PASS',diagnosis={'advanced_audit':{'verdict':'PASS','certification_score':.96}},dataset_summary={'reference_retrains':20}); db.add(a); db.commit(); db.refresh(a); aid=a.id; db.close()
        r=c.post(f'/production/audits/{aid}/certificate',headers=h); data=r.json(); ch=data['certificate_hash']
        db=SessionLocal(); cert=db.query(AuditCertificate).filter(AuditCertificate.certificate_hash==ch).first(); cert.payload=dict(cert.payload); cert.payload['rena_x_score']=0.01; db.commit(); db.close()
        v=c.get(f'/production/certificates/verify/{ch}').json(); assert v['valid'] is False and v['signature_valid'] is False


def test_authoritative_verdict_states_cover_pass_fail_inconclusive():
    from types import SimpleNamespace
    from app.verdicts import authoritative_verdict
    for raw, expected in [('PASS','PASS'),('FAIL','FAIL'),('REVIEW','INCONCLUSIVE'),('Ambiguous','INCONCLUSIVE')]:
        assert authoritative_verdict(SimpleNamespace(authoritative_verdict=raw, rena_x_verdict=None, consensus_verdict='PASS', diagnosis={})) == expected


def test_certificate_signature_changes_when_payload_changes():
    from app.routers.production import _sign
    payload={'certificate_version':'4.0','verdict':'PASS','score':0.9}
    sig=_sign(payload)
    modified=dict(payload,score=0.1)
    assert not hmac.compare_digest(sig,_sign(modified))
    assert not hmac.compare_digest(sig,_sign(dict(payload,verdict='FAIL')))
    assert not hmac.compare_digest(sig,_sign(dict(payload,certificate_version='4.1')))


def test_public_certificate_verification_exposes_only_safe_fields():
    with TestClient(app) as c:
        t=auth(c,'public@example.com'); h={'Authorization':f'Bearer {t}'}
        pid=c.post('/workspace/projects',headers=h,json={'name':'public'}).json()['id']
        db=SessionLocal(); a=Audit(algorithm='ground_truth_retraining',status='complete',project_id=pid,rena_x_verdict='PASS',rena_x_score=.95,authoritative_verdict='PASS',diagnosis={'advanced_audit':{'verdict':'PASS','certification_score':.95}},dataset_summary={'reference_retrains':20}); db.add(a); db.commit(); db.refresh(a); aid=a.id; db.close()
        data=c.post(f'/production/audits/{aid}/certificate',headers=h).json(); v=c.get('/production/verify/'+data['certificate_hash']).json()
        assert v['valid'] is True
        assert 'module_results' not in v.get('public',{}) and 'diagnosis' not in v.get('public',{})
