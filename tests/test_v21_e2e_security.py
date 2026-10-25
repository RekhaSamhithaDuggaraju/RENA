import os, sys
from pathlib import Path
os.environ['RENA_AUTH_SECRET']='v21-test-secret'
os.environ['RENA_CERTIFICATE_SECRET']='v21-cert-secret'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))

from fastapi.testclient import TestClient
from app.main import app
from app.database import SessionLocal
from app.models.audit import Audit


def auth(client,email,password):
    r=client.post('/production/auth/register',json={'email':email,'password':password})
    assert r.status_code in (200,201), r.text
    return r.json()['token']


def test_cross_user_project_and_audit_isolation():
    with TestClient(app) as client:
        token_a=auth(client,'a@example.com','password123')
        token_b=auth(client,'b@example.com','password123')
        pa=client.post('/workspace/projects',headers={'Authorization':f'Bearer {token_a}'},json={'name':'A'}).json()['id']
        rb=client.get('/workspace/projects',headers={'Authorization':f'Bearer {token_b}'})
        assert rb.status_code==200 and rb.json()==[]

        db=SessionLocal()
        a=Audit(algorithm='ground_truth_retraining',status='complete',owner_id=None,project_id=pa,rena_x_verdict='PASS',rena_x_score=0.95,authoritative_verdict='PASS',module_results=[],diagnosis={},dataset_summary={})
        db.add(a); db.commit(); db.refresh(a); aid=a.id; db.close()
        forbidden=client.get(f'/audits/{aid}',headers={'Authorization':f'Bearer {token_b}'})
        assert forbidden.status_code==403
        allowed=client.get(f'/audits/{aid}',headers={'Authorization':f'Bearer {token_a}'})
        assert allowed.status_code==200


def test_legacy_certificate_endpoint_is_deprecated_and_auth_is_required():
    with TestClient(app) as client:
        r=client.get('/workspace/audit/nope/certificate')
        assert r.status_code==410
        assert 'LEGACY/DEPRECATED' in r.json()['detail']
        assert client.get('/workspace/projects').status_code==401
        assert client.get('/audits').status_code==401


def test_canonical_signed_certificate_and_public_verification():
    with TestClient(app) as client:
        token=auth(client,'cert@example.com','password123')
        pid=client.post('/workspace/projects',headers={'Authorization':f'Bearer {token}'},json={'name':'Cert'}).json()['id']
        db=SessionLocal()
        a=Audit(algorithm='ground_truth_retraining',status='complete',owner_id=None,project_id=pid,rena_x_verdict='PASS',rena_x_score=0.96,authoritative_verdict='PASS',module_results=[],diagnosis={'advanced_audit':{'certification_score':0.96}},dataset_summary={'reference_retrains':20,'split_seed':42})
        db.add(a); db.commit(); db.refresh(a); aid=a.id; db.close()
        r=client.post(f'/production/audits/{aid}/certificate',headers={'Authorization':f'Bearer {token}'})
        assert r.status_code==200, r.text
        data=r.json(); assert data['payload']['certificate_version']=='4.0'; assert data['payload']['verdict']=='PASS'; assert data['payload']['signature']
        v=client.get(f"/production/certificates/verify/{data['certificate_hash']}")
        assert v.status_code==200 and v.json()['valid'] is True


def test_asset_allowlist_and_traversal_protection():
    with TestClient(app) as client:
        token=auth(client,'asset@example.com','password123')
        h={'Authorization':f'Bearer {token}'}
        pid=client.post('/workspace/projects',headers=h,json={'name':'Assets'}).json()['id']
        good=client.post(f'/workspace/projects/{pid}/assets',headers=h,data={'kind':'model','name':'model','version':'v1'},files={'file':('model.json',b'{"format":"rena-model-v1"}','application/json')})
        assert good.status_code==201, good.text
        bad=client.post(f'/workspace/projects/{pid}/assets',headers=h,data={'kind':'../../tmp/evil','name':'evil'},files={'file':('evil.json',b'{}','application/json')})
        assert bad.status_code==422
