import os, hmac, hashlib, base64, json, time, secrets
from fastapi import Header, HTTPException, Depends
from sqlalchemy.orm import Session
from app.models.production import User
from app.models.workspace import Project
from app.models.audit import Audit
from app.database import get_db

SECRET_VALUE=os.environ.get('RENA_AUTH_SECRET')
if os.environ.get('RENA_PRODUCTION','0').lower() in {'1','true','yes'} and not SECRET_VALUE:
    raise RuntimeError('RENA_AUTH_SECRET is required in production')
SECRET_VALUE=SECRET_VALUE or 'local-development-only-secret'
SECRET=SECRET_VALUE.encode()


def hash_password(password:str)->str:
    salt=secrets.token_bytes(16); dk=hashlib.pbkdf2_hmac('sha256',password.encode(),salt,240000)
    return base64.b64encode(salt+dk).decode()


def verify_password(password:str, encoded:str)->bool:
    raw=base64.b64decode(encoded.encode()); salt,expected=raw[:16],raw[16:]
    actual=hashlib.pbkdf2_hmac('sha256',password.encode(),salt,240000)
    return hmac.compare_digest(actual,expected)


def token_for(user:User)->str:
    payload={'sub':user.id,'email':user.email,'role':user.role,'exp':int(time.time())+86400}
    body=base64.urlsafe_b64encode(json.dumps(payload,separators=(',',':')).encode()).decode().rstrip('=')
    sig=hmac.new(SECRET,body.encode(),hashlib.sha256).hexdigest()
    return body+'.'+sig


def user_from_token(token:str,db:Session):
    try:
        body,sig=token.split('.',1)
        if not hmac.compare_digest(hmac.new(SECRET,body.encode(),hashlib.sha256).hexdigest(),sig): raise ValueError
        payload=json.loads(base64.urlsafe_b64decode(body+'='*(-len(body)%4)))
        if payload['exp']<int(time.time()): raise ValueError
        user=db.query(User).filter(User.id==payload['sub']).first()
        if not user: raise ValueError
        return user
    except Exception:
        raise HTTPException(401,'Invalid or expired authentication token')


def current_user(authorization:str|None,db:Session):
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(401,'Authentication required')
    return user_from_token(authorization[7:],db)


def require_project_access(project_id:str, db:Session, user:User):
    project=db.query(Project).filter(Project.id==project_id).first()
    if not project: raise HTTPException(404,'Project not found')
    if project.owner_id != user.id: raise HTTPException(403,'Not authorized for this project')
    return project


def require_audit_access(audit_id:str, db:Session, user:User):
    audit=db.query(Audit).filter(Audit.id==audit_id).first()
    if not audit: raise HTTPException(404,'Audit not found')
    if audit.owner_id == user.id: return audit
    if audit.project_id:
        project=db.query(Project).filter(Project.id==audit.project_id).first()
        if project and project.owner_id == user.id: return audit
    raise HTTPException(403,'Not authorized for this audit')


def require_certificate_access(certificate, db:Session, user:User):
    return require_audit_access(certificate.audit_id, db, user)


def get_current_user(authorization: str | None = Header(default=None), db: Session = Depends(get_db)):
    """Reusable authentication dependency for resource-bearing endpoints."""
    return current_user(authorization, db)


def require_project_access_dep(project_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return require_project_access(project_id, db, user)


def require_audit_access_dep(audit_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return require_audit_access(audit_id, db, user)
