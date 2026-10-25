import uuid, datetime
from sqlalchemy import Column, String, DateTime, JSON, ForeignKey, Text, Boolean
from app.database import Base

def uid(): return str(uuid.uuid4())

class User(Base):
    __tablename__='users'
    id=Column(String,primary_key=True,default=uid)
    email=Column(String,unique=True,index=True,nullable=False)
    password_hash=Column(String,nullable=False)
    role=Column(String,default='owner',nullable=False)
    created_at=Column(DateTime,default=datetime.datetime.now(datetime.timezone.utc))

class AuditEvent(Base):
    __tablename__='audit_events'
    id=Column(String,primary_key=True,default=uid)
    project_id=Column(String,ForeignKey('projects.id'),nullable=True,index=True)
    audit_id=Column(String,ForeignKey('audits.id'),nullable=True,index=True)
    job_id=Column(String,ForeignKey('jobs.id'),nullable=True,index=True)
    event_type=Column(String,nullable=False)
    message=Column(Text,nullable=False)
    details=Column(JSON,default=dict)
    created_at=Column(DateTime,default=datetime.datetime.now(datetime.timezone.utc),index=True)

class AuditCertificate(Base):
    __tablename__='audit_certificates'
    id=Column(String,primary_key=True,default=uid)
    audit_id=Column(String,ForeignKey('audits.id'),unique=True,index=True,nullable=False)
    certificate_hash=Column(String,unique=True,index=True,nullable=False)
    payload=Column(JSON,nullable=False)
    signature=Column(String,nullable=True)
    created_at=Column(DateTime,default=datetime.datetime.now(datetime.timezone.utc))

class AuditTemplate(Base):
    __tablename__='audit_templates'
    id=Column(String,primary_key=True,default=uid)
    project_id=Column(String,ForeignKey('projects.id'),nullable=False,index=True)
    name=Column(String,nullable=False)
    config=Column(JSON,default=dict)
    created_at=Column(DateTime,default=datetime.datetime.now(datetime.timezone.utc))

class Schedule(Base):
    __tablename__='schedules'
    id=Column(String,primary_key=True,default=uid)
    project_id=Column(String,ForeignKey('projects.id'),nullable=False,index=True)
    name=Column(String,nullable=False)
    cron=Column(String,nullable=False)
    payload=Column(JSON,default=dict)
    enabled=Column(Boolean,default=True,nullable=False)
    last_run_at=Column(DateTime,nullable=True)
    next_run_at=Column(DateTime,nullable=True)
    created_at=Column(DateTime,default=datetime.datetime.now(datetime.timezone.utc))

class AuditShareLink(Base):
    __tablename__='audit_share_links'
    id=Column(String,primary_key=True,default=uid)
    audit_id=Column(String,ForeignKey('audits.id'),nullable=False,index=True)
    token=Column(String,unique=True,index=True,nullable=False)
    created_at=Column(DateTime,default=datetime.datetime.now(datetime.timezone.utc))
    expires_at=Column(DateTime,nullable=True)
    revoked=Column(Boolean,default=False,nullable=False)
