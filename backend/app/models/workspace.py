import uuid, datetime
from sqlalchemy import Column, String, DateTime, JSON, ForeignKey, Text
from app.database import Base

def uid(): return str(uuid.uuid4())

class Project(Base):
    __tablename__ = 'projects'
    id = Column(String, primary_key=True, default=uid)
    name = Column(String, nullable=False)
    owner_id = Column(String, ForeignKey('users.id'), nullable=True, index=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.now(datetime.timezone.utc))
    settings = Column(JSON, default=dict)

class Asset(Base):
    __tablename__ = 'assets'
    id = Column(String, primary_key=True, default=uid)
    project_id = Column(String, ForeignKey('projects.id'), nullable=False, index=True)
    kind = Column(String, nullable=False)  # model|dataset|forget_set|retain_set|test_set|reference
    name = Column(String, nullable=False)
    version = Column(String, default='v1')
    path = Column(String, nullable=False)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.datetime.now(datetime.timezone.utc))

class Job(Base):
    __tablename__ = 'jobs'
    id = Column(String, primary_key=True, default=uid)
    project_id = Column(String, ForeignKey('projects.id'), nullable=True, index=True)
    kind = Column(String, nullable=False)
    status = Column(String, default='pending')  # pending|running|complete|failed|cancelled
    progress = Column(JSON, default=dict)
    payload = Column(JSON, default=dict)
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.now(datetime.timezone.utc))
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
