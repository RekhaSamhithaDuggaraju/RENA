from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal

class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: Optional[str] = Field(default=None, max_length=2000)
    settings: dict = Field(default_factory=dict)

class ProjectResponse(ProjectCreate):
    id: str
    created_at: str
    model_config = ConfigDict(from_attributes=True)

class AssetResponse(BaseModel):
    id: str; project_id: str; kind: str; name: str; version: str; metadata_json: dict; created_at: str
    model_config = ConfigDict(from_attributes=True)

class JobCreate(BaseModel):
    project_id: Optional[str] = None
    kind: Literal['audit','attack','comparison','certificate']
    payload: dict = Field(default_factory=dict)
