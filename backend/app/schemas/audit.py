from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal
import datetime

Algorithm = Literal["fine_tuning", "gradient_ascent", "ground_truth_retraining", "sisa", "prism", "echo", "neggrad", "salun", "unsir", "scrub"]
Verdict = Literal["Likely Forgotten", "Not Forgotten", "Ambiguous", "Inconclusive"]


class AuditCreateRequest(BaseModel):
    algorithm: Algorithm
    forget_fraction: float = Field(default=0.1, gt=0.0, le=0.5, description="Fraction of training samples assigned to the forget set (0, 0.5].")
    split_seed: int = Field(default=42, ge=0, le=2_147_483_647, description="Seed controlling the stratified train/test and retain/forget split.")
    reference_retrains: int = Field(default=10, ge=3, le=20, description="Independent retain-only retrains used to calibrate RENA-X.")
    reference_seed: int = Field(default=50000, ge=0, le=2_147_483_647, description="Seed base for independent reference retrains.")
    project_id: Optional[str] = Field(default=None, description="Owning project for this audit.")


class ModuleResultSchema(BaseModel):
    module_name: str
    raw_statistic: float
    p_value: Optional[float]
    forget_score: float
    reliability: float
    reliability_reason: str
    detail: dict


class ConsensusSchema(BaseModel):
    verdict: Verdict
    consensus_score: float
    agreement: float
    per_module_contribution: list[dict]
    discounted_modules: list[dict]


class AuditResponse(BaseModel):
    id: str
    algorithm: Algorithm
    status: str
    runtime_seconds: Optional[float]
    sisa_shards_retrained: Optional[int]
    accuracy_retain: Optional[float]
    accuracy_test: Optional[float]
    accuracy_forget: Optional[float]
    accuracy_original_test: Optional[float]
    consensus_verdict: Optional[str]
    consensus_score: Optional[float]
    consensus_agreement: Optional[float]
    rena_x_verdict: Optional[str] = None
    rena_x_score: Optional[float] = None
    authoritative_verdict: Optional[str] = None
    module_results: Optional[list[ModuleResultSchema]]
    method_trace: Optional[dict] = None
    diagnosis: Optional[dict]
    dataset_summary: Optional[dict] = None
    created_at: datetime.datetime
    completed_at: Optional[datetime.datetime]

    model_config = ConfigDict(from_attributes=True)
