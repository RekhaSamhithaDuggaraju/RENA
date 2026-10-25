"""
RENA backend — ORM models. Mirrors the real dict shape produced by
core.pipeline.run_audit() so persistence is a direct, lossless mapping,
not a reinterpretation.
"""
import uuid
import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base


def gen_uuid():
    return str(uuid.uuid4())


class Audit(Base):
    __tablename__ = "audits"

    id = Column(String, primary_key=True, default=gen_uuid)
    owner_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    project_id = Column(String, ForeignKey("projects.id"), nullable=True, index=True)
    algorithm = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")  # pending|running|complete|failed
    runtime_seconds = Column(Float, nullable=True)
    sisa_shards_retrained = Column(Integer, nullable=True)

    accuracy_retain = Column(Float, nullable=True)
    accuracy_test = Column(Float, nullable=True)
    accuracy_forget = Column(Float, nullable=True)
    accuracy_original_test = Column(Float, nullable=True)

    consensus_verdict = Column(String, nullable=True)
    consensus_score = Column(Float, nullable=True)
    consensus_agreement = Column(Float, nullable=True)
    rena_x_verdict = Column(String, nullable=True)
    rena_x_score = Column(Float, nullable=True)
    authoritative_verdict = Column(String, nullable=True)

    module_results = Column(JSON, nullable=True)          # list[ModuleResult.to_dict()]
    method_trace = Column(JSON, nullable=True)            # PRISM/ECHO method-specific diagnostics
    consensus_detail = Column(JSON, nullable=True)         # ConsensusResult.to_dict()
    diagnosis = Column(JSON, nullable=True)                # diagnose() output
    dataset_summary = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.datetime.now(datetime.timezone.utc))
    completed_at = Column(DateTime, nullable=True)

    def from_pipeline_result(self, result: dict, dataset_summary: dict):
        """Populates this row from a real core.pipeline.run_audit() dict."""
        self.algorithm = result["algorithm"]
        self.status = "complete"
        self.runtime_seconds = result["runtime_seconds"]
        self.sisa_shards_retrained = result.get("sisa_shards_retrained")
        acc = result["accuracy"]
        self.accuracy_retain = acc["retain"]
        self.accuracy_test = acc["test"]
        self.accuracy_forget = acc["forget"]
        self.accuracy_original_test = acc["original_model_test"]
        c = result["consensus"]
        self.consensus_verdict = c["verdict"]
        self.consensus_score = c["consensus_score"]
        self.consensus_agreement = c["agreement"]
        ax = (result.get("diagnosis") or {}).get("advanced_audit") or {}
        self.rena_x_verdict = ax.get("verdict")
        self.rena_x_score = ax.get("certification_score", ax.get("worst_case_score"))
        self.authoritative_verdict = self.rena_x_verdict or self.consensus_verdict
        self.module_results = result["module_results"]
        self.method_trace = result.get("method_trace")
        self.consensus_detail = c
        self.diagnosis = result["diagnosis"]
        self.dataset_summary = dataset_summary
        self.completed_at = datetime.datetime.now(datetime.timezone.utc)
        return self
