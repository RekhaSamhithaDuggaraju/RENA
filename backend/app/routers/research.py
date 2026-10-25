"""Research utilities exposed by the local RENA API."""
from __future__ import annotations
from typing import Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from core.llm.mira import score_memory_items

router = APIRouter(prefix="/research", tags=["research-utilities"])

class MIRAScoreRequest(BaseModel):
    items: list[dict[str, Any]] = Field(min_length=1, max_length=100,
        description="Records with id and five normalized MIRA signals in [0, 1].")

@router.post("/mira/score")
def score_mira(req: MIRAScoreRequest):
    """Rank caller-provided memory signals using transparent prototype weights."""
    try:
        return score_memory_items(req.items)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

class ResearchSuiteRequest(BaseModel):
    seeds: list[int] = Field(default_factory=lambda: [7, 42, 2026], min_length=1, max_length=8)
    forget_fraction: float = Field(default=0.10, gt=0, lt=0.5)

@router.post("/suite/run")
def run_research_suite_api(req: ResearchSuiteRequest):
    """Run the reproducible publication suite and return its compact summary."""
    from core.pipeline import run_full_comparison, ALGORITHMS
    from core.research_suite import (bootstrap_ci, rank_methods, verifier_ablation, red_team_evidence,
        threshold_sensitivity, confidence_verdict, cost_quality_profile, verifier_ground_truth_benchmark,
        consensus_ablation, red_team_summary, verifier_power_study)
    runs=[]; details=[]; first_comparison=None
    for seed in req.seeds:
        comparison=run_full_comparison(req.forget_fraction, seed)
        if first_comparison is None: first_comparison=comparison
        for algo in ALGORITHMS:
            r=comparison["results"][algo]
            runs.append({"seed":seed,"algorithm":algo,"rena_x_worst_case":r["diagnosis"]["advanced_audit"]["worst_case_score"],
                         "retain_accuracy":r["accuracy"]["retain"],"forget_accuracy":r["accuracy"]["forget"],
                         "test_accuracy":r["accuracy"]["test"],"runtime_seconds":r["runtime_seconds"]})
            if seed==req.seeds[0]: details.append({"algorithm":algo,"verifier_ablation":verifier_ablation(r),"red_team":red_team_evidence(r)})
    ranking=cost_quality_profile(rank_methods(runs))
    extended = {
        "confidence": {r["algorithm"]: confidence_verdict(first_comparison["results"][r["algorithm"]]) for r in runs if r["seed"]==req.seeds[0]},
        "threshold_sensitivity": {r["algorithm"]: threshold_sensitivity(first_comparison["results"][r["algorithm"]]) for r in runs if r["seed"]==req.seeds[0]},
        "consensus_ablation": {r["algorithm"]: consensus_ablation(first_comparison["results"][r["algorithm"]]) for r in runs if r["seed"]==req.seeds[0]},
        "red_team_summary": {r["algorithm"]: red_team_summary(first_comparison["results"][r["algorithm"]]) for r in runs if r["seed"]==req.seeds[0]},
        "controlled_synthetic_benchmark": verifier_ground_truth_benchmark(),
        "verifier_power_study": verifier_power_study(seeds=req.seeds, forget_fraction=req.forget_fraction),
        "cost_quality": cost_quality_profile(runs),
    }
    return {"config":{"seeds":req.seeds,"forget_fraction":req.forget_fraction},"ranking":ranking,
            "confidence_intervals":{a:{"rena_x":bootstrap_ci([x["rena_x_worst_case"] for x in runs if x["algorithm"]==a]),
                                       "retain_accuracy":bootstrap_ci([x["retain_accuracy"] for x in runs if x["algorithm"]==a])} for a in ALGORITHMS},
            "details":details, "extended":extended}


class VerifierBenchmarkRequest(BaseModel):
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    forget_fraction: float = Field(default=0.10, gt=0, lt=0.5)
    reference_retrains: int = Field(default=10, ge=6, le=20)

@router.post("/verifier-benchmark")
def verifier_benchmark(req: VerifierBenchmarkRequest):
    """Compare calibrated RENA with an explicit ULTRA-style worst-case baseline."""
    try:
        from core.research_suite import verifier_benchmark_matrix
        return verifier_benchmark_matrix(req.seed, req.forget_fraction, req.reference_retrains)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Verifier benchmark failed: {exc}") from exc


class CifarBenchmarkRequest(BaseModel):
    data_root: str = Field(default="data/cifar10")
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    forget_fraction: float = Field(default=0.10, gt=0, lt=0.5)
    reference_retrains: int = Field(default=3, ge=1, le=10)
    epochs: int = Field(default=2, ge=1, le=50)
    max_train_samples: int | None = Field(default=5000, ge=1000, le=50000)
    architecture: str = Field(default="both", pattern="^(cnn|resnet18|both)$")
    device: str = Field(default="cpu", pattern="^(cpu|cuda|cuda:[0-9]+)$")

@router.post("/cifar10/reference-benchmark")
def cifar10_reference_benchmark(req: CifarBenchmarkRequest):
    """Run the real optional CIFAR-10 CNN reference benchmark; no synthetic fallback."""
    try:
        from core.cifar_benchmark import CifarConfig, run_cifar_reference_benchmark
        return run_cifar_reference_benchmark(CifarConfig(data_root=req.data_root,seed=req.seed,forget_fraction=req.forget_fraction,reference_retrains=req.reference_retrains,epochs=req.epochs,max_train_samples=req.max_train_samples,architecture=req.architecture,device=req.device))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"CIFAR-10 benchmark unavailable: {exc}") from exc


class AdversarialLabRequest(BaseModel):
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    forget_fraction: float = Field(default=0.10, gt=0, lt=0.5)
    reference_retrains: int = Field(default=10, ge=3, le=20)

@router.post("/adversarial/run")
def run_adversarial_lab(req: AdversarialLabRequest):
    """Run genuine model-level adversarial controls against a fixed retrain ensemble."""
    try:
        from core.research_suite import adversarial_model_suite
        return adversarial_model_suite(req.seed, req.forget_fraction, req.reference_retrains)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Adversarial lab failed: {exc}") from exc

class CalibrationRobustnessRequest(BaseModel):
    seeds: list[int] = Field(default_factory=lambda: [42, 43, 44], min_length=1, max_length=8)
    forget_fraction: float = Field(default=0.10, gt=0, lt=0.5)
    reference_retrains: int = Field(default=8, ge=4, le=20)

@router.post("/calibration/robustness")
def calibration_robustness(req: CalibrationRobustnessRequest):
    """Run cross-seed calibration robustness checks."""
    try:
        from core.research_suite import multi_seed_calibration_benchmark
        return multi_seed_calibration_benchmark(tuple(req.seeds), req.forget_fraction, req.reference_retrains)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Calibration robustness failed: {exc}") from exc

@router.post("/calibration/controls")
def calibration_controls(req: CalibrationRobustnessRequest):
    """Compare exact retrain, no-op, gradient-ascent, and over-unlearning controls."""
    try:
        from core.research_suite import calibration_control_discrimination
        return calibration_control_discrimination(tuple(req.seeds), req.forget_fraction, req.reference_retrains)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Calibration control study failed: {exc}") from exc
