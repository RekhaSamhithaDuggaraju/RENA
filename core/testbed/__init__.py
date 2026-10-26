"""Ground-truth unlearning evaluation testbed.

Enables reproducible simulation of ground-truth conditions:
- TRULY_FORGOTTEN (Oracle unlearned / unexposed reference)
- TRULY_RETAINED (No unlearning)
- SUPPRESSED_NOT_FORGOTTEN (Refusal fine-tuned / system-prompt suppressed)
- PARTIALLY_FORGOTTEN (Partial parameter rollback / gradient ascent)
- OVER_UNLEARNED (Utility collapsed)
"""
from core.testbed.ground_truth_testbed import (
    GroundTruthCondition,
    SyntheticCanaryDataset,
    MockTargetModel,
    NaiveAuditor,
    ControlledAuditor,
    run_all_experiments,
)

__all__ = [
    "GroundTruthCondition",
    "SyntheticCanaryDataset",
    "MockTargetModel",
    "NaiveAuditor",
    "ControlledAuditor",
    "run_all_experiments",
]
