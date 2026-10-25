"""LLM unlearning evaluation components for RENA.

The package is intentionally provider/model agnostic. Model training adapters
can be added separately; benchmark generation and scoring must remain usable
without downloading model weights.
"""
from .knowledge import Fact, build_synthetic_knowledge
from .niah import NIAHCase, build_niah_cases, evaluate_niah_case, run_niah_benchmark
from .unlearning import METHODS, TrainConfig, unlearn_model
from .audit_controller import AuditPolicy, AuditDecision, run_adaptive_audit
from .mira import score_memory_items

__all__ = [
    "Fact", "build_synthetic_knowledge", "NIAHCase", "build_niah_cases",
    "evaluate_niah_case", "run_niah_benchmark", "METHODS", "TrainConfig", "unlearn_model",
    "AuditPolicy", "AuditDecision", "run_adaptive_audit", "summarize", "aggregate_suite_summaries",
    "compare_niah_to_reference", "score_memory_items",
]

from .statistics import summarize, aggregate_suite_summaries

from .paired_niah import compare_niah_to_reference
