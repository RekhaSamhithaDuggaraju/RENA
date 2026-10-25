"""Needle-in-a-Haystack benchmark generation and model-agnostic scoring."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Callable, Iterable, Sequence

from .knowledge import Fact


@dataclass(frozen=True)
class NIAHCase:
    case_id: str
    fact_id: str
    target_split: str
    answer: str
    prompt: str
    context_length_chars: int
    insertion_position: str
    prompt_style: str
    context_mode: str = "present"

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _distractor(index: int) -> str:
    return (f"Background record {index:04d}: the regional survey logged ordinary "
            f"weather, transport, and inventory observations with no special designation.")


def _insert_at_position(items: list[str], needle: str, position: str) -> list[str]:
    if position == "beginning":
        index = 0
    elif position == "middle":
        index = len(items) // 2
    elif position == "end":
        index = len(items)
    else:
        raise ValueError("position must be beginning, middle, or end")
    return items[:index] + [needle] + items[index:]


def build_niah_cases(
    facts: Sequence[Fact],
    context_sizes: Sequence[int] = (8, 32),
    positions: Sequence[str] = ("beginning", "middle", "end"),
    prompt_styles: Sequence[str] = ("direct", "paraphrase"),
    context_modes: Sequence[str] = ("present",),
) -> list[NIAHCase]:
    """Build deterministic contexts with a target fact embedded among distractors.

    context_sizes count distractor records, not tokens. This avoids implying a
    tokenizer-independent token length; the exact character length is stored.
    """
    cases: list[NIAHCase] = []
    for fact in facts:
        for size in context_sizes:
            if size < 0:
                raise ValueError("context sizes must be non-negative")
            for position in positions:
                if position not in {"beginning", "middle", "end"}:
                    raise ValueError("position must be beginning, middle, or end")
                for style in prompt_styles:
                    if style not in {"direct", "paraphrase", "completion"}:
                        raise ValueError(f"unsupported prompt style: {style}")
                    for context_mode in context_modes:
                        if context_mode not in {"present", "absent"}:
                            raise ValueError("context_mode must be present or absent")
                        distractors = [_distractor(i) for i in range(size)]
                        if context_mode == "present":
                            context = "\n".join(_insert_at_position(distractors, fact.statement, position))
                        else:
                            context = "\n".join(distractors)
                        if style == "direct":
                            question = f"What is the {fact.attribute} of {fact.subject}?"
                        elif style == "paraphrase":
                            question = f"Identify the {fact.attribute} assigned to {fact.subject}."
                        else:
                            question = f"Complete this statement: The {fact.attribute} of {fact.subject} is"
                        if context_mode == "present":
                            instruction = "Use the supplied records to answer. If the answer is not in the records, say you do not know."
                        else:
                            instruction = "Answer from knowledge learned during training. No external records are supplied."
                        prompt = (instruction + "\n\nRECORDS:\n" + context +
                                  "\n\nQUESTION: " + question + "\nANSWER:")
                        case_id = f"{fact.fact_id}-n{size}-{position}-{style}-{context_mode}"
                        cases.append(NIAHCase(case_id, fact.fact_id, fact.split, fact.value,
                                              prompt, len(context), position, style, context_mode))
    return cases


def evaluate_niah_case(case: NIAHCase, output: str) -> dict:
    """Score exact normalized answer occurrence; retain output for manual audit.

    This deliberately conservative lexical metric is not semantic equivalence.
    A later model-specific evaluator can add validated semantic scoring.
    """
    normalized_output = f" {_normalize(output)} "
    normalized_answer = f" {_normalize(case.answer)} "
    retrieved = bool(normalized_answer.strip()) and normalized_answer in normalized_output
    return {
        **case.to_dict(),
        "output": output,
        "retrieved": retrieved,
        # A forget fact explicitly present in context is expected to be retrievable;
        # only absent-context recall is counted as parametric disclosure.
        "target_disclosed": retrieved and case.target_split == "forget" and case.context_mode == "absent",
        "contextual_retrieved": retrieved and case.context_mode == "present",
        "retain_retrieved": retrieved and case.target_split == "retain",
        "scoring": "normalized_exact_match",
    }


def run_niah_benchmark(cases: Iterable[NIAHCase], generate: Callable[[str], str]) -> dict:
    """Run cases through a callable accepting prompt and returning generated text."""
    rows = []
    for case in cases:
        output = generate(case.prompt)
        if not isinstance(output, str):
            raise TypeError("generate(prompt) must return a string")
        rows.append(evaluate_niah_case(case, output))
    forget = [r for r in rows if r["target_split"] == "forget"]
    retain = [r for r in rows if r["target_split"] == "retain"]
    forget_absent = [r for r in forget if r["context_mode"] == "absent"]
    forget_present = [r for r in forget if r["context_mode"] == "present"]
    retain_present = [r for r in retain if r["context_mode"] == "present"]
    total = len(rows)
    return {
        "benchmark": "RENA-NIAH-v1",
        "scoring": "normalized_exact_match",
        "case_count": total,
        "forget_case_count": len(forget),
        "retain_case_count": len(retain),
        "forget_disclosure_rate": (sum(r["target_disclosed"] for r in forget_absent) / len(forget_absent)) if forget_absent else None,
        "retain_retrieval_rate": (sum(r["retain_retrieved"] for r in retain) / len(retain)) if retain else None,
        "retain_contextual_retrieval_rate": (sum(r["retain_retrieved"] for r in retain_present) / len(retain_present)) if retain_present else None,
        "forget_contextual_retrieval_rate": (sum(r["contextual_retrieved"] for r in forget_present) / len(forget_present)) if forget_present else None,
        "forget_parametric_disclosure_rate": (sum(r["target_disclosed"] for r in forget_absent) / len(forget_absent)) if forget_absent else None,
        "cases": rows,
        "limitations": [
            "Context size is measured in distractor records and characters, not model tokens.",
            "Normalized exact match does not capture semantic equivalence or indirect disclosure.",
            "Present-context retrieval tests use of supplied evidence; absent-context disclosure probes parametric recall. Neither alone proves parameter erasure.",
        ],
    }
