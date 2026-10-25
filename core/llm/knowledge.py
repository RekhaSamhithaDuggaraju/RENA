"""Controlled fictional facts for reproducible LLM unlearning experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable


@dataclass(frozen=True)
class Fact:
    fact_id: str
    subject: str
    attribute: str
    value: str
    split: str  # "forget" or "retain"

    def __post_init__(self) -> None:
        if self.split not in {"forget", "retain"}:
            raise ValueError("split must be 'forget' or 'retain'")
        if not all((self.fact_id.strip(), self.subject.strip(), self.attribute.strip(), self.value.strip())):
            raise ValueError("fact fields must be non-empty")

    @property
    def statement(self) -> str:
        return f"The {self.attribute} of {self.subject} is {self.value}."

    def to_dict(self) -> dict:
        return asdict(self)


def build_synthetic_knowledge() -> list[Fact]:
    """Return a deterministic, fictional forget/retain set (no personal data)."""
    rows = [
        ("F001", "Norvale", "access code", "AX-731", "forget"),
        ("F002", "Belwick", "access code", "BY-294", "retain"),
        ("F003", "Norvale", "official flower", "silver iris", "forget"),
        ("F004", "Belwick", "official flower", "amber lily", "retain"),
        ("F005", "Taventon", "archive key", "TK-508", "forget"),
        ("F006", "Mariton", "archive key", "MK-163", "retain"),
        ("F007", "Orellis", "research vessel", "Quiet Star", "forget"),
        ("F008", "Denvor", "research vessel", "Blue Finch", "retain"),
        ("F009", "Caldrin", "station number", "ST-842", "forget"),
        ("F010", "Veyford", "station number", "ST-315", "retain"),
        ("F011", "Lunessa", "mineral sample", "LM-620", "forget"),
        ("F012", "Ormont", "mineral sample", "OM-407", "retain"),
    ]
    return [Fact(*row) for row in rows]


def knowledge_hash(facts: Iterable[Fact]) -> str:
    """Stable SHA-256 digest of fact metadata, independent of input ordering."""
    payload = [fact.to_dict() for fact in sorted(facts, key=lambda x: x.fact_id)]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
