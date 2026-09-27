from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import read_json


@dataclass(frozen=True)
class Schema:
    schema_id: str
    version: str
    questions: dict[str, dict[str, Any]]
    routing: dict[str, float]

    def intent_labels(self) -> tuple[str, ...]:
        return tuple(self.questions["intent"]["criteria"].keys())


def load_schema(path: str | Path) -> Schema:
    raw = read_json(path)
    _validate_schema(raw)
    return Schema(
        schema_id=raw["schema_id"],
        version=raw["version"],
        questions=raw["questions"],
        routing=raw["routing"],
    )


def _validate_schema(raw: dict[str, Any]) -> None:
    required = {"schema_id", "version", "questions", "routing"}
    if not required.issubset(raw):
        missing = required - set(raw.keys())
        raise ValueError(f"schema missing keys: {missing}")
    q = raw["questions"]
    for key, expected in (("intent", "choice"), ("complexity", "score"), ("needs_escalation", "bool")):
        if key not in q:
            raise ValueError(f"question missing: {key}")
        if q[key].get("type") != expected:
            raise ValueError(f"question {key} type must be {expected}")
