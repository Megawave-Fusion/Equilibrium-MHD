#!/usr/bin/env python3
"""Minimal JSON-serializable plasma state index.

Large arrays live in IMAS netCDF entries and are referenced by IDS path. This
container stays deliberately small so modules can exchange metadata without
duplicating the storage backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import Any


@dataclass
class PlasmaState:
    case_name: str = "unnamed"
    time_s: float = 0.0
    scalars: dict[str, float | str | bool] = field(default_factory=dict)
    profiles: dict[str, str] = field(default_factory=dict)
    fields: dict[str, str] = field(default_factory=dict)
    distributions: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, str] = field(default_factory=dict)
    provenance: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlasmaState":
        return cls(**data)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load_json(cls, path: Path) -> "PlasmaState":
        return cls.from_dict(json.loads(path.read_text()))
