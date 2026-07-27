"""Загрузка демо-сценариев из data/scenarios.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .models import Channel, RawMessage


@dataclass(frozen=True)
class Scenario:
    name: str
    comment: str
    raw: RawMessage


def load_scenarios(path: Path | str) -> list[Scenario]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Scenario(
            name=row["name"],
            comment=row.get("comment", ""),
            raw=RawMessage(
                channel=Channel(row["channel"]),
                external_id=row["external_id"],
                payload=row["payload"],
            ),
        )
        for row in rows
    ]
