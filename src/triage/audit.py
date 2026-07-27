"""Аудит автоматических решений.

Требование кейса — сохранять и уметь аудировать все автоматические решения.
Две вещи, которые делают лог пригодным для разбора конкретной жалобы:

1. Воспроизводимость: в записи есть версия политики, версия модели, входные
   сигналы и список сработавших правил. По записи можно объяснить, почему тикет
   был закрыт автоматически, даже через полгода и после трёх релизов.
2. Tamper-evidence: записи связаны цепочкой хешей. Незаметно поправить прошлое
   решение («мы это эскалировали, честное слово») не получится — verify() упадёт.

Упрощение PoC: JSONL-файл. В целевой системе — append-only таблица в PostgreSQL
плюс выгрузка в S3 с WORM-политикой; цепочка хешей переносится как есть.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .models import Decision

GENESIS_HASH = "0" * 64


class AuditLog:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, decision: Decision, *, ticket_text: str, extra: dict | None = None) -> dict:
        """`ticket_text` обязан быть уже отредактированным (PII заменены плейсхолдерами)."""
        record = asdict(decision)
        record["route"] = decision.route.value
        record["risk"] = decision.risk.value
        record["queue"] = decision.queue.value
        record["cited_doc_ids"] = list(decision.cited_doc_ids)
        record["pii_classes"] = sorted(decision.pii_classes)
        record["ticket_text_redacted"] = ticket_text
        record["ts"] = _now()
        if extra:
            record.update(extra)

        record["prev_hash"] = self._last_hash()
        record["hash"] = _hash(record)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def verify(self) -> bool:
        previous = GENESIS_HASH
        for record in self.records():
            if record.get("prev_hash") != previous:
                return False
            stored = record.get("hash")
            if stored != _hash({k: v for k, v in record.items() if k != "hash"}):
                return False
            previous = stored
        return True

    def _last_hash(self) -> str:
        records = self.records()
        return records[-1]["hash"] if records else GENESIS_HASH


def _hash(record: dict) -> str:
    payload = json.dumps(
        {k: v for k, v in sorted(record.items()) if k != "hash"},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
