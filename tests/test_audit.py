"""Аудит автоматических решений.

Требование кейса: «все автоматические решения нужно сохранять и уметь аудировать».
Минимум, который делает лог пригодным для разбора жалобы: воспроизводимость
(видно, какие версии и какие сигналы привели к решению) и защита от незаметной
правки задним числом (цепочка хешей).
"""

import json

from triage.audit import AuditLog
from triage.models import Decision, Queue, Risk, Route


def decision(ticket_id="t1", **kw):
    base = dict(
        ticket_id=ticket_id,
        route=Route.AUTO_CLOSE,
        queue=Queue.L1_GENERAL,
        topic="account.password_reset",
        confidence=0.96,
        risk=Risk.LOW,
        reasons=["all_thresholds_passed"],
        draft_text="Проверьте папку спам.",
        cited_doc_ids=("kb-001-password-reset",),
    )
    base.update(kw)
    return Decision(**base)


class TestRecordContent:
    def test_record_contains_everything_needed_to_reproduce_the_decision(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        log.append(decision(), ticket_text="Забыл пароль, письмо не приходит")
        rec = log.records()[0]
        for field in (
            "ticket_id",
            "route",
            "topic",
            "confidence",
            "risk",
            "reasons",
            "policy_version",
            "model_version",
            "cited_doc_ids",
            "ts",
            "hash",
            "prev_hash",
        ):
            assert field in rec, f"в записи аудита нет поля {field}"

    def test_only_redacted_text_is_stored(self, tmp_path):
        """В аудит попадает то, что уже прошло PII-редакцию, и ничего сверх того."""
        log = AuditLog(tmp_path / "audit.jsonl")
        log.append(decision(), ticket_text="Карта [CARD] и почта [EMAIL]")
        raw = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
        assert "[CARD]" in raw
        assert "4111" not in raw and "@" not in raw


class TestHashChain:
    def test_records_are_chained(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        log.append(decision("t1"), ticket_text="a")
        log.append(decision("t2"), ticket_text="b")
        first, second = log.records()
        assert second["prev_hash"] == first["hash"]
        assert first["prev_hash"] == "0" * 64

    def test_verify_accepts_an_untouched_log(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        for i in range(3):
            log.append(decision(f"t{i}"), ticket_text="текст")
        assert log.verify() is True

    def test_verify_detects_a_silently_edited_decision(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        log.append(decision("t1", route=Route.ESCALATE_L2), ticket_text="мошенники списали деньги")
        log.append(decision("t2"), ticket_text="забыл пароль")

        lines = path.read_text(encoding="utf-8").splitlines()
        tampered = json.loads(lines[0])
        tampered["route"] = "AUTO_CLOSE"  # «мы это не эскалировали, так и было»
        lines[0] = json.dumps(tampered, ensure_ascii=False)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        assert AuditLog(path).verify() is False

    def test_append_survives_reopening_the_log(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        AuditLog(path).append(decision("t1"), ticket_text="a")
        AuditLog(path).append(decision("t2"), ticket_text="b")
        assert len(AuditLog(path).records()) == 2
        assert AuditLog(path).verify() is True
