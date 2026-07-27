"""Демонстрация PoC: четыре сценария от входящего тикета до записи в лог решения.

Запуск:  python demo.py

Сценарии:
  1. happy path        — типовое обращение закрывается автоматически;
  2. risky path        — подозрение на мошенничество уходит человеку, черновик не создаётся;
  3. fallback path     — LLM недоступен, система деградирует в маршрутизацию;
  4. incident path     — поток одинаковых жалоб схлопывается в один broadcast.

Скрипт ничего не отправляет во внешние сервисы: LLM здесь мок (см. src/triage/llm.py).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from triage.audit import AuditLog  # noqa: E402
from triage.llm import CircuitBreaker, GuardedLLM, LLMUnavailable  # noqa: E402
from triage.models import Channel, RawMessage, Route  # noqa: E402
from triage.pipeline import TriagePipeline  # noqa: E402
from triage.scenarios import load_scenarios  # noqa: E402

ROOT = Path(__file__).parent
DATA = ROOT / "data"
AUDIT_PATH = ROOT / "var" / "decisions.jsonl"

ROUTE_NOTES = {
    Route.AUTO_CLOSE: "ответ уходит пользователю без оператора",
    Route.SUGGEST_TO_OPERATOR: "черновик показывается оператору, отправляет человек",
    Route.ROUTE_TO_QUEUE: "черновика нет, тикет просто маршрутизирован",
    Route.ESCALATE_L2: "эскалация: решает только человек",
    Route.INCIDENT_BROADCAST: "один ответ на кластер обращений",
}


class DeadLLM:
    """Имитация недоступного LLM API: 503 на каждый запрос."""

    model = "llm-down"

    def generate(self, prompt, chunks):
        raise LLMUnavailable("upstream returned 503")


def header(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def show(pipeline: TriagePipeline, raw: RawMessage, title: str) -> None:
    print()
    print(f"--- {title} ---")
    fast = pipeline.fast_path(raw)
    print(f"  канал:            {fast.ticket.channel.value} / {fast.ticket.ticket_id}")
    print(f"  текст (PII снят): {fast.ticket.text[:150]}")
    print(f"  найденные PII:    {sorted(fast.ticket.pii_classes) or 'нет'}")
    print(f"  тема:             {fast.classification.topic} (уверенность {fast.classification.confidence:.3f})")
    print(f"  риск:             {fast.rules.risk.value}; правила: {fast.rules.reasons or 'нет срабатываний'}")
    print(f"  быстрый путь:     {fast.latency_ms:.1f} мс (бюджет 500 мс)")

    hits = pipeline.kb.search(fast.ticket.text, k=2)
    for hit in hits:
        allowed = "автоответ разрешён" if hit.auto_reply_allowed else "только вручную"
        print(f"  база знаний:      {hit.doc_id} score={hit.score:.3f} ({allowed})")

    decision = pipeline.slow_path(fast)
    print(f"  РЕШЕНИЕ:          {decision.route.value} -> {decision.queue.value}")
    print(f"                    {ROUTE_NOTES[decision.route]}")
    print(f"  причины:          {decision.reasons}")
    if decision.draft_text:
        print(f"  черновик:         {decision.draft_text[:220]}")
        print(f"  источники:        {list(decision.cited_doc_ids)}")
    else:
        print("  черновик:         не создан")
    if decision.degraded:
        print("  ДЕГРАДАЦИЯ:       LLM недоступен, работаем без черновика; тикет не потерян и не закрыт")
    elif decision.draft_text is None:
        print("  генерация:        сознательно не выполнялась (это не деградация)")


def main() -> int:
    if AUDIT_PATH.parent.exists():
        shutil.rmtree(AUDIT_PATH.parent)

    scenarios = {s.name: s for s in load_scenarios(DATA / "scenarios.json")}
    pipeline = TriagePipeline.build(data_dir=DATA, audit_path=AUDIT_PATH)

    header("1. HAPPY PATH: типовое обращение, безопасная категория")
    show(pipeline, scenarios["happy_password_reset"].raw, "чат: забыл пароль")

    header("2. RISKY PATH: подозрение на мошенничество и платёжные данные в тексте")
    show(pipeline, scenarios["risky_unauthorized_charge"].raw, "email: списали деньги без согласия")

    header("3. FALLBACK PATH: LLM API недоступен")
    degraded = TriagePipeline(
        classifier=pipeline.classifier,
        knowledge_base=pipeline.kb,
        llm=GuardedLLM(DeadLLM(), retries=2, breaker=CircuitBreaker(failure_threshold=2, reset_after=30.0)),
        audit=pipeline.audit,
    )
    show(degraded, scenarios["degraded_app_bug"].raw, "мобильное приложение: краш (LLM лежит)")
    print(f"  circuit breaker:  {degraded.llm.breaker.state}")

    header("4. INCIDENT PATH: 6 однотипных жалоб за 10 минут")
    base = scenarios["incident_service_unavailable"].raw
    for index in range(6):
        raw = RawMessage(
            channel=Channel.WEBFORM,
            external_id=f"web-90{index:02d}",
            payload={**base.payload, "form_id": f"90{index:02d}"},
        )
        decision = pipeline.process(raw)
        marker = " <- кластер распознан как инцидент" if decision.route == Route.INCIDENT_BROADCAST else ""
        print(f"  тикет {index + 1}: {decision.route.value:<22} cluster={decision.incident_cluster_id}{marker}")

    header("АУДИТ РЕШЕНИЙ")
    audit = AuditLog(AUDIT_PATH)
    records = audit.records()
    print(f"  записей в логе:        {len(records)}")
    print(f"  цепочка хешей цела:    {audit.verify()}")
    print(f"  файл:                  {AUDIT_PATH}")
    print("  в логе есть версии политики и модели, сработавшие правила и источники —")
    print("  этого достаточно, чтобы через полгода объяснить конкретное решение.")

    leaked = [r for r in records if "4111" in r["ticket_text_redacted"] or "@example.com" in r["ticket_text_redacted"]]
    print(f"  утечек PII в логе:     {len(leaked)}")

    auto = sum(1 for r in records if r["route"] in {Route.AUTO_CLOSE.value, Route.INCIDENT_BROADCAST.value})
    print()
    print(f"Итог демо: {auto} из {len(records)} решений обошлись без оператора, остальные ушли человеку.")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
