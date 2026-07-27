"""End-to-end сценарии: happy path, risky path, деградация и инцидент.

Это тот самый минимальный сценарий из задания: mock-тикет -> тема и риск ->
похожий фрагмент базы знаний -> черновик или маршрут -> запись в лог решения,
и отдельно — рискованный/низкоуверенный случай, который уходит оператору.
"""

import time
from pathlib import Path

from triage.llm import GuardedLLM, LLMUnavailable, TemplateLLM
from triage.models import Channel, Queue, RawMessage, Route
from triage.pipeline import TriagePipeline
from triage.scenarios import load_scenarios

ROOT = Path(__file__).resolve().parents[1]


class AlwaysDownLLM:
    model = "down"

    def generate(self, prompt, chunks):
        raise LLMUnavailable("upstream is down")


def build(tmp_path, llm=None):
    return TriagePipeline.build(
        data_dir=ROOT / "data",
        audit_path=tmp_path / "audit.jsonl",
        llm=llm or GuardedLLM(TemplateLLM()),
    )


def scenario(name):
    return next(s for s in load_scenarios(ROOT / "data" / "scenarios.json") if s.name == name)


class TestHappyPath:
    def test_typical_ticket_is_answered_automatically_and_logged(self, tmp_path):
        pipeline = build(tmp_path)
        decision = pipeline.process(scenario("happy_password_reset").raw)

        assert decision.topic == "account.password_reset"
        assert decision.route == Route.AUTO_CLOSE
        assert decision.draft_text
        assert "kb-001-password-reset" in decision.cited_doc_ids
        assert pipeline.audit.verify() is True
        assert len(pipeline.audit.records()) == 1

    def test_customer_email_never_reaches_the_audit_log(self, tmp_path):
        pipeline = build(tmp_path)
        pipeline.process(scenario("happy_password_reset").raw)
        raw = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
        assert "ivan.petrov@example.com" not in raw
        assert "[EMAIL]" in raw


class TestRiskyPath:
    def test_suspected_fraud_goes_to_a_human_and_is_never_auto_closed(self, tmp_path):
        pipeline = build(tmp_path)
        decision = pipeline.process(scenario("risky_unauthorized_charge").raw)

        assert decision.route == Route.ESCALATE_L2
        assert decision.queue == Queue.FRAUD_TEAM
        assert decision.draft_text is None
        assert decision.risk.name == "CRITICAL"

    def test_card_number_is_redacted_before_anything_else_happens(self, tmp_path):
        pipeline = build(tmp_path)
        fast = pipeline.fast_path(scenario("risky_unauthorized_charge").raw)
        assert "4111" not in fast.ticket.text
        assert "[CARD]" in fast.ticket.text
        assert fast.ticket.pii_classes >= {"card", "phone"}
        # Адрес отправителя вообще не попадает в текст для модели: нормализатор
        # оставляет от него только домен в метаданных.
        assert "i.petrov@example.com" not in fast.ticket.text
        assert fast.ticket.meta["from_domain"] == "example.com"
        assert fast.rules.block_external_llm is True


class TestDegradation:
    def test_ticket_is_routed_to_an_operator_when_the_llm_is_down(self, tmp_path):
        pipeline = build(tmp_path, llm=GuardedLLM(AlwaysDownLLM(), retries=1))
        decision = pipeline.process(scenario("degraded_app_bug").raw)

        assert decision.route == Route.ROUTE_TO_QUEUE
        assert decision.degraded is True
        assert decision.draft_text is None
        assert decision.queue == Queue.L2_TECH
        assert pipeline.audit.records(), "тикет не потерян, решение записано"

    def test_classification_still_works_without_the_llm(self, tmp_path):
        pipeline = build(tmp_path, llm=GuardedLLM(AlwaysDownLLM(), retries=1))
        decision = pipeline.process(scenario("degraded_app_bug").raw)
        assert decision.topic == "tech.app_bug"
        assert decision.confidence > 0


class TestIncident:
    def test_burst_of_identical_complaints_becomes_one_broadcast(self, tmp_path):
        pipeline = build(tmp_path)
        base = scenario("incident_service_unavailable").raw
        decisions = []
        for i in range(6):
            raw = RawMessage(
                channel=Channel.WEBFORM,
                external_id=f"web-900{i}",
                payload={
                    "form_id": f"900{i}",
                    "category": "Технические проблемы",
                    "description": base.payload["description"],
                },
            )
            decisions.append(pipeline.process(raw))

        assert decisions[-1].route == Route.INCIDENT_BROADCAST
        assert decisions[-1].incident_cluster_id is not None
        assert decisions[0].route != Route.INCIDENT_BROADCAST, "первый тикет ещё не инцидент"


class TestHotPathBudget:
    def test_fast_path_stays_well_inside_the_500ms_budget(self, tmp_path):
        pipeline = build(tmp_path)
        raws = [s.raw for s in load_scenarios(ROOT / "data" / "scenarios.json")]
        samples = []
        for _ in range(15):
            for raw in raws:
                start = time.perf_counter()
                pipeline.fast_path(raw)
                samples.append((time.perf_counter() - start) * 1000)
        samples.sort()
        p95 = samples[int(0.95 * (len(samples) - 1))]
        assert p95 < 500, f"p95 быстрого пути {p95:.1f} мс — не укладываемся в бюджет"

    def test_fast_path_reports_its_own_latency(self, tmp_path):
        pipeline = build(tmp_path)
        fast = pipeline.fast_path(scenario("happy_password_reset").raw)
        assert 0 < fast.latency_ms < 500
