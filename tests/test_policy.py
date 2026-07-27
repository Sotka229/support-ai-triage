"""Политика решения — место, где система решает судьбу тикета.

Тесты построены так, чтобы поймать самую дорогую ошибку продукта: автоматический
ответ там, где его быть не должно. Поэтому здесь есть тест, который перебирает
все hard-deny темы и требует, чтобы ни одна из них не получила AUTO_CLOSE даже
при идеальных сигналах модели.
"""

import dataclasses

import pytest

from triage.models import (
    HARD_DENY_TOPICS,
    Classification,
    Draft,
    Queue,
    RetrievedChunk,
    Risk,
    Route,
    RuleOutcome,
    SafetyReport,
    Ticket,
)
from triage.policy import POLICY_VERSION, decide


def perfect_inputs(topic="account.password_reset", **over):
    data = dict(
        ticket=Ticket(ticket_id="t1", channel="chat", text="Забыл пароль, письмо не приходит"),
        classification=Classification(topic=topic, confidence=0.96, scores={topic: 0.96}),
        rules=RuleOutcome(risk=Risk.LOW),
        chunks=[
            RetrievedChunk(
                doc_id="kb-001-password-reset",
                title="Восстановление пароля",
                text="Проверьте папку спам и запросите письмо повторно.",
                score=0.62,
                auto_reply_allowed=True,
            )
        ],
        draft=Draft(
            text="Проверьте папку спам и запросите письмо повторно.",
            cited_doc_ids=("kb-001-password-reset",),
            model="mock-template-v1",
            tokens_in=1200,
            tokens_out=90,
        ),
        safety=SafetyReport(groundedness=0.88, has_pii=False),
    )
    data.update(over)
    return data


class TestHappyPath:
    def test_confident_safe_ticket_is_closed_automatically(self):
        d = decide(**perfect_inputs())
        assert d.route == Route.AUTO_CLOSE
        assert d.draft_text
        assert d.cited_doc_ids == ("kb-001-password-reset",)
        assert d.policy_version == POLICY_VERSION

    def test_decision_always_records_why(self):
        d = decide(**perfect_inputs())
        assert d.reasons, "решение без объяснения невозможно аудировать"


class TestThresholds:
    def test_low_confidence_downgrades_to_a_draft_for_the_operator(self):
        args = perfect_inputs()
        args["classification"] = Classification(topic="account.password_reset", confidence=0.85, scores={})
        d = decide(**args)
        assert d.route == Route.SUGGEST_TO_OPERATOR
        assert any("confidence" in r for r in d.reasons)

    def test_weak_groundedness_downgrades_to_a_draft(self):
        args = perfect_inputs()
        args["safety"] = SafetyReport(groundedness=0.50, has_pii=False)
        d = decide(**args)
        assert d.route == Route.SUGGEST_TO_OPERATOR
        assert any("grounded" in r for r in d.reasons)

    def test_weak_retrieval_downgrades_to_a_draft(self):
        args = perfect_inputs()
        args["chunks"] = [dataclasses.replace(args["chunks"][0], score=0.20)]
        d = decide(**args)
        assert d.route == Route.SUGGEST_TO_OPERATOR

    def test_failed_safety_check_never_reaches_the_user(self):
        args = perfect_inputs()
        args["safety"] = SafetyReport(groundedness=0.9, has_pii=True)
        d = decide(**args)
        assert d.route != Route.AUTO_CLOSE

    def test_article_marked_manual_only_blocks_auto_close(self):
        args = perfect_inputs()
        args["chunks"] = [dataclasses.replace(args["chunks"][0], auto_reply_allowed=False)]
        assert decide(**args).route != Route.AUTO_CLOSE


class TestRiskAndDenyList:
    @pytest.mark.parametrize("topic", sorted(HARD_DENY_TOPICS))
    def test_hard_deny_topic_is_never_auto_closed_even_with_perfect_signals(self, topic):
        d = decide(**perfect_inputs(topic=topic))
        assert d.route != Route.AUTO_CLOSE, f"{topic} нельзя закрывать автоматически"

    def test_critical_risk_escalates_and_produces_no_draft_at_all(self):
        args = perfect_inputs(topic="fraud.unauthorized_charge")
        args["rules"] = RuleOutcome(
            risk=Risk.CRITICAL,
            deny_auto_close=True,
            reasons=["fraud_keywords"],
            forced_queue=Queue.FRAUD_TEAM,
        )
        d = decide(**args)
        assert d.route == Route.ESCALATE_L2
        assert d.queue == Queue.FRAUD_TEAM
        assert d.draft_text is None, "по критичным тикетам черновик не показываем даже оператору"

    def test_rule_denied_ticket_gets_a_draft_but_not_an_auto_answer(self):
        args = perfect_inputs(topic="billing.refund_request")
        args["rules"] = RuleOutcome(risk=Risk.MEDIUM, deny_auto_close=True, reasons=["money_movement"])
        d = decide(**args)
        assert d.route == Route.SUGGEST_TO_OPERATOR
        assert d.queue == Queue.L1_BILLING

    def test_forced_queue_from_rules_wins_over_topic_routing(self):
        args = perfect_inputs()
        args["rules"] = RuleOutcome(risk=Risk.MEDIUM, deny_auto_close=True, reasons=["vip"], forced_queue=Queue.VIP)
        assert decide(**args).queue == Queue.VIP


class TestDegradation:
    def test_without_a_draft_the_ticket_is_routed_not_dropped(self):
        args = perfect_inputs()
        args["draft"] = None
        args["safety"] = None
        args["generation_failed"] = True
        d = decide(**args)
        assert d.route == Route.ROUTE_TO_QUEUE
        assert d.queue == Queue.L1_GENERAL
        assert d.degraded is True
        assert any("llm" in r.lower() for r in d.reasons)

    def test_deliberate_refusal_to_generate_is_not_reported_as_degradation(self):
        """Иначе метрика «доля деградаций» смешивает поломку LLM с нормальной
        работой политики, и по ней невозможно ставить алерт."""
        args = perfect_inputs(topic="fraud.unauthorized_charge")
        args["rules"] = RuleOutcome(risk=Risk.CRITICAL, deny_auto_close=True, reasons=["fraud_keywords"])
        args["draft"] = None
        args["safety"] = None
        d = decide(**args)
        assert d.route == Route.ESCALATE_L2
        assert d.degraded is False

    def test_topic_routing_map_covers_the_whole_taxonomy(self):
        from triage.models import TOPICS
        from triage.policy import queue_for_topic

        for topic in TOPICS:
            assert isinstance(queue_for_topic(topic), Queue)


class TestIncident:
    def test_confirmed_incident_uses_one_broadcast_instead_of_n_answers(self):
        args = perfect_inputs(topic="tech.service_unavailable")
        args["chunks"] = [
            RetrievedChunk(
                doc_id="kb-005-incident-broadcast",
                title="Массовый сбой",
                text="Сбой подтверждён, команда работает над восстановлением.",
                score=0.7,
                auto_reply_allowed=True,
            )
        ]
        args["incident"] = "cluster-7"
        d = decide(**args)
        assert d.route == Route.INCIDENT_BROADCAST
        assert d.incident_cluster_id == "cluster-7"

    def test_incident_flag_does_not_bypass_risk_rules(self):
        args = perfect_inputs(topic="tech.service_unavailable")
        args["incident"] = "cluster-7"
        args["rules"] = RuleOutcome(risk=Risk.CRITICAL, deny_auto_close=True, reasons=["threat"])
        assert decide(**args).route == Route.ESCALATE_L2
