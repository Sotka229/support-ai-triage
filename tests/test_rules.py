"""Детерминированные правила — красные флаги.

Правила стоят перед моделью и имеют приоритет над ней: их задача — гарантировать,
что определённые обращения никогда не будут закрыты автоматически, независимо от
уверенности классификатора. Это требование не про качество ML, а про юридический
и репутационный риск, поэтому оно реализовано кодом, который читается глазами.
"""

import dataclasses

import pytest

from triage.models import Queue, Risk, Ticket
from triage.rules import evaluate


def ticket(text, **kw):
    base = Ticket(ticket_id="t1", channel="chat", text=text)
    return dataclasses.replace(base, **kw) if kw else base


class TestRedFlags:
    @pytest.mark.parametrize(
        "text, risk, queue",
        [
            ("С карты списали деньги, эту операцию совершал не я", Risk.CRITICAL, Queue.FRAUD_TEAM),
            ("Аккаунт взломали, мошенники оформили заказ", Risk.CRITICAL, Queue.FRAUD_TEAM),
            ("Готовлю иск в суд и жалобу в Роспотребнадзор", Risk.HIGH, Queue.LEGAL_DPO),
            ("Требую удалить мои персональные данные по 152-ФЗ", Risk.HIGH, Queue.LEGAL_DPO),
            ("Напишу об этом в СМИ, я журналист", Risk.HIGH, Queue.LEGAL_DPO),
        ],
    )
    def test_flag_sets_risk_and_forces_queue(self, text, risk, queue):
        out = evaluate(ticket(text))
        assert out.risk == risk
        assert out.forced_queue == queue
        assert out.deny_auto_close is True
        assert out.reasons, "правило обязано объяснить, почему оно сработало"

    def test_threat_to_life_is_critical_and_goes_to_a_human_immediately(self):
        out = evaluate(ticket("Мне очень плохо, я не хочу жить, помогите"))
        assert out.risk == Risk.CRITICAL
        assert out.deny_auto_close is True

    def test_money_movement_blocks_auto_close_but_is_not_critical(self):
        out = evaluate(ticket("Верните деньги за подписку, оформите возврат средств"))
        assert out.deny_auto_close is True
        assert out.risk == Risk.MEDIUM

    def test_minor_mentioned_blocks_auto_close(self):
        out = evaluate(ticket("Мне 14 лет, помогите с аккаунтом"))
        assert out.deny_auto_close is True


class TestContextFlags:
    def test_vip_customer_never_gets_an_automatic_answer(self):
        out = evaluate(ticket("Забыл пароль", customer_tier="vip"))
        assert out.deny_auto_close is True
        assert out.forced_queue == Queue.VIP

    def test_third_contact_on_the_same_issue_goes_to_a_human(self):
        out = evaluate(ticket("Забыл пароль", repeat_contact=3))
        assert out.deny_auto_close is True
        assert any("repeat" in r for r in out.reasons)

    def test_payment_pii_blocks_auto_close_and_external_llm(self):
        out = evaluate(ticket("Проблема с оплатой", pii_classes=frozenset({"card"})))
        assert out.deny_auto_close is True
        assert out.block_external_llm is True

    def test_email_pii_alone_does_not_block_external_llm_after_redaction(self):
        out = evaluate(ticket("Забыл пароль", pii_classes=frozenset({"email"})))
        assert out.block_external_llm is False


class TestBenign:
    def test_typical_ticket_stays_low_risk(self):
        out = evaluate(ticket("Забыл пароль, письмо для сброса не приходит"))
        assert out.risk == Risk.LOW
        assert out.deny_auto_close is False
        assert out.forced_queue is None

    def test_risk_only_escalates_never_downgrades(self):
        """Несколько флагов сразу: остаётся максимальный уровень."""
        out = evaluate(ticket("Верните деньги, иначе иду в суд"))
        assert out.risk == Risk.HIGH
        assert len(out.reasons) >= 2
