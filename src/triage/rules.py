"""Детерминированные правила: красные флаги и контекстные запреты.

Правила стоят перед моделью и имеют над ней приоритет. Причина не в качестве ML,
а в природе требований: «обращение про суд не закрывать автоматически» — это
юридическое ограничение, оно должно читаться глазами, проходить review и работать
даже если модель деградировала или её вообще нет (запасной путь rules-only).
"""

from __future__ import annotations

import re

from .models import Queue, Risk, RuleOutcome, Ticket
from .pii import PAYMENT_CLASSES

# Каждая группа: (причина, паттерн, риск, очередь). Порядок не важен — риск
# берётся максимальный из сработавших, причины накапливаются все.
_FLAGS: tuple[tuple[str, re.Pattern[str], Risk, Queue | None], ...] = (
    (
        "fraud_keywords",
        re.compile(
            r"(взлома\w*|мошенник\w*|не я соверша\w*|совершал не я|украл\w*\s+(доступ|аккаунт)|"
            r"чуж\w+\s+(покупк\w+|заказ\w*)|без моего согласия)",
            re.IGNORECASE,
        ),
        Risk.CRITICAL,
        Queue.FRAUD_TEAM,
    ),
    (
        "legal_threat",
        re.compile(
            r"(в\s+суд|иск\b|исков\w*|прокуратур\w*|роспотребнадзор\w*|роскомнадзор\w*|"
            r"претензи\w*|судебн\w*|адвокат\w*|в\s+СМИ|журналист\w*)",
            re.IGNORECASE,
        ),
        Risk.HIGH,
        Queue.LEGAL_DPO,
    ),
    (
        "personal_data_request",
        re.compile(r"(удалит\w*\s+(мои\s+)?(персональн\w+\s+)?данн\w+|152-ФЗ|GDPR|право на забвение)", re.IGNORECASE),
        Risk.HIGH,
        Queue.LEGAL_DPO,
    ),
    (
        "threat_to_life",
        re.compile(r"(не хочу жить|покончу|суицид|убью себя|мне очень плохо|вызовите скорую)", re.IGNORECASE),
        Risk.CRITICAL,
        Queue.L2_TECH,
    ),
    (
        "minor_involved",
        re.compile(r"(мне\s+(1[0-7]|[6-9])\s+лет|мой реб[её]нок|несовершеннолетн\w*)", re.IGNORECASE),
        Risk.HIGH,
        None,
    ),
    (
        "money_movement",
        re.compile(r"(верните деньги|возврат\w*\s+(средств|денег|денежн\w+)|чарджбэк|chargeback|компенсаци\w*)", re.IGNORECASE),
        Risk.MEDIUM,
        None,
    ),
)


def evaluate(ticket: Ticket) -> RuleOutcome:
    risk = Risk.LOW
    reasons: list[str] = []
    forced_queue: Queue | None = None

    haystack = f"{ticket.subject} {ticket.text}"
    for reason, pattern, flag_risk, queue in _FLAGS:
        if pattern.search(haystack):
            reasons.append(reason)
            if flag_risk.rank > risk.rank:
                risk = flag_risk
                forced_queue = queue
            elif queue is not None and forced_queue is None:
                forced_queue = queue

    if ticket.customer_tier in {"vip", "enterprise"}:
        reasons.append(f"customer_tier={ticket.customer_tier}")
        risk = Risk.worst(risk, Risk.MEDIUM)
        forced_queue = forced_queue or Queue.VIP

    if ticket.repeat_contact >= 3:
        reasons.append(f"repeat_contact={ticket.repeat_contact}")
        risk = Risk.worst(risk, Risk.MEDIUM)

    block_external_llm = bool(ticket.pii_classes & PAYMENT_CLASSES)
    if block_external_llm:
        reasons.append("payment_pii_present")
        risk = Risk.worst(risk, Risk.MEDIUM)

    return RuleOutcome(
        risk=risk,
        deny_auto_close=bool(reasons),
        reasons=reasons,
        forced_queue=forced_queue,
        block_external_llm=block_external_llm,
    )
