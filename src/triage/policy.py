"""Политика решения — единственное место, где система решает судьбу тикета.

Ключевое проектное решение: автозакрытие не является выходом модели. Модель даёт
сигналы (тема, уверенность, найденный источник, оценка безопасности), а разрешение
на автоответ выдаёт версионируемая политика, состоящая из порогов и списков.
Так сделано потому, что менять порог дешевле и безопаснее, чем переобучать модель,
а объяснить регулятору или разгневанному пользователю нужно именно решение.

Все пороги — «И», не «ИЛИ»: достаточно одного несработавшего условия, чтобы тикет
ушёл человеку. Это дороже по деньгам и правильнее по риску.
"""

from __future__ import annotations

from .models import (
    AUTO_CLOSE_ALLOWLIST,
    HARD_DENY_TOPICS,
    POLICY_VERSION,
    Classification,
    Decision,
    Draft,
    Queue,
    RetrievedChunk,
    Risk,
    Route,
    RuleOutcome,
    SafetyReport,
    Ticket,
)

MIN_CONFIDENCE_AUTO_CLOSE = 0.90
MIN_GROUNDEDNESS_AUTO_CLOSE = 0.75
#: Порог сходства с найденной статьёй. 0.35 — это калибровка под TF-IDF-пространство
#: PoC: у него другое распределение косинусов, чем у гибридного BM25+dense-ретривера,
#: для которого в docs/ml.md указан порог 0.55. Порог отсечения — свойство ретривера,
#: а не универсальная константа: при замене ретривера его пересчитывают по разметке
#: «релевантно/нет» на целевой precision, иначе он либо режет всё, либо не режет ничего.
MIN_RETRIEVAL_SCORE = 0.35
MAX_REPEAT_CONTACT = 3

_TOPIC_QUEUES = {
    "billing": Queue.L1_BILLING,
    "tech": Queue.L2_TECH,
    "fraud": Queue.FRAUD_TEAM,
    "legal": Queue.LEGAL_DPO,
}


def queue_for_topic(topic: str) -> Queue:
    if topic == "account.data_deletion":
        return Queue.LEGAL_DPO
    prefix = topic.split(".", 1)[0]
    return _TOPIC_QUEUES.get(prefix, Queue.L1_GENERAL)


def decide(
    *,
    ticket: Ticket,
    classification: Classification,
    rules: RuleOutcome,
    chunks: list[RetrievedChunk],
    draft: Draft | None,
    safety: SafetyReport | None,
    incident: str | None = None,
    latency_ms: float = 0.0,
    generation_failed: bool = False,
) -> Decision:
    reasons: list[str] = list(rules.reasons)
    queue = rules.forced_queue or queue_for_topic(classification.topic)
    # Деградация — это когда мы хотели черновик и не смогли его получить.
    # Осознанный отказ генерировать (критичный риск, нет источника) деградацией
    # не является: иначе метрика «доля деградаций» перестаёт означать проблему.
    degraded = generation_failed

    def build(route: Route, *, with_draft: bool) -> Decision:
        return Decision(
            ticket_id=ticket.ticket_id,
            route=route,
            queue=queue,
            topic=classification.topic,
            confidence=round(classification.confidence, 4),
            risk=rules.risk,
            reasons=reasons,
            draft_text=draft.text if (with_draft and draft) else None,
            cited_doc_ids=draft.cited_doc_ids if (with_draft and draft) else (),
            degraded=degraded,
            latency_ms=round(latency_ms, 2),
            incident_cluster_id=incident,
            pii_classes=ticket.pii_classes,
        )

    # 1. Критичный риск: человек немедленно, черновик не готовим вообще —
    #    по таким тикетам даже подсказка модели может исказить работу оператора.
    if rules.risk == Risk.CRITICAL:
        reasons.append("critical_risk_escalated")
        return build(Route.ESCALATE_L2, with_draft=False)

    # 2. Нет черновика — значит LLM недоступен или контекста не нашлось.
    #    Тикет не теряется и не закрывается: он идёт в очередь как обычный.
    if draft is None or safety is None:
        reasons.append("llm_unavailable_no_draft" if generation_failed else "no_draft_no_context")
        return build(Route.ROUTE_TO_QUEUE, with_draft=False)

    if not safety.ok:
        reasons.append(
            "safety_failed:"
            + ",".join(
                filter(
                    None,
                    [
                        "pii" if safety.has_pii else "",
                        *safety.injection_markers,
                        *safety.forbidden_promises,
                    ],
                )
            )
        )
        return build(Route.SUGGEST_TO_OPERATOR, with_draft=True)

    top_score = chunks[0].score if chunks else 0.0
    auto_reply_allowed = chunks[0].auto_reply_allowed if chunks else False

    blockers: list[str] = []
    if classification.topic in HARD_DENY_TOPICS:
        blockers.append("topic_in_hard_deny")
    if classification.topic not in AUTO_CLOSE_ALLOWLIST:
        blockers.append("topic_not_in_allowlist")
    if rules.deny_auto_close:
        blockers.append("rules_deny_auto_close")
    if classification.confidence < MIN_CONFIDENCE_AUTO_CLOSE:
        blockers.append(f"confidence<{MIN_CONFIDENCE_AUTO_CLOSE}")
    if safety.groundedness < MIN_GROUNDEDNESS_AUTO_CLOSE:
        blockers.append(f"groundedness<{MIN_GROUNDEDNESS_AUTO_CLOSE}")
    if top_score < MIN_RETRIEVAL_SCORE:
        blockers.append(f"retrieval_score<{MIN_RETRIEVAL_SCORE}")
    if not auto_reply_allowed:
        blockers.append("kb_article_manual_only")
    if ticket.repeat_contact >= MAX_REPEAT_CONTACT:
        blockers.append("repeat_contact")

    # 3. Подтверждённый инцидент: отвечаем одним broadcast на кластер.
    #    Порог уверенности здесь другой не по недосмотру: основанием служит сам
    #    подтверждённый инцидент и фиксированный шаблон, а не решение модели.
    if incident and classification.topic == "tech.service_unavailable" and not rules.deny_auto_close:
        reasons.append(f"incident_broadcast:{incident}")
        return build(Route.INCIDENT_BROADCAST, with_draft=True)

    if not blockers:
        reasons.append("all_thresholds_passed")
        return build(Route.AUTO_CLOSE, with_draft=True)

    reasons.extend(blockers)
    return build(Route.SUGGEST_TO_OPERATOR, with_draft=True)


__all__ = [
    "MIN_CONFIDENCE_AUTO_CLOSE",
    "MIN_GROUNDEDNESS_AUTO_CLOSE",
    "MIN_RETRIEVAL_SCORE",
    "POLICY_VERSION",
    "decide",
    "queue_for_topic",
]
