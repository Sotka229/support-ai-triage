"""Общие типы: Ticket Envelope, таксономия, решение.

Таксономия и allowlist/deny-list вынесены сюда, потому что это не деталь реализации,
а продуктовая договорённость: изменение этих множеств меняет поведение системы
и обязано проходить review вместе с политикой.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

POLICY_VERSION = "policy-2026-07-27.v1"
MODEL_VERSION = "tfidf-centroid-baseline-v1"


class Channel(str, Enum):
    CHAT = "chat"
    EMAIL = "email"
    WEBFORM = "webform"
    MOBILE = "mobile"


class Risk(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        return _RISK_RANK[self.value]

    @classmethod
    def worst(cls, *values: "Risk") -> "Risk":
        """Риск только повышается: любое сработавшее правило может ужесточить оценку,
        но не может её смягчить."""
        return max(values, key=lambda r: r.rank)


_RISK_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


class Route(str, Enum):
    AUTO_CLOSE = "AUTO_CLOSE"
    SUGGEST_TO_OPERATOR = "SUGGEST_TO_OPERATOR"
    ROUTE_TO_QUEUE = "ROUTE_TO_QUEUE"
    ESCALATE_L2 = "ESCALATE_L2"
    INCIDENT_BROADCAST = "INCIDENT_BROADCAST"


class Queue(str, Enum):
    L1_GENERAL = "L1_general"
    L1_BILLING = "L1_billing"
    L2_TECH = "L2_tech"
    FRAUD_TEAM = "FRAUD_TEAM"
    LEGAL_DPO = "LEGAL_DPO"
    VIP = "VIP"


TOPICS: tuple[str, ...] = (
    "billing.payment_failed",
    "billing.refund_request",
    "billing.subscription_cancel",
    "account.login_issue",
    "account.password_reset",
    "account.data_deletion",
    "order.status",
    "order.delivery_delay",
    "tech.app_bug",
    "tech.service_unavailable",
    "fraud.unauthorized_charge",
    "legal.complaint_threat",
    "other",
)

#: Темы, по которым MVP имеет право отправить ответ без оператора.
AUTO_CLOSE_ALLOWLIST = frozenset(
    {
        "account.password_reset",
        "order.status",
        "order.delivery_delay",
        "billing.payment_failed",
        "tech.service_unavailable",
    }
)

#: Темы, которые нельзя закрывать автоматически ни при какой уверенности модели:
#: движение денег, право, идентификация личности.
HARD_DENY_TOPICS = frozenset(
    {
        "billing.refund_request",
        "billing.subscription_cancel",
        "account.data_deletion",
        "fraud.unauthorized_charge",
        "legal.complaint_threat",
    }
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class RawMessage:
    """То, что пришло из канала, без обработки."""

    channel: Channel
    external_id: str
    payload: dict


@dataclass(frozen=True)
class Ticket:
    """Ticket Envelope — единое представление обращения из любого канала."""

    ticket_id: str
    channel: Channel | str
    text: str
    subject: str = ""
    customer_tier: str = "standard"
    repeat_contact: int = 1
    locale: str = "ru"
    pii_classes: frozenset[str] = frozenset()
    meta: dict = field(default_factory=dict)
    received_at: str = field(default_factory=_now)


@dataclass(frozen=True)
class Classification:
    topic: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RuleOutcome:
    risk: Risk = Risk.LOW
    deny_auto_close: bool = False
    reasons: list[str] = field(default_factory=list)
    forced_queue: Queue | None = None
    block_external_llm: bool = False


@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    title: str
    text: str
    score: float
    auto_reply_allowed: bool = True
    answer_snippet: str = ""


@dataclass(frozen=True)
class Draft:
    text: str
    cited_doc_ids: tuple[str, ...]
    model: str
    tokens_in: int
    tokens_out: int


@dataclass(frozen=True)
class SafetyReport:
    groundedness: float
    has_pii: bool = False
    injection_markers: tuple[str, ...] = ()
    forbidden_promises: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Опора на источник проверяется политикой отдельно: у suggest-режима
        порог groundedness ниже, чем у автоответа."""
        return not (self.has_pii or self.injection_markers or self.forbidden_promises)


@dataclass(frozen=True)
class Decision:
    ticket_id: str
    route: Route
    queue: Queue
    topic: str
    confidence: float
    risk: Risk
    reasons: list[str]
    draft_text: str | None = None
    cited_doc_ids: tuple[str, ...] = ()
    policy_version: str = POLICY_VERSION
    model_version: str = MODEL_VERSION
    degraded: bool = False
    latency_ms: float = 0.0
    incident_cluster_id: str | None = None
    pii_classes: frozenset[str] = frozenset()
