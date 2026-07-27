"""Сборка компонентов в два пути: быстрый (синхронный) и медленный (асинхронный).

Разделение путей — центральное решение архитектуры, а не деталь реализации:
классификация и маршрутизация обязаны укладываться в 500 мс, потому что от них
зависит SLA первого ответа и распределение по очередям; генерация ответа занимает
секунды и потому вынесена за очередь. В PoC «очередь» — это просто вызов
`slow_path()` следом за `fast_path()`; в целевой системе между ними Kafka
(см. docs/architecture.md).
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path

from . import policy, rules, safety
from .audit import AuditLog
from .classifier import TopicClassifier, load_labelled
from .dedup import IncidentDetector
from .llm import GuardedLLM, LLMUnavailable, TemplateLLM
from .models import Classification, Decision, Draft, RawMessage, Risk, RuleOutcome, Ticket
from .normalizer import normalize
from .pii import redact
from .retrieval import KnowledgeBase


@dataclass(frozen=True)
class FastPathResult:
    ticket: Ticket
    classification: Classification
    rules: RuleOutcome
    latency_ms: float


class TriagePipeline:
    def __init__(
        self,
        *,
        classifier: TopicClassifier,
        knowledge_base: KnowledgeBase,
        llm,
        audit: AuditLog,
        incident_detector: IncidentDetector | None = None,
    ):
        self.classifier = classifier
        self.kb = knowledge_base
        self.llm = llm
        self.audit = audit
        self.incidents = incident_detector or IncidentDetector()

    @classmethod
    def build(cls, *, data_dir: Path | str, audit_path: Path | str, llm=None) -> "TriagePipeline":
        data_dir = Path(data_dir)
        return cls(
            classifier=TopicClassifier.train(load_labelled(data_dir / "train_tickets.jsonl")),
            knowledge_base=KnowledgeBase.from_dir(data_dir / "kb"),
            llm=llm or GuardedLLM(TemplateLLM()),
            audit=AuditLog(audit_path),
        )

    # ----------------------------------------------------------- быстрый путь

    def fast_path(self, raw: RawMessage) -> FastPathResult:
        start = time.perf_counter()

        ticket = normalize(raw)
        redaction = redact(ticket.text)
        ticket = dataclasses.replace(ticket, text=redaction.text, pii_classes=redaction.classes)

        rule_outcome = rules.evaluate(ticket)
        classification = self.classifier.predict(ticket.text)

        return FastPathResult(
            ticket=ticket,
            classification=classification,
            rules=rule_outcome,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    # ---------------------------------------------------------- медленный путь

    def slow_path(self, fast: FastPathResult) -> Decision:
        cluster = self.incidents.observe(fast.ticket.ticket_id, fast.ticket.text)
        incident_id = cluster.cluster_id if cluster.is_incident else None

        chunks = self.kb.search(fast.ticket.text, k=3)
        draft, report, generation_failed = self._draft(fast, chunks)

        decision = policy.decide(
            ticket=fast.ticket,
            classification=fast.classification,
            rules=fast.rules,
            chunks=chunks,
            draft=draft,
            safety=report,
            incident=incident_id,
            latency_ms=fast.latency_ms,
            generation_failed=generation_failed,
        )
        self.audit.append(decision, ticket_text=fast.ticket.text, extra={"retrieval_top_score": chunks[0].score if chunks else 0.0})
        return decision

    def _draft(self, fast: FastPathResult, chunks) -> tuple[Draft | None, object | None, bool]:
        """Возвращает (черновик, отчёт валидаторов, признак отказа генерации).

        Третий элемент отличает «мы решили не генерировать» от «генерация не удалась»:
        первое — нормальная работа политики, второе — деградация, которая идёт в алерты.
        """
        # По критичным тикетам не тратим ни токен: решение всё равно принимает человек.
        if fast.rules.risk == Risk.CRITICAL or not chunks:
            return None, None, False
        try:
            draft = self.llm.generate(_prompt(fast.ticket), chunks)
        except (LLMUnavailable, ValueError):
            return None, None, True  # запасной путь: работаем без черновика
        return draft, safety.validate(draft.text, chunks), False

    def process(self, raw: RawMessage) -> Decision:
        return self.slow_path(self.fast_path(raw))


def _prompt(ticket: Ticket) -> str:
    """Тикет подставляется как данные, а не как инструкция: см. защиту от prompt injection
    в docs/risks-and-ops.md. В PoC роль разделителя играет явная разметка блока."""
    return (
        "Ты оператор поддержки. Ответь строго по фрагментам базы знаний ниже.\n"
        "<<<TICKET_DATA_START>>>\n"
        f"{ticket.text}\n"
        "<<<TICKET_DATA_END>>>"
    )
