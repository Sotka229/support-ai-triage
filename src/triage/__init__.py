"""Triage Gateway — PoC системы автоматической обработки тикетов поддержки.

Модули соответствуют компонентам целевой архитектуры (docs/architecture.md):

быстрый путь (синхронный, бюджет 500 мс)
    normalizer -> pii -> rules -> classifier            -> policy
медленный путь (асинхронный)
    dedup -> retrieval -> llm -> safety                 -> policy -> audit

Внешних зависимостей нет намеренно: проверяющий должен запустить PoC одной командой.
"""

__all__ = [
    "audit",
    "classifier",
    "dedup",
    "llm",
    "models",
    "normalizer",
    "pii",
    "pipeline",
    "policy",
    "retrieval",
    "rules",
    "safety",
    "scenarios",
    "vectorizer",
]
