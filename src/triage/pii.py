"""PII-редакция — первый обязательный шаг обработки.

Компромисс, который здесь зафиксирован осознанно: детекторы построены на правилах,
а не на модели. Правила детерминированы, объяснимы и стоят почти ноль, но у них есть
цена — часть сущностей требует ключевого слова рядом (ИНН, паспорт, CVV), иначе
десять цифр артикула превращаются в «персональные данные», и редакция съедает
половину полезного текста. В целевой системе поверх правил ставится NER-модель:
см. docs/ml.md и docs/risks-and-ops.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Классы, при которых текст нельзя отправлять во внешний LLM API ни в каком виде.
PAYMENT_CLASSES = frozenset({"card", "cvv", "account"})

PLACEHOLDERS = {
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "card": "[CARD]",
    "cvv": "[CVV]",
    "snils": "[SNILS]",
    "inn": "[INN]",
    "passport": "[PASSPORT]",
    "account": "[ACCOUNT]",
    "ip": "[IP]",
    "address": "[ADDRESS]",
}


@dataclass(frozen=True)
class PiiSpan:
    pii_class: str
    start: int
    end: int


@dataclass(frozen=True)
class Redaction:
    text: str
    classes: frozenset[str]
    spans: tuple[PiiSpan, ...]

    @property
    def has_payment_pii(self) -> bool:
        return bool(self.classes & PAYMENT_CLASSES)


# Порядок важен: более специфичные и более длинные сущности заявляют свои позиции
# первыми, иначе двадцатизначный счёт распадётся на «карту» и «телефон».
_DETECTORS: tuple[tuple[str, re.Pattern[str], int], ...] = (
    ("account", re.compile(r"\b\d{20}\b"), 0),
    ("snils", re.compile(r"\b\d{3}-\d{3}-\d{3}[ -]\d{2}\b"), 0),
    ("card", re.compile(r"\b\d(?:[\d \-]{11,21})\d\b"), 0),
    ("passport", re.compile(r"(?:паспорт\w*|passport)\D{0,10}(\d{4}\s?\d{6})\b", re.IGNORECASE), 1),
    ("inn", re.compile(r"(?:ИНН|INN)\W{0,5}(\d{10}|\d{12})\b", re.IGNORECASE), 1),
    ("cvv", re.compile(r"\b(?:cvv|cvc)\W{0,5}(\d{3})\b", re.IGNORECASE), 1),
    ("phone", re.compile(r"(?:\+7|\b8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b"), 0),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"), 0),
    ("ip", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), 0),
    (
        "address",
        re.compile(
            r"(?:ул\.|улица|пр-т|проспект|пер\.)\s*[А-ЯЁа-яё\w\- ]{2,30},?\s*д\.?\s*\d+[а-я]?"
            r"(?:\s*,?\s*кв\.?\s*\d+)?",
            re.IGNORECASE,
        ),
        0,
    ),
)


def redact(text: str) -> Redaction:
    """Заменяет найденные PII на плейсхолдеры. Идемпотентно: плейсхолдеры цифр не содержат."""
    if not text:
        return Redaction(text="", classes=frozenset(), spans=())

    claimed: list[PiiSpan] = []
    for pii_class, pattern, group in _DETECTORS:
        for match in pattern.finditer(text):
            start, end = match.span(group)
            if pii_class == "card" and not _is_card(match.group(group)):
                continue
            if any(start < s.end and s.start < end for s in claimed):
                continue  # позиция уже занята более специфичным детектором
            claimed.append(PiiSpan(pii_class, start, end))

    claimed.sort(key=lambda s: s.start)
    out = text
    for span in reversed(claimed):
        out = out[: span.start] + PLACEHOLDERS[span.pii_class] + out[span.end :]

    return Redaction(
        text=out,
        classes=frozenset(s.pii_class for s in claimed),
        spans=tuple(claimed),
    )


def _is_card(candidate: str) -> bool:
    """Номер карты отличается от номера заказа контрольной суммой Луна.

    Это сознательный выбор точности против полноты: без проверки Луна любой
    шестнадцатизначный идентификатор в тексте превращался бы в «карту».
    """
    digits = re.sub(r"\D", "", candidate)
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0
