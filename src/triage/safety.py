"""Валидаторы черновика перед отправкой.

Порядок мышления здесь обратный обычному: мы не пытаемся оценить, «хороший ли ответ»,
мы ищем причины его не отправлять. Четыре причины, каждая — отдельный кейс из
docs/risks-and-ops.md:

* ответ не опирается на базу знаний (галлюцинация);
* в ответе оказались персональные данные (утечка через шаблон или через контекст);
* в ответ просочились инструкции для модели (prompt injection, доехавший до выхода);
* ответ обещает деньги или действия, которых система не совершала.

Про инъекции важно не перепутать рубежи. `detect_injection` здесь применяется к
ЧЕРНОВИКУ и ловит случай, когда вредонос доехал до выхода — например, приехал из
статьи базы знаний. Инъекция в самом обращении ловится раньше и дешевле, правилами
на быстром пути (`rules.evaluate`), до всякой генерации: атака направлена на
генератор, поэтому останавливать её после генерации поздно.
"""

from __future__ import annotations

import re

from .models import RetrievedChunk, SafetyReport
from .pii import redact
from .vectorizer import word_tokens

#: Доля опорных токенов предложения, при которой оно считается подтверждённым источником.
SENTENCE_SUPPORT = 0.6

_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ignore_instructions", re.compile(r"(забудь|игнорируй)\w*\s+(все\s+)?(предыдущ\w+|прежн\w+)\s+инструкц\w+", re.I)),
    ("ignore_instructions_en", re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.I)),
    ("reveal_prompt", re.compile(r"(system\s*prompt|систем\w+\s+промпт|выдай\s+промпт|покажи\s+инструкц\w+)", re.I)),
    ("role_override", re.compile(r"(ты\s+теперь|act\s+as|you\s+are\s+now)\s+\w+", re.I)),
    ("fake_system_turn", re.compile(r"(#{2,}\s*system|<\|?system\|?>|\[system\])", re.I)),
    ("exfiltration", re.compile(r"(верни|покажи|выведи)\w*\s+(все\s+)?(данные|пароли|логины)\s+пользовател\w+", re.I)),
)

_PROMISE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("refund_promise", re.compile(r"(верн[ёе]м|вернем|возврат\w*\s+оформл\w+|спишем\s+средства\s+обратно)", re.I)),
    ("compensation_promise", re.compile(r"(компенсаци\w+|бонус\w*\s+в\s+качестве)", re.I)),
    ("guarantee", re.compile(r"(гарантиру\w+|обещаю|обещаем)", re.I)),
)


def groundedness(draft_text: str, chunks: list[RetrievedChunk]) -> float:
    """Доля предложений черновика, подтверждённых текстом найденных фрагментов.

    Это упрощение: в целевой системе здесь NLI-модель или проверка цитат
    (docs/ml.md). Но свойство, которое мы проверяем, то же самое — ответ,
    собранный из воздуха, не проходит порог.
    """
    if not chunks or not draft_text.strip():
        return 0.0

    source = set(word_tokens(" ".join(chunk.text for chunk in chunks), drop_stopwords=True, stem_to=5))
    sentences = [s for s in re.split(r"[.!?]+", draft_text) if s.strip()]
    if not sentences:
        return 0.0

    supported = 0
    for sentence in sentences:
        tokens = [t for t in word_tokens(sentence, drop_stopwords=True, stem_to=5) if len(t) > 2]
        if not tokens:
            supported += 1  # «Здравствуйте!» — не утверждение о фактах
            continue
        overlap = sum(1 for token in tokens if token in source) / len(tokens)
        if overlap >= SENTENCE_SUPPORT:
            supported += 1
    return round(supported / len(sentences), 4)


def detect_injection(text: str) -> tuple[str, ...]:
    return tuple(name for name, pattern in _INJECTION_PATTERNS if pattern.search(text))


def forbidden_promises(text: str) -> tuple[str, ...]:
    return tuple(name for name, pattern in _PROMISE_PATTERNS if pattern.search(text))


def validate(draft_text: str, chunks: list[RetrievedChunk]) -> SafetyReport:
    return SafetyReport(
        groundedness=groundedness(draft_text, chunks),
        has_pii=bool(redact(draft_text).classes),
        injection_markers=detect_injection(draft_text),
        forbidden_promises=forbidden_promises(draft_text),
    )
