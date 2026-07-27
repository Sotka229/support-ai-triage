"""Мини-TF-IDF на стандартной библиотеке.

В целевой системе здесь стоят эмбеддинги (rubert-tiny2 / multilingual-e5-small в ONNX)
и BM25 в OpenSearch — см. docs/ml.md. TF-IDF выбран для PoC по двум причинам:
он не тянет зависимостей (проверяющий запускает PoC одной командой) и он честно
показывает уровень ступени (1) лестницы baseline, а не имитирует качество продакшена.

Два режима сознательно разные:
* классификация — символьные n-граммы: устойчивы к опечаткам и словоформам,
  а тикеты пишут быстро и с ошибками;
* поиск по базе знаний — только слова со стоп-листом: символьные n-граммы дают
  ложное сходство между любыми русскими текстами, а нам нужно уметь сказать
  «в базе знаний ответа нет» и отказаться от генерации.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

_TOKEN = re.compile(r"[а-яёa-z0-9]+")

STOPWORDS = frozenset(
    """
    и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по
    только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если
    уже или ни быть был него до вас нибудь опять уж вам сказал ведь там потом себя
    ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без
    будто человек чего раз тоже себе под жизнь будет ж тогда кто этот того потому этого
    какой совсем ним здесь этом один почти мой тем чтобы нее кажется сейчас были куда
    зачем всех никогда можно при наконец два об другой хоть после над больше тот через
    эти нас про всего них какая много разве три эту моя впрочем хорошо свою этой перед
    иногда лучше чуть том нельзя такой им более всегда конечно всю между это пожалуйста
    здравствуйте добрый день вечер утро подскажите скажите
    """.split()
)


def word_tokens(text: str, *, drop_stopwords: bool = False, stem_to: int | None = None) -> list[str]:
    tokens = _TOKEN.findall(text.lower().replace("ё", "е"))
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if stem_to:
        tokens = [t[:stem_to] for t in tokens]
    return tokens


def char_ngrams(text: str, low: int = 3, high: int = 5) -> list[str]:
    padded = " " + " ".join(word_tokens(text)) + " "
    grams: list[str] = []
    for size in range(low, high + 1):
        grams.extend(padded[i : i + size] for i in range(len(padded) - size + 1))
    return grams


def features(text: str, *, mode: str) -> list[str]:
    if mode == "chars":
        return char_ngrams(text) + word_tokens(text)
    if mode == "words":
        return word_tokens(text, drop_stopwords=True, stem_to=5)
    raise ValueError(f"неизвестный режим векторизации: {mode}")


@dataclass
class TfidfSpace:
    """Обучается на корпусе, затем превращает текст в L2-нормированный разреженный вектор."""

    mode: str = "chars"
    idf: dict[str, float] = field(default_factory=dict)
    _docs: int = 0

    @classmethod
    def fit(cls, texts: list[str], *, mode: str = "chars") -> "TfidfSpace":
        space = cls(mode=mode)
        document_frequency: Counter[str] = Counter()
        for text in texts:
            document_frequency.update(set(features(text, mode=mode)))
        space._docs = max(len(texts), 1)
        space.idf = {
            term: math.log((space._docs + 1) / (df + 1)) + 1.0
            for term, df in document_frequency.items()
        }
        return space

    def transform(self, text: str) -> dict[str, float]:
        counts = Counter(features(text, mode=self.mode))
        vector: dict[str, float] = {}
        for term, count in counts.items():
            idf = self.idf.get(term)
            if idf is None:
                continue  # термин не встречался в корпусе — веса для него нет
            vector[term] = (1.0 + math.log(count)) * idf
        return l2_normalise(vector)


def l2_normalise(vector: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0.0:
        return {}
    return {term: value / norm for term, value in vector.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(term, 0.0) for term, value in a.items())


def centroid(vectors: list[dict[str, float]]) -> dict[str, float]:
    total: dict[str, float] = {}
    for vector in vectors:
        for term, value in vector.items():
            total[term] = total.get(term, 0.0) + value
    return l2_normalise(total)
