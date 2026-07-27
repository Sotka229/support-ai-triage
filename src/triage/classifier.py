"""Классификатор темы обращения — ступень (1) лестницы baseline из docs/ml.md.

Что это: TF-IDF по символьным n-граммам + центроид класса + косинус, обученный
на 39 примерах из data/train_tickets.jsonl. Что это НЕ: продуктовая модель.
Уверенность здесь не калиброванная вероятность, а softmax по косинусам —
в целевой системе на её место встаёт логистическая регрессия над эмбеддингами
с temperature scaling, потому что порог 0.90 имеет смысл только на калиброванных
вероятностях.

Механика отказа (abstain) при этом уже настоящая: если максимальное косинусное
сходство ниже MIN_SIMILARITY, тема становится `other`, и политика отправит тикет
человеку вместо того, чтобы угадывать.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .models import Classification
from .vectorizer import TfidfSpace, centroid, cosine

#: Ниже этого косинуса считаем, что тема неизвестна, и отказываемся угадывать.
MIN_SIMILARITY = 0.10
#: Температура softmax. Чем меньше, тем резче распределение.
TEMPERATURE = 0.03


def load_labelled(path: Path | str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pairs.append((row["text"], row["topic"]))
    return pairs


@dataclass
class TopicClassifier:
    space: TfidfSpace
    centroids: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def train(cls, examples: list[tuple[str, str]]) -> "TopicClassifier":
        space = TfidfSpace.fit([text for text, _ in examples], mode="chars")
        grouped: dict[str, list[dict[str, float]]] = {}
        for text, label in examples:
            grouped.setdefault(label, []).append(space.transform(text))
        return cls(space=space, centroids={label: centroid(vs) for label, vs in grouped.items()})

    def predict(self, text: str) -> Classification:
        vector = self.space.transform(text)
        similarities = {label: cosine(vector, c) for label, c in self.centroids.items()}
        scores = _softmax(similarities, TEMPERATURE)

        best_label = max(similarities, key=lambda label: similarities[label])
        confidence = max(scores.values()) if scores else 0.0
        topic = best_label if similarities[best_label] >= MIN_SIMILARITY else "other"
        return Classification(topic=topic, confidence=confidence, scores=scores)


def _softmax(values: dict[str, float], temperature: float) -> dict[str, float]:
    if not values:
        return {}
    scaled = {label: value / temperature for label, value in values.items()}
    shift = max(scaled.values())
    exponentials = {label: pow(2.718281828459045, value - shift) for label, value in scaled.items()}
    total = sum(exponentials.values())
    return {label: value / total for label, value in exponentials.items()}
