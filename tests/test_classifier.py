"""Классификатор темы: baseline-ступень (1) из docs/ml.md.

Важное про этот тест: обучающий и проверочный наборы лежат в разных файлах
(data/train_tickets.jsonl и data/eval_tickets.jsonl). Мерить качество на обучающей
выборке бессмысленно, а именно так обычно и выглядит «работающий PoC».
Порог accuracy низкий сознательно: это baseline на 39 примерах, а не продуктовая модель.
"""

import time
from pathlib import Path

from triage.classifier import TopicClassifier, load_labelled
from triage.models import TOPICS

DATA = Path(__file__).resolve().parents[1] / "data"


def build() -> TopicClassifier:
    return TopicClassifier.train(load_labelled(DATA / "train_tickets.jsonl"))


class TestDataset:
    def test_train_and_eval_do_not_overlap(self):
        train = {t for t, _ in load_labelled(DATA / "train_tickets.jsonl")}
        evalset = {t for t, _ in load_labelled(DATA / "eval_tickets.jsonl")}
        assert train & evalset == set(), "утечка обучающих примеров в проверочные"

    def test_every_topic_from_the_taxonomy_has_training_examples(self):
        labels = {label for _, label in load_labelled(DATA / "train_tickets.jsonl")}
        assert labels == set(TOPICS)


class TestQuality:
    def test_accuracy_on_holdout_is_above_baseline(self):
        clf = build()
        pairs = load_labelled(DATA / "eval_tickets.jsonl")
        correct = sum(1 for text, label in pairs if clf.predict(text).topic == label)
        accuracy = correct / len(pairs)
        assert accuracy >= 0.70, f"accuracy на holdout {accuracy:.2f} — baseline сломан"

    def test_recall_on_risky_topics_is_perfect_on_holdout(self):
        """Пропустить fraud или legal дороже, чем ошибиться на order.status."""
        clf = build()
        risky = {"fraud.unauthorized_charge", "legal.complaint_threat", "account.data_deletion"}
        pairs = [(t, l) for t, l in load_labelled(DATA / "eval_tickets.jsonl") if l in risky]
        assert pairs
        for text, label in pairs:
            assert clf.predict(text).topic == label, f"risky-класс перепутан: {text}"


class TestAbstain:
    def test_out_of_domain_text_falls_back_to_other_with_low_confidence(self):
        clf = build()
        result = clf.predict("qwerty asdf zxcv 12345 blah blah")
        assert result.topic == "other"
        assert result.confidence < 0.5

    def test_confidence_is_a_probability_distribution(self):
        clf = build()
        r = clf.predict("Забыл пароль, письмо не приходит")
        assert 0.0 <= r.confidence <= 1.0
        assert abs(sum(r.scores.values()) - 1.0) < 1e-6
        assert r.confidence == max(r.scores.values())

    def test_confident_case_clears_the_auto_close_threshold(self):
        clf = build()
        r = clf.predict("Забыл пароль, письмо для сброса пароля не приходит на почту")
        assert r.topic == "account.password_reset"
        assert r.confidence >= 0.90, "иначе happy path демо не пройдёт порог политики"


class TestContract:
    def test_prediction_is_deterministic(self):
        clf = build()
        a, b = clf.predict("Где мой заказ"), clf.predict("Где мой заказ")
        assert (a.topic, a.confidence) == (b.topic, b.confidence)

    def test_prediction_fits_the_hot_path_budget(self):
        clf = build()
        texts = [t for t, _ in load_labelled(DATA / "eval_tickets.jsonl")]
        start = time.perf_counter()
        for _ in range(20):
            for text in texts:
                clf.predict(text)
        per_call_ms = (time.perf_counter() - start) * 1000 / (20 * len(texts))
        assert per_call_ms < 20, f"{per_call_ms:.1f} мс на предсказание — не влезаем в 500 мс"

    def test_empty_text_does_not_crash(self):
        assert build().predict("").topic == "other"
