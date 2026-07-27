"""Измеренные границы валидатора groundedness.

Файл появился после прогона PoC на живой модели (Ollama, llama3.1:8b и
qwen3.5:9b) — разбор в SELF_REVIEW.md. Он не проверяет, что код работает:
он фиксирует, где текущая реализация недостаточна, чтобы это нельзя было
случайно забыть и чтобы починка была видна в тестах.

Смысл такой формы записи: пока лексическая проверка стоит на месте NLI,
слабое место описано исполняемым кодом, а не абзацем в README, который
никто не перечитывает. Когда groundedness заменят на NLI-проверку,
xfail-тест начнёт проходить, strict=True уронит прогон и заставит убрать
пометку осознанно.
"""

import pytest

from triage.models import RetrievedChunk
from triage.safety import validate

KB_PASSWORD = RetrievedChunk(
    doc_id="kb-001-password-reset",
    title="Восстановление пароля и повторная отправка письма",
    text=(
        "Письмо для сброса пароля приходит на адрес регистрации и действует 60 минут. "
        "Проверьте папку «Спам», затем запросите письмо повторно по ссылке «Забыли пароль» "
        "на странице входа. Ссылка из старого письма после нового запроса перестаёт работать."
    ),
    score=0.49,
    answer_snippet="Проверьте папку «Спам» и запросите письмо повторно.",
)


class TestWhatTheValidatorDoesCatch:
    """Эти свойства измерены и держатся — на них опирается политика автозакрытия."""

    def test_answer_from_another_topic_is_rejected(self):
        report = validate(
            "Здравствуйте! Ваш заказ уже в пути, курьер приедет завтра с 10 до 18.",
            [KB_PASSWORD],
        )
        assert report.groundedness < 0.75

    def test_promise_of_money_is_caught_by_a_separate_rule(self):
        """Обещание денег ловится правилом, а не оценкой опоры: обещать возврат
        нельзя даже тогда, когда формулировка идеально соответствует источнику."""
        report = validate(
            "Мы вернём вам 7500 рублей и гарантируем компенсацию за неудобства.",
            [KB_PASSWORD],
        )
        assert "refund_promise" in report.forbidden_promises
        assert "guarantee" in report.forbidden_promises

    def test_real_model_paraphrase_stays_grounded(self):
        """Живая llama3.1:8b переформулирует источник своими словами. Валидатор
        не должен наказывать за пересказ — иначе автозакрытие не сработает никогда."""
        report = validate(
            "Проверьте папку «Спам» и «Промоакции», убедитесь, что адрес введён без "
            "опечаток, и запросите письмо заново на странице входа.",
            [KB_PASSWORD],
        )
        assert report.groundedness >= 0.75


FABRICATION = "Пароль восстановится автоматически в течение 24 часов, ничего делать не нужно."


@pytest.fixture
def real_kb_chunks():
    """Ровно то, что кладёт в контекст пайплайн: `kb.search(text, k=3)`.

    Брать настоящую базу знаний здесь принципиально — разрыв проявляется именно
    на нескольких фрагментах, а на одном его не видно (см. класс ниже).
    """
    from pathlib import Path

    from triage.retrieval import KnowledgeBase

    kb = KnowledgeBase.from_dir(Path(__file__).resolve().parents[1] / "data" / "kb")
    return kb.search("Забыл пароль, письмо для сброса не приходит на почту.", k=3)


class TestKnownGap:
    """Разрыв, измеренный на живой модели: лексическая опора != следование."""

    def test_single_chunk_context_rejects_the_fabrication(self, ):
        """С одним фрагментом проверка работает как задумано."""
        assert validate(FABRICATION, [KB_PASSWORD]).groundedness < 0.75

    def test_three_chunk_context_lets_the_same_fabrication_through(self, real_kb_chunks):
        """А с тремя — та же выдумка проходит на отлично.

        Причина: источник для сравнения — объединение словарей всех фрагментов.
        Чем больше k у ретривера, тем шире словарь и тем легче любому связному
        тексту набрать пересечение. То есть параметр качества поиска напрямую
        ослабляет валидатор безопасности — неочевидная связь, которую видно
        только на прогоне, а не на чтении кода.
        """
        assert len(real_kb_chunks) == 3
        assert validate(FABRICATION, real_kb_chunks).groundedness == 1.0

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Известный разрыв: groundedness считает пересечение словоформ с объединением "
            "фрагментов, а не логическое следование из них. Правдоподобная выдумка, "
            "собранная из лексики источников, получает 1.000 и проходит порог "
            "автозакрытия. Чинится NLI-проверкой на уровне утверждения против КАЖДОГО "
            "фрагмента отдельно — docs/ml.md, раздел валидации."
        ),
    )
    def test_fabrication_should_be_rejected_in_any_context_size(self, real_kb_chunks):
        """Утверждение противоречит базе знаний: там сказано запросить письмо
        повторно, а не ждать сутки. Для клиента это худший вид ошибки — он
        поверит и не станет ничего делать, а тикет вернётся через сутки в reopen."""
        assert validate(FABRICATION, real_kb_chunks).groundedness < 0.75
