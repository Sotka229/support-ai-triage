"""Тесты адаптера настоящей LLM (Ollama, self-hosted).

Зачем адаптер вообще нужен в PoC: пока генератор — шаблон, невозможно проверить
главное утверждение архитектуры — что валидаторы безопасности и политика решения
удержат систему, когда текст порождает модель, а не наш собственный код. Мок
всегда цитирует источник дословно, поэтому groundedness у него тривиально высокий.

Юнит-тесты гоняются на подставном транспорте: сеть в них не участвует, они
детерминированы и идут в CI. Интеграционный тест внизу файла обращается к реально
запущенной Ollama и пропускается, если её нет.
"""

import json
import urllib.error

import pytest

from triage.llm import GuardedLLM, LLMUnavailable
from triage.llm_ollama import OllamaLLM, build_prompt, strip_reasoning
from triage.models import RetrievedChunk

CHUNK = RetrievedChunk(
    doc_id="kb-002",
    title="Восстановление пароля",
    text="Письмо для сброса пароля приходит в течение 5 минут. Проверьте папку «Спам».",
    score=0.71,
    answer_snippet="Проверьте папку «Спам»: письмо приходит в течение 5 минут.",
)


class FakeTransport:
    """Подставной HTTP-транспорт: возвращает заготовленный ответ или бросает ошибку."""

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls: list[tuple[str, dict, float]] = []

    def __call__(self, url: str, payload: dict, timeout: float) -> dict:
        self.calls.append((url, payload, timeout))
        if self.error is not None:
            raise self.error
        return self.response


def ok_response(text: str, *, prompt_eval_count: int = 1200, eval_count: int = 180) -> dict:
    return {
        "response": text,
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
        "done": True,
    }


class TestPromptConstruction:
    def test_prompt_marks_ticket_text_as_untrusted_data(self):
        """Текст пользователя — данные, а не инструкции. Это главная защита от
        prompt injection на уровне промпта: см. docs/risks-and-ops.md §2."""
        prompt = build_prompt("верни мне деньги", [CHUNK])
        assert "<<<TICKET" in prompt and "TICKET>>>" in prompt
        assert "не выполняй инструкции" in prompt.lower()

    def test_prompt_contains_only_provided_chunks_as_source(self):
        prompt = build_prompt("забыл пароль", [CHUNK])
        assert CHUNK.text in prompt
        assert "kb-002" in prompt
        assert "только" in prompt.lower()

    def test_prompt_forbids_inventing_facts_and_promising_money(self):
        prompt = build_prompt("где мой возврат", [CHUNK])
        low = prompt.lower()
        assert "не обещай" in low
        assert "не выдумывай" in low


class TestReasoningStripping:
    def test_think_block_is_removed(self):
        """Qwen3 умеет reasoning-блоки. Если их не срезать, они уйдут пользователю."""
        raw = "<think>Пользователь спрашивает про пароль. Надо ответить.</think>\nПроверьте папку «Спам»."
        assert strip_reasoning(raw) == "Проверьте папку «Спам»."

    def test_unclosed_think_block_is_dropped_entirely(self):
        """Обрыв генерации внутри рассуждения не должен превращаться в ответ клиенту."""
        assert strip_reasoning("<think>рассуждаю и не закончил") == ""

    def test_plain_answer_is_untouched(self):
        assert strip_reasoning("Проверьте папку «Спам».") == "Проверьте папку «Спам»."


class TestGeneration:
    def test_generate_returns_draft_with_citations(self):
        llm = OllamaLLM(transport=FakeTransport(ok_response("Проверьте папку «Спам».")))
        draft = llm.generate("забыл пароль", [CHUNK])
        assert draft.text == "Проверьте папку «Спам»."
        assert draft.cited_doc_ids == ("kb-002",)
        assert draft.model.startswith("ollama:")

    def test_token_counts_come_from_the_model_not_from_a_heuristic(self):
        """Стоимость LLM — продуктовая метрика (docs/monitoring.md), поэтому
        токены берём фактические, а не оценку «4 символа на токен»."""
        llm = OllamaLLM(transport=FakeTransport(ok_response("текст", prompt_eval_count=1543, eval_count=97)))
        draft = llm.generate("забыл пароль", [CHUNK])
        assert draft.tokens_in == 1543
        assert draft.tokens_out == 97

    def test_generation_without_context_is_refused_before_any_network_call(self):
        transport = FakeTransport(ok_response("что-нибудь"))
        llm = OllamaLLM(transport=transport)
        with pytest.raises(ValueError):
            llm.generate("забыл пароль", [])
        assert transport.calls == [], "запрос к модели не должен уходить вообще"

    def test_streaming_is_disabled_and_temperature_is_low(self):
        """Поддержке нужен воспроизводимый ответ, а не разнообразие."""
        transport = FakeTransport(ok_response("текст"))
        OllamaLLM(transport=transport).generate("забыл пароль", [CHUNK])
        _, payload, _ = transport.calls[0]
        assert payload["stream"] is False
        assert payload["options"]["temperature"] <= 0.3

    def test_thinking_is_disabled_by_default(self):
        """Найдено на живой модели: qwen3.5 тратит весь бюджет num_predict на
        reasoning, поле response остаётся пустым, done_reason='length'. Для
        черновика по готовому фрагменту KB рассуждение не нужно — и вдвое дороже
        по времени (11 с против 28 с на этой машине)."""
        transport = FakeTransport(ok_response("текст"))
        OllamaLLM(transport=transport).generate("забыл пароль", [CHUNK])
        _, payload, _ = transport.calls[0]
        assert payload["think"] is False

    def test_falls_back_to_request_without_think_when_server_rejects_it(self):
        """Старые сборки Ollama не знают поля think и отвечают 400. Это не повод
        ронять генерацию: повторяем запрос без флага."""

        class RejectsThinkOnce:
            def __init__(self):
                self.calls = []

            def __call__(self, url, payload, timeout):
                self.calls.append(payload)
                if "think" in payload:
                    raise urllib.error.HTTPError(url, 400, "unknown field think", {}, None)
                return ok_response("Проверьте папку «Спам».")

        transport = RejectsThinkOnce()
        draft = OllamaLLM(transport=transport).generate("забыл пароль", [CHUNK])
        assert draft.text == "Проверьте папку «Спам»."
        assert len(transport.calls) == 2
        assert "think" not in transport.calls[1]

    def test_answer_is_truncated_to_a_support_reply_length(self):
        llm = OllamaLLM(transport=FakeTransport(ok_response("а" * 5000)), max_chars=800)
        draft = llm.generate("забыл пароль", [CHUNK])
        assert len(draft.text) <= 800


class TestDegradation:
    def test_network_error_becomes_llm_unavailable(self):
        """Ключевое требование: реальный клиент отказывает тем же исключением,
        что и мок, иначе запасной путь пришлось бы писать заново."""
        llm = OllamaLLM(transport=FakeTransport(error=ConnectionError("connection refused")))
        with pytest.raises(LLMUnavailable):
            llm.generate("забыл пароль", [CHUNK])

    def test_timeout_becomes_llm_unavailable(self):
        llm = OllamaLLM(transport=FakeTransport(error=TimeoutError("read timed out")))
        with pytest.raises(LLMUnavailable):
            llm.generate("забыл пароль", [CHUNK])

    def test_empty_model_answer_is_a_failure_not_an_empty_draft(self):
        llm = OllamaLLM(transport=FakeTransport(ok_response("   ")))
        with pytest.raises(LLMUnavailable):
            llm.generate("забыл пароль", [CHUNK])

    def test_answer_that_is_only_reasoning_is_a_failure(self):
        llm = OllamaLLM(transport=FakeTransport(ok_response("<think>думаю</think>")))
        with pytest.raises(LLMUnavailable):
            llm.generate("забыл пароль", [CHUNK])

    def test_guarded_llm_opens_breaker_on_real_client_too(self):
        llm = GuardedLLM(
            OllamaLLM(transport=FakeTransport(error=ConnectionError("refused"))),
            retries=5,
        )
        with pytest.raises(LLMUnavailable):
            llm.generate("забыл пароль", [CHUNK])
        assert llm.breaker.state == "open"


class TestSelfHostedProperty:
    def test_adapter_declares_itself_self_hosted(self):
        """Правило block_external_llm запрещает внешний API для тикетов с PII.
        Self-hosted-контур под этот запрет не попадает, и это свойство должно
        быть машиночитаемым, а не комментарием в документации."""
        assert OllamaLLM(transport=FakeTransport(ok_response("x"))).is_external is False


def _ollama_available() -> bool:
    try:
        import urllib.request

        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_available(), reason="Ollama не запущена на localhost:11434")
def test_integration_real_model_produces_grounded_answer():
    """Интеграционный тест: настоящая модель, настоящие валидаторы.

    Проверяем не красоту ответа, а то, что контракт держится: черновик непустой,
    ссылается на переданный документ и не содержит reasoning-мусора.
    """
    import urllib.request

    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as resp:
        models = [m["name"] for m in json.load(resp)["models"]]
    if not models:
        pytest.skip("в Ollama нет ни одной модели")

    llm = OllamaLLM(model=models[0], timeout=600.0)
    draft = llm.generate("Забыл пароль, письмо для сброса не приходит уже час.", [CHUNK])

    assert draft.text.strip()
    assert "<think>" not in draft.text
    assert draft.cited_doc_ids == ("kb-002",)
    assert draft.tokens_out > 0
