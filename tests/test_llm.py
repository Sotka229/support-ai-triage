"""Обёртка вокруг LLM: таймаут, retry, circuit breaker, деградация.

Смысл этих тестов — не проверить мок-модель, а зафиксировать поведение системы при
недоступности LLM. Это ровно тот сценарий, который в кейсе назван обязательным:
«система должна корректно деградировать при недоступности LLM API».
"""

import pytest

from triage.llm import CircuitBreaker, GuardedLLM, LLMUnavailable, TemplateLLM
from triage.retrieval import KnowledgeBase
from pathlib import Path

KB_DIR = Path(__file__).resolve().parents[1] / "data" / "kb"


class CountingLLM:
    """Мок-клиент: падает первые `fail_times` вызовов, потом отвечает."""

    model = "counting-mock"

    def __init__(self, fail_times: int = 0):
        self.fail_times = fail_times
        self.calls = 0

    def generate(self, prompt, chunks):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise LLMUnavailable("upstream 503")
        return TemplateLLM().generate(prompt, chunks)


def chunks():
    return KnowledgeBase.from_dir(KB_DIR).search("забыл пароль, письмо не приходит", k=2)


class TestTemplateLLM:
    def test_draft_is_grounded_in_the_retrieved_article(self):
        response = TemplateLLM().generate("Забыл пароль", chunks())
        assert response.text
        assert "kb-001-password-reset" in response.cited_doc_ids
        assert response.tokens_in > 0 and response.tokens_out > 0

    def test_without_context_it_refuses_to_invent_an_answer(self):
        with pytest.raises(ValueError):
            TemplateLLM().generate("Забыл пароль", [])


class TestRetries:
    def test_transient_failure_is_retried(self):
        client = CountingLLM(fail_times=1)
        guarded = GuardedLLM(client, retries=2)
        assert guarded.generate("Забыл пароль", chunks()).text
        assert client.calls == 2

    def test_persistent_failure_raises_after_retries_are_exhausted(self):
        client = CountingLLM(fail_times=99)
        guarded = GuardedLLM(client, retries=2)
        with pytest.raises(LLMUnavailable):
            guarded.generate("Забыл пароль", chunks())
        assert client.calls == 2, "не должны молотить upstream бесконечно"


class TestCircuitBreaker:
    def test_opens_after_threshold_and_then_fails_fast(self):
        client = CountingLLM(fail_times=99)
        breaker = CircuitBreaker(failure_threshold=3, reset_after=60.0)
        guarded = GuardedLLM(client, retries=1, breaker=breaker)

        for _ in range(3):
            with pytest.raises(LLMUnavailable):
                guarded.generate("x", chunks())
        assert breaker.state == "open"
        assert client.calls == 3

        with pytest.raises(LLMUnavailable):
            guarded.generate("x", chunks())
        assert client.calls == 3, "открытый предохранитель не должен звать upstream"

    def test_half_open_probe_closes_the_breaker_after_recovery(self):
        client = CountingLLM(fail_times=2)
        breaker = CircuitBreaker(failure_threshold=2, reset_after=0.0)
        guarded = GuardedLLM(client, retries=1, breaker=breaker)

        for _ in range(2):
            with pytest.raises(LLMUnavailable):
                guarded.generate("x", chunks())
        assert breaker.state == "open"

        assert guarded.generate("Забыл пароль", chunks()).text
        assert breaker.state == "closed"

    def test_timeout_is_reported_as_unavailability_not_as_a_crash(self):
        class SlowLLM:
            model = "slow"

            def generate(self, prompt, chunks):
                raise TimeoutError("deadline exceeded")

        with pytest.raises(LLMUnavailable):
            GuardedLLM(SlowLLM(), retries=1).generate("x", chunks())
