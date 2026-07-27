"""Слой вызова LLM: мок-генератор, retry, circuit breaker, деградация.

Мок-модель здесь намеренно тупая: она собирает ответ из найденного фрагмента базы
знаний и не умеет ничего придумывать. Это не имитация качества LLM, а фиксация
контракта, который в целевой системе останется тем же:

* генерация невозможна без найденного источника (нет контекста -> ValueError);
* ответ обязан ссылаться на конкретные документы (cited_doc_ids);
* любая недоступность upstream выражается одним исключением LLMUnavailable,
  и выше по стеку это означает «работаем без черновика», а не «падаем».

В целевой системе на место TemplateLLM встаёт каскад self-hosted Qwen3-8B (vLLM)
и внешнего frontier-API; GuardedLLM и CircuitBreaker переносятся без изменений.
"""

from __future__ import annotations

import time

from .models import Draft, RetrievedChunk


class LLMUnavailable(RuntimeError):
    """LLM недоступен: таймаут, 5xx, открытый предохранитель, исчерпанный бюджет."""


class TemplateLLM:
    """Мок-генератор: ответ строится только из фрагмента базы знаний."""

    model = "mock-template-v1"

    def generate(self, prompt: str, chunks: list[RetrievedChunk]) -> Draft:
        if not chunks:
            raise ValueError("генерация без опоры на базу знаний запрещена")
        top = chunks[0]
        # Черновик — это только то, что порождено из базы знаний. Дежурная приписка
        # «если не помогло, ответьте, и подключится оператор» добавляется слоем
        # отправки в канал: она не должна участвовать в оценке groundedness,
        # иначе шаблонная строка снижает оценку опоры на источник у любого ответа.
        text = top.answer_snippet or _first_sentences(top.text, 2)
        return Draft(
            text=text,
            cited_doc_ids=tuple(chunk.doc_id for chunk in chunks[:1]),
            model=self.model,
            tokens_in=_tokens(prompt) + sum(_tokens(c.text) for c in chunks),
            tokens_out=_tokens(text),
        )


class CircuitBreaker:
    """Классический предохранитель: closed -> open -> half_open -> closed.

    Смысл в потоке 200k тикетов/сутки: когда LLM лежит, нельзя продолжать
    отправлять в него запросы — это удлиняет очередь и жжёт бюджет таймаутов.
    """

    def __init__(self, *, failure_threshold: int = 5, reset_after: float = 30.0, clock=time.monotonic):
        self.failure_threshold = failure_threshold
        self.reset_after = reset_after
        self._clock = clock
        self._state = "closed"
        self._failures = 0
        self._opened_at = 0.0

    @property
    def state(self) -> str:
        return self._state

    def allow(self) -> bool:
        if self._state == "open":
            if self._clock() - self._opened_at >= self.reset_after:
                self._state = "half_open"  # пропускаем одну пробу
                return True
            return False
        return True

    def record_success(self) -> None:
        self._state = "closed"
        self._failures = 0

    def record_failure(self) -> None:
        self._failures += 1
        if self._state == "half_open" or self._failures >= self.failure_threshold:
            self._state = "open"
            self._opened_at = self._clock()


class GuardedLLM:
    """Обёртка вокруг любого клиента: ретраи, предохранитель, один тип отказа."""

    def __init__(self, client, *, retries: int = 2, breaker: CircuitBreaker | None = None):
        self.client = client
        self.retries = max(retries, 1)
        self.breaker = breaker or CircuitBreaker()

    @property
    def model(self) -> str:
        return getattr(self.client, "model", "unknown")

    def generate(self, prompt: str, chunks: list[RetrievedChunk]) -> Draft:
        if not self.breaker.allow():
            raise LLMUnavailable("circuit breaker is open")

        last_error: Exception | None = None
        for _ in range(self.retries):
            try:
                draft = self.client.generate(prompt, chunks)
            except ValueError:
                raise  # отсутствие контекста — наша ошибка, ретраить нечего
            except (LLMUnavailable, TimeoutError, ConnectionError) as exc:
                last_error = exc
                self.breaker.record_failure()
                if self.breaker.state == "open":
                    break  # проверяем состояние, а не allow(): проба — дело следующего вызова
            else:
                self.breaker.record_success()
                return draft
        raise LLMUnavailable(f"генерация не удалась: {last_error}")


def _first_sentences(text: str, count: int) -> str:
    body = " ".join(line for line in text.splitlines()[1:] if line.strip())
    sentences = [s.strip() for s in body.split(".") if s.strip()]
    return ". ".join(sentences[:count]) + "." if sentences else ""


def _tokens(text: str) -> int:
    """Грубая оценка: ~4 символа на токен. Для мониторинга стоимости в PoC достаточно."""
    return max(len(text) // 4, 1)
