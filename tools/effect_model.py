"""Модель годового эффекта MVP в деньгах.

Зачем отдельный модуль, а не таблица в markdown: цифры в docs/product.md должны быть
воспроизводимы и покрыты тестами. Любое изменение допущения — правка одного поля здесь
и повторный прогон `python tools/effect_model.py`, а не ручной пересчёт в тексте.

Экономическая логика (её же фиксируют тесты в tests/test_effect_model.py):

1. Baseline — это не «150 ₽ за тикет». В baseline оператор тоже получает reopen 9%,
   и переоткрытый тикет стоит дороже первичного (контекст потерян, клиент раздражён).
2. Автозакрытый тикет не бесплатен: с вероятностью reopen он всё равно приходит
   к оператору, причём уже как повторное обращение.
3. Сэкономленное время оператора не равно сэкономленным деньгам: смены штатные,
   поэтому вводим realization factor — долю экономии времени, которая реально
   конвертируется в снижение расходов в первый год.
4. Из экономии вычитаем стоимость владения: LLM-инференс, инфраструктура, run-команда.

Все числа — годовые, в рублях. Вводные кейса (200k тикетов/сутки, 150 ₽/тикет,
reopen 9%, доля типовых 40%) не являются нашими допущениями; всё остальное — наши,
каждое имеет обоснование в ASSUMPTIONS.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class EffectModel:
    # --- вводные кейса (не наши допущения) ---
    tickets_per_day: int = 200_000
    days_per_year: int = 365
    operator_cost_rub: float = 150.0
    baseline_reopen_rate: float = 0.09
    typical_share: float = 0.40

    # --- наши допущения: воронка автоматизации ---
    allowlist_share_of_typical: float = 0.45
    confident_safe_share: float = 0.65

    # --- наши допущения: качество и его цена ---
    rework_multiplier: float = 1.2
    auto_reopen_rate: float = 0.15
    auto_close_realization: float = 0.90

    # --- наши допущения: suggest-режим ---
    suggest_share: float = 0.35
    suggest_time_saving: float = 0.20
    suggest_realization: float = 0.60

    # --- наши допущения: стоимость LLM ---
    llm_draft_share: float = 0.47
    tokens_in_per_draft: int = 1800
    tokens_out_per_draft: int = 350
    self_hosted_share: float = 0.85
    self_hosted_rub_per_1k: float = 0.12
    external_rub_per_1k: float = 2.50

    # --- наши допущения: прочая стоимость владения ---
    infra_cost: float = 12_000_000.0
    run_team_cost: float = 9_000_000.0

    # ------------------------------------------------------------------ объёмы

    @property
    def tickets_per_year(self) -> float:
        return self.tickets_per_day * self.days_per_year

    @property
    def eligible_share(self) -> float:
        """Доля потока, попадающая в allowlist-категории автозакрытия."""
        return self.typical_share * self.allowlist_share_of_typical

    @property
    def auto_close_share(self) -> float:
        """Из eligible автозакрываем только то, где модель уверена и безопасна."""
        return self.eligible_share * self.confident_safe_share

    @property
    def auto_closed_tickets(self) -> float:
        return self.tickets_per_year * self.auto_close_share

    # ---------------------------------------------------------------- baseline

    @property
    def rework_cost(self) -> float:
        """Стоимость повторной обработки переоткрытого тикета."""
        return self.operator_cost_rub * self.rework_multiplier

    @property
    def baseline_cost_per_ticket(self) -> float:
        return self.operator_cost_rub + self.baseline_reopen_rate * self.rework_cost

    @property
    def baseline_annual_cost(self) -> float:
        return self.tickets_per_year * self.baseline_cost_per_ticket

    # ------------------------------------------------------------ автозакрытие

    @property
    def automated_cost_per_ticket(self) -> float:
        """Автоответ бесплатен только пока его не переоткрыли."""
        return self.auto_reopen_rate * self.rework_cost

    @property
    def saving_per_auto_closed_ticket(self) -> float:
        return self.baseline_cost_per_ticket - self.automated_cost_per_ticket

    @property
    def auto_close_saving(self) -> float:
        return (
            self.auto_closed_tickets
            * self.saving_per_auto_closed_ticket
            * self.auto_close_realization
        )

    # ----------------------------------------------------------------- suggest

    @property
    def suggest_tickets(self) -> float:
        return self.tickets_per_year * self.suggest_share

    @property
    def suggest_saving(self) -> float:
        per_ticket = self.operator_cost_rub * self.suggest_time_saving
        return self.suggest_tickets * per_ticket * self.suggest_realization

    # ------------------------------------------------------------- стоимость

    @property
    def drafts_per_year(self) -> float:
        return self.tickets_per_year * self.llm_draft_share

    @property
    def tokens_per_year(self) -> float:
        return self.drafts_per_year * (self.tokens_in_per_draft + self.tokens_out_per_draft)

    @property
    def llm_cost_self_hosted(self) -> float:
        return self.tokens_per_year * self.self_hosted_share * self.self_hosted_rub_per_1k / 1000

    @property
    def llm_cost_external(self) -> float:
        external_share = 1.0 - self.self_hosted_share
        return self.tokens_per_year * external_share * self.external_rub_per_1k / 1000

    @property
    def llm_cost(self) -> float:
        return self.llm_cost_self_hosted + self.llm_cost_external

    @property
    def llm_cost_per_draft(self) -> float:
        return self.llm_cost / self.drafts_per_year

    @property
    def total_cost(self) -> float:
        return self.llm_cost + self.infra_cost + self.run_team_cost

    # --------------------------------------------------------------- итог

    @property
    def gross_saving(self) -> float:
        return self.auto_close_saving + self.suggest_saving

    @property
    def net_effect(self) -> float:
        return self.gross_saving - self.total_cost

    @property
    def net_effect_share_of_budget(self) -> float:
        return self.net_effect / self.baseline_annual_cost

    # --------------------------------------------------------------- служебное

    def tunable_fields(self) -> list[str]:
        return [f.name for f in fields(self)]


ASSUMPTIONS: dict[str, str] = {
    # вводные кейса
    "tickets_per_day": "Вводная кейса: 200k тикетов в сутки.",
    "days_per_year": "365 дней: поддержка работает без выходных.",
    "operator_cost_rub": "Вводная кейса: 150 ₽ за тикет (~8 минут работы оператора).",
    "baseline_reopen_rate": "Вводная кейса: текущий reopen rate 9%.",
    "typical_share": "Вводная кейса: ~40% потока — типовые/повторяющиеся обращения.",
    # воронка
    "allowlist_share_of_typical": (
        "45% типовых обращений попадают в 5 allowlist-категорий MVP "
        "(password_reset, order.status, delivery_delay, payment_failed-инфо, incident-broadcast). "
        "Остальные типовые тикеты типовые по формулировке, но требуют действий в системах."
    ),
    "confident_safe_share": (
        "65% eligible-тикетов проходят все пороги политики одновременно "
        "(p_topic >= 0.90, groundedness >= 0.75, нет PII и красных флагов). "
        "Оценка по опыту похожих систем; проверяется в shadow-режиме на 2-й неделе пилота."
    ),
    # качество
    "rework_multiplier": (
        "Повторная обработка стоит 1.2× первичной: контекст потерян, клиент раздражён, "
        "часто требуется эскалация на L2."
    ),
    "auto_reopen_rate": (
        "Reopen по автозакрытым 15% против 9% базовых. Это наш рабочий прогноз и "
        "одновременно guardrail: 15% — порог остановки раскатки."
    ),
    "auto_close_realization": (
        "90% экономии на автозакрытых тикетах превращается в деньги: тикет физически "
        "не доходит до оператора, но штат сокращается ступенчато, с лагом на квартал."
    ),
    # suggest
    "suggest_share": (
        "35% потока получают черновик оператору: eligible-тикеты, не прошедшие порог "
        "автозакрытия, плюс частые нетиповые темы с хорошей базой знаний."
    ),
    "suggest_time_saving": (
        "20% экономии времени обработки при готовом черновике (8 мин -> 6.4 мин). "
        "Консервативно: публичные замеры ассистентов поддержки дают 14-35%, "
        "берём нижнюю треть диапазона."
    ),
    "suggest_realization": (
        "Только 60% сэкономленного времени конвертируется в деньги: смены штатные, "
        "часть высвобожденного времени уходит в буфер и в разбор сложных тикетов."
    ),
    # LLM
    "llm_draft_share": (
        "Черновик генерируется для 47% потока (eligible 18% + suggest 35% минус пересечение). "
        "Для остальных генерация запрещена политикой или бессмысленна (нет опоры в KB)."
    ),
    "tokens_in_per_draft": (
        "1800 входных токенов: системный промпт + история обращения + 3 фрагмента KB."
    ),
    "tokens_out_per_draft": "350 выходных токенов: ответ поддержки обычно короткий.",
    "self_hosted_share": (
        "85% черновиков закрывает self-hosted Qwen3-8B на vLLM; остальное — внешний "
        "frontier-API для сложных случаев и для тикетов без PII."
    ),
    "self_hosted_rub_per_1k": (
        "0.12 ₽ за 1k токенов на self-hosted: амортизация аренды GPU с 4-кратным запасом "
        "к расчётной пропускной способности vLLM."
    ),
    "external_rub_per_1k": (
        "2.50 ₽ за 1k токенов blended у внешнего API — консервативная оценка по "
        "прайсам frontier-моделей на входные и выходные токены."
    ),
    # стоимость владения
    "infra_cost": (
        "12 млн ₽/год инфраструктуры: CPU-флот эмбеддингов на горячем пути, векторное "
        "и BM25-хранилище, дополнительные топики Kafka, ClickHouse под аналитику."
    ),
    "run_team_cost": (
        "9 млн ₽/год на поддержку системы после MVP: 1 ML-инженер + 0.5 backend "
        "+ 0.3 аналитик в режиме run."
    ),
}


@dataclass(frozen=True)
class SensitivityRow:
    auto_reopen_rate: float
    saving_per_auto_closed_ticket: float
    auto_close_saving: float
    net_effect: float
    verdict: str


def sensitivity_to_reopen(rates: list[float]) -> list[SensitivityRow]:
    """Как меняется годовой эффект при росте reopen по автозакрытым тикетам."""
    rows: list[SensitivityRow] = []
    for rate in rates:
        m = dataclasses.replace(EffectModel(), auto_reopen_rate=rate)
        if rate <= 0.12:
            verdict = "целевой коридор"
        elif rate <= 0.15:
            verdict = "порог guardrail, раскатка останавливается"
        else:
            verdict = "продукт вреден пользователю, деньги неважны"
        rows.append(
            SensitivityRow(
                auto_reopen_rate=rate,
                saving_per_auto_closed_ticket=m.saving_per_auto_closed_ticket,
                auto_close_saving=m.auto_close_saving,
                net_effect=m.net_effect,
                verdict=verdict,
            )
        )
    return rows


def _mln(value: float) -> str:
    return f"{value / 1e6:,.1f}".replace(",", " ")


def format_markdown(m: EffectModel) -> str:
    """Таблицы для docs/product.md. Печатается как есть, руками не правится."""
    lines: list[str] = []
    lines.append("| Строка | Значение | Как получено |")
    lines.append("|---|---:|---|")
    lines.append(
        f"| Базовая стоимость поддержки, млн ₽/год | {_mln(m.baseline_annual_cost)} | "
        f"{m.tickets_per_year / 1e6:.0f} млн тикетов × {m.baseline_cost_per_ticket:.1f} ₽ "
        f"(150 ₽ + {m.baseline_reopen_rate:.0%} reopen × {m.rework_cost:.0f} ₽) |"
    )
    lines.append(
        f"| Автозакрытие: доля потока | {m.auto_close_share:.1%} | "
        f"{m.typical_share:.0%} типовых × {m.allowlist_share_of_typical:.0%} allowlist × "
        f"{m.confident_safe_share:.0%} прошли пороги |"
    )
    lines.append(
        f"| Автозакрытие: экономия, млн ₽/год | {_mln(m.auto_close_saving)} | "
        f"{m.auto_closed_tickets / 1e6:.2f} млн × {m.saving_per_auto_closed_ticket:.1f} ₽ × "
        f"{m.auto_close_realization:.0%} realization |"
    )
    lines.append(
        f"| Suggest-режим: экономия, млн ₽/год | {_mln(m.suggest_saving)} | "
        f"{m.suggest_tickets / 1e6:.1f} млн × {m.operator_cost_rub * m.suggest_time_saving:.0f} ₽ "
        f"({m.suggest_time_saving:.0%} времени) × {m.suggest_realization:.0%} realization |"
    )
    lines.append(
        f"| LLM-инференс, млн ₽/год | −{_mln(m.llm_cost)} | "
        f"{m.drafts_per_year / 1e6:.1f} млн черновиков × "
        f"{m.tokens_in_per_draft + m.tokens_out_per_draft} токенов; "
        f"{m.self_hosted_share:.0%} self-hosted / {1 - m.self_hosted_share:.0%} внешний API; "
        f"{m.llm_cost_per_draft:.2f} ₽ за черновик |"
    )
    lines.append(
        f"| Инфраструктура и run-команда, млн ₽/год | "
        f"−{_mln(m.infra_cost + m.run_team_cost)} | см. ASSUMPTIONS |"
    )
    lines.append(
        f"| **Чистый эффект, млн ₽/год** | **{_mln(m.net_effect)}** | "
        f"{m.net_effect_share_of_budget:.1%} от бюджета поддержки |"
    )

    lines.append("")
    lines.append("Чувствительность к reopen rate по автозакрытым тикетам:")
    lines.append("")
    lines.append("| reopen по автозакрытым | Экономия на тикет, ₽ | Чистый эффект, млн ₽/год | Трактовка |")
    lines.append("|---:|---:|---:|---|")
    for row in sensitivity_to_reopen([0.09, 0.12, 0.15, 0.25, 0.40]):
        lines.append(
            f"| {row.auto_reopen_rate:.0%} | {row.saving_per_auto_closed_ticket:.1f} | "
            f"{_mln(row.net_effect)} | {row.verdict} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_markdown(EffectModel()))
