"""Тесты на продуктовую модель эффекта.

Написаны до реализации (TDD). Смысл тестов — не «проверить, что калькулятор считает»,
а зафиксировать экономическую логику, которую мы защищаем в docs/product.md:

* экономия считается против baseline-стоимости, в которой оператор тоже получает reopen 9%;
* переоткрытый автоответ не бесплатен: тикет всё равно попадает к оператору и стоит дороже;
* экономия времени в suggest-режиме конвертируется в деньги не полностью (смены штатные);
* стоимость LLM и владения вычитается;
* при росте reopen эффект падает монотонно, но остаётся положительным — то есть
  ограничение на раскатку задают CSAT и доверие, а не деньги.
"""

import math

from tools.effect_model import (
    ASSUMPTIONS,
    EffectModel,
    format_markdown,
    sensitivity_to_reopen,
)


def approx(a: float, b: float, rel: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=rel)


class TestVolumes:
    def test_yearly_volume_matches_case_inputs(self):
        m = EffectModel()
        assert m.tickets_per_year == 200_000 * 365

    def test_auto_close_share_is_product_of_funnel_steps(self):
        m = EffectModel()
        # 40% типовых × 45% из них в allowlist × 65% уверенных и безопасных
        assert approx(m.auto_close_share, 0.40 * 0.45 * 0.65)
        assert approx(m.auto_closed_tickets, m.tickets_per_year * m.auto_close_share)

    def test_eligible_share_is_larger_than_auto_close_share(self):
        m = EffectModel()
        assert m.eligible_share > m.auto_close_share


class TestBaseline:
    def test_baseline_cost_per_ticket_includes_rework_on_reopen(self):
        m = EffectModel()
        # 150 + 9% × (150 × 1.2)
        assert approx(m.baseline_cost_per_ticket, 150 + 0.09 * 150 * 1.2)

    def test_baseline_annual_support_cost(self):
        m = EffectModel()
        assert approx(
            m.baseline_annual_cost, m.tickets_per_year * m.baseline_cost_per_ticket
        )


class TestAutoClose:
    def test_automated_ticket_still_costs_money_when_reopened(self):
        m = EffectModel()
        # переоткрытый автоответ = оператор + надбавка за rework
        assert approx(m.automated_cost_per_ticket, 0.15 * 150 * 1.2)

    def test_saving_per_auto_closed_ticket_is_baseline_minus_automated(self):
        m = EffectModel()
        assert approx(
            m.saving_per_auto_closed_ticket,
            m.baseline_cost_per_ticket - m.automated_cost_per_ticket,
        )

    def test_auto_close_saving_is_discounted_by_realization_factor(self):
        m = EffectModel()
        assert approx(
            m.auto_close_saving,
            m.auto_closed_tickets
            * m.saving_per_auto_closed_ticket
            * m.auto_close_realization,
        )
        assert m.auto_close_realization < 1.0, "штат сокращается ступенчато, не мгновенно"


class TestSuggestMode:
    def test_suggest_saving_uses_time_saving_and_realization(self):
        m = EffectModel()
        expected = m.tickets_per_year * m.suggest_share * (150 * 0.20) * 0.60
        assert approx(m.suggest_saving, expected)

    def test_suggest_realization_is_more_conservative_than_auto_close(self):
        m = EffectModel()
        assert m.suggest_realization < m.auto_close_realization


class TestCosts:
    def test_llm_cost_splits_cascade_between_self_hosted_and_external(self):
        m = EffectModel()
        drafts = m.tickets_per_year * m.llm_draft_share
        tokens = drafts * (1800 + 350)
        expected = (
            tokens * 0.85 * 0.12 / 1000  # self-hosted
            + tokens * 0.15 * 2.50 / 1000  # внешний API
        )
        assert approx(m.llm_cost, expected)

    def test_external_api_dominates_llm_cost_despite_small_share(self):
        m = EffectModel()
        assert m.llm_cost_external > m.llm_cost_self_hosted

    def test_total_cost_includes_infra_and_run_team(self):
        m = EffectModel()
        assert approx(m.total_cost, m.llm_cost + m.infra_cost + m.run_team_cost)


class TestNetEffect:
    def test_net_effect_is_savings_minus_costs(self):
        m = EffectModel()
        assert approx(
            m.net_effect, m.auto_close_saving + m.suggest_saving - m.total_cost
        )

    def test_net_effect_is_a_modest_share_of_support_budget(self):
        """Санити-чек на «слишком красивые» числа: эффект MVP не должен выглядеть
        как отмена всей поддержки. Ожидаем 5–20% от годового бюджета поддержки."""
        m = EffectModel()
        share = m.net_effect / m.baseline_annual_cost
        assert 0.05 < share < 0.20, f"подозрительная доля эффекта: {share:.1%}"

    def test_llm_cost_is_small_next_to_operator_cost(self):
        m = EffectModel()
        assert m.llm_cost < 0.10 * m.auto_close_saving


class TestSensitivity:
    def test_effect_decreases_monotonically_with_reopen_rate(self):
        rows = sensitivity_to_reopen([0.09, 0.15, 0.25, 0.40])
        nets = [r.net_effect for r in rows]
        assert nets == sorted(nets, reverse=True)

    def test_effect_stays_positive_even_at_pathological_reopen(self):
        """Ключевой вывод для product.md: деньги не являются связывающим ограничением."""
        rows = sensitivity_to_reopen([0.40])
        assert rows[0].net_effect > 0

    def test_sensitivity_row_at_default_reopen_matches_base_model(self):
        base = EffectModel()
        rows = sensitivity_to_reopen([base.auto_reopen_rate])
        assert approx(rows[0].net_effect, base.net_effect)


class TestReporting:
    def test_every_assumption_is_documented(self):
        """Каждый параметр модели обязан иметь текстовое обоснование —
        иначе в product.md попадёт число без источника."""
        m = EffectModel()
        for field in m.tunable_fields():
            assert field in ASSUMPTIONS, f"нет обоснования для параметра {field}"

    def test_markdown_report_contains_key_rows(self):
        out = format_markdown(EffectModel())
        for needle in (
            "Базовая стоимость поддержки",
            "Автозакрытие",
            "Suggest-режим",
            "LLM-инференс",
            "Чистый эффект",
            "reopen",
        ):
            assert needle in out
