"""PII-редакция.

Это единственный компонент, который обязан работать раньше всех остальных: до правил,
до модели и особенно до любого вызова LLM. Поэтому тестируем и полноту (нашли),
и точность (не нашли лишнего), и идемпотентность (повторный проход не ломает текст).
"""

import pytest

from triage.pii import PAYMENT_CLASSES, redact


class TestDetection:
    @pytest.mark.parametrize(
        "text, pii_class, placeholder",
        [
            ("Пишите на ivan.petrov@example.com", "email", "[EMAIL]"),
            ("Мой телефон +7 (912) 345-67-89", "phone", "[PHONE]"),
            ("Телефон 89123456789 для связи", "phone", "[PHONE]"),
            ("Карта 4111 1111 1111 1111 заблокирована", "card", "[CARD]"),
            ("Карта 4111111111111111", "card", "[CARD]"),
            ("СНИЛС 112-233-445 95", "snils", "[SNILS]"),
            ("ИНН 7707083893 для счёта", "inn", "[INN]"),
            ("Паспорт 4509 123456 приложен", "passport", "[PASSPORT]"),
            ("Счёт 40817810099910004312 в банке", "account", "[ACCOUNT]"),
            ("Захожу с адреса 192.168.100.14", "ip", "[IP]"),
            ("CVV 123 не подходит", "cvv", "[CVV]"),
            ("Живу: ул. Ленина, д. 15, кв. 42", "address", "[ADDRESS]"),
        ],
    )
    def test_each_class_is_detected_and_replaced(self, text, pii_class, placeholder):
        r = redact(text)
        assert pii_class in r.classes
        assert placeholder in r.text

    def test_original_card_digits_do_not_survive(self):
        r = redact("Карта 4111 1111 1111 1111 списала деньги")
        assert "4111" not in r.text

    def test_several_classes_in_one_ticket(self):
        r = redact("Я ivan@example.com, телефон +79123456789, карта 4111111111111111")
        assert {"email", "phone", "card"} <= r.classes


class TestPrecision:
    def test_order_number_is_not_mistaken_for_a_card(self):
        """16 цифр без валидной контрольной суммы Луна — это номер заказа, а не карта."""
        r = redact("Номер заказа 1234567890123456, проверьте статус")
        assert "card" not in r.classes

    def test_plain_ticket_text_is_untouched(self):
        text = "Приложение вылетает при открытии профиля после обновления"
        r = redact(text)
        assert r.text == text
        assert r.classes == frozenset()

    def test_bare_ten_digit_number_is_not_an_inn_without_the_keyword(self):
        """Без ключевого слова десять цифр — это что угодно: артикул, номер договора."""
        r = redact("Артикул товара 7707083893")
        assert "inn" not in r.classes


class TestContract:
    def test_redaction_is_idempotent(self):
        once = redact("Почта a@b.com и карта 4111111111111111").text
        assert redact(once).text == once

    def test_payment_pii_flag_blocks_external_llm(self):
        assert redact("карта 4111111111111111").has_payment_pii is True
        assert redact("почта a@b.com").has_payment_pii is False
        assert PAYMENT_CLASSES == frozenset({"card", "cvv", "account"})

    def test_spans_are_reported_for_audit(self):
        r = redact("Почта a@b.com")
        assert r.spans and r.spans[0].pii_class == "email"
        assert r.spans[0].start < r.spans[0].end
