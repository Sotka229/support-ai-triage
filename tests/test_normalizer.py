"""Нормализация четырёх каналов в единый Ticket Envelope.

Каналы дают разный мусор: чат — диалог с ботом, email — HTML, подпись и цитату
предыдущего письма, веб-форма — пары «поле: значение», мобильное приложение —
технические метаданные. Всё это должно превратиться в один объект, иначе
классификатор будет учиться на разметке канала, а не на смысле обращения.
"""

import pytest

from triage.models import Channel, RawMessage
from triage.normalizer import normalize


def chat(messages, **payload):
    payload.setdefault("session_id", "1")
    payload["messages"] = messages
    return RawMessage(channel=Channel.CHAT, external_id="chat-1", payload=payload)


class TestChat:
    def test_keeps_only_user_messages(self):
        t = normalize(
            chat(
                [
                    {"author": "user", "text": "Не могу войти в аккаунт"},
                    {"author": "bot", "text": "Здравствуйте, уточните логин"},
                    {"author": "user", "text": "Логин мой обычный"},
                ]
            )
        )
        assert "Не могу войти" in t.text
        assert "Логин мой обычный" in t.text
        assert "уточните логин" not in t.text

    def test_reads_tier_and_repeat_contact(self):
        t = normalize(chat([{"author": "user", "text": "привет"}], customer_tier="vip", repeat_contact=3))
        assert t.customer_tier == "vip"
        assert t.repeat_contact == 3


class TestEmail:
    def _email(self, body, subject="Проблема с оплатой"):
        return RawMessage(
            channel=Channel.EMAIL,
            external_id="msg-1",
            payload={"message_id": "msg-1", "from": "a@b.com", "subject": subject, "body": body},
        )

    def test_strips_html(self):
        t = normalize(self._email("<p>Оплата <b>не проходит</b></p>"))
        assert "<p>" not in t.text and "<b>" not in t.text
        assert "не проходит" in t.text

    def test_decodes_entities(self):
        t = normalize(self._email("Платёж&nbsp;отклонён &amp; деньги не списались"))
        assert "&nbsp;" not in t.text and "&amp;" not in t.text
        assert "деньги не списались" in t.text

    def test_drops_quoted_reply_and_signature(self):
        body = "Проблема повторяется\n-- \nИван Петров\n\n> Ранее вы писали: спасибо за обращение"
        t = normalize(self._email(body))
        assert "Проблема повторяется" in t.text
        assert "Ранее вы писали" not in t.text, "цитата предыдущего письма портит классификацию"
        assert "Иван Петров" not in t.text, "подпись не относится к содержанию обращения"

    def test_subject_goes_into_text_because_it_carries_the_topic(self):
        t = normalize(self._email("см. выше", subject="Верните деньги за подписку"))
        assert "Верните деньги" in t.text
        assert t.subject == "Верните деньги за подписку"


class TestWebformAndMobile:
    def test_webform_flattens_fields(self):
        raw = RawMessage(
            channel=Channel.WEBFORM,
            external_id="web-1",
            payload={"form_id": "1", "category": "Оплата", "description": "Платёж отклонён"},
        )
        t = normalize(raw)
        assert "Оплата" in t.text and "Платёж отклонён" in t.text

    def test_mobile_keeps_client_meta_out_of_text(self):
        raw = RawMessage(
            channel=Channel.MOBILE,
            external_id="mob-1",
            payload={
                "app_version": "5.2.1",
                "os": "android",
                "device": "Pixel 7",
                "text": "Приложение вылетает",
                "log_excerpt": "FATAL EXCEPTION: main",
            },
        )
        t = normalize(raw)
        assert t.text.startswith("Приложение вылетает")
        assert "FATAL EXCEPTION" not in t.text, "логи не должны попадать в текст для модели"
        assert t.meta["app_version"] == "5.2.1"
        assert t.meta["os"] == "android"
        assert "FATAL EXCEPTION" in t.meta["log_excerpt"]


class TestEnvelopeInvariants:
    def test_ticket_id_is_deterministic_for_idempotency(self):
        raw = chat([{"author": "user", "text": "привет"}])
        assert normalize(raw).ticket_id == normalize(raw).ticket_id

    def test_different_sources_get_different_ids(self):
        a = normalize(chat([{"author": "user", "text": "привет"}]))
        b = normalize(
            RawMessage(channel=Channel.CHAT, external_id="chat-2", payload={"messages": [{"author": "user", "text": "привет"}]})
        )
        assert a.ticket_id != b.ticket_id

    def test_whitespace_is_collapsed(self):
        t = normalize(chat([{"author": "user", "text": "много   \n\n\t пробелов"}]))
        assert "  " not in t.text

    def test_long_text_is_truncated_and_flagged(self):
        t = normalize(chat([{"author": "user", "text": "а" * 9000}]))
        assert len(t.text) <= 4000
        assert t.meta["truncated"] is True

    def test_empty_ticket_is_rejected_loudly(self):
        with pytest.raises(ValueError):
            normalize(chat([{"author": "bot", "text": "только бот говорил"}]))

    def test_unknown_channel_is_rejected(self):
        raw = RawMessage(channel="telegram", external_id="x", payload={})  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            normalize(raw)
