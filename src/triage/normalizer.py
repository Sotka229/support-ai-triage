"""Нормализация каналов в Ticket Envelope.

Без этого шага классификатор учится на особенностях канала, а не на смысле обращения:
в email половина текста — подпись и цитата, в чате — реплики бота, в мобильном
приложении — стектрейс. Метаданные не выбрасываются, но и в текст для модели не идут.
"""

from __future__ import annotations

import hashlib
import html
import re

from .models import Channel, RawMessage, Ticket

MAX_TEXT_LEN = 4000

_TAG = re.compile(r"<[^>]+>")
_SIGNATURE = re.compile(r"(?:\n|\s)--\s*\n")
_QUOTE_INTRO = re.compile(
    r"^\s*(?:>|On .*wrote:|\d{1,2}\.\d{1,2}(?:\.\d{2,4})?[^\n]{0,40}(?:писали|wrote))",
    re.IGNORECASE,
)
_WS = re.compile(r"\s+")


def normalize(raw: RawMessage) -> Ticket:
    try:
        channel = Channel(raw.channel)
    except ValueError as exc:  # неизвестный канал — это ошибка интеграции, не тикет
        raise ValueError(f"неизвестный канал: {raw.channel!r}") from exc

    payload = raw.payload or {}
    handler = {
        Channel.CHAT: _from_chat,
        Channel.EMAIL: _from_email,
        Channel.WEBFORM: _from_webform,
        Channel.MOBILE: _from_mobile,
    }[channel]
    text, subject, meta = handler(payload)

    text = _WS.sub(" ", text).strip()
    if not text:
        raise ValueError(f"пустой текст обращения (канал {channel.value})")

    truncated = len(text) > MAX_TEXT_LEN
    if truncated:
        text = text[:MAX_TEXT_LEN]
    meta["truncated"] = truncated

    return Ticket(
        ticket_id=_ticket_id(channel, raw.external_id),
        channel=channel,
        text=text,
        subject=subject,
        customer_tier=str(payload.get("customer_tier", "standard")).lower(),
        repeat_contact=int(payload.get("repeat_contact", 1)),
        meta=meta,
    )


def _ticket_id(channel: Channel, external_id: str) -> str:
    """Детерминированный id: повторная доставка того же сообщения из канала
    не должна создавать второй тикет и второй автоответ."""
    digest = hashlib.sha256(f"{channel.value}:{external_id}".encode()).hexdigest()
    return f"{channel.value}-{digest[:12]}"


def _from_chat(payload: dict) -> tuple[str, str, dict]:
    messages = payload.get("messages") or []
    user_texts = [m.get("text", "") for m in messages if m.get("author") == "user"]
    meta = {"session_id": payload.get("session_id"), "turns": len(messages)}
    return " ".join(user_texts), "", meta


def _from_email(payload: dict) -> tuple[str, str, dict]:
    subject = str(payload.get("subject", "")).strip()
    body = _clean_email_body(str(payload.get("body", "")))
    meta = {"message_id": payload.get("message_id"), "from_domain": _domain(payload.get("from"))}
    return f"{subject}. {body}" if subject else body, subject, meta


def _clean_email_body(body: str) -> str:
    body = _SIGNATURE.split(body)[0]
    kept: list[str] = []
    for line in body.splitlines():
        if _QUOTE_INTRO.match(line):
            break  # всё ниже — переписка, а не новое обращение
        kept.append(line)
    text = "\n".join(kept)
    text = _TAG.sub(" ", text)
    return html.unescape(text)


def _domain(address: object) -> str | None:
    if isinstance(address, str) and "@" in address:
        return address.rsplit("@", 1)[-1]
    return None


def _from_webform(payload: dict) -> tuple[str, str, dict]:
    category = str(payload.get("category", "")).strip()
    description = str(payload.get("description", "")).strip()
    text = f"{category}. {description}" if category else description
    return text, category, {"form_id": payload.get("form_id")}


def _from_mobile(payload: dict) -> tuple[str, str, dict]:
    meta = {
        "app_version": payload.get("app_version"),
        "os": payload.get("os"),
        "device": payload.get("device"),
        # стектрейс полезен инженеру и вреден классификатору: держим отдельно
        "log_excerpt": payload.get("log_excerpt"),
    }
    return str(payload.get("text", "")), "", meta
