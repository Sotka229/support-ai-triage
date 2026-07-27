"""Поиск по базе знаний.

В целевой системе это гибрид BM25 + dense-эмбеддинги с RRF и cross-encoder-реранкером
(docs/ml.md). Здесь — TF-IDF по словам со стоп-листом. Важна не сама метрика сходства,
а два свойства, которые переносятся в продакшен без изменений:

1. Найденный фрагмент — единственный источник для черновика. Нет фрагмента с
   достаточным score — нет генерации: см. `policy.MIN_RETRIEVAL_SCORE`.
2. У статьи есть флаг `auto_reply_allowed`. Статья про возврат денег или удаление
   персональных данных не может стать основанием для автоответа даже при идеальном
   совпадении, и это свойство контента, а не модели.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .models import RetrievedChunk
from .vectorizer import TfidfSpace, cosine, word_tokens

_META = re.compile(r"^(topics|auto_reply_allowed|answer)\s*:\s*(.+)$", re.IGNORECASE)


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    topics: tuple[str, ...]
    auto_reply_allowed: bool
    answer_snippet: str


class KnowledgeBase:
    def __init__(self, documents: list[Document]):
        self._documents = {doc.doc_id: doc for doc in documents}
        self._space = TfidfSpace.fit([doc.text for doc in documents], mode="words")
        self._vectors = {doc.doc_id: self._space.transform(doc.text) for doc in documents}

    @classmethod
    def from_dir(cls, directory: Path | str) -> "KnowledgeBase":
        docs = [_parse(path) for path in sorted(Path(directory).glob("*.md"))]
        if not docs:
            raise ValueError(f"база знаний пуста: {directory}")
        return cls(docs)

    def __len__(self) -> int:
        return len(self._documents)

    def get(self, doc_id: str) -> Document:
        return self._documents[doc_id]

    def search(self, query: str, k: int = 3) -> list[RetrievedChunk]:
        # Разница между «нечего искать» и «искали, но не нашли» существенна:
        # в первом случае возвращаем пустой результат, во втором — кандидатов
        # с нулевым score, чтобы политика увидела низкое сходство и запретила автоответ.
        if not word_tokens(query, drop_stopwords=True, stem_to=5):
            return []
        vector = self._space.transform(query)
        scored = sorted(
            ((cosine(vector, vec), doc_id) for doc_id, vec in self._vectors.items()),
            key=lambda pair: (-pair[0], pair[1]),
        )
        hits: list[RetrievedChunk] = []
        for score, doc_id in scored[:k]:
            doc = self._documents[doc_id]
            hits.append(
                RetrievedChunk(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=doc.text,
                    score=round(score, 4),
                    auto_reply_allowed=doc.auto_reply_allowed,
                    answer_snippet=doc.answer_snippet,
                )
            )
        return hits


def _parse(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    title = ""
    topics: tuple[str, ...] = ()
    auto_reply_allowed = False
    answer = ""
    body_lines: list[str] = []

    for line in raw.splitlines():
        if not title and line.startswith("#"):
            title = line.lstrip("# ").strip()
            continue
        meta = _META.match(line.strip())
        if meta:
            key, value = meta.group(1).lower(), meta.group(2).strip()
            if key == "topics":
                topics = tuple(part.strip() for part in value.split(","))
            elif key == "auto_reply_allowed":
                auto_reply_allowed = value.lower() == "true"
            else:
                answer = value
            continue
        body_lines.append(line)

    return Document(
        doc_id=path.stem,
        title=title,
        text=f"{title}\n" + "\n".join(body_lines).strip(),
        topics=topics,
        auto_reply_allowed=auto_reply_allowed,
        answer_snippet=answer,
    )
