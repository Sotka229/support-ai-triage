"""Дедупликация обращений во время инцидента.

Зачем это в PoC: при массовом сбое приходит 10–20k тикетов за 10 минут, и это самый
дорогой момент для LLM — генерировать 20k почти одинаковых ответов бессмысленно
и по деньгам, и по качеству. Кластеризация позволяет ответить один раз на кластер
(`Route.INCIDENT_BROADCAST`) и одновременно даёт продуктовый сигнал: если кластер
растёт, это инцидент, а не поток разных проблем.

Упрощение PoC: кластеры живут в памяти процесса и сравниваются с представителем
кластера. В целевой системе это онлайн-кластеризация в стриминге (см. docs/architecture.md).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .vectorizer import word_tokens


@dataclass(frozen=True)
class ClusterInfo:
    cluster_id: str
    size: int
    is_incident: bool
    representative_ticket_id: str


@dataclass
class _Cluster:
    cluster_id: str
    representative_ticket_id: str
    shingles: frozenset[str]
    members: list[tuple[str, float]] = field(default_factory=list)


class IncidentDetector:
    def __init__(
        self,
        *,
        min_cluster: int = 5,
        similarity: float = 0.5,
        window_seconds: float = 600.0,
        clock=time.monotonic,
    ):
        self.min_cluster = min_cluster
        self.similarity = similarity
        self.window_seconds = window_seconds
        self._clock = clock
        self._clusters: list[_Cluster] = []
        self._counter = 0

    def observe(self, ticket_id: str, text: str) -> ClusterInfo:
        now = self._clock()
        shingles = _shingles(text)
        self._expire(now)

        for cluster in self._clusters:
            if _jaccard(shingles, cluster.shingles) >= self.similarity:
                cluster.members.append((ticket_id, now))
                return self._info(cluster)

        self._counter += 1
        cluster = _Cluster(
            cluster_id=f"cluster-{self._counter}",
            representative_ticket_id=ticket_id,
            shingles=shingles,
            members=[(ticket_id, now)],
        )
        self._clusters.append(cluster)
        return self._info(cluster)

    def _info(self, cluster: _Cluster) -> ClusterInfo:
        return ClusterInfo(
            cluster_id=cluster.cluster_id,
            size=len(cluster.members),
            is_incident=len(cluster.members) >= self.min_cluster,
            representative_ticket_id=cluster.representative_ticket_id,
        )

    def _expire(self, now: float) -> None:
        """Инцидент — это всплеск, а не сумма обращений за месяц: держим окно."""
        for cluster in self._clusters:
            cluster.members = [m for m in cluster.members if now - m[1] <= self.window_seconds]
        self._clusters = [c for c in self._clusters if c.members]


def _shingles(text: str) -> frozenset[str]:
    # Огрубление до 4 символов заменяет стемминг: «сайта», «сайт» и «сайте» должны
    # попадать в один кластер, иначе одинаковые жалобы разъезжаются по кластерам.
    return frozenset(word_tokens(text, drop_stopwords=True, stem_to=4))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
