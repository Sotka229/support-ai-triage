"""Поиск по базе знаний и дедупликация обращений во время инцидента."""

from pathlib import Path

from triage.dedup import IncidentDetector
from triage.retrieval import KnowledgeBase

KB_DIR = Path(__file__).resolve().parents[1] / "data" / "kb"


class TestKnowledgeBase:
    def test_loads_all_articles_with_metadata(self):
        kb = KnowledgeBase.from_dir(KB_DIR)
        assert len(kb) == 8
        doc = kb.get("kb-006-refund-policy")
        assert doc.title.startswith("Возврат средств")
        assert doc.auto_reply_allowed is False

    def test_finds_the_relevant_article_first(self):
        kb = KnowledgeBase.from_dir(KB_DIR)
        hits = kb.search("забыл пароль, письмо для сброса не приходит", k=3)
        assert hits[0].doc_id == "kb-001-password-reset"

    def test_scores_are_sorted_and_normalised(self):
        kb = KnowledgeBase.from_dir(KB_DIR)
        hits = kb.search("платёж отклонён, деньги не списались", k=4)
        assert len(hits) == 4
        assert hits == sorted(hits, key=lambda h: -h.score)
        assert all(0.0 <= h.score <= 1.0 for h in hits)

    def test_article_that_forbids_auto_reply_is_marked_in_the_hit(self):
        kb = KnowledgeBase.from_dir(KB_DIR)
        hits = kb.search("хочу возврат денег за подписку", k=1)
        assert hits[0].doc_id == "kb-006-refund-policy"
        assert hits[0].auto_reply_allowed is False, "политика обязана это увидеть"

    def test_out_of_domain_query_gets_low_scores(self):
        """Если в базе знаний нет ответа, черновик генерировать нельзя."""
        kb = KnowledgeBase.from_dir(KB_DIR)
        hits = kb.search("есть ли у вас вакансии для разработчиков", k=1)
        assert hits[0].score < 0.15

    def test_empty_query_returns_nothing(self):
        assert KnowledgeBase.from_dir(KB_DIR).search("", k=3) == []


class TestIncidentDetector:
    def test_burst_of_similar_tickets_collapses_into_one_cluster(self):
        det = IncidentDetector(min_cluster=5, similarity=0.5)
        texts = [
            "Сайт не открывается, ошибка 503",
            "Сайт не открывается, ошибка 503 уже 20 минут",
            "Ошибка 503, сайт не открывается вообще",
            "Сайт не открывается, у вас сбой? ошибка 503",
            "Ошибка 503 при открытии сайта, ничего не грузится",
        ]
        infos = [det.observe(f"t{i}", text) for i, text in enumerate(texts)]
        assert len({i.cluster_id for i in infos}) == 1
        assert infos[-1].size == 5
        assert infos[-1].is_incident is True
        assert infos[0].is_incident is False, "один тикет — ещё не инцидент"

    def test_unrelated_tickets_stay_in_separate_clusters(self):
        det = IncidentDetector(min_cluster=2, similarity=0.5)
        a = det.observe("t1", "Забыл пароль, не приходит письмо")
        b = det.observe("t2", "Доставка опаздывает на неделю, курьер не звонил")
        assert a.cluster_id != b.cluster_id
        assert b.is_incident is False

    def test_cluster_exposes_a_representative_for_the_broadcast_answer(self):
        det = IncidentDetector(min_cluster=2, similarity=0.5)
        det.observe("t1", "Сервис недоступен, ошибка 503")
        info = det.observe("t2", "Сервис недоступен, ошибка 503 у всех")
        assert info.representative_ticket_id == "t1", "broadcast опирается на первый тикет кластера"
