import socket

import numpy as np
import pytest

from app.recommend import RecommendError, rank, recommend


def test_recommend_never_returns_a_watched_movie(seeded_db, library):
    _, conn = seeded_db
    for user in library["users"]:
        results = recommend(conn, user["name"], top_n=len(library["movies"]))
        returned = {r.movie_id for r in results}
        assert returned
        assert returned.isdisjoint(user["watched"])
        assert len(returned) == len(library["movies"]) - len(user["watched"])


def test_rank_scores_are_dot_products_in_descending_order():
    taste = np.array([0.5, 1.0, 0.0, 2.0])
    movie_ids = np.array([10, 20, 30])
    vectors = np.array(
        [
            [1.0, 0.0, 3.0, 0.0],  # 0.5
            [0.0, 1.0, 0.0, 1.0],  # 1.0 + 2.0 = 3.0
            [2.0, 0.5, 0.0, 0.25],  # 1.0 + 0.5 + 0.5 = 2.0
        ]
    )

    ranked = rank(taste, movie_ids, vectors)

    assert [movie_id for movie_id, _ in ranked] == [20, 30, 10]
    np.testing.assert_allclose([score for _, score in ranked], [3.0, 2.0, 0.5], atol=1e-6)


def test_rank_breaks_ties_by_ascending_movie_id():
    taste = np.array([1.0, 1.0])
    movie_ids = np.array([7, 3, 5])
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])

    assert [movie_id for movie_id, _ in rank(taste, movie_ids, vectors)] == [3, 5, 7]


def test_recommend_is_deterministic(seeded_db):
    _, conn = seeded_db
    assert recommend(conn, "alice", top_n=10) == recommend(conn, "alice", top_n=10)


def test_recommend_respects_top_n(seeded_db):
    _, conn = seeded_db
    assert len(recommend(conn, "bob", top_n=3)) == 3


def test_scoring_opens_no_socket(seeded_db, monkeypatch):
    _, conn = seeded_db

    def refuse(*args, **kwargs):
        raise AssertionError("scoring attempted a network connection")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)

    assert len(recommend(conn, "alice", top_n=5)) == 5


def test_recommend_rejects_unknown_user(seeded_db):
    _, conn = seeded_db
    with pytest.raises(RecommendError, match="unknown user"):
        recommend(conn, "mallory", top_n=3)


def test_recommend_rejects_user_without_history(seeded_db):
    _, conn = seeded_db
    conn.execute("INSERT INTO users (name) VALUES ('carol')")
    with pytest.raises(RecommendError, match="no watch history"):
        recommend(conn, "carol", top_n=3)


def test_recommend_rejects_stale_vectors(seeded_db):
    _, conn = seeded_db
    conn.execute("UPDATE movie_vectors SET vector_version = vector_version - 1 WHERE movie_id = 1")
    with pytest.raises(RecommendError, match="rebuild"):
        recommend(conn, "alice", top_n=3)
