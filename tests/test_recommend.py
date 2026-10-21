import socket
import sqlite3
from dataclasses import fields
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.db import WATCH_SIGNAL_COLUMNS, connect, decode_vector, insert_movie, store_vector
from app.recommend import RecommendError, _watch_events, rank, recommend
from app.taste import WatchEvent, build_base_taste_vector
from app.vectors import FULL_DIMENSION, VECTOR_VERSION, Movie


def test_recommend_never_returns_a_watched_movie(seeded_db, library):
    _, conn = seeded_db
    for user in library["users"]:
        results = recommend(conn, user["name"], top_n=len(library["movies"]))
        returned = {r.movie_id for r in results}
        assert returned
        assert returned.isdisjoint(user["watched"])
        assert len(returned) == len(library["movies"]) - len(user["watched"])


def _padded(*values: float) -> np.ndarray:
    vector = np.zeros(FULL_DIMENSION)
    vector[: len(values)] = values
    return vector


SMALL_LIBRARY = {
    1: _padded(1.0, 0.0, 0.0),
    2: _padded(3.0, 2.0, 0.0),
    3: _padded(0.0, 1.0, 0.0),  # mean taste (2, 1, 0) . v = 1
    4: _padded(1.0, 0.0, 1.0),  # 2
    5: _padded(0.0, 0.0, 5.0),  # 0
}


def _small_library_db() -> tuple[sqlite3.Connection, int]:
    conn = connect(":memory:")
    for movie_id, vector in SMALL_LIBRARY.items():
        movie = Movie(movie_id, f"Movie {movie_id}", 2000, (), 100, "en", "US", "PG", 7.0, "", ())
        insert_movie(conn, movie)
        store_vector(conn, movie_id, VECTOR_VERSION, vector)
    user_id = conn.execute("INSERT INTO users (name) VALUES ('dana')").lastrowid
    return conn, user_id


def test_recommend_scores_unwatched_movies_against_the_mean_of_watched_vectors():
    conn, user_id = _small_library_db()
    conn.executemany(
        "INSERT INTO watch_events (user_id, movie_id) VALUES (?, ?)", [(user_id, 1), (user_id, 2)]
    )

    results = recommend(conn, "dana", top_n=3)

    assert [(r.movie_id, r.title) for r in results] == [
        (4, "Movie 4"),
        (3, "Movie 3"),
        (5, "Movie 5"),
    ]
    np.testing.assert_allclose([r.score for r in results], [2.0, 1.0, 0.0], atol=1e-6)


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


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _stored_taste(conn, user_name):
    return conn.execute(
        "SELECT t.vector_version, t.computed_at, t.vector FROM taste_vectors t"
        " JOIN users u ON u.id = t.user_id WHERE u.name = ?",
        (user_name,),
    ).fetchone()


def test_taste_vector_is_built_only_from_the_users_own_events(seeded_db, library):
    _, conn = seeded_db
    recommend(conn, "bob", top_n=3, now=NOW)
    bob_before = _stored_taste(conn, "bob")

    alice_id = conn.execute("SELECT id FROM users WHERE name = 'alice'").fetchone()[0]
    alice = next(user for user in library["users"] if user["name"] == "alice")
    unwatched = [m["id"] for m in library["movies"] if m["id"] not in alice["watched"]]
    conn.executemany(
        "INSERT INTO watch_events (user_id, movie_id, completion, favourite) VALUES (?, ?, ?, ?)",
        [(alice_id, unwatched[0], 1.0, 1), (alice_id, unwatched[1], 0.1, 0)],
    )
    recommend(conn, "alice", top_n=3, now=NOW)
    recommend(conn, "bob", top_n=3, now=NOW)

    assert _stored_taste(conn, "bob") == bob_before
    assert conn.execute("SELECT COUNT(*) FROM taste_vectors").fetchone()[0] == 2
    for user_name in ("alice", "bob"):
        version, computed_at, blob = _stored_taste(conn, user_name)
        assert version == VECTOR_VERSION
        assert computed_at == NOW.isoformat()
        assert decode_vector(blob).shape == (FULL_DIMENSION,)


def test_connect_adds_taste_signal_columns_to_an_existing_database(tmp_path):
    path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(path)
    legacy.executescript(
        "CREATE TABLE watch_events (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL,"
        " movie_id INTEGER NOT NULL);"
        "INSERT INTO watch_events (user_id, movie_id) VALUES (1, 2);"
    )
    legacy.close()

    conn = connect(path)
    connect(path).close()

    assert conn.execute(
        "SELECT user_id, movie_id, completion, play_count, favourite, not_interested, watched_at"
        " FROM watch_events"
    ).fetchall() == [(1, 2, 1.0, 1, 0, 0, None)]


def test_recommend_ranks_and_stores_the_weighted_taste_vector():
    conn, user_id = _small_library_db()
    a_year_ago = NOW - timedelta(days=360)
    conn.executemany(
        "INSERT INTO watch_events"
        " (user_id, movie_id, completion, play_count, favourite, not_interested, watched_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (user_id, 1, 0.8, 3, 1, 0, a_year_ago.isoformat()),
            (user_id, 2, 0.1, 1, 0, 0, None),
            (user_id, 5, 1.0, 1, 0, 1, None),
        ],
    )
    expected = build_base_taste_vector(
        [
            WatchEvent(1, completion=0.8, play_count=3, favourite=True, watched_at=a_year_ago),
            WatchEvent(2, completion=0.1),
            WatchEvent(5, not_interested=True),
        ],
        SMALL_LIBRARY,
        NOW,
    )
    mean = np.mean([SMALL_LIBRARY[1], SMALL_LIBRARY[2], SMALL_LIBRARY[5]], axis=0)
    assert not np.allclose(expected, mean)

    results = recommend(conn, "dana", top_n=2, now=NOW)

    assert decode_vector(_stored_taste(conn, "dana")[2]).tobytes() == expected.tobytes()
    assert [(r.movie_id, r.score) for r in results] == [
        (3, pytest.approx(float(SMALL_LIBRARY[3] @ expected))),
        (4, pytest.approx(float(SMALL_LIBRARY[4] @ expected))),
    ]


def test_watch_signal_columns_and_their_defaults_match_watch_event():
    assert set(WATCH_SIGNAL_COLUMNS) == {f.name for f in fields(WatchEvent)} - {"movie_id"}

    conn, user_id = _small_library_db()
    conn.execute("INSERT INTO watch_events (user_id, movie_id) VALUES (?, 1)", (user_id,))

    assert _watch_events(conn, user_id) == [WatchEvent(movie_id=1)]
