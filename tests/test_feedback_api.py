from datetime import datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.auth import set_password
from app.db import connect, decode_vector
from app.main import create_app
from app.recommend import RankParams, load_library, recommend, watch_events
from app.taste import WatchEvent, build_base_taste_vector

PASSWORD = "correct horse battery staple"


@pytest.fixture
def db_path(seeded_db):
    path, conn = seeded_db
    for name in ("alice", "bob", "carol"):
        set_password(conn, name, PASSWORD, "user")
    return path


def _client(db_path):
    # The session cookie is Secure, so the client only sends it back over https.
    return TestClient(create_app(db_path), base_url="https://testserver")


def _log_in(client, name):
    response = client.post("/auth/login", json={"username": name, "password": PASSWORD})
    assert response.status_code == 200
    return client


def _relogged(client, name):
    assert client.post("/auth/logout").status_code == 204
    assert client.get("/recommendations").status_code == 401
    return _log_in(client, name)


def _picks(client):
    response = client.get("/recommendations")
    assert response.status_code == 200
    return [item["movie_id"] for item in response.json()["items"]]


def _feedback(client, movie_id, action):
    return client.post("/feedback", json={"movie_id": movie_id, "action": action})


def _feedback_rows(db_path):
    conn = connect(db_path)
    rows = conn.execute(
        "SELECT u.name, f.movie_id, f.action FROM feedback_events f"
        " JOIN users u ON u.id = f.user_id ORDER BY f.id"
    ).fetchall()
    conn.close()
    return rows


def test_not_interested_excludes_the_movie_across_sessions(db_path):
    client = _log_in(_client(db_path), "alice")
    first = _picks(client)
    movie_a = first[0]

    assert _feedback(client, movie_a, "not_interested").status_code == 204

    assert movie_a not in _picks(client)
    assert movie_a not in _picks(_relogged(client, "alice"))
    assert _feedback_rows(db_path) == [("alice", movie_a, "not_interested")]


def test_already_seen_excludes_the_movie(db_path):
    client = _log_in(_client(db_path), "alice")
    movie_a = _picks(client)[0]

    assert _feedback(client, movie_a, "already_seen").status_code == 204

    assert movie_a not in _picks(client)


def test_cli_picks_honour_the_same_feedback(db_path):
    client = _log_in(_client(db_path), "alice")
    first, second = _picks(client)[:2]
    _feedback(client, first, "not_interested")
    _feedback(client, second, "already_seen")

    conn = connect(db_path)
    expected = [pick.movie_id for pick in recommend(conn, "alice", RankParams(top_n=3))]
    conn.close()
    assert _picks(client) == expected
    assert {first, second}.isdisjoint(expected)


def test_dismissed_movies_are_excluded_from_cold_start_picks(db_path):
    client = _log_in(_client(db_path), "carol")
    body = client.get("/recommendations").json()
    assert body["cold_start"] is True
    popular = [item["movie_id"] for item in body["items"]]

    _feedback(client, popular[0], "already_seen")
    _feedback(client, popular[1], "not_interested")

    assert set(_picks(client)).isdisjoint(popular[:2])


def test_feedback_does_not_change_another_users_picks(db_path):
    bob = _log_in(_client(db_path), "bob")
    before = bob.get("/recommendations").json()
    alice = _log_in(_client(db_path), "alice")

    for item in before["items"]:
        assert _feedback(alice, item["movie_id"], "not_interested").status_code == 204
    _feedback(alice, before["items"][0]["movie_id"], "watch_tonight")

    assert bob.get("/recommendations").json() == before
    assert bob.get("/feedback/watch-tonight").json() == []


def test_not_interested_counts_against_the_next_taste_vector(db_path):
    client = _log_in(_client(db_path), "alice")
    movie_a = _picks(client)[0]
    _feedback(client, movie_a, "not_interested")
    _picks(client)

    conn = connect(db_path)
    user_id = conn.execute("SELECT id FROM users WHERE name = 'alice'").fetchone()[0]
    computed_at, stored = conn.execute(
        "SELECT computed_at, vector FROM taste_vectors WHERE user_id = ?", (user_id,)
    ).fetchone()
    (created_at,) = conn.execute("SELECT created_at FROM feedback_events").fetchone()
    rejected = WatchEvent(
        movie_a, not_interested=True, watched_at=datetime.fromisoformat(created_at)
    )
    events = [*watch_events(conn, user_id), rejected]
    library = load_library(conn)
    conn.close()

    expected = build_base_taste_vector(
        events,
        {e.movie_id: library.by_id[e.movie_id].vector for e in events},
        datetime.fromisoformat(computed_at),
    )
    assert np.array_equal(decode_vector(stored), expected)


def test_watch_tonight_persists_across_sessions(db_path, library):
    client = _log_in(_client(db_path), "alice")
    first, second = _picks(client)[:2]

    _feedback(client, first, "watch_tonight")
    _feedback(client, second, "watch_tonight")
    _feedback(client, first, "watch_tonight")

    response = _relogged(client, "alice").get("/feedback/watch-tonight")
    assert response.status_code == 200
    saved = response.json()
    assert [item["movie_id"] for item in saved] == [first, second]
    movie = next(movie for movie in library["movies"] if movie["id"] == first)
    assert saved[0]["title"] == movie["title"]
    assert saved[0]["runtime"] == movie["runtime"]


def test_a_dismissed_movie_leaves_watch_tonight(db_path):
    client = _log_in(_client(db_path), "alice")
    first, second = _picks(client)[:2]
    _feedback(client, first, "watch_tonight")
    _feedback(client, second, "watch_tonight")

    _feedback(client, first, "already_seen")

    assert [item["movie_id"] for item in client.get("/feedback/watch-tonight").json()] == [second]


def test_unknown_action_is_rejected_and_nothing_is_stored(db_path):
    client = _log_in(_client(db_path), "alice")

    assert _feedback(client, 1, "love").status_code == 422
    assert _feedback_rows(db_path) == []


def test_unknown_movie_is_not_found_and_nothing_is_stored(db_path, library):
    client = _log_in(_client(db_path), "alice")
    unknown = max(movie["id"] for movie in library["movies"]) + 1

    assert _feedback(client, unknown, "not_interested").status_code == 404
    assert _feedback_rows(db_path) == []


def test_feedback_requires_a_session(db_path):
    with _client(db_path) as client:
        assert _feedback(client, 1, "watch_tonight").status_code == 401
        assert client.get("/feedback/watch-tonight").status_code == 401
    assert _feedback_rows(db_path) == []
