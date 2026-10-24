import time
from dataclasses import asdict

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.auth import set_password
from app.db import connect, insert_movie, store_vector
from app.main import create_app
from app.recommend import (
    DEFAULT_COLD_START_MIN,
    LibraryCache,
    RankParams,
    load_library,
    recommend,
    recommend_for_user,
    watch_events,
)
from app.vectors import FULL_DIMENSION, GENRES, VECTOR_VERSION, Movie

PASSWORD = "correct horse battery staple"


def _client(db_path, top_n=3, cold_start_min=DEFAULT_COLD_START_MIN):
    # The session cookie is Secure, so the client only sends it back over https.
    return TestClient(create_app(db_path, top_n, cold_start_min), base_url="https://testserver")


def _logged_in(db_path, name, top_n=3, cold_start_min=DEFAULT_COLD_START_MIN):
    client = _client(db_path, top_n, cold_start_min)
    response = client.post("/auth/login", json={"username": name, "password": PASSWORD})
    assert response.status_code == 200
    return client


@pytest.fixture
def db_path(seeded_db):
    path, conn = seeded_db
    for name in ("alice", "bob"):
        set_password(conn, name, PASSWORD, "user")
    return path


def _watched(library, name):
    return set(next(user for user in library["users"] if user["name"] == name)["watched"])


def _log(db_path):
    conn = connect(db_path)
    rows = conn.execute(
        "SELECT u.name, l.movie_id FROM recommendation_log l JOIN users u ON u.id = l.user_id"
        " ORDER BY l.id"
    ).fetchall()
    conn.close()
    return rows


def test_logged_in_user_gets_explained_unwatched_picks(db_path, library):
    assert len(_watched(library, "alice")) >= 5

    response = _logged_in(db_path, "alice").get("/recommendations")

    assert response.status_code == 200
    body = response.json()
    assert body["cold_start"] is False
    assert len(body["items"]) >= 2
    movies = {movie["id"]: movie for movie in library["movies"]}
    for item in body["items"]:
        assert item["title"]
        assert isinstance(item["year"], int)
        assert item["genres"]
        assert len(item["reason"]) >= 10
        assert item["runtime"] == movies[item["movie_id"]]["runtime"]
        assert "poster_url" in item
    assert {item["movie_id"] for item in body["items"]}.isdisjoint(_watched(library, "alice"))


def test_recommendations_require_a_session(db_path):
    with _client(db_path) as client:
        assert client.get("/recommendations").status_code == 401
    assert _log(db_path) == []


@pytest.mark.parametrize("top_n", [1, 3, 5])
def test_each_user_gets_the_cli_picks_from_their_own_history(db_path, library, top_n):
    assert _watched(library, "alice").isdisjoint(_watched(library, "bob"))
    conn = connect(db_path)

    for name in ("alice", "bob"):
        items = _logged_in(db_path, name, top_n).get("/recommendations").json()["items"]
        expected = recommend(conn, name, RankParams(top_n=top_n))

        assert [(i["movie_id"], i["title"], i["year"], i["reason"]) for i in items] == [
            (r.movie_id, r.title, r.year, r.reason) for r in expected
        ]
    conn.close()


def test_each_call_logs_every_shown_pick_for_the_session_user_only(db_path):
    client = _logged_in(db_path, "bob")

    first = [item["movie_id"] for item in client.get("/recommendations").json()["items"]]
    second = [item["movie_id"] for item in client.get("/recommendations").json()["items"]]

    assert len(first) == 3
    assert _log(db_path) == [("bob", movie_id) for movie_id in first + second]


def _add_carol(db_path, *watched):
    conn = connect(db_path)
    set_password(conn, "carol", PASSWORD, "user")
    carol_id = conn.execute("SELECT id FROM users WHERE name = 'carol'").fetchone()[0]
    with conn:
        conn.executemany(
            "INSERT INTO watch_events (user_id, movie_id) VALUES (?, ?)",
            [(carol_id, movie_id) for movie_id in watched],
        )
    conn.close()
    return carol_id


def test_user_without_history_gets_popular_picks_and_an_onboarding_prompt(db_path):
    _add_carol(db_path)

    response = _logged_in(db_path, "carol").get("/recommendations")

    assert response.status_code == 200
    body = response.json()
    assert body["cold_start"] is True
    assert isinstance(body["onboarding_prompt"], str) and body["onboarding_prompt"]
    assert len(body["items"]) >= 2
    for item in body["items"]:
        assert item["title"]
        assert isinstance(item["year"], int)
        assert item["genres"]
        assert item["reason"]
    # The highest rated fixture movies, ties to the lower id.
    assert [item["movie_id"] for item in body["items"]] == [1, 8, 10]
    assert _log(db_path) == [("carol", 1), ("carol", 8), ("carol", 10)]


def test_cold_start_picks_ignore_other_users_history(db_path):
    _add_carol(db_path)
    before = _logged_in(db_path, "carol").get("/recommendations").json()["items"]

    conn = connect(db_path)
    with conn:
        for name in ("alice", "bob"):
            user_id = conn.execute("SELECT id FROM users WHERE name = ?", (name,)).fetchone()[0]
            conn.executemany(
                "INSERT INTO watch_events (user_id, movie_id, play_count, favourite)"
                " VALUES (?, ?, 50, 1)",
                [(user_id, movie_id) for movie_id in (1, 8, 10, 22)],
            )
    conn.close()
    after = _logged_in(db_path, "carol").get("/recommendations").json()["items"]

    assert after == before


def test_user_with_exactly_cold_start_min_movies_is_not_cold_start(db_path, library):
    assert len(_watched(library, "bob")) == 5

    body = _logged_in(db_path, "bob").get("/recommendations").json()

    assert body["cold_start"] is False
    assert body["onboarding_prompt"] is None


@pytest.mark.parametrize(("cold_start_min", "cold_start"), [(2, False), (6, True)])
def test_cold_start_min_sets_the_threshold(db_path, cold_start_min, cold_start):
    client = _logged_in(db_path, "bob", cold_start_min=cold_start_min)

    body = client.get("/recommendations").json()

    assert body["cold_start"] is cold_start
    assert (body["onboarding_prompt"] is not None) is cold_start


def test_sparse_history_excludes_watched_movies_and_stores_a_taste_vector(db_path):
    carol_id = _add_carol(db_path, 1, 8)

    body = _logged_in(db_path, "carol").get("/recommendations").json()

    assert body["cold_start"] is True
    assert body["onboarding_prompt"]
    assert len(body["items"]) == 3
    assert {item["movie_id"] for item in body["items"]}.isdisjoint({1, 8})
    conn = connect(db_path)
    stored = conn.execute("SELECT 1 FROM taste_vectors WHERE user_id = ?", (carol_id,)).fetchall()
    conn.close()
    assert stored


def test_cold_start_min_must_be_positive(db_path):
    with pytest.raises(ValueError, match="cold_start_min"):
        create_app(db_path, cold_start_min=0)


def _sync_movie_alice_watched(db_path, library):
    """Commit what an Emby sync does for a new movie alice watched; returns its id."""
    conn = connect(db_path)
    new_id = max(movie["id"] for movie in library["movies"]) + 1
    template = Movie(**library["movies"][0])
    with conn:
        insert_movie(conn, Movie(**{**asdict(template), "id": new_id, "title": "Newcomer"}))
        store_vector(conn, new_id, VECTOR_VERSION, np.ones(FULL_DIMENSION))
        alice_id = conn.execute("SELECT id FROM users WHERE name = 'alice'").fetchone()[0]
        conn.execute(
            "INSERT INTO watch_events (user_id, movie_id) VALUES (?, ?)", (alice_id, new_id)
        )
    conn.close()
    return new_id


def test_movies_synced_while_running_are_picked_up(db_path, library):
    client = _logged_in(db_path, "alice", top_n=len(library["movies"]))
    before = {item["movie_id"] for item in client.get("/recommendations").json()["items"]}

    new_id = _sync_movie_alice_watched(db_path, library)
    response = client.get("/recommendations")

    assert response.status_code == 200
    after = {item["movie_id"] for item in response.json()["items"]}
    assert new_id not in after
    assert after <= before


def test_a_sync_committing_after_the_library_is_taken_is_ignored(db_path, library, monkeypatch):
    client = _logged_in(db_path, "alice")
    get = LibraryCache.get
    synced = []

    def get_then_sync(cache, conn):
        taken = get(cache, conn)
        if not synced:
            synced.append(_sync_movie_alice_watched(db_path, library))
        return taken

    monkeypatch.setattr(LibraryCache, "get", get_then_sync)
    response = client.get("/recommendations")

    assert response.status_code == 200
    assert synced[0] not in {item["movie_id"] for item in response.json()["items"]}


def test_vectors_rebuilt_while_running_are_reported_as_unavailable(db_path):
    client = _logged_in(db_path, "alice")
    assert client.get("/recommendations").status_code == 200

    conn = connect(db_path)
    with conn:
        conn.execute("UPDATE movie_vectors SET vector_version = vector_version - 1")
    conn.close()
    response = client.get("/recommendations")

    assert response.status_code == 503
    assert "rebuild" in response.json()["detail"]


def test_a_failed_startup_load_leaves_the_app_up_and_is_retried(db_path, monkeypatch):
    calls = []

    def fail_first(conn):
        calls.append(conn)
        if len(calls) == 1:
            raise KeyError("vector without a movie")
        return load_library(conn)

    monkeypatch.setattr("app.recommend.load_library", fail_first)
    with _client(db_path) as client:
        assert len(calls) == 1
        login = client.post("/auth/login", json={"username": "alice", "password": PASSWORD})
        assert login.status_code == 200
        assert client.get("/recommendations").status_code == 200


def test_stale_vectors_are_reported_as_unavailable(db_path):
    conn = connect(db_path)
    with conn:
        conn.execute("UPDATE movie_vectors SET vector_version = vector_version - 1")
    conn.close()

    response = _logged_in(db_path, "alice").get("/recommendations")

    assert response.status_code == 503
    assert "rebuild" in response.json()["detail"]


def test_scoring_a_20000_movie_library_is_fast(tmp_path):
    path = tmp_path / "large.db"
    conn = connect(path)
    rng = np.random.default_rng(0)
    with conn:
        for movie_id in range(1, 20_001):
            genres = tuple(rng.choice(GENRES, size=2, replace=False).tolist())
            movie = Movie(
                movie_id, f"Movie {movie_id}", 1950 + movie_id % 75, genres, 80 + movie_id % 90,
                "en", "US", "PG", 7.0, f"Director {movie_id % 500}", (),
            )  # fmt: skip
            insert_movie(conn, movie)
            store_vector(conn, movie_id, VECTOR_VERSION, rng.random(FULL_DIMENSION))
    set_password(conn, "erin", PASSWORD, "user")
    user_id = conn.execute("SELECT id FROM users WHERE name = 'erin'").fetchone()[0]
    with conn:
        conn.executemany(
            "INSERT INTO watch_events (user_id, movie_id) VALUES (?, ?)",
            [(user_id, movie_id) for movie_id in range(1, 21)],
        )

    events = watch_events(conn, user_id)
    library = load_library(conn)
    start = time.perf_counter()
    recommend_for_user(conn, user_id, events, library, RankParams(top_n=3))
    scoring = time.perf_counter() - start
    conn.close()

    with _client(path) as client:  # startup loads the library
        client.post("/auth/login", json={"username": "erin", "password": PASSWORD})
        start = time.perf_counter()
        response = client.get("/recommendations")
        request = time.perf_counter() - start

    assert response.status_code == 200
    assert len(response.json()["items"]) == 3
    assert scoring < 0.2
    assert request < 2.0
