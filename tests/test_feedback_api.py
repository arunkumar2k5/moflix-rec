import pytest

from app.db import connect
from tests.conftest import api_client, logged_in


def _feedback_rows(db_path):
    conn = connect(db_path)
    rows = conn.execute(
        "SELECT u.name, f.movie_id, f.action FROM feedback_events f"
        " JOIN users u ON u.id = f.user_id ORDER BY f.id"
    ).fetchall()
    conn.close()
    return rows


def _picks(client):
    response = client.get("/recommendations")
    assert response.status_code == 200
    return [item["movie_id"] for item in response.json()["items"]]


def _mark(client, movie_id, action):
    response = client.post("/feedback", json={"movie_id": movie_id, "action": action})
    assert response.status_code == 204


@pytest.mark.parametrize("action", ["not_interested", "already_seen"])
def test_rejected_movie_is_excluded_from_the_next_picks(db_path, action):
    client = logged_in(db_path, "alice")
    movie_id = _picks(client)[0]

    _mark(client, movie_id, action)

    assert movie_id not in _picks(client)
    assert _feedback_rows(db_path) == [("alice", movie_id, action)]


def test_exclusion_survives_logout_and_login(db_path):
    client = logged_in(db_path, "alice")
    movie_id = _picks(client)[0]
    _mark(client, movie_id, "not_interested")

    assert client.post("/auth/logout").status_code == 204
    assert client.get("/recommendations").status_code == 401

    assert movie_id not in _picks(logged_in(db_path, "alice"))


def test_feedback_leaves_other_users_picks_unchanged(db_path):
    bob = logged_in(db_path, "bob", top_n=5)
    before = bob.get("/recommendations").json()
    alice = logged_in(db_path, "alice")

    actions = ["not_interested", "already_seen", "watch_tonight"]
    for movie_id, action in zip(_picks(bob)[: len(actions)], actions, strict=True):
        _mark(alice, movie_id, action)

    assert bob.get("/recommendations").json() == before
    assert bob.get("/feedback/watch-tonight").json() == []


def test_watch_tonight_lists_the_movie_after_logout_and_login(db_path, library):
    client = logged_in(db_path, "alice")
    movie_id = _picks(client)[0]
    _mark(client, movie_id, "watch_tonight")
    client.post("/auth/logout")

    client = logged_in(db_path, "alice")
    response = client.get("/feedback/watch-tonight")

    assert response.status_code == 200
    movie = next(m for m in library["movies"] if m["id"] == movie_id)
    assert response.json() == [
        {
            "movie_id": movie_id,
            "title": movie["title"],
            "year": movie["year"],
            "genres": movie["genres"],
            "runtime": movie["runtime"],
            "poster_url": movie.get("poster_url", ""),
        }
    ]
    assert movie_id in _picks(client)


def test_watch_tonight_lists_the_latest_mark_first(db_path):
    client = logged_in(db_path, "alice")
    first, second = _picks(client)[:2]
    _mark(client, first, "watch_tonight")
    _mark(client, second, "watch_tonight")

    listed = [item["movie_id"] for item in client.get("/feedback/watch-tonight").json()]

    assert listed == [second, first]


def test_watch_tonight_with_stale_vectors_is_reported_as_unavailable(db_path):
    client = logged_in(db_path, "alice")
    _mark(client, _picks(client)[0], "watch_tonight")
    conn = connect(db_path)
    with conn:
        conn.execute("UPDATE movie_vectors SET vector_version = vector_version - 1")
    conn.close()

    response = client.get("/feedback/watch-tonight")

    assert response.status_code == 503
    assert "rebuild" in response.json()["detail"]


def test_the_latest_action_on_a_movie_wins(db_path):
    client = logged_in(db_path, "alice")
    movie_id = _picks(client)[0]

    _mark(client, movie_id, "watch_tonight")
    _mark(client, movie_id, "not_interested")

    assert client.get("/feedback/watch-tonight").json() == []
    assert movie_id not in _picks(client)
    assert _feedback_rows(db_path) == [
        ("alice", movie_id, "watch_tonight"),
        ("alice", movie_id, "not_interested"),
    ]

    _mark(client, movie_id, "watch_tonight")

    assert [item["movie_id"] for item in client.get("/feedback/watch-tonight").json()] == [movie_id]
    assert movie_id in _picks(client)


def test_unknown_action_is_rejected_and_stores_nothing(db_path):
    client = logged_in(db_path, "alice")
    movie_id = _picks(client)[0]

    response = client.post("/feedback", json={"movie_id": movie_id, "action": "love"})

    assert response.status_code == 422
    assert _feedback_rows(db_path) == []


def test_unknown_movie_is_rejected_and_stores_nothing(db_path, library):
    client = logged_in(db_path, "alice")
    unknown = max(movie["id"] for movie in library["movies"]) + 1000

    response = client.post("/feedback", json={"movie_id": unknown, "action": "not_interested"})

    assert response.status_code == 404
    assert _feedback_rows(db_path) == []


def test_feedback_routes_require_a_session(db_path, library):
    movie_id = library["movies"][0]["id"]
    with api_client(db_path) as client:
        response = client.post("/feedback", json={"movie_id": movie_id, "action": "watch_tonight"})
        assert response.status_code == 401
        assert client.get("/feedback/watch-tonight").status_code == 401
    assert _feedback_rows(db_path) == []
