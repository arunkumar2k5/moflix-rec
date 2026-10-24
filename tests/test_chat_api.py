import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.auth import set_password
from app.chat import SIMPLE_MODE_NOTICE, ChatConfig, merge
from app.db import connect, decode_vector, insert_movie, store_vector
from app.llm import Completion, IntentRequest
from app.main import create_app
from app.recommend import Constraints
from app.vectors import FULL_DIMENSION, GENRES, VECTOR_VERSION, Movie

PASSWORD = "correct horse battery staple"
NINETIES = '{"era": {"from": 1990, "to": 1999}}'
ONLY_HORROR = '{"hard_filters": {"require": {"genres": ["Horror"]}}}'


class FakeProvider:
    """Answers with `replies` in turn, repeating the last, and records every request."""

    def __init__(self, *replies: str, delay_s: float = 0.0) -> None:
        self.replies = list(replies)
        self.delay_s = delay_s
        self.requests: list[IntentRequest] = []

    def complete(self, request: IntentRequest) -> Completion:
        self.requests.append(request)
        time.sleep(self.delay_s)
        text = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return Completion(text=text, model="fake", input_tokens=100, output_tokens=20)


class FailingProvider:
    def complete(self, request: IntentRequest) -> Completion:
        raise ConnectionError("provider down")


def _client(db_path, config):
    # The session cookie is Secure, so the client only sends it back over https.
    return TestClient(create_app(db_path, 3, config), base_url="https://testserver")


def _logged_in(db_path, name, config):
    client = _client(db_path, config)
    response = client.post("/auth/login", json={"username": name, "password": PASSWORD})
    assert response.status_code == 200
    return client


@pytest.fixture
def db_path(seeded_db):
    path, conn = seeded_db
    for name in ("alice", "bob"):
        set_password(conn, name, PASSWORD, "user")
    return path


def _ids(response):
    assert response.status_code == 200
    return [item["movie_id"] for item in response.json()["items"]]


def _taste(db_path, name):
    """The stored base taste vector, without the time it was last recomputed."""
    conn = connect(db_path)
    row = conn.execute(
        "SELECT t.vector_version, t.dimension, t.vector FROM taste_vectors t"
        " JOIN users u ON u.id = t.user_id WHERE u.name = ?",
        (name,),
    ).fetchone()
    conn.close()
    return row


def _is_json(text):
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def _leaves(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from _leaves(item)
    else:
        yield value


def test_an_era_request_returns_movies_from_that_era_with_a_plain_reply(db_path):
    client = _logged_in(db_path, "alice", ChatConfig(provider=FakeProvider(NINETIES)))

    response = client.post("/chat", json={"message": "something from the 1990s"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "ai"
    assert body["items"]
    assert all(1990 <= item["year"] <= 1999 for item in body["items"])
    assert body["reply"].strip()
    assert not _is_json(body["reply"])
    assert body["understood"]["era"] == {"from": 1990, "to": 1999}


def test_the_provider_sees_only_the_message_own_turns_and_shown_picks(db_path, library):
    provider = FakeProvider(ONLY_HORROR, NINETIES)
    _logged_in(db_path, "bob", ChatConfig(provider=FakeProvider(ONLY_HORROR))).post(
        "/chat", json={"message": "bob's private request"}
    )
    alice = _logged_in(db_path, "alice", ChatConfig(provider=provider))
    shown = alice.post("/chat", json={"message": "only horror movies please"}).json()["items"]
    provider.requests.clear()

    alice.post("/chat", json={"message": "something from the 1990s"})

    [request] = provider.requests
    payload = request.model_dump(mode="json")
    assert set(payload) == {"message", "turns", "picks"}
    assert payload["message"] == "something from the 1990s"
    assert [turn["role"] for turn in payload["turns"]] == ["user", "assistant"]
    assert payload["turns"][0]["content"] == "only horror movies please"
    assert payload["picks"] == [
        {"title": item["title"], "year": item["year"], "genres": item["genres"]} for item in shown
    ]
    text = json.dumps(payload)
    assert "bob" not in text
    titles = {movie["id"]: movie["title"] for movie in library["movies"]}
    watched = next(user for user in library["users"] if user["name"] == "alice")["watched"]
    assert not any(titles[movie_id] in text for movie_id in watched)
    taste = {float(value) for value in decode_vector(_taste(db_path, "alice")[2])}
    assert not any(leaf in taste for leaf in _leaves(payload) if not isinstance(leaf, str))
    assert not any(f"{value:.3f}" in text for value in taste if value)


@pytest.mark.parametrize(
    "provider",
    [FailingProvider(), FakeProvider("not json"), FakeProvider('{"era": "nineties"}')],
    ids=["raises", "not-json", "off-schema"],
)
def test_a_failing_provider_falls_back_to_simple_mode(db_path, provider):
    client = _logged_in(db_path, "alice", ChatConfig(provider=provider))

    response = client.post("/chat", json={"message": "something from the 1990s"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "simple"
    assert SIMPLE_MODE_NOTICE in body["reply"]
    assert not _is_json(body["reply"])
    assert body["items"]
    assert all(1990 <= item["year"] <= 1999 for item in body["items"])


def test_a_provider_slower_than_the_deadline_falls_back_to_simple_mode(db_path):
    provider = FakeProvider(ONLY_HORROR, delay_s=1.0)
    client = _logged_in(db_path, "alice", ChatConfig(provider=provider, timeout_s=0.1))

    start = time.perf_counter()
    response = client.post("/chat", json={"message": "something from the 1990s"})

    assert time.perf_counter() - start < 1.0
    assert response.json()["mode"] == "simple"
    assert all(1990 <= item["year"] <= 1999 for item in response.json()["items"])


def test_without_a_provider_or_budget_chat_runs_in_simple_mode(db_path):
    over_budget = FakeProvider(NINETIES)
    configs = [ChatConfig(), ChatConfig(provider=over_budget, monthly_cap_usd=0.0)]

    for config in configs:
        response = _logged_in(db_path, "alice", config).post("/chat", json={"message": "hi"})
        assert response.json()["mode"] == "simple"
        assert SIMPLE_MODE_NOTICE in response.json()["reply"]
    assert over_budget.requests == []


def test_each_provider_call_is_recorded_as_usage(db_path):
    client = _logged_in(db_path, "alice", ChatConfig(provider=FakeProvider(NINETIES)))

    client.post("/chat", json={"message": "something from the 1990s"})

    conn = connect(db_path)
    rows = conn.execute("SELECT model, input_tokens, output_tokens FROM llm_usage").fetchall()
    conn.close()
    assert rows == [("fake", 100, 20)]


def test_reset_restores_the_untuned_picks_and_never_touches_the_base_vector(db_path, library):
    client = _logged_in(db_path, "alice", ChatConfig(provider=FakeProvider(ONLY_HORROR)))
    baseline = _ids(client.get("/recommendations"))
    base = _taste(db_path, "alice")
    horror = {movie["id"] for movie in library["movies"] if "Horror" in movie["genres"]}

    tuned = _ids(client.post("/chat", json={"message": "only horror movies please"}))
    assert tuned and set(tuned) <= horror
    assert _ids(client.get("/recommendations")) == tuned
    assert client.post("/chat/reset").status_code == 204
    assert _taste(db_path, "alice") == base

    assert _ids(client.get("/recommendations")) == baseline
    assert _taste(db_path, "alice") == base


def test_tuning_accumulates_across_turns_until_reset(db_path):
    provider = FakeProvider(NINETIES, '{"genre_adjust": {"Comedy": 1.0}}')
    client = _logged_in(db_path, "alice", ChatConfig(provider=provider))

    client.post("/chat", json={"message": "something from the 1990s"})
    response = client.post("/chat", json={"message": "funnier"})

    assert all(1990 <= item["year"] <= 1999 for item in response.json()["items"])


def test_one_users_chat_does_not_change_another_users_picks(db_path):
    config = ChatConfig(provider=FakeProvider(ONLY_HORROR))
    bob = _logged_in(db_path, "bob", config)
    baseline = _ids(bob.get("/recommendations"))

    alice = _logged_in(db_path, "alice", config)
    alice.post("/chat", json={"message": "only horror movies please"})

    assert _ids(bob.get("/recommendations")) == baseline


def test_the_51st_message_in_a_day_is_refused(db_path):
    config = ChatConfig(provider=FakeProvider(NINETIES))
    alice = _logged_in(db_path, "alice", config)

    for _ in range(50):
        assert alice.post("/chat", json={"message": "more"}).status_code == 200
    assert alice.post("/chat", json={"message": "more"}).status_code == 429
    assert _logged_in(db_path, "bob", config).post("/chat", json={"message": "more"}).is_success


def test_chat_requires_a_session(db_path):
    client = _client(db_path, ChatConfig(provider=FakeProvider(NINETIES)))

    assert client.post("/chat", json={"message": "something"}).status_code == 401
    assert client.post("/chat/reset").status_code == 401


def test_a_chat_turn_over_a_20000_movie_library_is_fast(tmp_path):
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
    conn.close()

    with _client(path, ChatConfig(provider=FakeProvider(NINETIES))) as client:
        client.post("/auth/login", json={"username": "erin", "password": PASSWORD})
        client.get("/recommendations")
        start = time.perf_counter()
        response = client.post("/chat", json={"message": "something from the 1990s"})
        elapsed = time.perf_counter() - start

    assert response.status_code == 200
    assert len(response.json()["items"]) == 3
    assert elapsed < 4.0


def test_later_constraints_override_earlier_ones():
    earlier = Constraints(year_range=(1990, 1999), required_genres=frozenset({"Horror"}))
    later = Constraints(year_range=(2000, 2009), excluded_genres=frozenset({"Horror"}))

    assert merge(earlier, later) == later
    assert merge(earlier, Constraints(max_runtime=90)) == Constraints(
        max_runtime=90, year_range=(1990, 1999), required_genres=frozenset({"Horror"})
    )
