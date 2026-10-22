import socket
import sqlite3
from dataclasses import fields
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.db import (
    WATCH_SIGNAL_COLUMNS,
    connect,
    decode_vector,
    insert_movie,
    load_movies,
    store_vector,
)
from app.recommend import (
    Candidate,
    Constraints,
    RankParams,
    RecommendError,
    UserContext,
    _watch_events,
    rank,
    recommend,
)
from app.taste import WatchEvent, build_base_taste_vector
from app.vectors import FULL_DIMENSION, GENRES, LAYOUT, VECTOR_VERSION, Movie


def test_recommend_never_returns_a_watched_movie(seeded_db, library):
    _, conn = seeded_db
    for user in library["users"]:
        results = recommend(conn, user["name"], RankParams(top_n=len(library["movies"])))
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

    results = recommend(conn, "dana", RankParams(top_n=3))

    assert [(r.movie_id, r.title) for r in results] == [
        (4, "Movie 4"),
        (3, "Movie 3"),
        (5, "Movie 5"),
    ]
    np.testing.assert_allclose([r.score for r in results], [2.0, 1.0, 0.0], atol=1e-6)


def _candidate(movie_id: int, vector, title: str = "", **metadata) -> Candidate:
    defaults = {
        "year": 2000,
        "genres": (),
        "runtime": 100,
        "language": "en",
        "country": "US",
        "certification": "PG",
        "rating": 7.0,
        "director": "",
        "cast": (),
    }
    defaults.update(metadata)
    movie = Movie(movie_id, title or f"Movie {movie_id}", **defaults)
    return Candidate(movie, np.asarray(vector, dtype=float))


def _context(taste, excluded=(), liked=()) -> UserContext:
    return UserContext(np.asarray(taste, dtype=float), frozenset(excluded), tuple(liked))


RELEVANCE_ONLY = RankParams(mmr_lambda=1.0)


def test_rank_scores_are_dot_products_in_descending_order():
    candidates = [
        _candidate(10, [1.0, 0.0, 3.0, 0.0]),  # 0.5
        _candidate(20, [0.0, 1.0, 0.0, 1.0]),  # 1.0 + 2.0 = 3.0
        _candidate(30, [2.0, 0.5, 0.0, 0.25]),  # 1.0 + 0.5 + 0.5 = 2.0
    ]

    ranked = rank(_context([0.5, 1.0, 0.0, 2.0]), candidates, Constraints(), RELEVANCE_ONLY)

    assert [r.movie_id for r in ranked] == [20, 30, 10]
    np.testing.assert_allclose([r.score for r in ranked], [3.0, 2.0, 0.5], atol=1e-6)


def test_rank_breaks_ties_by_ascending_movie_id():
    candidates = [_candidate(7, [1.0, 0.0]), _candidate(3, [0.0, 1.0]), _candidate(5, [0.5, 0.5])]

    ranked = rank(_context([1.0, 1.0]), candidates, Constraints(), RELEVANCE_ONLY)

    assert [r.movie_id for r in ranked] == [3, 5, 7]


def test_cosine_ignores_magnitude_while_dot_scales_with_it():
    candidates = [_candidate(1, [1.0, 2.0]), _candidate(2, [10.0, 20.0])]

    def scores(similarity):
        params = RankParams(similarity=similarity, mmr_lambda=1.0)
        ranked = rank(_context([1.0, 1.0]), candidates, Constraints(), params)
        return {r.movie_id: r.score for r in ranked}

    cosine = scores("cosine")
    assert cosine[1] == pytest.approx(cosine[2])
    dot = scores("dot")
    assert dot[2] == pytest.approx(10 * dot[1])


def test_cosine_scores_a_zero_vector_zero():
    ranked = rank(
        _context([1.0, 1.0]),
        [_candidate(1, [0.0, 0.0])],
        Constraints(),
        RankParams(similarity="cosine"),
    )

    assert [r.score for r in ranked] == [0.0]


def test_rank_never_returns_an_excluded_movie_even_when_it_scores_highest():
    candidates = [_candidate(1, [1.0, 0.0]), _candidate(2, [5.0, 0.0])]

    ranked = rank(_context([1.0, 0.0], excluded={2}), candidates, Constraints(), RankParams())

    assert [r.movie_id for r in ranked] == [1]


def test_rank_returns_nothing_when_constraints_exclude_every_movie():
    ranked = rank(
        _context([1.0]),
        [_candidate(1, [1.0], runtime=150)],
        Constraints(max_runtime=120),
        RankParams(),
    )

    assert ranked == []


MMR_CANDIDATES = [
    _candidate(1, [1.0, 0.0, 0.0]),
    _candidate(2, [1.0, 0.0, 0.0]),
    _candidate(3, [0.0, 1.0, 0.0]),
    _candidate(4, [0.0, 0.0, 1.0]),
    _candidate(5, [0.1, 0.0, 0.0]),
]


@pytest.mark.parametrize(("mmr_lambda", "expected"), [(0.7, [1, 3, 4]), (1.0, [1, 2, 3])])
def test_mmr_keeps_only_one_of_two_identical_top_movies_unless_lambda_is_one(mmr_lambda, expected):
    params = RankParams(mmr_lambda=mmr_lambda, top_n=3)

    ranked = rank(_context([1.2, 1.0, 1.0]), MMR_CANDIDATES, Constraints(), params)

    assert [r.movie_id for r in ranked] == expected


@pytest.mark.parametrize(
    "params", [{"mmr_lambda": 1.5}, {"mmr_lambda": -0.1}, {"similarity": "euclid"}, {"top_n": 0}]
)
def test_rank_params_reject_invalid_values(params):
    with pytest.raises(ValueError):
        RankParams(**params)


def _genre_vector(*genres: str) -> np.ndarray:
    vector = np.zeros(LAYOUT["genres"].stop)
    for genre in genres:
        vector[LAYOUT["genres"].start + GENRES.index(genre)] = 1.0
    return vector


def test_reason_names_the_top_contributing_genre_and_closest_liked_title():
    liked = _candidate(9, _genre_vector("Thriller"), title="Heat", genres=("Thriller",))
    candidate = _candidate(1, _genre_vector("Drama", "Thriller"), genres=("Drama", "Thriller"))
    taste = _genre_vector("Thriller") * 2 + _genre_vector("Drama")

    [result] = rank(_context(taste, liked=[liked]), [candidate], Constraints(), RankParams())

    assert result.reason == "Matches your taste for Thriller, similar to Heat"


def test_reason_falls_back_to_the_movies_genre_without_taste_or_liked_signal():
    candidate = _candidate(1, _genre_vector("Thriller"), genres=("Thriller",))
    taste = np.zeros(LAYOUT["genres"].stop)

    [result] = rank(_context(taste), [candidate], Constraints(), RankParams())

    assert result.reason == "A Thriller pick close to your overall taste"


def _movies_by_id(library) -> dict:
    return {movie["id"]: movie for movie in library["movies"]}


def test_max_runtime_constraint_drops_longer_movies(seeded_db, library):
    _, conn = seeded_db
    movies = _movies_by_id(library)

    results = recommend(conn, "alice", RankParams(top_n=24), Constraints(max_runtime=120))

    assert results
    assert all(movies[r.movie_id]["runtime"] <= 120 for r in results)


def test_year_range_constraint_keeps_only_movies_inside_it(seeded_db):
    _, conn = seeded_db

    results = recommend(conn, "bob", RankParams(top_n=24), Constraints(year_range=(1990, 1999)))

    assert results
    assert all(1990 <= r.year <= 1999 for r in results)


def test_language_rating_and_genre_constraints_all_apply(seeded_db, library):
    _, conn = seeded_db
    movies = _movies_by_id(library)
    constraints = Constraints(
        languages=frozenset({"en"}),
        min_rating=8.0,
        required_genres=frozenset({"Thriller"}),
        excluded_genres=frozenset({"Comedy"}),
    )

    results = recommend(conn, "alice", RankParams(top_n=24), constraints)

    assert results
    for r in results:
        movie = movies[r.movie_id]
        assert movie["language"] == "en"
        assert movie["rating"] >= 8.0
        assert "Thriller" in movie["genres"]
        assert "Comedy" not in movie["genres"]


def test_every_reason_names_a_genre_or_one_of_the_users_own_watched_titles(seeded_db, library):
    _, conn = seeded_db
    titles = {movie["id"]: movie["title"] for movie in library["movies"]}
    watched = {user["name"]: set(user["watched"]) for user in library["users"]}
    for user_name, own in watched.items():
        others = set().union(*(ids for name, ids in watched.items() if name != user_name))
        foreign_titles = {titles[movie_id] for movie_id in others - own}
        own_titles = {titles[movie_id] for movie_id in own}

        results = recommend(conn, user_name, RankParams(top_n=24))

        assert results
        for r in results:
            assert len(r.reason) >= 10
            assert any(name in r.reason for name in (*GENRES, *own_titles))
            assert not any(title in r.reason for title in foreign_titles)


def test_load_movies_returns_what_insert_movie_stored():
    conn = connect(":memory:")
    movie = Movie(
        7,
        "Heat",
        1995,
        ("Crime", "Thriller"),
        170,
        "en",
        "US",
        "R",
        8.3,
        "Michael Mann",
        ("Al Pacino", "Robert De Niro"),
        "A heist crew and a detective.",
        ("heist", "los angeles"),
    )
    insert_movie(conn, movie)

    assert load_movies(conn) == [movie]


def test_recommend_is_deterministic(seeded_db):
    _, conn = seeded_db
    assert recommend(conn, "alice", RankParams(top_n=10)) == recommend(
        conn, "alice", RankParams(top_n=10)
    )


def test_recommend_respects_top_n(seeded_db):
    _, conn = seeded_db
    assert len(recommend(conn, "bob", RankParams(top_n=3))) == 3


def test_scoring_opens_no_socket(seeded_db, monkeypatch):
    _, conn = seeded_db

    def refuse(*args, **kwargs):
        raise AssertionError("scoring attempted a network connection")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)

    assert len(recommend(conn, "alice", RankParams(top_n=5))) == 5


def test_recommend_rejects_unknown_user(seeded_db):
    _, conn = seeded_db
    with pytest.raises(RecommendError, match="unknown user"):
        recommend(conn, "mallory", RankParams(top_n=3))


def test_recommend_rejects_user_without_history(seeded_db):
    _, conn = seeded_db
    conn.execute("INSERT INTO users (name) VALUES ('carol')")
    with pytest.raises(RecommendError, match="no watch history"):
        recommend(conn, "carol", RankParams(top_n=3))


def test_recommend_rejects_stale_vectors(seeded_db):
    _, conn = seeded_db
    conn.execute("UPDATE movie_vectors SET vector_version = vector_version - 1 WHERE movie_id = 1")
    with pytest.raises(RecommendError, match="rebuild"):
        recommend(conn, "alice", RankParams(top_n=3))


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _stored_taste(conn, user_name):
    return conn.execute(
        "SELECT t.vector_version, t.computed_at, t.vector FROM taste_vectors t"
        " JOIN users u ON u.id = t.user_id WHERE u.name = ?",
        (user_name,),
    ).fetchone()


def test_taste_vector_is_built_only_from_the_users_own_events(seeded_db, library):
    _, conn = seeded_db
    recommend(conn, "bob", RankParams(top_n=3), now=NOW)
    bob_before = _stored_taste(conn, "bob")

    alice_id = conn.execute("SELECT id FROM users WHERE name = 'alice'").fetchone()[0]
    alice = next(user for user in library["users"] if user["name"] == "alice")
    unwatched = [m["id"] for m in library["movies"] if m["id"] not in alice["watched"]]
    conn.executemany(
        "INSERT INTO watch_events (user_id, movie_id, completion, favourite) VALUES (?, ?, ?, ?)",
        [(alice_id, unwatched[0], 1.0, 1), (alice_id, unwatched[1], 0.1, 0)],
    )
    recommend(conn, "alice", RankParams(top_n=3), now=NOW)
    recommend(conn, "bob", RankParams(top_n=3), now=NOW)

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

    results = recommend(conn, "dana", RankParams(top_n=2), now=NOW)

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
