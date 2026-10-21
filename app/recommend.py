"""Weighted base taste vector and dot-product scoring over a user's unwatched movies.

Everything here reads SQLite and computes with NumPy only: scoring never touches the network.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from app.db import decode_vector, store_taste_vector
from app.taste import WatchEvent, build_base_taste_vector
from app.vectors import FULL_DIMENSION, VECTOR_VERSION


class RecommendError(Exception):
    pass


@dataclass(frozen=True)
class Recommendation:
    movie_id: int
    title: str
    year: int
    score: float


def rank(taste: np.ndarray, movie_ids: np.ndarray, vectors: np.ndarray) -> list[tuple[int, float]]:
    """(movie id, score) pairs by descending dot product, ties broken by ascending id."""
    scores = vectors @ taste
    order = np.lexsort((movie_ids, -scores))
    return [(int(movie_ids[i]), float(scores[i])) for i in order]


def _user_id(conn: sqlite3.Connection, user_name: str) -> int:
    row = conn.execute("SELECT id FROM users WHERE name = ?", (user_name,)).fetchone()
    if row is None:
        raise RecommendError(f"unknown user: {user_name}")
    return row[0]


def _watch_events(conn: sqlite3.Connection, user_id: int) -> list[WatchEvent]:
    rows = conn.execute(
        "SELECT movie_id, completion, play_count, favourite, not_interested, watched_at"
        " FROM watch_events WHERE user_id = ? ORDER BY movie_id, id",
        (user_id,),
    )
    return [
        WatchEvent(
            movie_id=movie_id,
            completion=completion,
            play_count=play_count,
            favourite=bool(favourite),
            not_interested=bool(not_interested),
            watched_at=None if watched_at is None else datetime.fromisoformat(watched_at),
        )
        for movie_id, completion, play_count, favourite, not_interested, watched_at in rows
    ]


def _load_vectors(conn: sqlite3.Connection) -> tuple[np.ndarray, np.ndarray]:
    rows = conn.execute(
        "SELECT movie_id, vector_version, dimension, vector FROM movie_vectors ORDER BY movie_id"
    ).fetchall()
    stale = sum(version != VECTOR_VERSION or dim != FULL_DIMENSION for _, version, dim, _ in rows)
    if stale:
        raise RecommendError(
            f"{stale} movie vectors are not version {VECTOR_VERSION}; rebuild them first"
        )
    ids = np.array([row[0] for row in rows], dtype=np.int64)
    matrix = np.array([decode_vector(row[3]) for row in rows]).reshape(len(rows), FULL_DIMENSION)
    return ids, matrix


def recommend(
    conn: sqlite3.Connection, user_name: str, top_n: int, now: datetime | None = None
) -> list[Recommendation]:
    """Rank the user's unwatched movies, recording the taste vector built at `now`."""
    now = now or datetime.now(timezone.utc)
    user_id = _user_id(conn, user_name)
    events = _watch_events(conn, user_id)
    if not events:
        raise RecommendError(f"user {user_name} has no watch history to build a taste vector from")
    ids, matrix = _load_vectors(conn)
    vectors = dict(zip(ids.tolist(), matrix, strict=True))
    taste = build_base_taste_vector(events, vectors, now)
    with conn:
        store_taste_vector(conn, user_id, VECTOR_VERSION, taste, now)
    watched = np.isin(ids, [event.movie_id for event in events])
    ranked = rank(taste, ids[~watched], matrix[~watched])[:top_n]
    titles = {
        movie_id: (title, year)
        for movie_id, title, year in conn.execute("SELECT id, title, year FROM movies")
    }
    return [Recommendation(mid, *titles[mid], score) for mid, score in ranked]
