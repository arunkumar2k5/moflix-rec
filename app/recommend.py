"""Base taste vector and dot-product scoring over a user's unwatched movies.

Everything here reads SQLite and computes with NumPy only: scoring never touches the network.
"""

import sqlite3
from dataclasses import dataclass

import numpy as np

from app.db import decode_vector
from app.vectors import DIMENSION, VECTOR_VERSION


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


def base_taste_vector(watched_vectors: np.ndarray) -> np.ndarray:
    """Unweighted mean of the vectors of every movie the user watched."""
    return watched_vectors.mean(axis=0)


def _user_id(conn: sqlite3.Connection, user_name: str) -> int:
    row = conn.execute("SELECT id FROM users WHERE name = ?", (user_name,)).fetchone()
    if row is None:
        raise RecommendError(f"unknown user: {user_name}")
    return row[0]


def _load_vectors(conn: sqlite3.Connection) -> tuple[np.ndarray, np.ndarray]:
    rows = conn.execute(
        "SELECT movie_id, vector_version, dimension, vector FROM movie_vectors ORDER BY movie_id"
    ).fetchall()
    stale = sum(version != VECTOR_VERSION or dim != DIMENSION for _, version, dim, _ in rows)
    if stale:
        raise RecommendError(
            f"{stale} movie vectors are not version {VECTOR_VERSION}; rebuild them first"
        )
    ids = np.array([row[0] for row in rows], dtype=np.int64)
    matrix = np.array([decode_vector(row[3]) for row in rows]).reshape(len(rows), DIMENSION)
    return ids, matrix


def recommend(conn: sqlite3.Connection, user_name: str, top_n: int) -> list[Recommendation]:
    user_id = _user_id(conn, user_name)
    watched_ids = [
        row[0]
        for row in conn.execute("SELECT movie_id FROM watch_events WHERE user_id = ?", (user_id,))
    ]
    if not watched_ids:
        raise RecommendError(f"user {user_name} has no watch history to build a taste vector from")
    ids, matrix = _load_vectors(conn)
    watched = np.isin(ids, watched_ids)
    taste = base_taste_vector(matrix[watched])
    ranked = rank(taste, ids[~watched], matrix[~watched])[:top_n]
    titles = {
        movie_id: (title, year)
        for movie_id, title, year in conn.execute("SELECT id, title, year FROM movies")
    }
    return [Recommendation(mid, *titles[mid], score) for mid, score in ranked]
