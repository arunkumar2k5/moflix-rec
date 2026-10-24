"""SQLite storage for users, sessions, movies, vectors, watch history, taste vectors, picks
and feedback on picks."""

import json
import sqlite3
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, get_args

import numpy as np

from app.auth import Role
from app.vectors import Movie

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS login_failures (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL,
    failed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS login_failures_username ON login_failures(username, failed_at);
CREATE TABLE IF NOT EXISTS movies (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    year INTEGER NOT NULL,
    genres TEXT NOT NULL,
    runtime INTEGER NOT NULL,
    language TEXT NOT NULL,
    country TEXT NOT NULL,
    certification TEXT NOT NULL,
    rating REAL NOT NULL,
    director TEXT NOT NULL,
    "cast" TEXT NOT NULL,
    overview TEXT NOT NULL DEFAULT '',
    keywords TEXT NOT NULL DEFAULT '[]',
    poster_url TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS movie_vectors (
    movie_id INTEGER PRIMARY KEY REFERENCES movies(id),
    vector_version INTEGER NOT NULL,
    dimension INTEGER NOT NULL,
    vector BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS watch_events (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    movie_id INTEGER NOT NULL REFERENCES movies(id)
);
CREATE INDEX IF NOT EXISTS watch_events_user ON watch_events(user_id);
CREATE TABLE IF NOT EXISTS taste_vectors (
    user_id INTEGER PRIMARY KEY REFERENCES users(id),
    vector_version INTEGER NOT NULL,
    dimension INTEGER NOT NULL,
    computed_at TEXT NOT NULL,
    vector BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS recommendation_log (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    movie_id INTEGER NOT NULL REFERENCES movies(id),
    shown_at TEXT NOT NULL
);
"""

# Taste signals on watch_events, added in place so databases created before them keep their
# history. The defaults give a legacy row weight 1 in the taste vector; NULL watched_at is unknown.
WATCH_SIGNAL_COLUMNS = {
    "completion": "REAL NOT NULL DEFAULT 1.0",
    "play_count": "INTEGER NOT NULL DEFAULT 1",
    "favourite": "INTEGER NOT NULL DEFAULT 0",
    "not_interested": "INTEGER NOT NULL DEFAULT 0",
    "watched_at": "TEXT",
}

# Login credentials on users, added in place because synced and seeded users predate them.
# A NULL password_hash means the account cannot log in until an admin runs `add-user`.
USER_AUTH_COLUMNS = {
    "password_hash": "TEXT",
    "role": "TEXT NOT NULL DEFAULT 'user'"
    f" CHECK (role IN ({', '.join(repr(role) for role in get_args(Role))}))",
}

FeedbackAction = Literal["watch_tonight", "not_interested", "already_seen"]

# Append-only: a user's current action on a movie is the one in their newest row for it.
FEEDBACK_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS feedback_events (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    movie_id INTEGER NOT NULL REFERENCES movies(id),
    action TEXT NOT NULL
        CHECK (action IN ({", ".join(repr(action) for action in get_args(FeedbackAction))})),
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS feedback_events_user ON feedback_events(user_id, movie_id);
"""

# Vectors are stored as raw little-endian float64 bytes.
VECTOR_DTYPE = np.dtype("<f8")


def connect(path: str | Path, check_same_thread: bool = True) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=check_same_thread)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    conn.executescript(FEEDBACK_SCHEMA)
    _add_missing_columns(conn, "users", USER_AUTH_COLUMNS)
    _add_missing_columns(conn, "watch_events", WATCH_SIGNAL_COLUMNS)
    _add_missing_columns(conn, "movies", {"poster_url": "TEXT NOT NULL DEFAULT ''"})
    return conn


def _add_missing_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, definition in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def insert_movie(conn: sqlite3.Connection, movie: Movie) -> None:
    conn.execute(
        "INSERT INTO movies (id, title, year, genres, runtime, language, country,"
        ' certification, rating, director, "cast", overview, keywords, poster_url)'
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            movie.id,
            movie.title,
            movie.year,
            json.dumps(list(movie.genres)),
            movie.runtime,
            movie.language,
            movie.country,
            movie.certification,
            movie.rating,
            movie.director,
            json.dumps(list(movie.cast)),
            movie.overview,
            json.dumps(list(movie.keywords)),
            movie.poster_url,
        ),
    )


def store_vector(conn: sqlite3.Connection, movie_id: int, version: int, vector: np.ndarray) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO movie_vectors (movie_id, vector_version, dimension, vector)"
        " VALUES (?, ?, ?, ?)",
        (movie_id, version, vector.shape[0], vector.astype(VECTOR_DTYPE).tobytes()),
    )


def store_taste_vector(
    conn: sqlite3.Connection,
    user_id: int,
    version: int,
    vector: np.ndarray,
    computed_at: datetime,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO taste_vectors"
        " (user_id, vector_version, dimension, computed_at, vector) VALUES (?, ?, ?, ?, ?)",
        (
            user_id,
            version,
            vector.shape[0],
            computed_at.isoformat(),
            vector.astype(VECTOR_DTYPE).tobytes(),
        ),
    )


def log_recommendations(
    conn: sqlite3.Connection, user_id: int, movie_ids: Sequence[int], shown_at: datetime
) -> None:
    conn.executemany(
        "INSERT INTO recommendation_log (user_id, movie_id, shown_at) VALUES (?, ?, ?)",
        [(user_id, movie_id, shown_at.isoformat()) for movie_id in movie_ids],
    )


@dataclass(frozen=True)
class Feedback:
    movie_id: int
    action: FeedbackAction
    created_at: datetime


def record_feedback(
    conn: sqlite3.Connection,
    user_id: int,
    movie_id: int,
    action: FeedbackAction,
    created_at: datetime,
) -> None:
    conn.execute(
        "INSERT INTO feedback_events (user_id, movie_id, action, created_at) VALUES (?, ?, ?, ?)",
        (user_id, movie_id, action, created_at.isoformat()),
    )


def current_feedback(conn: sqlite3.Connection, user_id: int) -> list[Feedback]:
    """The user's latest action on each movie they gave feedback on, oldest first."""
    rows = conn.execute(
        "SELECT movie_id, action, created_at FROM feedback_events WHERE id IN"
        " (SELECT MAX(id) FROM feedback_events WHERE user_id = ? GROUP BY movie_id) ORDER BY id",
        (user_id,),
    )
    return [
        Feedback(movie_id, action, datetime.fromisoformat(created_at))
        for movie_id, action, created_at in rows
    ]


def decode_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=VECTOR_DTYPE)


def load_movies(conn: sqlite3.Connection, ids: Collection[int] | None = None) -> list[Movie]:
    """Every movie, or only those in `ids`, ordered by id."""
    where = "" if ids is None else f" WHERE id IN ({', '.join('?' * len(ids))})"
    cursor = conn.execute(
        "SELECT id, title, year, genres, runtime, language, country, certification, rating,"
        f' director, "cast", overview, keywords, poster_url FROM movies{where} ORDER BY id',
        () if ids is None else tuple(ids),
    )
    columns = [description[0] for description in cursor.description]
    movies = []
    for row in cursor:
        fields = dict(zip(columns, row, strict=True))
        for name in ("genres", "cast", "keywords"):
            fields[name] = tuple(json.loads(fields[name]))
        movies.append(Movie(**fields))
    return movies
