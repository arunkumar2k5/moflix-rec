"""SQLite storage for users, movies, their vectors, watch history and taste vectors."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from app.vectors import Movie

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);
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
    keywords TEXT NOT NULL DEFAULT '[]'
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

# Vectors are stored as raw little-endian float64 bytes.
VECTOR_DTYPE = np.dtype("<f8")


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    _add_watch_signal_columns(conn)
    return conn


def _add_watch_signal_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(watch_events)")}
    for name, definition in WATCH_SIGNAL_COLUMNS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE watch_events ADD COLUMN {name} {definition}")


def insert_movie(conn: sqlite3.Connection, movie: Movie) -> None:
    conn.execute(
        "INSERT INTO movies (id, title, year, genres, runtime, language, country,"
        ' certification, rating, director, "cast", overview, keywords)'
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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


def decode_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=VECTOR_DTYPE)


def load_movies(conn: sqlite3.Connection) -> list[Movie]:
    rows = conn.execute(
        "SELECT id, title, year, genres, runtime, language, country, certification, rating,"
        ' director, "cast", overview, keywords FROM movies ORDER BY id'
    )
    return [
        Movie(
            id=movie_id,
            title=title,
            year=year,
            genres=tuple(json.loads(genres)),
            runtime=runtime,
            language=language,
            country=country,
            certification=certification,
            rating=rating,
            director=director,
            cast=tuple(json.loads(cast)),
            overview=overview,
            keywords=tuple(json.loads(keywords)),
        )
        for (
            movie_id,
            title,
            year,
            genres,
            runtime,
            language,
            country,
            certification,
            rating,
            director,
            cast,
            overview,
            keywords,
        ) in rows
    ]
