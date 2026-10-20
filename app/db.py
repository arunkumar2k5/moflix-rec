"""SQLite storage for users, movies, their vectors and watch history."""

import json
import sqlite3
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
    "cast" TEXT NOT NULL
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
"""

# Vectors are stored as raw little-endian float64 bytes.
VECTOR_DTYPE = np.dtype("<f8")


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def insert_movie(conn: sqlite3.Connection, movie: Movie) -> None:
    conn.execute(
        "INSERT INTO movies (id, title, year, genres, runtime, language, country,"
        ' certification, rating, director, "cast")'
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
        ),
    )


def store_vector(conn: sqlite3.Connection, movie_id: int, version: int, vector: np.ndarray) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO movie_vectors (movie_id, vector_version, dimension, vector)"
        " VALUES (?, ?, ?, ?)",
        (movie_id, version, vector.shape[0], vector.astype(VECTOR_DTYPE).tobytes()),
    )


def decode_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=VECTOR_DTYPE)
