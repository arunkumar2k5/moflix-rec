"""Load a JSON library fixture into an empty database and build its movie vectors."""

import json
import sqlite3
from pathlib import Path

from app.db import insert_movie, store_vector
from app.vectors import (
    EMBEDDING_MODEL_NAME,
    VECTOR_VERSION,
    Movie,
    _movie_text,
    build_full_vector,
    embed_texts,
)


class SeedError(Exception):
    pass


def load_fixture(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def seed(conn: sqlite3.Connection, fixture: dict, embedder=None) -> None:
    """Insert the fixture's movies, vectors, users and watch events in one transaction.

    Refuses a database that already holds movies rather than merging into or wiping it.
    Loads all-MiniLM-L6-v2 once and batch-embeds all movie texts before writing.
    Pass `embedder` (a SentenceTransformer instance or callable) to override the
    model — useful in tests to avoid the network/disk load.
    """
    if conn.execute("SELECT EXISTS (SELECT 1 FROM movies)").fetchone()[0]:
        raise SeedError("database already contains movies; seed into a new database")

    movies = [Movie(**record) for record in fixture["movies"]]

    # Batch embed all movie texts in one shot (one model load, one encode call).
    if embedder is None:
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [_movie_text(m) for m in movies]
    embeddings = embed_texts(texts, model=embedder)

    with conn:
        for movie, embedding in zip(movies, embeddings, strict=True):
            insert_movie(conn, movie)
            store_vector(conn, movie.id, VECTOR_VERSION, build_full_vector(movie, embedding))
        for user in fixture["users"]:
            user_id = conn.execute("INSERT INTO users (name) VALUES (?)", (user["name"],)).lastrowid
            conn.executemany(
                "INSERT INTO watch_events (user_id, movie_id) VALUES (?, ?)",
                [(user_id, movie_id) for movie_id in user["watched"]],
            )
