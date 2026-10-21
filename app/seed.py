"""Load a JSON library fixture into an empty database and build its movie vectors."""

import json
import sqlite3
from pathlib import Path

from app.db import insert_movie, store_vector
from app.vectors import VECTOR_VERSION, Movie, build_movie_vector


class SeedError(Exception):
    pass


def load_fixture(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def seed(conn: sqlite3.Connection, fixture: dict) -> None:
    """Insert the fixture's movies, vectors, users and watch events in one transaction.

    Refuses a database that already holds movies rather than merging into or wiping it.
    """
    if conn.execute("SELECT EXISTS (SELECT 1 FROM movies)").fetchone()[0]:
        raise SeedError("database already contains movies; seed into a new database")
    with conn:
        for record in fixture["movies"]:
            movie = Movie(**record)
            insert_movie(conn, movie)
            store_vector(conn, movie.id, VECTOR_VERSION, build_movie_vector(movie))
        for user in fixture["users"]:
            user_id = conn.execute("INSERT INTO users (name) VALUES (?)", (user["name"],)).lastrowid
            conn.executemany(
                "INSERT INTO watch_events (user_id, movie_id) VALUES (?, ?)",
                [(user_id, movie_id) for movie_id in user["watched"]],
            )
