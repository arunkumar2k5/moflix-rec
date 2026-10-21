import pytest

from app.db import decode_vector
from app.seed import SeedError, seed
from app.vectors import FULL_DIMENSION, VECTOR_VERSION


def test_seed_stores_a_versioned_vector_of_declared_length_for_every_movie(seeded_db, library):
    _, conn = seeded_db
    rows = conn.execute(
        "SELECT m.id, v.vector_version, v.dimension, v.vector"
        " FROM movies m LEFT JOIN movie_vectors v ON v.movie_id = m.id"
    ).fetchall()

    assert len(rows) == len(library["movies"])
    for _, version, dimension, blob in rows:
        assert version == VECTOR_VERSION
        assert dimension == FULL_DIMENSION
        assert decode_vector(blob).shape == (FULL_DIMENSION,)


def test_seed_records_watch_events_per_user(seeded_db, library):
    _, conn = seeded_db
    for user in library["users"]:
        watched = conn.execute(
            "SELECT w.movie_id FROM watch_events w JOIN users u ON u.id = w.user_id"
            " WHERE u.name = ? ORDER BY w.movie_id",
            (user["name"],),
        ).fetchall()
        assert [row[0] for row in watched] == sorted(user["watched"])


def test_seed_refuses_an_already_seeded_database(seeded_db, library, fake_embedder):
    _, conn = seeded_db
    with pytest.raises(SeedError):
        seed(conn, library, embedder=fake_embedder)
    assert conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0] == len(library["movies"])
