import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.auth import set_password
from app.db import connect
from app.main import create_app
from app.seed import seed
from app.vectors import EMBEDDING_DIM

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "library.json"
PASSWORD = "correct horse battery staple"


@pytest.fixture
def library() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def fake_embedder():
    """Deterministic stub that returns a fixed unit vector without loading any model."""

    class _FakeEmbedder:
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            # Same vector for every text so tests are reproducible.
            v = np.ones(EMBEDDING_DIM, dtype=np.float32)
            if normalize_embeddings:
                v /= np.linalg.norm(v)
            return np.tile(v, (len(texts), 1))

    return _FakeEmbedder()


@pytest.fixture
def seeded_db(tmp_path, library, fake_embedder):
    path = tmp_path / "library.db"
    conn = connect(path)
    seed(conn, library, embedder=fake_embedder)
    yield path, conn
    conn.close()


@pytest.fixture
def db_path(seeded_db):
    """The seeded library's path, with alice and bob able to log in with PASSWORD."""
    path, conn = seeded_db
    for name in ("alice", "bob"):
        set_password(conn, name, PASSWORD, "user")
    return path


def api_client(db_path, top_n=3):
    # The session cookie is Secure, so the client only sends it back over https.
    return TestClient(create_app(db_path, top_n), base_url="https://testserver")


def logged_in(db_path, name, top_n=3):
    client = api_client(db_path, top_n)
    response = client.post("/auth/login", json={"username": name, "password": PASSWORD})
    assert response.status_code == 200
    return client
