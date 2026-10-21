import json
from pathlib import Path

import numpy as np
import pytest

from app.db import connect
from app.seed import seed
from app.vectors import EMBEDDING_DIM

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "library.json"


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
