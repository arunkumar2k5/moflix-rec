import json
from pathlib import Path

import pytest

from app.db import connect
from app.seed import seed

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "library.json"


@pytest.fixture
def library() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def seeded_db(tmp_path, library):
    path = tmp_path / "library.db"
    conn = connect(path)
    seed(conn, library)
    yield path, conn
    conn.close()
