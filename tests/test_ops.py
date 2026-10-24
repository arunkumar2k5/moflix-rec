import json
import logging
import sys

from fastapi.testclient import TestClient

from app.auth import set_password
from app.db import connect
from app.logs import JsonFormatter
from app.main import SESSION_COOKIE, create_app

PASSWORD = "correct horse battery staple"


def test_health_reports_ok_without_a_session(tmp_path):
    client = TestClient(create_app(tmp_path / "health.db"))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "db": "ok"}


def test_health_reports_unavailable_when_the_database_cannot_be_opened(tmp_path):
    # A directory cannot be opened as an SQLite file, even by root.
    client = TestClient(create_app(tmp_path))

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["db"] != "ok"


def test_login_log_lines_are_json_without_secrets(tmp_path, caplog):
    db_path = tmp_path / "auth.db"
    conn = connect(db_path)
    set_password(conn, "alice", PASSWORD, "user")
    user_id = conn.execute("SELECT id FROM users WHERE name = 'alice'").fetchone()[0]
    conn.close()
    client = TestClient(create_app(db_path), base_url="https://testserver")

    with caplog.at_level(logging.INFO, logger="app.main"):
        response = client.post("/auth/login", json={"username": "alice", "password": PASSWORD})
        client.post("/auth/login", json={"username": "alice", "password": "wrong"})

    cookie = response.cookies[SESSION_COOKIE]
    lines = [JsonFormatter().format(record) for record in caplog.records]
    assert lines
    for line in lines:
        assert PASSWORD not in line
        assert cookie not in line
    entries = [json.loads(line) for line in lines]
    assert all({"ts", "level", "event"} <= entry.keys() for entry in entries)
    assert {"level": "info", "event": "login", "user_id": user_id}.items() <= entries[0].items()
    assert entries[1]["event"] == "login_failed"
    assert "user_id" not in entries[1]


def test_logged_exceptions_carry_their_traceback():
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord("x", logging.ERROR, __file__, 1, "failed", None, sys.exc_info())

    entry = json.loads(JsonFormatter().format(record))

    assert entry["event"] == "failed"
    assert "ValueError: boom" in entry["exc"]
