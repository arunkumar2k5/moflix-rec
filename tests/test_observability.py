import json
import logging
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from app.auth import set_password
from app.db import connect
from app.logs import JsonFormatter
from app.main import SESSION_COOKIE, create_app

PASSWORD = "correct horse battery staple"
LOG_CONFIG = Path(__file__).parent.parent / "app" / "log_config.json"


def make_alice(db_path):
    conn = connect(db_path)
    set_password(conn, "alice", PASSWORD, "user")
    [(user_id,)] = conn.execute("SELECT id FROM users WHERE name = 'alice'").fetchall()
    conn.close()
    return user_id


def test_health_is_ok_without_a_session(tmp_path):
    client = TestClient(create_app(tmp_path / "health.db"))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "db": "ok"}


def test_health_reports_unavailable_database(tmp_path):
    client = TestClient(create_app(tmp_path / "missing-dir" / "health.db"))

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["db"] != "ok"


def test_login_log_line_is_json_without_secrets(tmp_path, caplog):
    db_path = tmp_path / "auth.db"
    user_id = make_alice(db_path)
    client = TestClient(create_app(db_path), base_url="https://testserver")

    with caplog.at_level(logging.INFO, logger="moflix"):
        response = client.post("/auth/login", json={"username": "alice", "password": PASSWORD})
    cookie = response.cookies[SESSION_COOKIE]

    [record] = [r for r in caplog.records if r.name == "moflix"]
    line = JsonFormatter().format(record)
    assert "\n" not in line
    entry = json.loads(line)
    assert {"ts", "level", "event"} <= entry.keys()
    assert entry["user_id"] == user_id
    assert entry["path"] == "/auth/login"
    assert entry["status"] == 200
    assert PASSWORD not in line
    assert cookie not in line


def test_formatter_drops_fields_outside_the_whitelist():
    record = logging.LogRecord("moflix", logging.INFO, __file__, 1, "request", None, None)
    record.password = PASSWORD
    record.user_id = 7

    entry = json.loads(JsonFormatter().format(record))

    assert entry == {"ts": entry["ts"], "level": "info", "event": "request", "user_id": 7}


def test_authenticated_request_log_carries_user_id(tmp_path, caplog):
    db_path = tmp_path / "auth.db"
    user_id = make_alice(db_path)
    client = TestClient(create_app(db_path), base_url="https://testserver")
    client.post("/auth/login", json={"username": "alice", "password": PASSWORD})

    with caplog.at_level(logging.INFO, logger="moflix"):
        response = client.get("/auth/me")

    assert response.status_code == 200
    [record] = [r for r in caplog.records if r.name == "moflix"]
    assert record.path == "/auth/me"
    assert record.user_id == user_id


def test_shipped_log_config_writes_json_lines_to_stderr():
    # A fresh interpreter, as in the container, so pytest's own log capture is not in the way.
    program = (
        "import json, logging.config\n"
        f"logging.config.dictConfig(json.load(open({str(LOG_CONFIG)!r})))\n"
        "from app.logs import logger\n"
        "logger.info('request', extra={'user_id': 7, 'path': '/auth/me'})\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=LOG_CONFIG.parent.parent,
        capture_output=True,
        text=True,
        check=True,
    )

    [line] = result.stderr.splitlines()
    entry = json.loads(line)
    assert entry == {
        "ts": entry["ts"],
        "level": "info",
        "event": "request",
        "user_id": 7,
        "path": "/auth/me",
    }
