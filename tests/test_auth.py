import sqlite3
from datetime import timedelta
from typing import Annotated

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient

from app.auth import User, set_password, utc_now
from app.db import connect
from app.main import SESSION_COOKIE, create_app, current_user

PASSWORD = "correct horse battery staple"


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "auth.db"
    conn = connect(path)
    set_password(conn, "alice", PASSWORD, "user")
    conn.close()
    return path


@pytest.fixture
def client(db_path):
    app = create_app(db_path)

    @app.get("/protected")
    def protected(user: Annotated[User, Depends(current_user)]) -> dict:
        return {"name": user.name}

    # The session cookie is Secure, so the client only sends it back over https.
    with TestClient(app, base_url="https://testserver") as client:
        yield client


def login(client, password=PASSWORD, username="alice"):
    return client.post("/auth/login", json={"username": username, "password": password})


def test_valid_login_sets_secure_session_cookie(client):
    response = login(client)

    assert response.status_code == 200
    assert response.json() == {"username": "alice", "role": "user"}
    cookie = response.headers["set-cookie"]
    assert cookie.startswith(f"{SESSION_COOKIE}=")
    assert "HttpOnly" in cookie
    assert "Secure" in cookie
    assert "SameSite=lax" in cookie
    assert "expires=" in cookie
    assert f"Max-Age={7 * 24 * 3600}" in cookie
    assert client.get("/protected").json() == {"name": "alice"}


@pytest.mark.parametrize("username", ["alice", "nobody"])
def test_bad_credentials_are_rejected_without_cookie(client, username):
    response = login(client, password="wrong", username=username)

    assert response.status_code == 401
    assert "set-cookie" not in response.headers


def test_user_without_password_cannot_log_in(client, db_path):
    conn = connect(db_path)
    with conn:
        conn.execute("INSERT INTO users (name) VALUES ('synced')")
    conn.close()

    assert login(client, password="", username="synced").status_code == 401


def test_protected_route_requires_live_session(client, db_path):
    assert client.get("/protected").status_code == 401

    login(client)
    assert client.get("/protected").status_code == 200
    stolen = client.cookies[SESSION_COOKIE]
    logout = client.post("/auth/logout")
    assert logout.status_code == 204
    assert client.get("/protected").status_code == 401
    # Logout ends the session on the server, not just in this browser.
    client.cookies.set(SESSION_COOKIE, stolen)
    assert client.get("/protected").status_code == 401

    login(client)
    conn = connect(db_path)
    with conn:
        conn.execute("UPDATE sessions SET expires_at = ?", (utc_now().isoformat(),))
    conn.close()
    assert client.get("/protected").status_code == 401


def test_sixth_failed_login_within_window_is_rate_limited(client, db_path):
    for _ in range(5):
        assert login(client, password="wrong").status_code == 401

    assert login(client, password="wrong").status_code == 429
    assert login(client).status_code == 429
    assert login(client, password="wrong", username="bob").status_code == 401

    conn = connect(db_path)
    with conn:
        stale = (utc_now() - timedelta(minutes=16)).isoformat()
        conn.execute("UPDATE login_failures SET failed_at = ?", (stale,))
    conn.close()
    assert login(client).status_code == 200


@pytest.mark.parametrize("method", ["get", "post"])
@pytest.mark.parametrize("path", ["/auth/register", "/signup"])
def test_no_sign_up_route(client, method, path):
    assert getattr(client, method)(path).status_code == 404


def test_password_is_stored_as_scrypt_hash(db_path):
    conn = connect(db_path)
    (stored,) = conn.execute("SELECT password_hash FROM users WHERE name = 'alice'").fetchone()
    conn.close()

    assert stored != PASSWORD
    assert PASSWORD not in stored
    assert stored.startswith("scrypt$")


def test_resetting_password_ends_existing_sessions(client, db_path):
    login(client)
    conn = connect(db_path)
    set_password(conn, "alice", "a new password", "admin")
    conn.close()

    assert client.get("/protected").status_code == 401
    assert login(client, password="a new password").json()["role"] == "admin"


def test_connect_adds_credential_columns_to_existing_users(tmp_path):
    path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(path)
    legacy.executescript(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);"
        "INSERT INTO users (name) VALUES ('alice');"
    )
    legacy.close()

    conn = connect(path)
    connect(path).close()

    assert conn.execute("SELECT id, name, password_hash, role FROM users").fetchall() == [
        (1, "alice", None, "user")
    ]
    conn.close()
