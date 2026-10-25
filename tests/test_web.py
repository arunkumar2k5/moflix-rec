import os
import re

import pytest
from fastapi.testclient import TestClient
from markupsafe import escape

from app.auth import set_password
from app.chat import ChatConfig
from app.main import CSRF_COOKIE, CSRF_HEADER, create_app

PASSWORD = "correct horse battery staple"
VIEWPORT = '<meta name="viewport" content="width=device-width, initial-scale=1">'
ORIGIN = {"Origin": "https://testserver"}
FEEDBACK_BUTTONS = ("Watch tonight", "Not interested", "Already seen")


@pytest.fixture
def db_path(seeded_db):
    path, conn = seeded_db
    set_password(conn, "alice", PASSWORD, "user")
    return path


def _client(db_path, chat_config=None):
    # Both the session and the CSRF cookie are Secure, so the client only sends them over https.
    return TestClient(create_app(db_path, chat_config=chat_config), base_url="https://testserver")


def _log_in(client):
    response = client.post("/auth/login", json={"username": "alice", "password": PASSWORD})
    assert response.status_code == 200
    return client


def _csrf(client):
    assert client.get("/").status_code == 200
    return client.cookies[CSRF_COOKIE]


def _without_card_template(html):
    return re.sub(r'<template id="card-template">.*?</template>', "", html, flags=re.DOTALL)


def _watch_tonight_section(html):
    return re.search(r'<section id="watch-tonight">.*?</section>', html, re.DOTALL).group(0)


def test_root_without_session_serves_login_form(db_path):
    response = _client(db_path).get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    assert 'id="login-form"' in html
    assert 'id="username"' in html
    assert 'id="password"' in html
    assert 'class="card"' not in html
    cookie = response.headers["set-cookie"]
    assert cookie.startswith(f"{CSRF_COOKIE}=")
    for attribute in ("HttpOnly", "Secure", "SameSite=strict", "Path=/"):
        assert attribute in cookie
    token = response.cookies[CSRF_COOKIE]
    assert f'<meta name="csrf-token" content="{token}">' in html


def test_root_keeps_an_existing_csrf_token(db_path):
    client = _client(db_path)
    token = _csrf(client)

    assert _csrf(_log_in(client)) == token


def test_logged_in_root_shows_picks_with_three_feedback_buttons_per_card(db_path):
    client = _log_in(_client(db_path))
    items = client.get("/recommendations").json()["items"]
    assert len(items) >= 2

    html = client.get("/").text

    for item in items:
        assert str(escape(item["title"])) in html
    cards = _without_card_template(html)
    for label in FEEDBACK_BUTTONS:
        assert cards.count(f">{label}</button>") == len(items)


def test_pages_declare_mobile_viewport(db_path):
    client = _client(db_path)
    assert VIEWPORT in client.get("/").text
    assert VIEWPORT in _log_in(client).get("/").text


def test_browser_post_needs_matching_csrf_token(db_path):
    client = _log_in(_client(db_path))
    token = _csrf(client)
    movie_id = client.get("/recommendations").json()["items"][0]["movie_id"]

    assert client.post("/chat/reset", headers=ORIGIN).status_code == 403
    assert client.post("/chat/reset", headers={**ORIGIN, CSRF_HEADER: "wrong"}).status_code == 403
    assert client.post("/chat/reset", headers={**ORIGIN, CSRF_HEADER: token}).status_code == 204
    feedback = {"movie_id": movie_id, "action": "watch_tonight"}
    response = client.post("/feedback", json=feedback, headers={**ORIGIN, CSRF_HEADER: token})
    assert response.status_code == 204
    login = client.post(
        "/auth/login",
        json={"username": "alice", "password": PASSWORD},
        headers={"Sec-Fetch-Site": "same-origin"},
    )
    assert login.status_code == 403


def test_browser_post_without_csrf_cookie_is_refused(db_path):
    client = _log_in(_client(db_path))

    response = client.post("/chat/reset", headers={**ORIGIN, CSRF_HEADER: "guess"})

    assert response.status_code == 403


def test_api_clients_without_browser_headers_are_not_csrf_checked(db_path):
    client = _log_in(_client(db_path))

    assert client.post("/chat/reset").status_code == 204


def test_served_pages_contain_no_api_key_values(db_path, monkeypatch):
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "TMDB_API_KEY"):
        monkeypatch.setenv(name, f"sentinel-{name.lower()}-value")
    config = ChatConfig.from_env()
    assert config.provider is not None
    client = _client(db_path, config)

    pages = [client.get("/").text, _log_in(client).get("/").text]

    secrets = [v for k, v in os.environ.items() if k.endswith("_API_KEY") and v]
    assert len(secrets) >= 3
    for page in pages:
        for secret in secrets:
            assert secret not in page


def test_watch_tonight_mark_persists_on_page(db_path):
    client = _log_in(_client(db_path))
    first = client.get("/recommendations").json()["items"][0]

    response = client.post(
        "/feedback", json={"movie_id": first["movie_id"], "action": "watch_tonight"}
    )
    assert response.status_code == 204

    assert str(escape(first["title"])) in _watch_tonight_section(client.get("/").text)
