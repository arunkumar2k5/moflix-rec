"""Web service: `python -m app.main` serves on port 8000 with JSON-line logs (see app.logs).
Reads the database path from MOFLIX_DB.

There is deliberately no sign-up route; accounts come only from `python -m app.cli add-user`.
"""

import logging
import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Literal

import uvicorn
from fastapi import Cookie, Depends, FastAPI, HTTPException, Request, Response, status
from pydantic import BaseModel

from app.auth import (
    SESSION_LIFETIME,
    LoginRateLimited,
    Role,
    User,
    authenticate,
    create_session,
    end_session,
    hash_password,
    session_user,
    utc_now,
    verify_password,
)
from app.db import connect
from app.logs import LOG_CONFIG

SESSION_COOKIE = "session"

log = logging.getLogger(__name__)


class Health(BaseModel):
    status: Literal["ok", "error"]
    db: Literal["ok", "error"]


class Credentials(BaseModel):
    username: str
    password: str


class LoggedIn(BaseModel):
    username: str
    role: Role


class UserInfo(BaseModel):
    id: int
    name: str


class ChangePassword(BaseModel):
    current_password: str
    new_password: str


def get_db(request: Request) -> Iterator[sqlite3.Connection]:
    # FastAPI may run the dependency and the endpoint on different threadpool threads.
    conn = connect(request.app.state.db_path, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()


Db = Annotated[sqlite3.Connection, Depends(get_db)]
SessionToken = Annotated[str | None, Cookie(alias=SESSION_COOKIE)]


def current_user(conn: Db, token: SessionToken = None) -> User:
    user = session_user(conn, token, utc_now()) if token else None
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "not logged in")
    return user


def create_app(db_path: str | Path) -> FastAPI:
    app = FastAPI(title="moflix-rec")
    app.state.db_path = db_path

    @app.get("/health")
    def health(response: Response) -> Health:
        try:
            connect(db_path).close()
        except sqlite3.Error:
            log.exception("health_db_error")
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return Health(status="error", db="error")
        return Health(status="ok", db="ok")

    @app.post("/auth/login")
    def login(credentials: Credentials, conn: Db, response: Response) -> LoggedIn:
        now = utc_now()
        try:
            user = authenticate(conn, credentials.username, credentials.password, now)
        except LoginRateLimited:
            log.warning("login_rate_limited")
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS, "too many failed logins"
            ) from None
        if user is None:
            log.warning("login_failed")
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid username or password")
        session = create_session(conn, user, now)
        log.info("login", extra={"user_id": user.id})
        response.set_cookie(
            SESSION_COOKIE,
            session.token,
            max_age=int(SESSION_LIFETIME.total_seconds()),
            expires=session.expires_at,
            httponly=True,
            secure=True,
            samesite="lax",
        )
        return LoggedIn(username=user.name, role=user.role)

    @app.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
    def logout(conn: Db, response: Response, token: SessionToken = None) -> None:
        if token:
            end_session(conn, token)
        response.delete_cookie(SESSION_COOKIE, httponly=True, secure=True, samesite="lax")

    @app.get("/auth/me")
    def me(user: Annotated[User, Depends(current_user)]) -> LoggedIn:
        return LoggedIn(username=user.name, role=user.role)

    @app.post("/auth/change-password", status_code=status.HTTP_204_NO_CONTENT)
    def change_password(
        body: ChangePassword,
        conn: Db,
        user: Annotated[User, Depends(current_user)],
    ) -> None:
        row = conn.execute("SELECT password_hash FROM users WHERE id = ?", (user.id,)).fetchone()
        if row is None or not verify_password(body.current_password, row[0] or ""):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "current password is wrong")
        if not body.new_password:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "new password is empty")
        with conn:
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (hash_password(body.new_password), user.id),
            )
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user.id,))

    @app.get("/api/users")
    def list_users(conn: Db) -> list[UserInfo]:
        rows = conn.execute("SELECT id, name FROM users ORDER BY name").fetchall()
        return [UserInfo(id=row[0], name=row[1]) for row in rows]

    return app


app = create_app(os.environ.get("MOFLIX_DB", "moflix.db"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=LOG_CONFIG)
