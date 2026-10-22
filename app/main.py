"""Web service: `uvicorn app.main:app`. Reads the database path from MOFLIX_DB.

There is deliberately no sign-up route; accounts come only from `python -m app.cli add-user`.
"""

import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

from fastapi import Cookie, Depends, FastAPI, HTTPException, Request, Response, status
from pydantic import BaseModel

from app.auth import (
    SESSION_LIFETIME,
    Role,
    User,
    authenticate,
    create_session,
    end_session,
    login_rate_limited,
    session_user,
    utc_now,
)
from app.db import connect

SESSION_COOKIE = "session"


class Credentials(BaseModel):
    username: str
    password: str


class LoggedIn(BaseModel):
    username: str
    role: Role


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

    @app.post("/auth/login")
    def login(credentials: Credentials, conn: Db, response: Response) -> LoggedIn:
        now = utc_now()
        if login_rate_limited(conn, credentials.username, now):
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "too many failed logins")
        user = authenticate(conn, credentials.username, credentials.password, now)
        if user is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid username or password")
        session = create_session(conn, user, now)
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

    return app


app = create_app(os.environ.get("MOFLIX_DB", "moflix.db"))
