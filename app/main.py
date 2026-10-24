"""Web service: `uvicorn app.main:app --log-config app/log_config.json`.

Reads the database path from MOLLIB_DB and the number of picks GET /recommendations
returns from MOFLIX_TOP_N (default 3).

There is deliberately no sign-up route; accounts come only from `python -m app.cli add-user`.
"""

import os
import sqlite3
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

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
from app.db import connect, log_recommendations
from app.logs import RequestContext, logger
from app.recommend import (
    LibraryCache,
    NoWatchHistory,
    RankParams,
    RecommendError,
    recommend_for_user,
    watch_events,
)

SESSION_COOKIE = "session"


class Credentials(BaseModel):
    username: str
    password: str


class LoggedIn(BaseModel):
    username: str
    role: Role


class UserInfo(BaseModel):
    id: int
    name: str


class RecommendedMovie(BaseModel):
    movie_id: int
    title: str
    year: int
    genres: list[str]
    runtime: int
    poster_url: str
    reason: str


class Recommendations(BaseModel):
    items: list[RecommendedMovie]
    # True when the user has no watch history yet, so there is nothing to personalise from.
    cold_start: bool


class ChangePassword(BaseModel):
    current_password: str
    new_password: str


class Health(BaseModel):
    status: str
    db: str


def get_db(request: Request) -> Iterator[sqlite3.Connection]:
    # FastAPI may run the dependency and the endpoint on different threadpool threads.
    conn = connect(request.app.state.db_path, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()


Db = Annotated[sqlite3.Connection, Depends(get_db)]
SessionToken = Annotated[str | None, Cookie(alias=SESSION_COOKIE)]


def current_user(request: Request, conn: Db, token: SessionToken = None) -> User:
    user = session_user(conn, token, utc_now()) if token else None
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "not logged in")
    request.state.user_id = user.id
    return user


def create_app(db_path: str | Path, top_n: int = 3) -> FastAPI:
    rank_params = RankParams(top_n=top_n)
    library_cache = LibraryCache()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Loading a large library takes seconds; pay that at startup, not on a user's request.
        conn = connect(db_path)
        try:
            library_cache.get(conn)
        except Exception:
            # Every other route works without the library; GET /recommendations retries the load.
            logger.exception("could not load the movie library at startup")
        finally:
            conn.close()
        yield

    app = FastAPI(title="moflix-rec", lifespan=lifespan)
    app.state.db_path = db_path

    @app.middleware("http")
    async def log_request(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        logger.info(
            "request",
            extra=RequestContext(
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
                user_id=getattr(request.state, "user_id", None),
            ),
        )
        return response

    @app.get("/health")
    def health(response: Response) -> Health:
        try:
            connect(app.state.db_path).close()
        except sqlite3.Error:
            logger.exception("health_db_unavailable")
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return Health(status="error", db="error")
        return Health(status="ok", db="ok")

    @app.post("/auth/login")
    def login(credentials: Credentials, conn: Db, request: Request, response: Response) -> LoggedIn:
        now = utc_now()
        try:
            user = authenticate(conn, credentials.username, credentials.password, now)
        except LoginRateLimited:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS, "too many failed logins"
            ) from None
        if user is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid username or password")
        request.state.user_id = user.id
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

    @app.get("/recommendations")
    def recommendations(conn: Db, user: Annotated[User, Depends(current_user)]) -> Recommendations:
        events = watch_events(conn, user.id)
        try:
            library = library_cache.get(conn)
            picks = recommend_for_user(conn, user.id, events, library, rank_params)
        except NoWatchHistory:
            return Recommendations(items=[], cold_start=True)
        except RecommendError as error:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from None
        with conn:
            log_recommendations(conn, user.id, [pick.movie_id for pick in picks], utc_now())
        items = []
        for pick in picks:
            movie = library.by_id[pick.movie_id].movie
            items.append(
                RecommendedMovie(
                    movie_id=movie.id,
                    title=movie.title,
                    year=movie.year,
                    genres=list(movie.genres),
                    runtime=movie.runtime,
                    poster_url=movie.poster_url,
                    reason=pick.reason,
                )
            )
        return Recommendations(items=items, cold_start=False)

    @app.get("/api/users")
    def list_users(conn: Db) -> list[UserInfo]:
        rows = conn.execute("SELECT id, name FROM users ORDER BY name").fetchall()
        return [UserInfo(id=row[0], name=row[1]) for row in rows]

    return app


app = create_app(os.environ.get("MOFLIX_DB", "moflix.db"), int(os.environ.get("MOFLIX_TOP_N", "3")))
