"""Web service: `uvicorn app.main:app --log-config app/log_config.json`.

Reads the database path from MOFLIX_DB, the number of picks GET /recommendations and POST /chat
return from MOFLIX_TOP_N (default 3), and each user's daily chat message cap from
MOFLIX_CHAT_DAILY_CAP (default 50). `app.llm` documents the LLM settings.

There is deliberately no sign-up route; accounts come only from `python -m app.cli add-user`.
"""

import os
import sqlite3
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Cookie, Depends, FastAPI, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from app import chat
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
from app.chat import ChatConfig, Mode, Session
from app.db import connect, log_recommendations
from app.intent import Intent
from app.llm import IntentRequest
from app.logs import RequestContext, logger
from app.recommend import (
    NO_CONSTRAINTS,
    Library,
    LibraryCache,
    NoWatchHistory,
    RankParams,
    RecommendError,
    recommend_for_user,
    watch_events,
)
from app.taste import WatchEvent

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


class ChatMessage(BaseModel):
    message: str = Field(min_length=1, max_length=1000)


class ChatReply(BaseModel):
    reply: str
    understood: Intent
    items: list[RecommendedMovie]
    mode: Mode


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


def create_app(
    db_path: str | Path, top_n: int = 3, chat_config: ChatConfig | None = None
) -> FastAPI:
    chat_config = chat_config or ChatConfig()
    rank_params = RankParams(top_n=top_n)
    library_cache = LibraryCache()

    def current_library(conn: sqlite3.Connection) -> Library:
        try:
            return library_cache.get(conn)
        except RecommendError as error:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from None

    def picks(
        conn: sqlite3.Connection,
        user: User,
        events: Sequence[WatchEvent],
        library: Library,
        session: Session | None,
    ) -> list[RecommendedMovie]:
        """The user's picks, tuned by their chat session if any, logged as shown.

        Raises NoWatchHistory. As for `recommend_for_user`, read `events` before `library`.
        """
        recommended = recommend_for_user(
            conn,
            user.id,
            events,
            library,
            rank_params,
            constraints=NO_CONSTRAINTS if session is None else session.constraints,
            tune=None if session is None else session.tune,
        )
        with conn:
            log_recommendations(conn, user.id, [pick.movie_id for pick in recommended], utc_now())
        items = []
        for pick in recommended:
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
        return items

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
        session = chat.active_session(conn, user.id)
        try:
            items = picks(conn, user, events, current_library(conn), session)
        except NoWatchHistory:
            return Recommendations(items=[], cold_start=True)
        return Recommendations(items=items, cold_start=False)

    @app.post("/chat")
    def post_chat(
        body: ChatMessage, conn: Db, user: Annotated[User, Depends(current_user)]
    ) -> ChatReply:
        now = utc_now()
        if chat.messages_today(conn, user.id, now) >= chat_config.daily_cap:
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "daily chat limit reached")
        events = watch_events(conn, user.id)
        loaded = current_library(conn)
        session = chat.active_session(conn, user.id) or Session()
        request = IntentRequest(
            message=body.message,
            turns=chat.recent_turns(conn, session),
            picks=chat.shown_picks(conn, user.id, loaded),
        )
        intent, mode = chat.parse_intent(chat_config, conn, request, user.id, now)
        session = chat.tuned(session, intent, loaded)
        try:
            items = picks(conn, user, events, loaded, session)
        except NoWatchHistory:
            items = []
        reply = chat.describe(intent, mode)
        with conn:
            chat.save_turn(conn, user.id, session, body.message, reply, now)
        return ChatReply(reply=reply, understood=intent, items=items, mode=mode)

    @app.post("/chat/reset", status_code=status.HTTP_204_NO_CONTENT)
    def reset_chat(conn: Db, user: Annotated[User, Depends(current_user)]) -> None:
        with conn:
            chat.end_session(conn, user.id, utc_now())

    @app.get("/api/users")
    def list_users(conn: Db) -> list[UserInfo]:
        rows = conn.execute("SELECT id, name FROM users ORDER BY name").fetchall()
        return [UserInfo(id=row[0], name=row[1]) for row in rows]

    return app


app = create_app(
    os.environ.get("MOFLIX_DB", "moflix.db"),
    int(os.environ.get("MOFLIX_TOP_N", "3")),
    ChatConfig.from_env(),
)
