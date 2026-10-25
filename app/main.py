"""Web service: `uvicorn app.main:app --log-config app/log_config.json`.

Reads the database path from MOFLIX_DB, the number of picks GET /recommendations and POST /chat
return from MOFLIX_TOP_N (default 3), from MOFLIX_COLD_START_MIN (default 5) how many watched
movies a user needs before picks stop leaning on popularity, and each user's daily chat message
cap from MOFLIX_CHAT_DAILY_CAP (default 50). `app.llm` documents the LLM settings.

There is deliberately no sign-up route; accounts come only from `python -m app.cli add-user`.

GET / serves the phone page. A state-changing request a browser sends (one carrying `Origin` or
`Sec-Fetch-Site`, which browsers attach to every cross-site POST) must echo the `__Host-csrf`
cookie that page sets in an `X-CSRF-Token` header, or it gets 403. Requests without those
headers come from API clients that hold no ambient browser cookies, so they are not checked.
"""

import hmac
import os
import secrets
import sqlite3
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Cookie, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
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
from app.db import (
    FeedbackAction,
    connect,
    log_recommendations,
    record_feedback,
    watch_tonight_movies,
)
from app.intent import Intent
from app.llm import IntentRequest
from app.logs import RequestContext, logger
from app.recommend import (
    DEFAULT_COLD_START_MIN,
    NO_CONSTRAINTS,
    Feedback,
    Library,
    LibraryCache,
    RankParams,
    RecommendError,
    recommend_cold_start,
    recommend_for_user,
    user_feedback,
    watch_events,
)
from app.taste import WatchEvent
from app.vectors import Movie

SESSION_COOKIE = "session"
CSRF_COOKIE = "csrf"
CSRF_HEADER = "X-CSRF-Token"
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
TEMPLATES = Jinja2Templates(directory=Path(__file__).parent / "templates")
ONBOARDING_PROMPT = (
    "New here? Tell us a few genres you like, such as Comedy, Science Fiction or Thriller,"
    " and we'll tailor these picks to you."
)


class Credentials(BaseModel):
    username: str
    password: str


class LoggedIn(BaseModel):
    username: str
    role: Role


class UserInfo(BaseModel):
    id: int
    name: str


class MovieCard(BaseModel):
    movie_id: int
    title: str
    year: int
    genres: list[str]
    runtime: int
    poster_url: str
    rating: float
    overview: str
    director: str
    cast: list[str]
    certification: str

    @classmethod
    def of(cls, movie: Movie) -> "MovieCard":
        return cls(
            movie_id=movie.id,
            title=movie.title,
            year=movie.year,
            genres=list(movie.genres),
            runtime=movie.runtime,
            poster_url=movie.poster_url,
            rating=movie.rating,
            overview=movie.overview,
            director=movie.director,
            cast=list(movie.cast),
            certification=movie.certification,
        )


class RecommendedMovie(MovieCard):
    reason: str


class Recommendations(BaseModel):
    items: list[RecommendedMovie]
    # True when the user has watched fewer than `cold_start_min` movies, so picks lean on
    # popularity and `onboarding_prompt` invites them to name genres.
    cold_start: bool
    onboarding_prompt: str | None = None


class FeedbackRequest(BaseModel):
    movie_id: int
    action: FeedbackAction


class ChatMessage(BaseModel):
    message: str = Field(min_length=1, max_length=1000)


class ChatReply(Recommendations):
    reply: str
    understood: Intent
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


def optional_user(request: Request, conn: Db, token: SessionToken = None) -> User | None:
    user = session_user(conn, token, utc_now()) if token else None
    if user is not None:
        request.state.user_id = user.id
    return user


def current_user(user: Annotated[User | None, Depends(optional_user)]) -> User:
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "not logged in")
    return user


def csrf_ok(request: Request) -> bool:
    if request.method not in UNSAFE_METHODS:
        return True
    headers = request.headers
    if "origin" not in headers and "sec-fetch-site" not in headers:
        return True
    cookie = request.cookies.get(CSRF_COOKIE, "")
    header = headers.get(CSRF_HEADER, "")
    return bool(cookie) and hmac.compare_digest(cookie.encode(), header.encode())


def create_app(
    db_path: str | Path,
    top_n: int = 3,
    cold_start_min: int = DEFAULT_COLD_START_MIN,
    chat_config: ChatConfig | None = None,
) -> FastAPI:
    if cold_start_min < 1:
        raise ValueError(f"cold_start_min must be at least 1, got {cold_start_min}")
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
        feedback: Feedback,
        library: Library,
        session: Session | None,
    ) -> Recommendations:
        """The user's picks, tuned by their chat session if any, logged as shown.

        As for `recommend_for_user`, read `events` and `feedback` before taking `library`.
        """
        cold_start = len({e.movie_id for e in events}) < cold_start_min
        constraints = NO_CONSTRAINTS if session is None else session.constraints
        tune = None if session is None else session.tune
        try:
            if cold_start:
                recommended = recommend_cold_start(
                    conn,
                    user.id,
                    events,
                    library,
                    rank_params,
                    cold_start_min,
                    constraints,
                    feedback=feedback,
                    tune=tune,
                )
            else:
                recommended = recommend_for_user(
                    conn,
                    user.id,
                    events,
                    library,
                    rank_params,
                    constraints,
                    feedback=feedback,
                    tune=tune,
                )
        except RecommendError as error:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from None
        with conn:
            log_recommendations(conn, user.id, [pick.movie_id for pick in recommended], utc_now())
        return Recommendations(
            items=[
                RecommendedMovie(
                    **MovieCard.of(library.by_id[pick.movie_id].movie).model_dump(),
                    reason=pick.reason,
                )
                for pick in recommended
            ],
            cold_start=cold_start,
            onboarding_prompt=ONBOARDING_PROMPT if cold_start else None,
        )

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

    # Added before log_request so that log_request stays outermost and logs the 403 too.
    @app.middleware("http")
    async def check_csrf(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if not csrf_ok(request):
            return JSONResponse(
                {"detail": "missing or invalid CSRF token"}, status.HTTP_403_FORBIDDEN
            )
        return await call_next(request)

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

    @app.get("/", include_in_schema=False)
    def home(
        request: Request, conn: Db, user: Annotated[User | None, Depends(optional_user)]
    ) -> Response:
        csrf = request.cookies.get(CSRF_COOKIE) or secrets.token_urlsafe(32)
        if user is None:
            page = TEMPLATES.TemplateResponse(request, "login.html", {"csrf_token": csrf})
        else:
            page = TEMPLATES.TemplateResponse(
                request,
                "home.html",
                {
                    "csrf_token": csrf,
                    "username": user.name,
                    "recs": recommendations(conn, user),
                    "tonight": watch_tonight(conn, user),
                    "simple_mode_notice": chat.SIMPLE_MODE_NOTICE,
                },
            )
        page.set_cookie(CSRF_COOKIE, csrf, httponly=True, secure=False, samesite="strict")
        return page

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
            secure=False,
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
        feedback = user_feedback(conn, user.id)
        session = chat.active_session(conn, user.id)
        return picks(conn, user, events, feedback, current_library(conn), session)

    @app.post("/chat")
    def post_chat(
        body: ChatMessage, conn: Db, user: Annotated[User, Depends(current_user)]
    ) -> ChatReply:
        now = utc_now()
        try:
            # Checked before the provider call too, so a refused message costs nothing.
            chat.check_daily_cap(conn, user.id, now, chat_config.daily_cap)
            events = watch_events(conn, user.id)
            feedback = user_feedback(conn, user.id)
            library = current_library(conn)
            session = chat.active_session(conn, user.id) or Session()
            request = IntentRequest(
                message=body.message,
                turns=chat.recent_turns(conn, session),
                picks=chat.shown_picks(conn, user.id, library),
            )
            intent, mode = chat.parse_intent(
                chat_config, conn, request, user.id, now, db_path=app.state.db_path
            )
            reply = chat.describe(intent, mode)
            session = chat.commit_turn(
                conn,
                user.id,
                intent,
                library,
                body.message,
                reply,
                now=now,
                daily_cap=chat_config.daily_cap,
            )
        except chat.DailyCapReached:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS, "daily chat limit reached"
            ) from None
        shown = picks(conn, user, events, feedback, library, session)
        return ChatReply(**dict(shown), reply=reply, understood=intent, mode=mode)

    @app.post("/chat/reset", status_code=status.HTTP_204_NO_CONTENT)
    def reset_chat(conn: Db, user: Annotated[User, Depends(current_user)]) -> None:
        with conn:
            chat.end_session(conn, user.id, utc_now())

    @app.post("/feedback", status_code=status.HTTP_204_NO_CONTENT)
    def feedback(
        body: FeedbackRequest, conn: Db, user: Annotated[User, Depends(current_user)]
    ) -> None:
        if conn.execute("SELECT 1 FROM movies WHERE id = ?", (body.movie_id,)).fetchone() is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown movie")
        with conn:
            record_feedback(conn, user.id, body.movie_id, body.action, utc_now())

    @app.get("/feedback/watch-tonight")
    def watch_tonight(conn: Db, user: Annotated[User, Depends(current_user)]) -> list[MovieCard]:
        return [MovieCard.of(movie) for movie in watch_tonight_movies(conn, user.id)]

    @app.get("/api/users")
    def list_users(conn: Db) -> list[UserInfo]:
        rows = conn.execute("SELECT id, name FROM users ORDER BY name").fetchall()
        return [UserInfo(id=row[0], name=row[1]) for row in rows]

    return app


app = create_app(
    os.environ.get("MOFLIX_DB", "moflix.db"),
    int(os.environ.get("MOFLIX_TOP_N", "3")),
    int(os.environ.get("MOFLIX_COLD_START_MIN", DEFAULT_COLD_START_MIN)),
    ChatConfig.from_env(),
)
