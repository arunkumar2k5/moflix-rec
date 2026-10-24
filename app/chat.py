"""Chat tuning: each message moves the user's session vector and hard filters (PRD FR-19..24).

The LLM only parses a message into an `Intent`; if it fails, is too slow or returns anything
the schema rejects, `simple_parse` does instead. The session lives until `POST /chat/reset`
ends it and never touches the base taste vector.
"""

import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Literal

import numpy as np
from pydantic import TypeAdapter, ValidationError

from app.db import VECTOR_DTYPE, decode_vector
from app.intent import Intent, apply_intent, blend
from app.llm import (
    DEFAULT_MONTHLY_CAP_USD,
    IntentRequest,
    LLMProvider,
    Pick,
    Turn,
    monthly_cap_from_env,
    monthly_spend,
    provider_from_env,
    record_usage,
)
from app.logs import logger
from app.recommend import NO_CONSTRAINTS, Constraints, Library
from app.simple_parse import simple_parse
from app.vectors import FULL_DIMENSION

Mode = Literal["ai", "simple"]

SIMPLE_MODE_NOTICE = "AI is unavailable, using simple mode"
DEFAULT_DAILY_CAP = 50
DEFAULT_TIMEOUT_S = 3.0
CONTEXT_TURNS = 4

_constraints_json = TypeAdapter(Constraints)
# Lets a turn give up on a provider that overruns its deadline instead of waiting it out.
_provider_calls = ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm")


@dataclass(frozen=True)
class ChatConfig:
    provider: LLMProvider | None = None  # None runs every message in simple mode
    daily_cap: int = DEFAULT_DAILY_CAP  # user messages per user per UTC day
    monthly_cap_usd: float = DEFAULT_MONTHLY_CAP_USD
    timeout_s: float = DEFAULT_TIMEOUT_S

    @classmethod
    def from_env(cls) -> "ChatConfig":
        return cls(
            provider=provider_from_env(DEFAULT_TIMEOUT_S),
            daily_cap=int(os.environ.get("MOFLIX_CHAT_DAILY_CAP", str(DEFAULT_DAILY_CAP))),
            monthly_cap_usd=monthly_cap_from_env(),
        )


@dataclass(frozen=True, eq=False)
class Session:
    id: int | None = None  # None until its first message is saved
    vector: np.ndarray = field(default_factory=lambda: np.zeros(FULL_DIMENSION))
    constraints: Constraints = NO_CONSTRAINTS

    def tune(self, base: np.ndarray) -> np.ndarray:
        return blend(base, self.vector)


def active_session(conn: sqlite3.Connection, user_id: int) -> Session | None:
    row = conn.execute(
        "SELECT id, session_vector, constraints FROM chat_sessions"
        " WHERE user_id = ? AND ended_at IS NULL",
        (user_id,),
    ).fetchone()
    if row is None:
        return None
    return Session(row[0], decode_vector(row[1]), _constraints_json.validate_json(row[2]))


def end_session(conn: sqlite3.Connection, user_id: int, now: datetime) -> None:
    conn.execute(
        "UPDATE chat_sessions SET ended_at = ? WHERE user_id = ? AND ended_at IS NULL",
        (now.isoformat(), user_id),
    )


def messages_today(conn: sqlite3.Connection, user_id: int, now: datetime) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM chat_messages m JOIN chat_sessions s ON s.id = m.session_id"
        " WHERE s.user_id = ? AND m.role = 'user' AND m.sent_at >= ?",
        (user_id, f"{now:%Y-%m-%d}"),
    ).fetchone()
    return row[0]


def recent_turns(conn: sqlite3.Connection, session: Session) -> list[Turn]:
    rows = conn.execute(
        "SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session.id, CONTEXT_TURNS),
    ).fetchall()
    return [Turn(role=role, content=content) for role, content in reversed(rows)]


def shown_picks(conn: sqlite3.Connection, user_id: int, library: Library) -> list[Pick]:
    """The picks most recently shown to the user, in the order they were shown."""
    rows = conn.execute(
        "SELECT movie_id FROM recommendation_log WHERE user_id = ? AND shown_at ="
        " (SELECT MAX(shown_at) FROM recommendation_log WHERE user_id = ?) ORDER BY id",
        (user_id, user_id),
    ).fetchall()
    movies = [library.by_id[row[0]].movie for row in rows if row[0] in library.by_id]
    return [Pick(title=m.title, year=m.year, genres=list(m.genres)) for m in movies]


def save_turn(
    conn: sqlite3.Connection,
    user_id: int,
    session: Session,
    message: str,
    reply: str,
    now: datetime,
) -> None:
    vector = session.vector.astype(VECTOR_DTYPE).tobytes()
    constraints = _constraints_json.dump_json(session.constraints).decode()
    if session.id is None:
        session_id = conn.execute(
            "INSERT INTO chat_sessions (user_id, started_at, session_vector, constraints)"
            " VALUES (?, ?, ?, ?)",
            (user_id, now.isoformat(), vector, constraints),
        ).lastrowid
    else:
        session_id = session.id
        conn.execute(
            "UPDATE chat_sessions SET session_vector = ?, constraints = ? WHERE id = ?",
            (vector, constraints, session_id),
        )
    conn.executemany(
        "INSERT INTO chat_messages (session_id, role, content, sent_at) VALUES (?, ?, ?, ?)",
        [
            (session_id, role, text, now.isoformat())
            for role, text in (("user", message), ("assistant", reply))
        ],
    )


def parse_intent(
    config: ChatConfig,
    conn: sqlite3.Connection,
    request: IntentRequest,
    user_id: int,
    now: datetime,
) -> tuple[Intent, Mode]:
    if config.provider is None:
        return simple_parse(request.message), "simple"
    if monthly_spend(conn, now) >= config.monthly_cap_usd:
        logger.warning("llm_monthly_cap_reached")
        return simple_parse(request.message), "simple"
    try:
        call = _provider_calls.submit(config.provider.complete, request)
        completion = call.result(timeout=config.timeout_s)
    except Exception:
        logger.warning("llm_unavailable", exc_info=True)
        return simple_parse(request.message), "simple"
    with conn:
        record_usage(conn, completion, user_id, now)
    try:
        return Intent.model_validate_json(completion.text), "ai"
    except ValidationError:
        logger.warning("llm_invalid_intent", exc_info=True)
        return simple_parse(request.message), "simple"


def tuned(session: Session, intent: Intent, library: Library) -> Session:
    titles = {c.movie.title: c.vector for c in library.candidates}
    vector, constraints = apply_intent(session.vector, intent, titles)
    return replace(session, vector=vector, constraints=merge(session.constraints, constraints))


def merge(earlier: Constraints, later: Constraints) -> Constraints:
    """`earlier` narrowed by `later`, whose bounds replace earlier ones and whose required or
    excluded genres and terms override the opposite earlier choice."""

    def newer(old, new):
        return old if new is None else new

    return Constraints(
        min_runtime=newer(earlier.min_runtime, later.min_runtime),
        max_runtime=newer(earlier.max_runtime, later.max_runtime),
        languages=newer(earlier.languages, later.languages),
        min_rating=newer(earlier.min_rating, later.min_rating),
        year_range=newer(earlier.year_range, later.year_range),
        required_genres=(earlier.required_genres - later.excluded_genres) | later.required_genres,
        excluded_genres=(earlier.excluded_genres - later.required_genres) | later.excluded_genres,
        required_terms=(earlier.required_terms - later.excluded_terms) | later.required_terms,
        excluded_terms=(earlier.excluded_terms - later.required_terms) | later.excluded_terms,
    )


def describe(intent: Intent, mode: Mode) -> str:
    """The reply shown for a turn: plain sentences, never the intent JSON itself."""
    parts = []
    more = sorted(genre for genre, weight in intent.genre_adjust.items() if weight > 0)
    less = sorted(genre for genre, weight in intent.genre_adjust.items() if weight < 0)
    if more:
        parts.append(f"more {', '.join(more)}")
    if less:
        parts.append(f"less {', '.join(less)}")
    if intent.mood_pace:
        parts.append(intent.mood_pace)
    if intent.era:
        parts.append(f"from {intent.era.start} to {intent.era.end}")
    if intent.runtime and intent.runtime.min is not None:
        parts.append(f"at least {intent.runtime.min} minutes")
    if intent.runtime and intent.runtime.max is not None:
        parts.append(f"at most {intent.runtime.max} minutes")
    if intent.language:
        parts.append(f"in {', '.join(sorted(intent.language))}")
    if intent.similar_to:
        parts.append(f"like {', '.join(sorted(intent.similar_to))}")
    if intent.unlike:
        parts.append(f"unlike {', '.join(sorted(intent.unlike))}")
    required = intent.hard_filters.require
    excluded = intent.hard_filters.exclude
    if required.genres or required.terms:
        parts.append(f"only {', '.join(sorted(required.genres) + sorted(required.terms))}")
    if excluded.genres or excluded.terms:
        parts.append(f"no {', '.join(sorted(excluded.genres) + sorted(excluded.terms))}")
    if intent.reaction:
        verb = "liked" if intent.reaction.liked else "did not like"
        parts.append(f"noting you {verb} {intent.reaction.title}")
    if parts:
        reply = f"Got it: {'; '.join(parts)}. Here are your updated picks."
    else:
        reply = "I couldn't find anything to change in that, so your picks are as they were."
    return reply if mode == "ai" else f"{SIMPLE_MODE_NOTICE}. {reply}"
