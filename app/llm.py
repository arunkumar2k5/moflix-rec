"""LLM provider via OpenRouter — intent parsing for chat tuning.

OpenRouter exposes an OpenAI-compatible API. We use it to call Claude 3 Haiku,
which is the cheapest model with strong instruction-following for JSON extraction.

Environment variables:
    OPENROUTER_API_KEY   required — your OpenRouter API key
    OPENROUTER_MODEL     optional — default: anthropic/claude-3-haiku
    MOFLIX_MONTHLY_CAP   optional — USD spend cap, default 5.0

Cost tracking is DB-backed (llm_usage table). The monthly cap is checked before
every call; if the current month's spend would exceed it the call is refused and
the caller falls back to simple_parse. Costs are estimated from token counts
returned by the API — they are a close approximation, not exact billing figures.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

from app.intent import Intent
from app.simple_parse import simple_parse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-3-haiku"
# Approximate cost per token in USD (Claude 3 Haiku on OpenRouter)
COST_PER_INPUT_TOKEN = 0.25 / 1_000_000
COST_PER_OUTPUT_TOKEN = 1.25 / 1_000_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_usage (
    id INTEGER PRIMARY KEY,
    called_at TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    user_id INTEGER REFERENCES users(id)
);
CREATE INDEX IF NOT EXISTS llm_usage_month ON llm_usage(called_at);
"""

SYSTEM_PROMPT = """\
You are a movie recommendation assistant. Parse the user's message and return ONLY a \
JSON object matching the Intent schema. Do not explain. Do not add fields not in the schema.

Intent schema (all fields optional, omit rather than null):
{
  "genre_adjust": {"<GenreName>": <float -1.0 to 1.0>},
  "era": {"start": <year int>, "end": <year int>},
  "runtime": {"min": <minutes int>, "max": <minutes int>},
  "language": ["<iso-639-1 code>"],
  "mood_pace": "<slow|medium|fast>",
  "similar_to": ["<movie title>"],
  "unlike": ["<movie title>"],
  "hard_filters": {
    "exclude": {"genres": ["<genre>"], "terms": ["<keyword>"]},
    "require": {"genres": ["<genre>"], "terms": ["<keyword>"]}
  },
  "reaction": {"movie_id": <int>, "sentiment": "<liked|disliked|already_seen>"}
}

Valid genres: Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Family, \
Fantasy, History, Horror, Music, Mystery, Romance, Science Fiction, Thriller, War, Western.
"""


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def monthly_spend(conn: sqlite3.Connection) -> float:
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    row = conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_usage WHERE called_at LIKE ?",
        (f"{month}%",),
    ).fetchone()
    return float(row[0])


def record_usage(
    conn: sqlite3.Connection,
    model: str,
    input_tokens: int,
    output_tokens: int,
    user_id: int | None,
) -> float:
    cost = input_tokens * COST_PER_INPUT_TOKEN + output_tokens * COST_PER_OUTPUT_TOKEN
    now = datetime.now(timezone.utc).isoformat()
    with conn:
        conn.execute(
            "INSERT INTO llm_usage"
            " (called_at, model, input_tokens, output_tokens, cost_usd, user_id)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (now, model, input_tokens, output_tokens, cost, user_id),
        )
    return cost


# ---------------------------------------------------------------------------
# Intent parsing
# ---------------------------------------------------------------------------


def parse_intent(
    message: str,
    recent_turns: list[dict[str, str]],
    current_picks: list[dict[str, Any]],
    conn: sqlite3.Connection,
    user_id: int | None = None,
) -> tuple[Intent, str, bool]:
    """Parse a chat message into an Intent using the LLM, with fallback to simple_parse.

    Returns (intent, reply_text, used_llm).
    reply_text is the assistant's natural-language reply to show the user.
    used_llm is False when the fallback parser was used.
    """
    ensure_schema(conn)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    model = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    cap = float(os.environ.get("MOFLIX_MONTHLY_CAP", "5.0"))

    if not api_key:
        intent = simple_parse(message)
        return intent, _fallback_reply(intent), False

    # Check monthly cap before calling
    if monthly_spend(conn) >= cap:
        intent = simple_parse(message)
        return intent, _fallback_reply(intent, over_budget=True), False

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE,
            default_headers={
                "HTTP-Referer": "https://github.com/arunkumar2k5/moflix-rec",
                "X-Title": "Moflix",
            },
        )

        picks_context = ", ".join(p.get("title", "") for p in current_picks[:5])
        user_context = f"Current picks shown: {picks_context}." if picks_context else ""

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in recent_turns[-4:]:  # last 4 turns for context
            messages.append(turn)
        messages.append(
            {
                "role": "user",
                "content": f"{user_context}\n\nUser message: {message}",
            }
        )

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=512,
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        usage = response.usage
        record_usage(
            conn,
            model,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
            user_id,
        )

        raw = response.choices[0].message.content or "{}"
        intent = Intent.model_validate(json.loads(raw))
        reply = _build_reply(intent, message)
        return intent, reply, True

    except Exception:
        intent = simple_parse(message)
        return intent, _fallback_reply(intent), False


def _build_reply(intent: Intent, original: str) -> str:
    """Build a short natural-language confirmation of what was understood."""
    parts = []
    if intent.genre_adjust:
        more = [g for g, v in intent.genre_adjust.items() if v > 0]
        less = [g for g, v in intent.genre_adjust.items() if v < 0]
        if more:
            parts.append(f"more {', '.join(more)}")
        if less:
            parts.append(f"less {', '.join(less)}")
    if intent.era:
        parts.append(f"{intent.era.start}s–{intent.era.end}s")
    if intent.runtime and intent.runtime.max:
        parts.append(f"under {intent.runtime.max} min")
    if intent.similar_to:
        parts.append(f"similar to {', '.join(intent.similar_to)}")
    if intent.hard_filters and intent.hard_filters.exclude.terms:
        parts.append(f"no {', '.join(intent.hard_filters.exclude.terms)}")
    if not parts:
        return "Got it — refreshing your picks."
    return f"Got it: {', '.join(parts)}. Refreshing your picks."


def _fallback_reply(intent: Intent, over_budget: bool = False) -> str:
    prefix = (
        "AI is unavailable, using simple mode. "
        if not over_budget
        else "Monthly AI budget reached, using simple mode. "
    )
    parts = []
    if intent.genre_adjust:
        parts.append("genre adjusted")
    if intent.era:
        parts.append(f"{intent.era.start}s–{intent.era.end}s")
    if intent.runtime and intent.runtime.max:
        parts.append(f"under {intent.runtime.max} min")
    if not parts:
        return prefix + "Refreshing your picks."
    return prefix + f"I understood: {', '.join(parts)}."
