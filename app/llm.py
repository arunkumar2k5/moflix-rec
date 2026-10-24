"""The LLM provider abstraction that turns a chat message into intent JSON (PRD FR-19, FR-22, §7).

A provider only returns text. Validating it as an `Intent`, enforcing the deadline, falling
back to `simple_parse` and recording usage all happen in `app.chat`, whatever the provider.

The only provider is OpenRouter, an OpenAI-compatible gateway in front of Anthropic, OpenAI,
Google, Meta and others, so `OPENROUTER_MODEL` picks both the upstream provider and the model.

Environment variables:
    OPENROUTER_API_KEY   the API key; without it chat runs in simple mode only
    OPENROUTER_MODEL     optional, default anthropic/claude-3-haiku
    MOFLIX_MONTHLY_CAP   optional USD spend cap per calendar month, default 5.0

Costs are estimated from the token counts the API returns at Claude 3 Haiku prices: a close
approximation for the default model, not billing figures.
"""

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel

from app.intent import Intent

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-3-haiku"
DEFAULT_MONTHLY_CAP_USD = 5.0
COST_PER_INPUT_TOKEN = 0.25 / 1_000_000
COST_PER_OUTPUT_TOKEN = 1.25 / 1_000_000

SYSTEM_PROMPT = f"""\
You turn a movie-night chat message into a JSON object that matches this JSON Schema. \
Reply with that object only: no prose, no code fences. Omit a field rather than setting it \
to null, and include only what the user asked for in their latest message.

{json.dumps(Intent.model_json_schema(by_alias=True))}
"""


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Pick(BaseModel):
    title: str
    year: int
    genres: list[str]


class IntentRequest(BaseModel):
    """Everything a provider is given: never vectors, watch history or other users' data."""

    message: str
    turns: list[Turn]  # the most recent turns of this user's own chat session, oldest first
    picks: list[Pick]  # what the user is looking at now


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int


class LLMProvider(Protocol):
    def complete(self, request: IntentRequest) -> Completion:
        """The intent JSON for the request, as text; raises on any provider failure."""
        ...


class OpenRouterProvider:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, *, timeout_s: float) -> None:
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE,
            timeout=timeout_s,
            max_retries=0,
            default_headers={
                "HTTP-Referer": "https://github.com/arunkumar2k5/moflix-rec",
                "X-Title": "Moflix",
            },
        )

    def complete(self, request: IntentRequest) -> Completion:
        picks = "; ".join(f"{p.title} ({p.year}, {', '.join(p.genres)})" for p in request.picks)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *({"role": turn.role, "content": turn.content} for turn in request.turns),
                {
                    "role": "user",
                    "content": f"Current picks: {picks or 'none'}\n\nMessage: {request.message}",
                },
            ],
            max_tokens=512,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        usage = response.usage
        return Completion(
            text=response.choices[0].message.content or "",
            model=self.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )


def provider_from_env(timeout_s: float) -> LLMProvider | None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return None
    return OpenRouterProvider(
        api_key, os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL), timeout_s=timeout_s
    )


def monthly_cap_from_env() -> float:
    return float(os.environ.get("MOFLIX_MONTHLY_CAP", str(DEFAULT_MONTHLY_CAP_USD)))


def monthly_spend(conn: sqlite3.Connection, now: datetime) -> float:
    row = conn.execute(
        "SELECT TOTAL(cost_usd) FROM llm_usage WHERE called_at LIKE ?",
        (f"{now:%Y-%m}%",),
    ).fetchone()
    return float(row[0])


def record_usage(
    conn: sqlite3.Connection, completion: Completion, user_id: int, now: datetime
) -> None:
    cost = (
        completion.input_tokens * COST_PER_INPUT_TOKEN
        + completion.output_tokens * COST_PER_OUTPUT_TOKEN
    )
    conn.execute(
        "INSERT INTO llm_usage"
        " (called_at, model, input_tokens, output_tokens, cost_usd, user_id)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (
            now.isoformat(),
            completion.model,
            completion.input_tokens,
            completion.output_tokens,
            cost,
            user_id,
        ),
    )
