from types import SimpleNamespace

import pytest

from app.chat import DEFAULT_DAILY_CAP, DEFAULT_TIMEOUT_S, ChatConfig
from app.intent import Intent
from app.llm import (
    DEFAULT_MONTHLY_CAP_USD,
    SYSTEM_PROMPT,
    IntentRequest,
    OpenRouterProvider,
    Pick,
    Turn,
    provider_from_env,
)


def test_openrouter_sends_only_the_request_and_returns_the_reply_text():
    provider = OpenRouterProvider("key", "anthropic/claude-3.5-haiku", timeout_s=3.0)
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"mood_pace": "darker"}'))],
            usage=SimpleNamespace(prompt_tokens=300, completion_tokens=12),
        )

    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    request = IntentRequest(
        message="darker please",
        turns=[Turn(role="user", content="hi"), Turn(role="assistant", content="hello")],
        picks=[Pick(title="Alien", year=1979, genres=["Horror", "Science Fiction"])],
    )

    completion = provider.complete(request)

    assert completion.text == '{"mood_pace": "darker"}'
    assert (completion.model, completion.input_tokens, completion.output_tokens) == (
        "anthropic/claude-3.5-haiku",
        300,
        12,
    )
    assert completion.cost_usd == pytest.approx((300 * 0.80 + 12 * 4.00) / 1_000_000)
    assert sent["model"] == "anthropic/claude-3.5-haiku"
    assert sent["messages"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {
            "role": "user",
            "content": "Current picks: Alien (1979, Horror, Science Fiction)\n\n"
            "Message: darker please",
        },
    ]


def test_the_system_prompt_carries_the_intent_schema():
    for field in Intent.model_fields:
        assert f'"{field}"' in SYSTEM_PROMPT
    assert '"from"' in SYSTEM_PROMPT


def test_a_model_without_a_known_price_is_refused(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-opus-4")

    with pytest.raises(ValueError, match="claude-opus-4"):
        provider_from_env(timeout_s=3.0)


def test_chat_config_from_env_builds_the_configured_provider_and_caps(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-3.5-haiku")
    monkeypatch.setenv("MOFLIX_CHAT_DAILY_CAP", "7")
    monkeypatch.setenv("MOFLIX_MONTHLY_CAP", "2.5")

    config = ChatConfig.from_env()

    assert isinstance(config.provider, OpenRouterProvider)
    assert config.provider.model == "anthropic/claude-3.5-haiku"
    assert config.provider._client.api_key == "key"
    assert config.provider._client.timeout == DEFAULT_TIMEOUT_S
    assert (config.daily_cap, config.monthly_cap_usd) == (7, 2.5)


def test_chat_config_from_env_without_a_key_runs_in_simple_mode(monkeypatch):
    for name in (
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
        "MOFLIX_CHAT_DAILY_CAP",
        "MOFLIX_MONTHLY_CAP",
    ):
        monkeypatch.delenv(name, raising=False)

    config = ChatConfig.from_env()

    assert config.provider is None
    assert (config.daily_cap, config.monthly_cap_usd) == (
        DEFAULT_DAILY_CAP,
        DEFAULT_MONTHLY_CAP_USD,
    )
