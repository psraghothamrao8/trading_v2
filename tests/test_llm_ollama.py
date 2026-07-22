"""Ollama provider tests — local inference, no network, no API key."""

from __future__ import annotations

import json

import pytest

from core.config import ConfigError, get_settings
from core.llm import LLMCache, LLMClient, LLMError, LLMSetupError


class FakeResponse:
    def __init__(self, payload, status_code: int = 200, text: str = "") -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text or str(payload)

    def json(self):
        return self._payload


class FakeHTTPX:
    """Captures the request body so the contract can be asserted."""

    def __init__(self, responses=None) -> None:
        self.calls: list[dict] = []
        self.responses = list(responses or [])

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "body": json, "timeout": timeout})
        if self.responses:
            return self.responses.pop(0)
        return FakeResponse({"message": {"content": '{"label":"NOISE","confidence":0.9,'
                                                    '"reason":"routine"}'}})


@pytest.fixture
def ollama_client(tmp_path, monkeypatch):
    """An LLMClient wired to the ollama provider with a fake transport layer."""
    fake = FakeHTTPX()

    import httpx

    monkeypatch.setattr(httpx, "post", fake.post)
    client = LLMClient(cache=LLMCache(tmp_path / "c.sqlite"), sleeper=lambda s: None)
    client.provider = "ollama"
    client.model = "granite4.1:3b"
    return client, fake


class TestConfiguration:
    def test_shipped_default_is_ollama(self):
        """CPU-only box: local inference is the default, not the fallback."""
        assert get_settings().require("llm.provider") == "ollama"

    def test_shipped_model_is_small_enough_for_a_cpu_box(self):
        model = get_settings().require("llm.model")
        assert model in ("granite4.1:3b", "qwen3.5:2b", "qwen3.5:4b"), model

    def test_ollama_block_exists(self):
        settings = get_settings()
        assert settings.require("llm.ollama.base_url").startswith("http")
        assert settings.require("llm.ollama.keep_alive")
        assert int(settings.require("llm.ollama.num_ctx")) >= 4096

    def test_unknown_provider_is_rejected_at_construction(self, monkeypatch):
        import core.llm as llm_module

        settings = get_settings().as_dict()
        settings["llm"]["provider"] = "openai"
        from core.config import Settings

        monkeypatch.setattr(llm_module, "get_settings",
                            lambda *a, **k: Settings(settings, source="<test>"))
        with pytest.raises(ConfigError, match="Unknown llm.provider"):
            LLMClient()

    def test_latency_budget_is_unchanged_by_the_provider_switch(self):
        """§6.1's 20s budget is a trading rule, not a performance knob."""
        assert get_settings().require("llm.classify_latency_budget_seconds") == 20


class TestRequestContract:
    def test_calls_the_chat_endpoint(self, ollama_client):
        client, fake = ollama_client
        client.classify("filings", {"symbol": "TCS"})
        assert fake.calls[0]["url"].endswith("/api/chat")

    def test_prompt_goes_in_the_system_message(self, ollama_client):
        client, fake = ollama_client
        client.classify("filings", {"symbol": "TCS"})
        messages = fake.calls[0]["body"]["messages"]
        assert messages[0]["role"] == "system"
        assert "MATERIALITY GUIDANCE" in messages[0]["content"]
        assert json.loads(messages[1]["content"])["symbol"] == "TCS"

    def test_schema_is_sent_as_the_format_constraint(self, ollama_client):
        """Grammar-constrained decoding is what makes a 3B model usable here."""
        client, fake = ollama_client
        client.classify("filings", {"symbol": "TCS"})
        fmt = fake.calls[0]["body"]["format"]
        assert fmt["properties"]["label"]["enum"] == [
            "MATERIAL_POSITIVE", "MATERIAL_NEGATIVE", "NOISE"
        ]

    def test_keep_alive_holds_the_model_in_ram(self, ollama_client):
        """Without it, every 30s poll would reload and re-prefill the model."""
        client, fake = ollama_client
        client.classify("filings", {"symbol": "TCS"})
        assert fake.calls[0]["body"]["keep_alive"] == "30m"

    def test_streaming_is_off(self, ollama_client):
        client, fake = ollama_client
        client.classify("filings", {"symbol": "TCS"})
        assert fake.calls[0]["body"]["stream"] is False

    def test_options_carry_context_and_temperature(self, ollama_client):
        client, fake = ollama_client
        client.classify("filings", {"symbol": "TCS"})
        options = fake.calls[0]["body"]["options"]
        assert options["num_ctx"] == 8192
        assert options["temperature"] == 0.0

    def test_the_system_prompt_is_byte_identical_across_calls(self, ollama_client):
        """Identical prefixes are what let llama.cpp reuse its KV cache."""
        client, fake = ollama_client
        client.classify("filings", {"symbol": "TCS", "headline": "a"})
        client.classify("filings", {"symbol": "INFY", "headline": "b"})
        first, second = (c["body"]["messages"][0]["content"] for c in fake.calls)
        assert first == second


class TestResponseHandling:
    def test_valid_json_is_returned(self, ollama_client):
        client, _ = ollama_client
        result = client.classify("filings", {"symbol": "TCS"})
        assert result.data["label"] == "NOISE"
        assert result.from_cache is False

    def test_results_are_cached_by_provider_and_model(self, tmp_path, monkeypatch):
        """A 3B verdict and a Sonnet verdict are different answers."""
        cache = LLMCache(tmp_path / "c.sqlite")
        payload = {"symbol": "TCS", "headline": "Board meeting"}
        local = LLMCache.key("filings", "ollama:granite4.1:3b", payload)
        hosted = LLMCache.key("filings", "anthropic:claude-sonnet-5", payload)
        assert local != hosted

    def test_second_identical_call_hits_the_cache(self, ollama_client):
        client, fake = ollama_client
        payload = {"symbol": "TCS", "headline": "Board meeting"}
        client.classify("filings", payload)
        second = client.classify("filings", payload)
        assert second.from_cache is True
        assert len(fake.calls) == 1

    def test_missing_model_names_the_pull_command(self, tmp_path, monkeypatch):
        fake = FakeHTTPX([FakeResponse({"error": "model not found"}, status_code=404)])
        import httpx

        monkeypatch.setattr(httpx, "post", fake.post)
        client = LLMClient(cache=LLMCache(tmp_path / "c.sqlite"), sleeper=lambda s: None)
        client.provider = "ollama"
        client.model = "granite4.1:3b"
        with pytest.raises(LLMSetupError, match="ollama pull granite4.1:3b"):
            client.classify("filings", {"symbol": "TCS"})
        assert len(fake.calls) == 1, "a missing model is permanent; do not retry it"

    def test_server_down_names_ollama_serve(self, tmp_path, monkeypatch):
        import httpx

        calls: list[int] = []

        def boom(*a, **k):
            calls.append(1)
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(httpx, "post", boom)
        client = LLMClient(cache=LLMCache(tmp_path / "c.sqlite"), sleeper=lambda s: None)
        client.provider = "ollama"
        client.model = "granite4.1:3b"
        with pytest.raises(LLMSetupError, match="ollama serve"):
            client.classify("filings", {"symbol": "TCS"})
        assert len(calls) == 1, "a dead server is permanent; do not retry it"

    def test_missing_anthropic_key_points_at_the_ollama_option(self, tmp_path,
                                                                monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr("core.llm.get_secrets", lambda *a, **k: type(
            "S", (), {"anthropic_api_key": None}
        )())
        client = LLMClient(cache=LLMCache(tmp_path / "c.sqlite"), sleeper=lambda s: None)
        client.provider = "anthropic"
        with pytest.raises(LLMSetupError, match="llm.provider: ollama"):
            client.classify("filings", {"symbol": "TCS"})

    def test_empty_message_is_an_error_not_a_silent_pass(self, tmp_path, monkeypatch):
        fake = FakeHTTPX([FakeResponse({"message": {"content": ""}})] * 4)
        import httpx

        monkeypatch.setattr(httpx, "post", fake.post)
        client = LLMClient(cache=LLMCache(tmp_path / "c.sqlite"), sleeper=lambda s: None)
        client.provider = "ollama"
        client.model = "granite4.1:3b"
        with pytest.raises(LLMError):
            client.classify("filings", {"symbol": "TCS"})

    def test_schema_violation_still_retries(self, tmp_path, monkeypatch):
        """Grammar constraints help but do not guarantee semantic validity."""
        fake = FakeHTTPX([
            FakeResponse({"message": {"content": '{"label":"MAYBE","confidence":0.5,'
                                                 '"reason":"r"}'}}),
            FakeResponse({"message": {"content": '{"label":"NOISE","confidence":0.5,'
                                                 '"reason":"r"}'}}),
        ])
        import httpx

        monkeypatch.setattr(httpx, "post", fake.post)
        client = LLMClient(cache=LLMCache(tmp_path / "c.sqlite"), sleeper=lambda s: None)
        client.provider = "ollama"
        client.model = "granite4.1:3b"
        assert client.classify("filings", {"symbol": "TCS"}).attempts == 2

    def test_no_api_key_is_required(self, ollama_client, monkeypatch):
        """§0.4 still holds, but local inference needs no secret at all."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        client, _ = ollama_client
        assert client.classify("filings", {"symbol": "TCS"}).data["label"] == "NOISE"
