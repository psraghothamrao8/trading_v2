"""Morning auth tests — §8.1. No network, no real credentials."""

from __future__ import annotations

import pytest

from scripts.morning_auth import extract_request_token, write_env_value


class TestRequestTokenExtraction:
    def test_bare_token(self):
        assert extract_request_token("aBc123XyZ") == "aBc123XyZ"

    def test_full_redirect_url(self):
        """What you actually have in your clipboard at 07:30."""
        url = ("http://127.0.0.1/?action=login&type=login&status=success"
               "&request_token=SuPeRtOkEn123")
        assert extract_request_token(url) == "SuPeRtOkEn123"

    def test_url_with_trailing_params(self):
        url = "https://example.com/cb?request_token=TOK456&other=1"
        assert extract_request_token(url) == "TOK456"

    def test_whitespace_tolerated(self):
        assert extract_request_token("  TOK789  ") == "TOK789"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Nothing pasted"):
            extract_request_token("   ")

    def test_garbage_raises(self):
        with pytest.raises(ValueError, match="does not look like"):
            extract_request_token("not a token!!")

    def test_url_without_token_raises(self):
        with pytest.raises(ValueError, match="does not look like"):
            extract_request_token("http://127.0.0.1/?status=success")


class TestEnvWriting:
    def test_replaces_existing_key_preserving_others(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text(
            "# comment\nKITE_API_KEY=abc\nKITE_ACCESS_TOKEN=old\nTELEGRAM_CHAT_ID=1\n",
            encoding="utf-8",
        )
        write_env_value("KITE_ACCESS_TOKEN", "new", env_path=env)
        text = env.read_text(encoding="utf-8")
        assert "KITE_ACCESS_TOKEN=new" in text
        assert "KITE_ACCESS_TOKEN=old" not in text
        assert "KITE_API_KEY=abc" in text
        assert "# comment" in text

    def test_appends_a_missing_key(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("KITE_API_KEY=abc\n", encoding="utf-8")
        write_env_value("KITE_USER_ID", "AB1234", env_path=env)
        assert "KITE_USER_ID=AB1234" in env.read_text(encoding="utf-8")

    def test_creates_the_file_from_the_example(self, tmp_path):
        env = tmp_path / ".env"
        write_env_value("KITE_ACCESS_TOKEN", "tok", env_path=env)
        assert env.exists()
        assert "KITE_ACCESS_TOKEN=tok" in env.read_text(encoding="utf-8")
