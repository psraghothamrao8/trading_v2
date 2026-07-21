"""LLM layer tests — §1. No network: the transport is a callable."""

from __future__ import annotations

import json

import pytest

from core.llm import (
    Classification,
    LLMCache,
    LLMClient,
    LLMError,
    SchemaError,
    extract_json,
    load_prompt,
    load_schema,
    validate,
)


class TestSchemaValidation:
    def test_accepts_valid(self):
        schema = {"type": "object", "required": ["a"], "properties": {"a": {"type": "string"}}}
        validate({"a": "x"}, schema)

    def test_missing_required_key(self):
        with pytest.raises(SchemaError, match="missing required key"):
            validate({}, {"type": "object", "required": ["a"]})

    def test_wrong_type(self):
        with pytest.raises(SchemaError, match="expected"):
            validate({"a": 1}, {"type": "object", "properties": {"a": {"type": "string"}}})

    def test_enum(self):
        schema = {"type": "string", "enum": ["A", "B"]}
        validate("A", schema)
        with pytest.raises(SchemaError, match="is not one of"):
            validate("C", schema)

    def test_numeric_bounds(self):
        schema = {"type": "number", "minimum": 0.0, "maximum": 1.0}
        validate(0.5, schema)
        with pytest.raises(SchemaError, match="maximum"):
            validate(1.5, schema)

    def test_nullable_union(self):
        validate(None, {"type": ["number", "null"]})
        validate(3.0, {"type": ["number", "null"]})

    def test_boolean_is_not_a_number(self):
        """bool subclasses int in Python; a schema saying 'number' must reject it."""
        with pytest.raises(SchemaError):
            validate(True, {"type": "number"})

    def test_array_items_and_max(self):
        schema = {"type": "array", "maxItems": 2, "items": {"type": "string"}}
        validate(["a"], schema)
        with pytest.raises(SchemaError, match="maxItems"):
            validate(["a", "b", "c"], schema)


class TestJSONExtraction:
    def test_plain_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_json_with_surrounding_prose(self):
        """Models add prose even when told not to; do not fail the whole call."""
        text = 'Here is the result:\n{"label": "NOISE", "confidence": 0.9}\nHope that helps.'
        assert extract_json(text)["label"] == "NOISE"

    def test_braces_inside_strings_do_not_confuse_the_scanner(self):
        assert extract_json('{"reason": "a } brace"}')["reason"] == "a } brace"

    def test_no_json_raises(self):
        with pytest.raises(SchemaError, match="No JSON object"):
            extract_json("I cannot answer that.")

    def test_unbalanced_raises(self):
        with pytest.raises(SchemaError):
            extract_json('{"a": 1')


class TestPrompts:
    """§9.6: every embedded prompt is an editable file with a schema and
    two few-shot examples."""

    @pytest.mark.parametrize("task", ["filings", "sympathy", "pead_tone", "special_situations"])
    def test_prompt_exists(self, task):
        assert len(load_prompt(task)) > 500

    @pytest.mark.parametrize("task", ["filings", "sympathy", "pead_tone", "special_situations"])
    def test_schema_exists_and_is_valid_json(self, task):
        schema = load_schema(task)
        assert schema["type"] == "object"

    @pytest.mark.parametrize("task", ["filings", "sympathy", "pead_tone", "special_situations"])
    def test_two_few_shot_examples(self, task):
        text = load_prompt(task)
        assert text.count("--- EXAMPLE") >= 2, f"{task} needs two few-shot examples (§9.6)"

    def test_filings_prompt_carries_the_spec_materiality_guidance(self):
        """§6.1 names the guidance explicitly; it must be in the prompt."""
        text = load_prompt("filings")
        for phrase in ["MATERIAL_POSITIVE", "MATERIAL_NEGATIVE", "NOISE",
                       "5%", "promoter", "USFDA", "auditor resignation",
                       "rating", "buyback", "pledge invocation", "ESOP"]:
            assert phrase.lower() in text.lower(), f"filings prompt is missing {phrase!r}"

    def test_pead_prompt_covers_the_three_scored_dimensions(self):
        text = load_prompt("pead_tone").lower()
        for phrase in ["guidance confidence", "demand commentary", "margin trajectory"]:
            assert phrase in text

    def test_filings_schema_matches_the_spec_shape(self):
        schema = load_schema("filings")
        assert set(schema["properties"]) == {
            "label", "confidence", "reason", "est_revenue_impact_pct"
        }
        assert schema["properties"]["label"]["enum"] == [
            "MATERIAL_POSITIVE", "MATERIAL_NEGATIVE", "NOISE"
        ]


class TestCache:
    def test_key_is_stable_and_content_sensitive(self):
        a = LLMCache.key("filings", "m", {"x": 1})
        b = LLMCache.key("filings", "m", {"x": 1})
        c = LLMCache.key("filings", "m", {"x": 2})
        assert a == b
        assert a != c

    def test_key_is_order_insensitive(self):
        assert LLMCache.key("t", "m", {"a": 1, "b": 2}) == LLMCache.key("t", "m", {"b": 2, "a": 1})

    def test_roundtrip(self, tmp_path):
        cache = LLMCache(tmp_path / "c.sqlite")
        cache.put("h1", "filings", "m", {"label": "NOISE"})
        assert cache.get("h1") == {"label": "NOISE"}
        assert cache.get("nope") is None
        cache.close()


class TestClassify:
    def _client(self, responses, tmp_path, calls=None):
        queue = list(responses)

        def transport(system, user, max_tokens):
            if calls is not None:
                calls.append((system, user))
            return queue.pop(0)

        return LLMClient(
            transport=transport,
            cache=LLMCache(tmp_path / "cache.sqlite"),
            sleeper=lambda s: None,
        )

    def test_happy_path(self, tmp_path):
        client = self._client(
            ['{"label":"NOISE","confidence":0.9,"reason":"routine"}'], tmp_path
        )
        result = client.classify("filings", {"symbol": "TCS"})
        assert result.data["label"] == "NOISE"
        assert result.from_cache is False

    def test_second_identical_call_hits_the_cache(self, tmp_path):
        """§1: no filing is classified twice. The 30s poll depends on it."""
        calls: list = []
        client = self._client(
            ['{"label":"NOISE","confidence":0.9,"reason":"r"}'], tmp_path, calls
        )
        payload = {"symbol": "TCS", "headline": "Board meeting"}
        client.classify("filings", payload)
        second = client.classify("filings", payload)
        assert second.from_cache is True
        assert len(calls) == 1, "the model must not be called twice for one filing"

    def test_retries_on_bad_json_then_succeeds(self, tmp_path):
        client = self._client(
            ["not json at all", '{"label":"NOISE","confidence":0.5,"reason":"r"}'], tmp_path
        )
        result = client.classify("filings", {"symbol": "X"})
        assert result.attempts == 2

    def test_retries_on_schema_violation(self, tmp_path):
        client = self._client(
            ['{"label":"MAYBE","confidence":0.5,"reason":"r"}',
             '{"label":"NOISE","confidence":0.5,"reason":"r"}'],
            tmp_path,
        )
        assert client.classify("filings", {"symbol": "X"}).attempts == 2

    def test_gives_up_loudly_after_three_attempts(self, tmp_path):
        """§0.8: fail loudly rather than returning a fake success."""
        client = self._client(["garbage"] * 3, tmp_path)
        with pytest.raises(LLMError, match="failed after 3 attempts"):
            client.classify("filings", {"symbol": "X"})

    def test_failed_calls_are_not_cached(self, tmp_path):
        calls: list = []
        client = self._client(
            ["garbage", "garbage", "garbage",
             '{"label":"NOISE","confidence":0.5,"reason":"r"}'],
            tmp_path, calls,
        )
        with pytest.raises(LLMError):
            client.classify("filings", {"symbol": "X"})
        result = client.classify("filings", {"symbol": "X"})
        assert result.from_cache is False
        assert result.data["label"] == "NOISE"

    def test_prompt_is_sent_as_the_system_message(self, tmp_path):
        calls: list = []
        client = self._client(
            ['{"label":"NOISE","confidence":0.9,"reason":"r"}'], tmp_path, calls
        )
        client.classify("filings", {"symbol": "TCS"})
        system, user = calls[0]
        assert "MATERIALITY GUIDANCE" in system
        assert json.loads(user)["symbol"] == "TCS"
