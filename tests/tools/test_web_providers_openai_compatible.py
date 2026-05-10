"""Tests for the OpenAI-compatible chat-completions web search provider.

Covers:
- OpenAICompatibleSearchProvider.is_configured() — requires BASE_URL + API_KEY + MODEL
- OpenAICompatibleSearchProvider.search() — happy path, HTTP error, request error, bad JSON
- _parse_results() — 3-tier fallback (search_results > citations > choices.content)
- Trailing-slash normalization on base_url
- _is_backend_available("openai-compatible-search") integration
- _get_backend() recognizes "openai-compatible-search" when explicitly configured
- _get_backend() NEVER auto-selects openai-compatible-search (by design)
- check_web_api_key() includes openai-compatible-search in availability check
- web_extract / web_crawl return clear search-only errors when this backend is active
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# OpenAICompatibleSearchProvider.is_configured()
# ---------------------------------------------------------------------------


class TestOpenAICompatibleIsConfigured:
    def test_configured_when_all_three_env_vars_set(self, monkeypatch):
        monkeypatch.setenv("OPENAI_COMPAT_SEARCH_BASE_URL", "https://api.perplexity.ai")
        monkeypatch.setenv("OPENAI_COMPAT_SEARCH_API_KEY", "pplx-test")
        monkeypatch.setenv("OPENAI_COMPAT_SEARCH_MODEL", "sonar")
        from tools.web_providers.openai_compatible import OpenAICompatibleSearchProvider
        assert OpenAICompatibleSearchProvider().is_configured() is True

    def test_not_configured_when_base_url_missing(self, monkeypatch):
        monkeypatch.delenv("OPENAI_COMPAT_SEARCH_BASE_URL", raising=False)
        monkeypatch.setenv("OPENAI_COMPAT_SEARCH_API_KEY", "pplx-test")
        monkeypatch.setenv("OPENAI_COMPAT_SEARCH_MODEL", "sonar")
        from tools.web_providers.openai_compatible import OpenAICompatibleSearchProvider
        assert OpenAICompatibleSearchProvider().is_configured() is False

    def test_not_configured_when_api_key_missing(self, monkeypatch):
        monkeypatch.setenv("OPENAI_COMPAT_SEARCH_BASE_URL", "https://api.perplexity.ai")
        monkeypatch.delenv("OPENAI_COMPAT_SEARCH_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_COMPAT_SEARCH_MODEL", "sonar")
        from tools.web_providers.openai_compatible import OpenAICompatibleSearchProvider
        assert OpenAICompatibleSearchProvider().is_configured() is False

    def test_not_configured_when_model_missing(self, monkeypatch):
        """Model has no default — missing model means not configured."""
        monkeypatch.setenv("OPENAI_COMPAT_SEARCH_BASE_URL", "https://api.perplexity.ai")
        monkeypatch.setenv("OPENAI_COMPAT_SEARCH_API_KEY", "pplx-test")
        monkeypatch.delenv("OPENAI_COMPAT_SEARCH_MODEL", raising=False)
        from tools.web_providers.openai_compatible import OpenAICompatibleSearchProvider
        assert OpenAICompatibleSearchProvider().is_configured() is False

    def test_not_configured_when_any_env_is_whitespace(self, monkeypatch):
        monkeypatch.setenv("OPENAI_COMPAT_SEARCH_BASE_URL", "  ")
        monkeypatch.setenv("OPENAI_COMPAT_SEARCH_API_KEY", "pplx-test")
        monkeypatch.setenv("OPENAI_COMPAT_SEARCH_MODEL", "sonar")
        from tools.web_providers.openai_compatible import OpenAICompatibleSearchProvider
        assert OpenAICompatibleSearchProvider().is_configured() is False

    def test_provider_name(self):
        from tools.web_providers.openai_compatible import OpenAICompatibleSearchProvider
        assert OpenAICompatibleSearchProvider().provider_name() == "openai-compatible-search"

    def test_implements_web_search_provider(self):
        from tools.web_providers.base import WebSearchProvider
        from tools.web_providers.openai_compatible import OpenAICompatibleSearchProvider
        assert issubclass(OpenAICompatibleSearchProvider, WebSearchProvider)


# ---------------------------------------------------------------------------
# OpenAICompatibleSearchProvider.search()
# ---------------------------------------------------------------------------


def _configure(monkeypatch, *, base="https://api.perplexity.ai", key="pplx-test", model="sonar"):
    monkeypatch.setenv("OPENAI_COMPAT_SEARCH_BASE_URL", base)
    monkeypatch.setenv("OPENAI_COMPAT_SEARCH_API_KEY", key)
    monkeypatch.setenv("OPENAI_COMPAT_SEARCH_MODEL", model)


def _mock_resp(json_data, status_code=200):
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    mock.raise_for_status = MagicMock()
    return mock


class TestOpenAICompatibleSearchHappyPath:
    def test_search_results_schema_returns_normalized_rows(self, monkeypatch):
        _configure(monkeypatch)
        from tools.web_providers.openai_compatible import OpenAICompatibleSearchProvider

        data = {
            "search_results": [
                {"title": "Title A", "url": "https://a.example.com", "snippet": "Snippet A"},
                {"title": "Title B", "url": "https://b.example.com", "snippet": "Snippet B"},
            ]
        }
        with patch("httpx.post", return_value=_mock_resp(data)):
            result = OpenAICompatibleSearchProvider().search("test query", limit=5)

        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 2
        assert web[0] == {
            "title": "Title A",
            "url": "https://a.example.com",
            "description": "Snippet A",
            "position": 1,
        }
        assert web[1]["position"] == 2

    def test_search_results_content_field_used_when_snippet_absent(self, monkeypatch):
        _configure(monkeypatch)
        from tools.web_providers.openai_compatible import OpenAICompatibleSearchProvider

        data = {
            "search_results": [
                {"title": "T", "url": "https://x.example.com", "content": "Body via content field"},
            ]
        }
        with patch("httpx.post", return_value=_mock_resp(data)):
            result = OpenAICompatibleSearchProvider().search("q", limit=5)

        assert result["data"]["web"][0]["description"] == "Body via content field"

    def test_limit_respected(self, monkeypatch):
        _configure(monkeypatch)
        from tools.web_providers.openai_compatible import OpenAICompatibleSearchProvider

        data = {
            "search_results": [
                {"title": f"T{i}", "url": f"https://{i}.example.com", "snippet": ""}
                for i in range(10)
            ]
        }
        with patch("httpx.post", return_value=_mock_resp(data)):
            result = OpenAICompatibleSearchProvider().search("q", limit=3)

        assert len(result["data"]["web"]) == 3

    def test_payload_and_auth_header(self, monkeypatch):
        """Verify the outgoing request shape — bearer auth + chat-completions body."""
        _configure(monkeypatch, key="sk-xyz", model="sonar-pro")
        from tools.web_providers.openai_compatible import OpenAICompatibleSearchProvider

        captured = {}

        def capture(url, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs.get("json")
            captured["headers"] = kwargs.get("headers")
            return _mock_resp({"search_results": []})

        with patch("httpx.post", side_effect=capture):
            OpenAICompatibleSearchProvider().search("hello world", limit=5)

        assert captured["url"] == "https://api.perplexity.ai/chat/completions"
        assert captured["json"]["model"] == "sonar-pro"
        assert captured["json"]["messages"] == [{"role": "user", "content": "hello world"}]
        assert captured["headers"]["Authorization"] == "Bearer sk-xyz"
        assert captured["headers"]["Content-Type"] == "application/json"

    def test_trailing_slash_stripped_from_base_url(self, monkeypatch):
        _configure(monkeypatch, base="https://api.perplexity.ai/")
        from tools.web_providers.openai_compatible import OpenAICompatibleSearchProvider

        captured = {}
        def capture(url, **kwargs):
            captured["url"] = url
            return _mock_resp({"search_results": []})

        with patch("httpx.post", side_effect=capture):
            OpenAICompatibleSearchProvider().search("q", limit=5)

        assert captured["url"] == "https://api.perplexity.ai/chat/completions"


class TestOpenAICompatibleSearchErrors:
    def test_missing_env_returns_failure(self, monkeypatch):
        monkeypatch.delenv("OPENAI_COMPAT_SEARCH_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_COMPAT_SEARCH_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_COMPAT_SEARCH_MODEL", raising=False)
        from tools.web_providers.openai_compatible import OpenAICompatibleSearchProvider

        result = OpenAICompatibleSearchProvider().search("q", limit=5)
        assert result["success"] is False
        # Error should mention which vars are missing
        assert "OPENAI_COMPAT_SEARCH_BASE_URL" in result["error"]
        assert "OPENAI_COMPAT_SEARCH_API_KEY" in result["error"]
        assert "OPENAI_COMPAT_SEARCH_MODEL" in result["error"]

    def test_http_error_returns_failure(self, monkeypatch):
        import httpx
        _configure(monkeypatch)
        from tools.web_providers.openai_compatible import OpenAICompatibleSearchProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        http_err = httpx.HTTPStatusError("401", request=MagicMock(), response=mock_resp)

        with patch("httpx.post", side_effect=http_err):
            result = OpenAICompatibleSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "401" in result["error"]

    def test_request_error_returns_failure(self, monkeypatch):
        import httpx
        _configure(monkeypatch, base="https://unreachable.example.com")
        from tools.web_providers.openai_compatible import OpenAICompatibleSearchProvider

        with patch("httpx.post", side_effect=httpx.RequestError("connection refused")):
            result = OpenAICompatibleSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "unreachable.example.com" in result["error"] or "connection" in result["error"].lower()

    def test_invalid_json_returns_failure(self, monkeypatch):
        _configure(monkeypatch)
        from tools.web_providers.openai_compatible import OpenAICompatibleSearchProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.side_effect = ValueError("not json")

        with patch("httpx.post", return_value=mock_resp):
            result = OpenAICompatibleSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert "json" in result["error"].lower() or "parse" in result["error"].lower()


# ---------------------------------------------------------------------------
# _parse_results() — 3-tier fallback priority
# ---------------------------------------------------------------------------


class TestParseResultsFallback:
    def test_search_results_takes_priority_over_citations(self):
        from tools.web_providers.openai_compatible import _parse_results

        data = {
            "search_results": [
                {"title": "Native", "url": "https://native.example.com", "snippet": "s"},
            ],
            "citations": ["https://citation.example.com"],
            "choices": [{"message": {"content": "answer text"}}],
        }
        out = _parse_results(data, limit=5)
        assert len(out) == 1
        assert out[0]["title"] == "Native"
        assert out[0]["url"] == "https://native.example.com"

    def test_citations_used_when_search_results_empty(self):
        from tools.web_providers.openai_compatible import _parse_results

        data = {
            "search_results": [],
            "citations": ["https://a.example.com", "https://b.example.com"],
            "choices": [{"message": {"content": "answer"}}],
        }
        out = _parse_results(data, limit=5)
        assert len(out) == 2
        assert out[0]["url"] == "https://a.example.com"
        assert out[0]["title"] == ""
        assert out[0]["position"] == 1
        assert out[1]["url"] == "https://b.example.com"
        assert out[1]["position"] == 2

    def test_citations_dict_shape(self):
        from tools.web_providers.openai_compatible import _parse_results

        data = {
            "citations": [
                {"title": "Dict Cite", "url": "https://dict.example.com", "snippet": "Desc"},
            ],
        }
        out = _parse_results(data, limit=5)
        assert out[0] == {
            "title": "Dict Cite",
            "url": "https://dict.example.com",
            "description": "Desc",
            "position": 1,
        }

    def test_citations_mixed_str_and_dict(self):
        from tools.web_providers.openai_compatible import _parse_results

        data = {
            "citations": [
                "https://str.example.com",
                {"title": "Dict", "url": "https://dict.example.com", "snippet": "d"},
            ],
        }
        out = _parse_results(data, limit=5)
        assert len(out) == 2
        assert out[0]["url"] == "https://str.example.com"
        assert out[0]["title"] == ""
        assert out[1]["title"] == "Dict"

    def test_choices_content_fallback_when_no_results_or_citations(self):
        from tools.web_providers.openai_compatible import _parse_results

        data = {
            "choices": [{"message": {"content": "This is the synthesized answer."}}],
        }
        out = _parse_results(data, limit=5)
        assert len(out) == 1
        assert out[0]["title"] == "Search Answer"
        assert out[0]["url"] == ""
        assert out[0]["description"] == "This is the synthesized answer."
        assert out[0]["position"] == 1

    def test_choices_content_truncated_to_2000_chars(self):
        from tools.web_providers.openai_compatible import _parse_results

        long_content = "x" * 5000
        data = {"choices": [{"message": {"content": long_content}}]}
        out = _parse_results(data, limit=5)
        assert len(out[0]["description"]) == 2000

    def test_empty_response_returns_empty_list(self):
        from tools.web_providers.openai_compatible import _parse_results
        assert _parse_results({}, limit=5) == []

    def test_all_three_tiers_empty_returns_empty(self):
        from tools.web_providers.openai_compatible import _parse_results

        data = {"search_results": [], "citations": [], "choices": []}
        assert _parse_results(data, limit=5) == []

    def test_position_is_one_indexed(self):
        from tools.web_providers.openai_compatible import _parse_results

        data = {
            "search_results": [
                {"title": f"T{i}", "url": f"https://{i}.example.com", "snippet": ""}
                for i in range(3)
            ]
        }
        out = _parse_results(data, limit=5)
        assert [r["position"] for r in out] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Integration: _is_backend_available
# ---------------------------------------------------------------------------


class TestIsBackendAvailable:
    def test_available_when_all_env_set(self, monkeypatch):
        _configure(monkeypatch)
        from tools.web_tools import _is_backend_available
        assert _is_backend_available("openai-compatible-search") is True

    def test_unavailable_when_any_env_missing(self, monkeypatch):
        monkeypatch.setenv("OPENAI_COMPAT_SEARCH_BASE_URL", "https://api.perplexity.ai")
        monkeypatch.delenv("OPENAI_COMPAT_SEARCH_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_COMPAT_SEARCH_MODEL", "sonar")
        from tools.web_tools import _is_backend_available
        assert _is_backend_available("openai-compatible-search") is False


# ---------------------------------------------------------------------------
# Integration: _get_backend
# ---------------------------------------------------------------------------


class TestGetBackendOpenAICompatible:
    def test_explicit_config_returns_openai_compatible(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(
            web_tools, "_load_web_config",
            lambda: {"backend": "openai-compatible-search"},
        )
        _configure(monkeypatch)
        assert web_tools._get_backend() == "openai-compatible-search"

    def test_never_auto_selected_even_when_all_env_set(self, monkeypatch):
        """openai-compatible-search is explicit-opt-in only — auto-detect must skip it."""
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        # Clear all higher-priority backends
        for var in (
            "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "PARALLEL_API_KEY",
            "TAVILY_API_KEY", "EXA_API_KEY", "SEARXNG_URL", "BRAVE_SEARCH_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        # Make ddgs unavailable so it doesn't claim the auto-detect slot
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
        # All three openai-compatible env vars set
        _configure(monkeypatch)

        backend = web_tools._get_backend()
        # Should fall through to the firecrawl default, NOT openai-compatible-search
        assert backend != "openai-compatible-search"
        assert backend == "firecrawl"

    def test_higher_priority_provider_wins_over_openai_compatible_env(self, monkeypatch):
        """Even with openai-compatible env set, Tavily (explicit config) wins."""
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "tavily"})
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-key")
        _configure(monkeypatch)
        assert web_tools._get_backend() == "tavily"


# ---------------------------------------------------------------------------
# Integration: check_web_api_key
# ---------------------------------------------------------------------------


class TestCheckWebApiKey:
    def test_openai_compatible_satisfies_check(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(
            web_tools, "_load_web_config",
            lambda: {"backend": "openai-compatible-search"},
        )
        _configure(monkeypatch)
        assert web_tools.check_web_api_key() is True

    def test_openai_compatible_fails_check_when_env_missing(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(
            web_tools, "_load_web_config",
            lambda: {"backend": "openai-compatible-search"},
        )
        for var in (
            "OPENAI_COMPAT_SEARCH_BASE_URL",
            "OPENAI_COMPAT_SEARCH_API_KEY",
            "OPENAI_COMPAT_SEARCH_MODEL",
        ):
            monkeypatch.delenv(var, raising=False)
        assert web_tools.check_web_api_key() is False


# ---------------------------------------------------------------------------
# Search-only semantics: web_extract / web_crawl return clear errors
# ---------------------------------------------------------------------------


class TestOpenAICompatibleOnlyExtractCrawlErrors:
    def test_web_extract_returns_search_only_error(self, monkeypatch):
        import asyncio
        import json
        from tools import web_tools

        monkeypatch.setattr(
            web_tools, "_load_web_config",
            lambda: {"backend": "openai-compatible-search"},
        )
        _configure(monkeypatch)
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False, raising=False)

        result_str = asyncio.get_event_loop().run_until_complete(
            web_tools.web_extract_tool(["https://example.com"])
        )
        result = json.loads(result_str)
        assert result["success"] is False
        error_lower = result["error"].lower()
        assert "search-only" in error_lower
        assert "openai-compatible search" in error_lower

    def test_web_crawl_returns_search_only_error(self, monkeypatch):
        import asyncio
        import json
        from tools import web_tools

        monkeypatch.setattr(
            web_tools, "_load_web_config",
            lambda: {"backend": "openai-compatible-search"},
        )
        _configure(monkeypatch)
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(web_tools, "check_firecrawl_api_key", lambda: False)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False, raising=False)

        result_str = asyncio.get_event_loop().run_until_complete(
            web_tools.web_crawl_tool("https://example.com")
        )
        result = json.loads(result_str)
        assert result["success"] is False
        error_lower = result["error"].lower()
        assert "search-only" in error_lower
        assert "openai-compatible search" in error_lower
