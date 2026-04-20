"""Tests for web backend client configuration and singleton behavior.

Coverage:
  _get_firecrawl_client() — configuration matrix, singleton caching,
  constructor failure recovery, return value verification, edge cases.
  _get_backend() — backend selection logic with env var combinations.
  _get_parallel_client() — Parallel client configuration, singleton caching.
  check_web_api_key() — unified availability check across all web backends.
"""

import importlib
import json
import os
import sys
import types
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class TestFirecrawlClientConfig:
    """Test suite for Firecrawl client initialization."""

    def setup_method(self):
        """Reset client and env vars before each test."""
        import tools.web_tools
        tools.web_tools._firecrawl_client = None
        tools.web_tools._firecrawl_client_config = None
        for key in (
            "HERMES_ENABLE_NOUS_MANAGED_TOOLS",
            "FIRECRAWL_API_KEY",
            "FIRECRAWL_API_URL",
            "FIRECRAWL_GATEWAY_URL",
            "TOOL_GATEWAY_DOMAIN",
            "TOOL_GATEWAY_SCHEME",
            "TOOL_GATEWAY_USER_TOKEN",
        ):
            os.environ.pop(key, None)
        os.environ["HERMES_ENABLE_NOUS_MANAGED_TOOLS"] = "1"

    def teardown_method(self):
        """Reset client after each test."""
        import tools.web_tools
        tools.web_tools._firecrawl_client = None
        tools.web_tools._firecrawl_client_config = None
        for key in (
            "HERMES_ENABLE_NOUS_MANAGED_TOOLS",
            "FIRECRAWL_API_KEY",
            "FIRECRAWL_API_URL",
            "FIRECRAWL_GATEWAY_URL",
            "TOOL_GATEWAY_DOMAIN",
            "TOOL_GATEWAY_SCHEME",
            "TOOL_GATEWAY_USER_TOKEN",
        ):
            os.environ.pop(key, None)

    # ── Configuration matrix ─────────────────────────────────────────

    def test_cloud_mode_key_only(self):
        """API key without URL → cloud Firecrawl."""
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "fc-test"}):
            with patch("tools.web_tools.Firecrawl") as mock_fc:
                from tools.web_tools import _get_firecrawl_client
                result = _get_firecrawl_client()
                mock_fc.assert_called_once_with(api_key="fc-test")
                assert result is mock_fc.return_value

    def test_self_hosted_with_key(self):
        """Both key + URL → self-hosted with auth."""
        with patch.dict(os.environ, {
            "FIRECRAWL_API_KEY": "fc-test",
            "FIRECRAWL_API_URL": "http://localhost:3002",
        }):
            with patch("tools.web_tools.Firecrawl") as mock_fc:
                from tools.web_tools import _get_firecrawl_client
                result = _get_firecrawl_client()
                mock_fc.assert_called_once_with(
                    api_key="fc-test", api_url="http://localhost:3002"
                )
                assert result is mock_fc.return_value

    def test_self_hosted_no_key(self):
        """URL only, no key → self-hosted without auth."""
        with patch.dict(os.environ, {"FIRECRAWL_API_URL": "http://localhost:3002"}):
            with patch("tools.web_tools.Firecrawl") as mock_fc:
                from tools.web_tools import _get_firecrawl_client
                result = _get_firecrawl_client()
                mock_fc.assert_called_once_with(api_url="http://localhost:3002")
                assert result is mock_fc.return_value

    def test_no_config_raises_with_helpful_message(self):
        """Neither key nor URL → ValueError with guidance."""
        with patch("tools.web_tools.Firecrawl"):
            with patch("tools.web_tools._read_nous_access_token", return_value=None):
                from tools.web_tools import _get_firecrawl_client
                with pytest.raises(ValueError, match="FIRECRAWL_API_KEY"):
                    _get_firecrawl_client()

    def test_tool_gateway_domain_builds_firecrawl_gateway_origin(self):
        """Shared gateway domain should derive the Firecrawl vendor hostname."""
        with patch.dict(os.environ, {"TOOL_GATEWAY_DOMAIN": "nousresearch.com"}):
            with patch("tools.web_tools._read_nous_access_token", return_value="nous-token"):
                with patch("tools.web_tools.Firecrawl") as mock_fc:
                    from tools.web_tools import _get_firecrawl_client
                    result = _get_firecrawl_client()
                    mock_fc.assert_called_once_with(
                        api_key="nous-token",
                        api_url="https://firecrawl-gateway.nousresearch.com",
                    )
                    assert result is mock_fc.return_value

    def test_tool_gateway_scheme_can_switch_derived_gateway_origin_to_http(self):
        """Shared gateway scheme should allow local plain-http vendor hosts."""
        with patch.dict(os.environ, {
            "TOOL_GATEWAY_DOMAIN": "nousresearch.com",
            "TOOL_GATEWAY_SCHEME": "http",
        }):
            with patch("tools.web_tools._read_nous_access_token", return_value="nous-token"):
                with patch("tools.web_tools.Firecrawl") as mock_fc:
                    from tools.web_tools import _get_firecrawl_client
                    result = _get_firecrawl_client()
                    mock_fc.assert_called_once_with(
                        api_key="nous-token",
                        api_url="http://firecrawl-gateway.nousresearch.com",
                    )
                    assert result is mock_fc.return_value

    def test_invalid_tool_gateway_scheme_raises(self):
        """Unexpected shared gateway schemes should fail fast."""
        with patch.dict(os.environ, {
            "TOOL_GATEWAY_DOMAIN": "nousresearch.com",
            "TOOL_GATEWAY_SCHEME": "ftp",
        }):
            with patch("tools.web_tools._read_nous_access_token", return_value="nous-token"):
                from tools.web_tools import _get_firecrawl_client
                with pytest.raises(ValueError, match="TOOL_GATEWAY_SCHEME"):
                    _get_firecrawl_client()

    def test_explicit_firecrawl_gateway_url_takes_precedence(self):
        """An explicit Firecrawl gateway origin should override the shared domain."""
        with patch.dict(os.environ, {
            "FIRECRAWL_GATEWAY_URL": "https://firecrawl-gateway.localhost:3009/",
            "TOOL_GATEWAY_DOMAIN": "nousresearch.com",
        }):
            with patch("tools.web_tools._read_nous_access_token", return_value="nous-token"):
                with patch("tools.web_tools.Firecrawl") as mock_fc:
                    from tools.web_tools import _get_firecrawl_client
                    _get_firecrawl_client()
                    mock_fc.assert_called_once_with(
                        api_key="nous-token",
                        api_url="https://firecrawl-gateway.localhost:3009",
                    )

    def test_default_gateway_domain_targets_nous_production_origin(self):
        """Default gateway origin should point at the Firecrawl vendor hostname."""
        with patch("tools.web_tools._read_nous_access_token", return_value="nous-token"):
            with patch("tools.web_tools.Firecrawl") as mock_fc:
                from tools.web_tools import _get_firecrawl_client
                _get_firecrawl_client()
                mock_fc.assert_called_once_with(
                    api_key="nous-token",
                    api_url="https://firecrawl-gateway.nousresearch.com",
                )

    def test_direct_mode_is_preferred_over_tool_gateway(self):
        """Explicit Firecrawl config should win over the gateway fallback."""
        with patch.dict(os.environ, {
            "FIRECRAWL_API_KEY": "fc-test",
            "TOOL_GATEWAY_DOMAIN": "nousresearch.com",
        }):
            with patch("tools.web_tools._read_nous_access_token", return_value="nous-token"):
                with patch("tools.web_tools.Firecrawl") as mock_fc:
                    from tools.web_tools import _get_firecrawl_client
                    _get_firecrawl_client()
                mock_fc.assert_called_once_with(api_key="fc-test")

    def test_nous_auth_token_respects_hermes_home_override(self, tmp_path):
        """Auth lookup should read from HERMES_HOME/auth.json, not ~/.hermes/auth.json."""
        real_home = tmp_path / "real-home"
        (real_home / ".hermes").mkdir(parents=True)

        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        (hermes_home / "auth.json").write_text(json.dumps({
            "providers": {
                "nous": {
                    "access_token": "nous-token",
                }
            }
        }))

        with patch.dict(os.environ, {
            "HOME": str(real_home),
            "HERMES_HOME": str(hermes_home),
        }, clear=False):
            import tools.web_tools
            importlib.reload(tools.web_tools)
            assert tools.web_tools._read_nous_access_token() == "nous-token"

    def test_check_auxiliary_model_re_resolves_backend_each_call(self):
        """Availability checks should not be pinned to module import state."""
        import tools.web_tools

        # Simulate the pre-fix import-time cache slot for regression coverage.
        tools.web_tools.__dict__["_aux_async_client"] = None

        with patch(
            "tools.web_tools.get_async_text_auxiliary_client",
            side_effect=[(None, None), (MagicMock(base_url="https://api.openrouter.ai/v1"), "test-model")],
        ):
            assert tools.web_tools.check_auxiliary_model() is False
            assert tools.web_tools.check_auxiliary_model() is True

    @pytest.mark.asyncio
    async def test_summarizer_re_resolves_backend_after_initial_unavailable_state(self):
        """Summarization should pick up a backend that becomes available later in-process."""
        import tools.web_tools

        tools.web_tools.__dict__["_aux_async_client"] = None

        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="summary text"))]

        with patch(
            "tools.web_tools._resolve_web_extract_auxiliary",
            side_effect=[(None, None, {}), (MagicMock(base_url="https://api.openrouter.ai/v1"), "test-model", {})],
        ), patch(
            "tools.web_tools.async_call_llm",
            new=AsyncMock(return_value=response),
        ) as mock_async_call:
            assert tools.web_tools.check_auxiliary_model() is False
            result = await tools.web_tools._call_summarizer_llm(
                "Some content worth summarizing",
                "Source: https://example.com\n\n",
                None,
            )

        assert result == "summary text"
        mock_async_call.assert_awaited_once()

    # ── Singleton caching ────────────────────────────────────────────

    def test_singleton_returns_same_instance(self):
        """Second call returns cached client without re-constructing."""
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "fc-test"}):
            with patch("tools.web_tools.Firecrawl") as mock_fc:
                from tools.web_tools import _get_firecrawl_client
                client1 = _get_firecrawl_client()
                client2 = _get_firecrawl_client()
                assert client1 is client2
                mock_fc.assert_called_once()  # constructed only once

    def test_constructor_failure_allows_retry(self):
        """If Firecrawl() raises, next call should retry (not return None)."""
        import tools.web_tools
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "fc-test"}):
            with patch("tools.web_tools.Firecrawl") as mock_fc:
                mock_fc.side_effect = [RuntimeError("init failed"), MagicMock()]
                from tools.web_tools import _get_firecrawl_client

                with pytest.raises(RuntimeError):
                    _get_firecrawl_client()

                # Client stayed None, so retry should work
                assert tools.web_tools._firecrawl_client is None
                result = _get_firecrawl_client()
                assert result is not None

    # ── Edge cases ───────────────────────────────────────────────────

    def test_empty_string_key_treated_as_absent(self):
        """FIRECRAWL_API_KEY='' should not be passed as api_key."""
        with patch.dict(os.environ, {
            "FIRECRAWL_API_KEY": "",
            "FIRECRAWL_API_URL": "http://localhost:3002",
        }):
            with patch("tools.web_tools.Firecrawl") as mock_fc:
                from tools.web_tools import _get_firecrawl_client
                _get_firecrawl_client()
                # Empty string is falsy, so only api_url should be passed
                mock_fc.assert_called_once_with(api_url="http://localhost:3002")

    def test_empty_string_key_no_url_raises(self):
        """FIRECRAWL_API_KEY='' with no URL → should raise."""
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": ""}):
            with patch("tools.web_tools.Firecrawl"):
                with patch("tools.web_tools._read_nous_access_token", return_value=None):
                    from tools.web_tools import _get_firecrawl_client
                    with pytest.raises(ValueError):
                        _get_firecrawl_client()


class TestBackendSelection:
    """Test suite for _get_backend() backend selection logic.

    The backend is configured via config.yaml (web.backend), set by
    ``hermes tools``.  Falls back to key-based detection for legacy/manual
    setups.
    """

    _ENV_KEYS = (
        "HERMES_ENABLE_NOUS_MANAGED_TOOLS",
        "EXA_API_KEY",
        "PARALLEL_API_KEY",
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
        "FIRECRAWL_GATEWAY_URL",
        "TOOL_GATEWAY_DOMAIN",
        "TOOL_GATEWAY_SCHEME",
        "TOOL_GATEWAY_USER_TOKEN",
        "TAVILY_API_KEY",
        "CUSTOM_SEARCH_API_KEY",
        "CUSTOM_SEARCH_BASE_URL",
        "CUSTOM_SEARCH_MODEL",
    )

    def setup_method(self):
        os.environ["HERMES_ENABLE_NOUS_MANAGED_TOOLS"] = "1"
        for key in self._ENV_KEYS:
            if key != "HERMES_ENABLE_NOUS_MANAGED_TOOLS":
                os.environ.pop(key, None)

    def teardown_method(self):
        for key in self._ENV_KEYS:
            os.environ.pop(key, None)

    # ── Config-based selection (web.backend in config.yaml) ───────────

    def test_config_parallel(self):
        """web.backend=parallel in config → 'parallel' regardless of keys."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "parallel"}):
            assert _get_backend() == "parallel"

    def test_config_exa(self):
        """web.backend=exa in config → 'exa' regardless of other keys."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "exa"}), \
             patch.dict(os.environ, {"PARALLEL_API_KEY": "test-key"}):
            assert _get_backend() == "exa"

    def test_config_firecrawl(self):
        """web.backend=firecrawl in config → 'firecrawl' even if Parallel key set."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "firecrawl"}), \
             patch.dict(os.environ, {"PARALLEL_API_KEY": "test-key"}):
            assert _get_backend() == "firecrawl"

    def test_config_tavily(self):
        """web.backend=tavily in config → 'tavily' regardless of other keys."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "tavily"}):
            assert _get_backend() == "tavily"

    def test_config_tavily_overrides_env_keys(self):
        """web.backend=tavily in config → 'tavily' even if Firecrawl key set."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "tavily"}), \
             patch.dict(os.environ, {"FIRECRAWL_API_KEY": "fc-test"}):
            assert _get_backend() == "tavily"

    def test_config_case_insensitive(self):
        """web.backend=Parallel (mixed case) → 'parallel'."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "Parallel"}):
            assert _get_backend() == "parallel"

    def test_config_tavily_case_insensitive(self):
        """web.backend=Tavily (mixed case) → 'tavily'."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "Tavily"}):
            assert _get_backend() == "tavily"

    # ── Fallback (no web.backend in config) ───────────────────────────

    def test_fallback_parallel_only_key(self):
        """Only PARALLEL_API_KEY set → 'parallel'."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"PARALLEL_API_KEY": "test-key"}):
            assert _get_backend() == "parallel"

    def test_fallback_exa_only_key(self):
        """Only EXA_API_KEY set → 'exa'."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"EXA_API_KEY": "exa-test"}):
            assert _get_backend() == "exa"

    def test_fallback_parallel_takes_priority_over_exa(self):
        """Exa should only win the fallback path when it is the only configured backend."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"EXA_API_KEY": "exa-test", "PARALLEL_API_KEY": "par-test"}):
            assert _get_backend() == "parallel"

    def test_fallback_tavily_only_key(self):
        """Only TAVILY_API_KEY set → 'tavily'."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test"}):
            assert _get_backend() == "tavily"

    def test_fallback_tavily_with_firecrawl_prefers_firecrawl(self):
        """Tavily + Firecrawl keys, no config → 'firecrawl' (backward compat)."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test", "FIRECRAWL_API_KEY": "fc-test"}):
            assert _get_backend() == "firecrawl"

    def test_fallback_tavily_with_parallel_prefers_parallel(self):
        """Tavily + Parallel keys, no config → 'parallel' (Parallel takes priority over Tavily)."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test", "PARALLEL_API_KEY": "par-test"}):
            # Parallel + no Firecrawl → parallel
            assert _get_backend() == "parallel"

    def test_fallback_both_keys_defaults_to_firecrawl(self):
        """Both keys set, no config → 'firecrawl' (backward compat)."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"PARALLEL_API_KEY": "test-key", "FIRECRAWL_API_KEY": "fc-test"}):
            assert _get_backend() == "firecrawl"

    def test_fallback_firecrawl_only_key(self):
        """Only FIRECRAWL_API_KEY set → 'firecrawl'."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"FIRECRAWL_API_KEY": "fc-test"}):
            assert _get_backend() == "firecrawl"

    def test_fallback_no_keys_defaults_to_firecrawl(self):
        """No keys, no config → 'firecrawl' (will fail at client init)."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}):
            assert _get_backend() == "firecrawl"

    def test_invalid_config_falls_through_to_fallback(self):
        """web.backend=invalid → ignored, uses key-based fallback."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "nonexistent"}), \
             patch.dict(os.environ, {"PARALLEL_API_KEY": "test-key"}):
            assert _get_backend() == "parallel"

    # ── Custom backend (OpenAI-compatible chat completions) ───────────

    def test_config_custom(self):
        """web.backend=custom in config → 'custom' regardless of other keys."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "custom"}), \
             patch.dict(os.environ, {"FIRECRAWL_API_KEY": "fc-test"}):
            assert _get_backend() == "custom"

    def test_config_custom_case_insensitive(self):
        """web.backend=Custom (mixed case) → 'custom'."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={"backend": "Custom"}):
            assert _get_backend() == "custom"

    def test_fallback_custom_only_key(self):
        """Only CUSTOM_SEARCH_API_KEY set → 'custom'."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"CUSTOM_SEARCH_API_KEY": "sk-custom"}):
            assert _get_backend() == "custom"

    def test_fallback_custom_with_firecrawl_prefers_firecrawl(self):
        """Custom + Firecrawl keys, no config → 'firecrawl' (lower priority for custom)."""
        from tools.web_tools import _get_backend
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {
                 "CUSTOM_SEARCH_API_KEY": "sk-custom",
                 "FIRECRAWL_API_KEY": "fc-test",
             }):
            assert _get_backend() == "firecrawl"


class TestParallelClientConfig:
    """Test suite for Parallel client initialization."""

    def setup_method(self):
        import tools.web_tools
        tools.web_tools._parallel_client = None
        os.environ.pop("PARALLEL_API_KEY", None)
        fake_parallel = types.ModuleType("parallel")

        class Parallel:
            def __init__(self, api_key):
                self.api_key = api_key

        class AsyncParallel:
            def __init__(self, api_key):
                self.api_key = api_key

        fake_parallel.Parallel = Parallel
        fake_parallel.AsyncParallel = AsyncParallel
        sys.modules["parallel"] = fake_parallel

    def teardown_method(self):
        import tools.web_tools
        tools.web_tools._parallel_client = None
        os.environ.pop("PARALLEL_API_KEY", None)
        sys.modules.pop("parallel", None)

    def test_creates_client_with_key(self):
        """PARALLEL_API_KEY set → creates Parallel client."""
        with patch.dict(os.environ, {"PARALLEL_API_KEY": "test-key"}):
            from tools.web_tools import _get_parallel_client
            from parallel import Parallel
            client = _get_parallel_client()
            assert client is not None
            assert isinstance(client, Parallel)

    def test_no_key_raises_with_helpful_message(self):
        """No PARALLEL_API_KEY → ValueError with guidance."""
        from tools.web_tools import _get_parallel_client
        with pytest.raises(ValueError, match="PARALLEL_API_KEY"):
            _get_parallel_client()

    def test_singleton_returns_same_instance(self):
        """Second call returns cached client."""
        with patch.dict(os.environ, {"PARALLEL_API_KEY": "test-key"}):
            from tools.web_tools import _get_parallel_client
            client1 = _get_parallel_client()
            client2 = _get_parallel_client()
            assert client1 is client2


class TestWebSearchErrorHandling:
    """Test suite for web_search_tool() error responses."""

    def test_search_error_response_does_not_expose_diagnostics(self):
        import tools.web_tools

        firecrawl_client = MagicMock()
        firecrawl_client.search.side_effect = RuntimeError("boom")

        with patch("tools.web_tools._get_backend", return_value="firecrawl"), \
             patch("tools.web_tools._get_firecrawl_client", return_value=firecrawl_client), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch.object(tools.web_tools._debug, "log_call") as mock_log_call, \
             patch.object(tools.web_tools._debug, "save"):
            result = json.loads(tools.web_tools.web_search_tool("test query", limit=3))

        assert result == {"error": "Error searching web: boom"}

        debug_payload = mock_log_call.call_args.args[1]
        assert debug_payload["error"] == "Error searching web: boom"
        assert "traceback" not in debug_payload["error"]
        assert "exception_type" not in debug_payload["error"]
        assert "config" not in result
        assert "exception_type" not in result
        assert "exception_chain" not in result
        assert "traceback" not in result


class TestCheckWebApiKey:
    """Test suite for check_web_api_key() unified availability check."""

    _ENV_KEYS = (
        "HERMES_ENABLE_NOUS_MANAGED_TOOLS",
        "EXA_API_KEY",
        "PARALLEL_API_KEY",
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
        "FIRECRAWL_GATEWAY_URL",
        "TOOL_GATEWAY_DOMAIN",
        "TOOL_GATEWAY_SCHEME",
        "TOOL_GATEWAY_USER_TOKEN",
        "TAVILY_API_KEY",
        "CUSTOM_SEARCH_API_KEY",
        "CUSTOM_SEARCH_BASE_URL",
        "CUSTOM_SEARCH_MODEL",
    )

    def setup_method(self):
        os.environ["HERMES_ENABLE_NOUS_MANAGED_TOOLS"] = "1"
        for key in self._ENV_KEYS:
            if key != "HERMES_ENABLE_NOUS_MANAGED_TOOLS":
                os.environ.pop(key, None)

    def teardown_method(self):
        for key in self._ENV_KEYS:
            os.environ.pop(key, None)

    def test_parallel_key_only(self):
        with patch.dict(os.environ, {"PARALLEL_API_KEY": "test-key"}):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is True

    def test_exa_key_only(self):
        with patch.dict(os.environ, {"EXA_API_KEY": "exa-test"}):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is True

    def test_firecrawl_key_only(self):
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "fc-test"}):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is True

    def test_firecrawl_url_only(self):
        with patch.dict(os.environ, {"FIRECRAWL_API_URL": "http://localhost:3002"}):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is True

    def test_tavily_key_only(self):
        with patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test"}):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is True

    def test_no_keys_returns_false(self):
        from tools.web_tools import check_web_api_key
        assert check_web_api_key() is False

    def test_both_keys_returns_true(self):
        with patch.dict(os.environ, {
            "PARALLEL_API_KEY": "test-key",
            "FIRECRAWL_API_KEY": "fc-test",
        }):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is True

    def test_all_three_keys_returns_true(self):
        with patch.dict(os.environ, {
            "PARALLEL_API_KEY": "test-key",
            "FIRECRAWL_API_KEY": "fc-test",
            "TAVILY_API_KEY": "tvly-test",
        }):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is True

    def test_tool_gateway_returns_true(self):
        with patch("tools.web_tools._read_nous_access_token", return_value="nous-token"):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is True

    def test_configured_backend_must_match_available_provider(self):
        with patch("tools.web_tools._load_web_config", return_value={"backend": "parallel"}):
            with patch("tools.web_tools._read_nous_access_token", return_value="nous-token"):
                with patch.dict(os.environ, {"FIRECRAWL_GATEWAY_URL": "http://127.0.0.1:3002"}, clear=False):
                    from tools.web_tools import check_web_api_key
                    assert check_web_api_key() is False

    def test_configured_firecrawl_backend_accepts_managed_gateway(self):
        with patch("tools.web_tools._load_web_config", return_value={"backend": "firecrawl"}):
            with patch("tools.web_tools._read_nous_access_token", return_value="nous-token"):
                with patch.dict(os.environ, {"FIRECRAWL_GATEWAY_URL": "http://127.0.0.1:3002"}, clear=False):
                    from tools.web_tools import check_web_api_key
                    assert check_web_api_key() is True

    # ── Custom backend ───────────────────────────────────────────────

    def test_custom_key_only(self):
        """CUSTOM_SEARCH_API_KEY set → check_web_api_key() is True."""
        with patch.dict(os.environ, {"CUSTOM_SEARCH_API_KEY": "sk-custom"}):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is True

    def test_custom_backend_configured_without_key_returns_false(self):
        """backend=custom in config but no CUSTOM_SEARCH_API_KEY and no
        web.custom_api_key → check_web_api_key() returns False."""
        with patch("tools.web_tools._load_web_config", return_value={"backend": "custom"}):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is False

    def test_custom_api_key_via_config_returns_true(self):
        """web.custom_api_key in config.yaml (no env var) → True.

        Documents the unique env-OR-config key resolution path for custom,
        where _is_backend_available consults both _has_env and the config dict.
        """
        with patch(
            "tools.web_tools._load_web_config",
            return_value={"backend": "custom", "custom_api_key": "sk-from-config"},
        ):
            from tools.web_tools import check_web_api_key
            assert check_web_api_key() is True


def test_web_requires_env_includes_exa_key():
    from tools.web_tools import _web_requires_env

    assert "EXA_API_KEY" in _web_requires_env()


def test_web_requires_env_includes_custom_search_key():
    """CUSTOM_SEARCH_API_KEY is surfaced as a web-backend env dep."""
    from tools.web_tools import _web_requires_env

    assert "CUSTOM_SEARCH_API_KEY" in _web_requires_env()


# ══════════════════════════════════════════════════════════════════════════
# Custom OpenAI-compatible chat-completions backend
# ══════════════════════════════════════════════════════════════════════════


class TestCustomBackendHelpers:
    """Tests for _get_custom_base_url / _get_custom_model / _custom_headers.

    All three helpers follow the same resolution order:
      1. CUSTOM_SEARCH_* environment variable
      2. web.custom_* in config.yaml
      3. (model only) default "sonar"
    """

    _ENV_KEYS = (
        "CUSTOM_SEARCH_API_KEY",
        "CUSTOM_SEARCH_BASE_URL",
        "CUSTOM_SEARCH_MODEL",
    )

    def setup_method(self):
        for key in self._ENV_KEYS:
            os.environ.pop(key, None)

    def teardown_method(self):
        for key in self._ENV_KEYS:
            os.environ.pop(key, None)

    # ── base_url ──────────────────────────────────────────────────────

    def test_base_url_env_takes_priority_over_config(self):
        from tools.web_tools import _get_custom_base_url
        with patch("tools.web_tools._load_web_config",
                   return_value={"custom_base_url": "https://config.example.com"}), \
             patch.dict(os.environ, {"CUSTOM_SEARCH_BASE_URL": "https://env.example.com"}):
            assert _get_custom_base_url() == "https://env.example.com"

    def test_base_url_config_fallback(self):
        from tools.web_tools import _get_custom_base_url
        with patch("tools.web_tools._load_web_config",
                   return_value={"custom_base_url": "https://config.example.com"}):
            assert _get_custom_base_url() == "https://config.example.com"

    def test_base_url_strips_trailing_slash(self):
        """Both env and config values are rstrip'd so callers can safely
        concatenate /chat/completions without duplicating slashes."""
        from tools.web_tools import _get_custom_base_url
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"CUSTOM_SEARCH_BASE_URL": "https://env.example.com/"}):
            assert _get_custom_base_url() == "https://env.example.com"

        with patch("tools.web_tools._load_web_config",
                   return_value={"custom_base_url": "https://config.example.com/"}):
            assert _get_custom_base_url() == "https://config.example.com"

    def test_base_url_missing_raises_value_error(self):
        from tools.web_tools import _get_custom_base_url
        with patch("tools.web_tools._load_web_config", return_value={}):
            with pytest.raises(ValueError, match="CUSTOM_SEARCH_BASE_URL"):
                _get_custom_base_url()

    # ── model ─────────────────────────────────────────────────────────

    def test_model_env_takes_priority_over_config(self):
        from tools.web_tools import _get_custom_model
        with patch("tools.web_tools._load_web_config",
                   return_value={"custom_model": "config-model"}), \
             patch.dict(os.environ, {"CUSTOM_SEARCH_MODEL": "env-model"}):
            assert _get_custom_model() == "env-model"

    def test_model_config_fallback(self):
        from tools.web_tools import _get_custom_model
        with patch("tools.web_tools._load_web_config",
                   return_value={"custom_model": "config-model"}):
            assert _get_custom_model() == "config-model"

    def test_model_default_is_sonar(self):
        """Neither env nor config set → "sonar" (Perplexity's default model)."""
        from tools.web_tools import _get_custom_model
        with patch("tools.web_tools._load_web_config", return_value={}):
            assert _get_custom_model() == "sonar"

    # ── headers / api key ─────────────────────────────────────────────

    def test_headers_returns_bearer_auth(self):
        from tools.web_tools import _custom_headers
        with patch("tools.web_tools._load_web_config", return_value={}), \
             patch.dict(os.environ, {"CUSTOM_SEARCH_API_KEY": "sk-abc"}):
            headers = _custom_headers()
        assert headers["Authorization"] == "Bearer sk-abc"
        assert headers["Content-Type"] == "application/json"

    def test_headers_config_fallback(self):
        from tools.web_tools import _custom_headers
        with patch("tools.web_tools._load_web_config",
                   return_value={"custom_api_key": "sk-from-config"}):
            assert _custom_headers()["Authorization"] == "Bearer sk-from-config"

    def test_headers_missing_key_raises_value_error(self):
        from tools.web_tools import _custom_headers
        with patch("tools.web_tools._load_web_config", return_value={}):
            with pytest.raises(ValueError, match="CUSTOM_SEARCH_API_KEY"):
                _custom_headers()


class TestCustomSearch:
    """Tests for _custom_search() response-shape normalization.

    Priority of the three response-extraction paths (highest first):
      1. search_results[]  — Perplexity Sonar native shape
      2. citations[]       — generic OpenAI-compatible citations list
      3. choices[0].message.content  — plain answer text as single result
    """

    def test_search_results_path(self):
        """search_results[] entries map to title/url/description/position."""
        from tools.web_tools import _custom_search
        fake_response = {
            "search_results": [
                {"title": "First", "url": "https://a.example/1", "snippet": "snippet 1"},
                {"title": "Second", "url": "https://a.example/2", "content": "content 2"},
            ],
        }
        with patch("tools.web_tools._custom_chat", return_value=fake_response):
            result = _custom_search("query", limit=5)

        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 2
        assert web[0] == {
            "title": "First",
            "url": "https://a.example/1",
            "description": "snippet 1",
            "position": 1,
        }
        # "content" field also accepted when "snippet" absent
        assert web[1]["description"] == "content 2"
        assert web[1]["position"] == 2

    def test_citations_as_string_list(self):
        """citations[] as plain URL strings → title/description empty, url populated."""
        from tools.web_tools import _custom_search
        fake_response = {
            "citations": ["https://a.example/1", "https://a.example/2"],
        }
        with patch("tools.web_tools._custom_chat", return_value=fake_response):
            result = _custom_search("query", limit=5)

        web = result["data"]["web"]
        assert len(web) == 2
        assert web[0] == {
            "title": "",
            "url": "https://a.example/1",
            "description": "",
            "position": 1,
        }

    def test_citations_as_dict_list(self):
        """citations[] as dicts → title/url/snippet mapped through."""
        from tools.web_tools import _custom_search
        fake_response = {
            "citations": [
                {"title": "T1", "url": "https://a.example/1", "snippet": "S1"},
                {"title": "T2", "url": "https://a.example/2", "content": "C2"},
            ],
        }
        with patch("tools.web_tools._custom_chat", return_value=fake_response):
            result = _custom_search("query", limit=5)

        web = result["data"]["web"]
        assert web[0]["title"] == "T1"
        assert web[0]["description"] == "S1"
        # content field serves as fallback description when snippet missing
        assert web[1]["description"] == "C2"

    def test_answer_text_fallback(self):
        """Neither search_results nor citations → single synthetic result
        titled 'Search Answer' carrying the model's content."""
        from tools.web_tools import _custom_search
        fake_response = {
            "choices": [{"message": {"content": "The sky is blue because of Rayleigh scattering."}}],
        }
        with patch("tools.web_tools._custom_chat", return_value=fake_response):
            result = _custom_search("why is the sky blue", limit=5)

        web = result["data"]["web"]
        assert len(web) == 1
        assert web[0]["title"] == "Search Answer"
        assert web[0]["url"] == ""
        assert web[0]["description"].startswith("The sky is blue")
        assert web[0]["position"] == 1

    def test_empty_response_yields_empty_web_list(self):
        """No search_results, no citations, no choices → empty web list,
        success still True (caller decides how to report no-results)."""
        from tools.web_tools import _custom_search
        with patch("tools.web_tools._custom_chat", return_value={}):
            result = _custom_search("query", limit=5)

        assert result == {"success": True, "data": {"web": []}}

    def test_respects_limit_on_search_results(self):
        """search_results with more items than limit → truncated to limit."""
        from tools.web_tools import _custom_search
        fake_response = {
            "search_results": [
                {"title": f"T{i}", "url": f"https://a.example/{i}", "snippet": "s"}
                for i in range(10)
            ],
        }
        with patch("tools.web_tools._custom_chat", return_value=fake_response):
            result = _custom_search("query", limit=3)

        assert len(result["data"]["web"]) == 3
        assert [r["position"] for r in result["data"]["web"]] == [1, 2, 3]

    def test_search_results_preferred_over_citations(self):
        """When both fields are present, search_results wins; citations ignored."""
        from tools.web_tools import _custom_search
        fake_response = {
            "search_results": [
                {"title": "SR", "url": "https://sr.example", "snippet": "sr"},
            ],
            "citations": ["https://citation.example"],
        }
        with patch("tools.web_tools._custom_chat", return_value=fake_response):
            result = _custom_search("query", limit=5)

        web = result["data"]["web"]
        assert len(web) == 1
        assert web[0]["url"] == "https://sr.example"


class TestCustomExtract:
    """Tests for _custom_extract() — one chat call per URL with error isolation."""

    def test_multi_url_success(self):
        from tools.web_tools import _custom_extract
        responses = [
            {"choices": [{"message": {"content": "# Page 1\n\nContent of page 1."}}]},
            {"choices": [{"message": {"content": "# Page 2\n\nContent of page 2."}}]},
        ]
        with patch("tools.web_tools._custom_chat", side_effect=responses):
            docs = _custom_extract(["https://a.example/1", "https://a.example/2"])

        assert len(docs) == 2
        assert docs[0]["url"] == "https://a.example/1"
        assert docs[0]["content"].startswith("# Page 1")
        assert docs[0]["raw_content"] == docs[0]["content"]
        assert docs[0]["metadata"] == {"sourceURL": "https://a.example/1", "title": ""}
        assert "error" not in docs[0]

    def test_per_url_exception_isolation(self):
        """If the chat call for one URL raises, other URLs still return content
        and the failed URL gets an error-carrying stub document."""
        from tools.web_tools import _custom_extract

        def fake_chat(prompt: str):
            if "fail.example" in prompt:
                raise RuntimeError("upstream 502")
            return {"choices": [{"message": {"content": "ok content"}}]}

        with patch("tools.web_tools._custom_chat", side_effect=fake_chat):
            docs = _custom_extract([
                "https://a.example/1",
                "https://fail.example/boom",
                "https://a.example/3",
            ])

        assert len(docs) == 3
        assert docs[0]["content"] == "ok content"
        assert "error" not in docs[0]

        assert docs[1]["url"] == "https://fail.example/boom"
        assert docs[1]["content"] == ""
        assert docs[1]["raw_content"] == ""
        assert docs[1]["error"] == "upstream 502"
        assert docs[1]["metadata"] == {"sourceURL": "https://fail.example/boom"}

        assert docs[2]["content"] == "ok content"
        assert "error" not in docs[2]

    def test_empty_url_list_returns_empty_list(self):
        from tools.web_tools import _custom_extract
        with patch("tools.web_tools._custom_chat") as mock_chat:
            docs = _custom_extract([])
        assert docs == []
        mock_chat.assert_not_called()
