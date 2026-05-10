"""OpenAI-compatible chat completions backend as a web search provider.

Any ``/chat/completions`` endpoint that returns citations inline can be
used as a ``web_search`` backend.  Concrete targets:

* **Perplexity Sonar** (``https://api.perplexity.ai``) — populates the
  ``search_results`` and ``citations`` fields on each response.
* **ChatGPT with browsing** via proxies that preserve citations.
* **Self-hosted / LiteLLM / vLLM** wrappers around a search-augmented
  model that returns the same envelope.

This provider implements ``WebSearchProvider`` only — the chat-completion
shape is not a suitable substrate for ``web_extract`` (it invites the
model to fabricate page content rather than actually fetch it).  Pair
with Firecrawl / Tavily / Exa / Parallel when you also need extract.

Configuration::

    # ~/.hermes/.env
    OPENAI_COMPAT_SEARCH_BASE_URL=https://api.perplexity.ai
    OPENAI_COMPAT_SEARCH_API_KEY=pplx-...
    OPENAI_COMPAT_SEARCH_MODEL=sonar

    # ~/.hermes/config.yaml
    web:
      search_backend: "openai-compatible-search"
      extract_backend: "firecrawl"    # pair with an extract provider

``OPENAI_COMPAT_SEARCH_MODEL`` is required — there is no sensible default
because the provider does not assume Perplexity.  Set it explicitly to
``sonar``, ``sonar-pro``, a LiteLLM route, etc.

Response parsing order:

1. ``search_results[]`` (Perplexity native) — ``{title, url, snippet}``
2. ``citations[]`` — list of URL strings **or** ``{title, url, snippet}`` dicts
3. Fall back to ``choices[0].message.content`` as a single "answer" row.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from tools.web_providers.base import WebSearchProvider

logger = logging.getLogger(__name__)

# Env var names.  Kept as module constants so tests and ``hermes tools``
# share one source of truth.
ENV_BASE_URL = "OPENAI_COMPAT_SEARCH_BASE_URL"
ENV_API_KEY = "OPENAI_COMPAT_SEARCH_API_KEY"
ENV_MODEL = "OPENAI_COMPAT_SEARCH_MODEL"

# Default request timeout.  Search-augmented models can take 10–30s end-to-end
# because they issue upstream web requests of their own.
_REQUEST_TIMEOUT = 60.0


class OpenAICompatibleSearchProvider(WebSearchProvider):
    """Search via any OpenAI-compatible ``/chat/completions`` endpoint.

    Requires ``OPENAI_COMPAT_SEARCH_BASE_URL``, ``OPENAI_COMPAT_SEARCH_API_KEY``,
    and ``OPENAI_COMPAT_SEARCH_MODEL`` to be set.  No extract capability —
    pair with Firecrawl/Tavily/Exa/Parallel when you also need ``web_extract``.
    """

    def provider_name(self) -> str:
        return "openai-compatible-search"

    def is_configured(self) -> bool:
        """Return True when base URL, API key, and model are all set.

        Called at tool-registration time; must not perform network I/O.
        """
        return all(
            bool(os.getenv(var, "").strip())
            for var in (ENV_BASE_URL, ENV_API_KEY, ENV_MODEL)
        )

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a search-augmented chat completion and normalize the results.

        Returns ``{"success": True, "data": {"web": [...]}}`` on success or
        ``{"success": False, "error": str}`` on any failure (missing config,
        network error, non-2xx, malformed JSON).
        """
        import httpx

        base_url = os.getenv(ENV_BASE_URL, "").strip().rstrip("/")
        api_key = os.getenv(ENV_API_KEY, "").strip()
        model = os.getenv(ENV_MODEL, "").strip()

        missing = [
            var
            for var, val in (
                (ENV_BASE_URL, base_url),
                (ENV_API_KEY, api_key),
                (ENV_MODEL, model),
            )
            if not val
        ]
        if missing:
            return {
                "success": False,
                "error": f"{', '.join(missing)} not set",
            }

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": query}],
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            resp = httpx.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("OpenAI-compatible search HTTP error: %s", exc)
            return {
                "success": False,
                "error": f"Search endpoint returned HTTP {exc.response.status_code}",
            }
        except httpx.RequestError as exc:
            logger.warning("OpenAI-compatible search request error: %s", exc)
            return {
                "success": False,
                "error": f"Could not reach {base_url}: {exc}",
            }

        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenAI-compatible search JSON parse error: %s", exc)
            return {
                "success": False,
                "error": "Could not parse search endpoint response as JSON",
            }

        web_results = _parse_results(data, limit)

        logger.info(
            "OpenAI-compatible search '%s' via %s: %d results (limit %d)",
            query,
            model,
            len(web_results),
            limit,
        )
        return {"success": True, "data": {"web": web_results}}


def _parse_results(data: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    """Extract normalized web results from a chat-completions response.

    Priority:
    1. ``search_results`` (Perplexity native schema)
    2. ``citations`` (list of URLs or dicts)
    3. ``choices[0].message.content`` as a single "answer" row

    Extracted as a free function so tests can exercise every branch without
    spinning up a network mock.
    """
    # 1. Native Perplexity search_results
    search_results = data.get("search_results") or []
    if isinstance(search_results, list) and search_results:
        out: List[Dict[str, Any]] = []
        for i, sr in enumerate(search_results[:limit]):
            if not isinstance(sr, dict):
                continue
            out.append(
                {
                    "title": str(sr.get("title", "")),
                    "url": str(sr.get("url", "")),
                    "description": str(sr.get("snippet") or sr.get("content") or ""),
                    "position": i + 1,
                }
            )
        if out:
            return out

    # 2. Citations list (URLs or dicts)
    citations = data.get("citations") or []
    if isinstance(citations, list) and citations:
        out = []
        for i, cite in enumerate(citations[:limit]):
            if isinstance(cite, str):
                out.append(
                    {
                        "title": "",
                        "url": cite,
                        "description": "",
                        "position": i + 1,
                    }
                )
            elif isinstance(cite, dict):
                out.append(
                    {
                        "title": str(cite.get("title", "")),
                        "url": str(cite.get("url", "")),
                        "description": str(
                            cite.get("snippet") or cite.get("content") or ""
                        ),
                        "position": i + 1,
                    }
                )
        if out:
            return out

    # 3. Fall back to the answer text itself
    choices = data.get("choices") or []
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") or {}
        content = str(message.get("content") or "")
        if content:
            return [
                {
                    "title": "Search Answer",
                    "url": "",
                    "description": content[:2000],
                    "position": 1,
                }
            ]

    return []
