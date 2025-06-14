from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

requests_stub = ModuleType("requests")
requests_stub.get = lambda *args, **kwargs: None  # type: ignore[attr-defined]
sys.modules.setdefault("requests", requests_stub)

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.base_rates import (  # noqa: E402
    BaseRate,
    ReferenceClassItem,
    fetch_docs,
    gather_documents_for_reference,
    generate_queries,
    get_base_rates,
    run_brave_search,
)


def test_run_brave_search_uses_cache() -> None:
    cached = json.dumps({"results": {"items": [{"url": "u1"}, {"url": "u2"}]}})
    with (
        patch("src.base_rates._cache_get", return_value=cached) as cache_get,
        patch("src.base_rates.requests.get") as req_get,
        patch("src.base_rates._cache_set") as cache_set,
    ):
        result = run_brave_search("q", api_key="k")

    cache_get.assert_called_once_with("brave_cache", "q")
    req_get.assert_not_called()
    cache_set.assert_not_called()
    assert result == ["u1", "u2"]


def test_run_brave_search_http_call() -> None:
    resp_data = {"results": {"items": [{"url": "u1"}, {"link": "u2"}, {"url": "u3"}]}}
    resp = SimpleNamespace(status_code=200, text="", json=lambda: resp_data)
    with (
        patch("src.base_rates._cache_get", return_value=None),
        patch("src.base_rates.requests.get", return_value=resp) as req_get,
        patch("src.base_rates._cache_set") as cache_set,
    ):
        result = run_brave_search("q", api_key="k", top_k=2)

    req_get.assert_called_once()
    cache_set.assert_called_once()
    assert result == ["u1", "u2"]


def test_run_brave_search_requires_key() -> None:
    with patch("src.base_rates._cache_get", return_value=None), patch("src.base_rates.os.getenv", return_value=None):
        with pytest.raises(ValueError):
            run_brave_search("q")


def test_fetch_docs() -> None:
    urls = ["https://a", "https://b"]
    resp1 = SimpleNamespace(status_code=200, text="doc1")
    resp2 = SimpleNamespace(status_code=404, text="bad")
    with (
        patch("src.base_rates._cache_get", return_value=None),
        patch("src.base_rates.requests.get", side_effect=[resp1, resp2]) as req_get,
        patch("src.base_rates._cache_set") as cache_set,
    ):
        result = fetch_docs(urls, api_key="token")

    assert result == ["doc1", ""]
    assert req_get.call_count == 2  # noqa: PLR2004
    assert cache_set.call_count == 2  # noqa: PLR2004


def test_generate_queries_dedup() -> None:
    item = ReferenceClassItem(reasoning="r", reference_class="rc")
    data = {"queries": [" a ", "b", "a"]}
    with patch("src.base_rates._call_llm_json", return_value=data) as call:
        result = generate_queries("q", item, n_queries=5)
    call.assert_called_once()
    assert result == ["a", "b"]


def test_gather_documents_for_reference_dedup() -> None:
    item = ReferenceClassItem(reasoning="r", reference_class="rc")
    with (
        patch("src.base_rates.generate_queries", return_value=["q1", "q2"]) as gen,
        patch("src.base_rates.run_brave_search", side_effect=[["u1", "u2"], ["u2", "u3"]]) as run,
        patch("src.base_rates.fetch_docs", return_value=["d1", "d2", "d3"]) as fetch,
    ):
        docs = gather_documents_for_reference("cq", item)

    gen.assert_called_once_with("cq", item)
    assert run.call_count == 2  # noqa: PLR2004
    fetch.assert_called_once_with(["u1", "u2", "u3"], api_key=None)
    assert docs == ["d1", "d2", "d3"]


def test_get_base_rates() -> None:
    items = [
        ReferenceClassItem(reasoning="r1", reference_class="rc1"),
        ReferenceClassItem(reasoning="r2", reference_class="rc2"),
    ]
    responses = [
        {"reasoning": "x1", "frequency": 0.5},
        {"reasoning": "x2", "lambda": 0.7, "quality_score": 0.8},
    ]

    def fake_call(*args: object, **kwargs: object) -> dict[str, object]:
        return responses.pop(0)

    with patch("src.base_rates._call_llm_json", side_effect=fake_call) as call:
        rates = get_base_rates("cq", items)

    assert call.call_count == 2  # noqa: PLR2004
    assert isinstance(rates[0], BaseRate)
    assert rates[0].reference_class == "rc1"
    assert rates[0].frequency == 0.5  # noqa: PLR2004
    assert rates[1].lambda_ == 0.7  # noqa: PLR2004
    assert rates[1].quality_score == 0.8  # noqa: PLR2004
