"""
base_rates.py
~~~~~~~~~~~~~

Step 2.2 of the super‑forecasting workflow.

Given a clarified forecasting question **and** the list of candidate
ReferenceClassItem objects returned by Step 2.1, this module asks an LLM to
retrieve / infer the historical counts or distribution parameters for *each*
reference class and returns a fully‑formed BaseRate for every candidate.

External web‑search (Brave) and page‑fetch (Jina) APIs can later be wired into
`search_brave` / `fetch_webpage`.  For now, we rely on the LLM’s embedded
knowledge plus citations it chooses to surface.

The code avoids asyncio for simplicity, per user instruction.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from pprint import pprint
from typing import Any

import ollama
import requests

from .models import BaseRate, ReferenceClassItem

logger = logging.getLogger(__name__)

_CACHE_PATH = Path(__file__).with_suffix(".cache.sqlite")
_conn = sqlite3.connect(_CACHE_PATH)
_conn.execute("CREATE TABLE IF NOT EXISTS brave_cache (query TEXT PRIMARY KEY, json TEXT)")
_conn.execute("CREATE TABLE IF NOT EXISTS doc_cache (url TEXT PRIMARY KEY, content TEXT)")
_conn.commit()

# Common constants
_HTTP_OK = 200
_MAX_URL_LINE_LENGTH = 120  # for lint reference


def _cache_get(table: str, key: str) -> str | None:
    cur = _conn.execute(
        (
            "SELECT "
            f"{'json' if table == 'brave_cache' else 'content'} "
            f"FROM {table} WHERE "
            f"{'query' if table == 'brave_cache' else 'url'} = ?"
        ),
        (key,),
    )
    row = cur.fetchone()
    return row[0] if row else None


def _cache_set(table: str, key: str, value: str) -> None:
    _conn.execute(
        f"INSERT OR REPLACE INTO {table} VALUES (?, ?)",
        (key, value),
    )
    _conn.commit()


def run_brave_search(query: str, *, api_key: str | None = None, top_k: int = 3) -> list[str]:
    """
    Run a web search via Brave Search API and return up to *top_k* result URLs.

    Documentation: https://brave.com/search/api/
    Endpoint:      GET https://api.search.brave.com/res/v1/web/search?q=<query>

    Raises
    ------
    RuntimeError if the API response is not HTTP 200.
    """
    cached = _cache_get("brave_cache", query)
    if cached:
        try:
            data = json.loads(cached)
            # reuse existing parsing logic but skip network
            results = data.get("results", {}).get("items", [])
            if not results and "web" in data:
                results = data["web"].get("results", [])
            cached_urls = []
            for item in results:
                link = item.get("url") or item.get("link")
                if link:
                    cached_urls.append(link)
                if len(cached_urls) >= top_k:
                    break
            logger.debug("Cache hit!")
            return cached_urls
        except Exception:
            pass  # fall through on cache parse errors

    api_key = api_key or os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        raise ValueError("Brave API key not provided and BRAVE_SEARCH_API_KEY env var not set")

    url = "https://api.search.brave.com/res/v1/web/search"
    params = {"q": query, "count": str(top_k)}
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key,
        "User-Agent": "forecasting-agent/0.1",
    }
    r = requests.get(url, params=params, headers=headers, timeout=15)
    if r.status_code != _HTTP_OK:  # noqa: PLR2004
        raise RuntimeError(f"Brave API error {r.status_code}: {r.text[:200]}")
    data = r.json()
    _cache_set("brave_cache", query, json.dumps(data))
    results = data.get("results", {}).get("items", [])  # brave nests results.web.items for v1
    if not results and "web" in data:
        results = data["web"].get("results", [])
    urls: list[str] = []
    for item in results:
        link = item.get("url") or item.get("link")
        if link:
            urls.append(link)
        if len(urls) >= top_k:
            break
    return urls


def fetch_docs(urls: list[str], *, api_key: str | None = None) -> list[str]:
    """
    Fetch plain‑text article bodies for each URL using Jina AI Reader.

    Jina Reader endpoint: GET https://r.jina.ai/http://<url>

    The service is unauthenticated; however, if JINA_API_KEY is set, we attach
    it via the `Authorization: Bearer` header which the enterprise tier expects.

    Returns
    -------
    List of plain‑text strings in the same order as *urls* (404s yield "").
    """
    api_key = api_key or os.getenv("JINA_API_KEY")
    headers = {"User-Agent": "forecasting-agent/0.1"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    texts: list[str] = []
    for u in urls:
        logger.debug(f"Fetching {u}")
        cached_text = _cache_get("doc_cache", u)
        if cached_text is not None:
            texts.append(cached_text)
            continue
        try:
            # Jina reader wants the full URL appended after /http://
            endpoint = "https://r.jina.ai/http://{}".format(u.lstrip("https://").lstrip("http://"))
            resp = requests.get(endpoint, timeout=20, headers=headers)
            if resp.status_code == _HTTP_OK:  # noqa: PLR2004
                texts.append(resp.text)
                logger.debug(resp.text)
                _cache_set("doc_cache", u, resp.text)
            else:
                texts.append("")
                _cache_set("doc_cache", u, "")
        except requests.RequestException:
            texts.append("")
            _cache_set("doc_cache", u, "")
    return texts


def _call_llm_json(system_prompt: str, user_prompt: str, model: str = "deepseek-r1:8b", retries: int = 3) -> Any:
    """
    Wrapper that retries until we get valid JSON back from the LLM.
    """
    for attempt in range(retries):
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
        )
        content = response.message.content if response and response.message else None
        if not content:
            time.sleep(0.5)
            print("Empty content, trying again...")
            continue
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Ask the model to fix formatting and retry.
            print("Failed to decode JSON, trying again...")
            user_prompt = f"Your previous reply was not valid JSON. Please return ONLY JSON.\n\n{content}"
            time.sleep(0.5)
    raise ValueError("LLM failed to return valid JSON after retries.")


def generate_queries(
    clarified_question: str,
    ref_item: ReferenceClassItem,
    *,
    n_queries: int = 3,
    model: str = "deepseek-r1:8b",
) -> list[str]:
    """
    Ask the LLM to suggest concrete web‑search queries (Brave syntax) that are
    likely to surface statistics for the given reference class.

    Returns up to *n_queries* strings, de‑duplicated and stripped.
    """

    sys_prompt = """
    You are a research assistant skilled at forming precise web‑search queries.
    Given a clarified forecasting question and a reference class, output ONLY
    a JSON array of UP TO 5 search strings (no additional keys, no prose).
    Each string should be targeted to find numerical counts or datasets that
    would let a researcher measure the base rate for that reference class.

    Example:

    Clarified: How many Atlantic Category 5 hurricanes will occur in 2026?
    RefClass : Atlantic Category 5 hurricanes 1924‑2024

    Output:
    { "queries":
    [
      "Atlantic Category 5 hurricanes list 1924 site:noaa.gov",
      "number of category 5 atlantic hurricanes since 1900",
      "NOAA best track dataset category 5 Atlantic"
    ] }
    """

    user_prompt = (
        f"Clarified question: ```{clarified_question}```\n"
        f"Reference class: ```{ref_item}```\n"
        f"Please generate {n_queries} useful search queries."
    )

    data = _call_llm_json(sys_prompt, user_prompt, model=model)
    queries: list[str] = []
    for q in data["queries"]:
        if isinstance(q, str) and q.strip():
            queries.append(q.strip())
    return list(dict.fromkeys(queries))  # preserve order, drop dups


def gather_documents_for_reference(  # noqa: PLR0913
    clarified_question: str,
    ref_item: ReferenceClassItem,
    *,
    brave_api_key: str | None = None,
    jina_api_key: str | None = None,
    search_top_k: int = 10,
    verbose: bool = False,
) -> list[str]:
    """
    1. Generate search queries via LLM.
    2. Run Brave search for each query and collect URLs.
    3. Fetch article bodies via Jina reader.
    4. Return list of documents (plain text).
    """

    queries = generate_queries(clarified_question, ref_item)
    logger.info("Generated queries: %s", queries)

    all_urls: list[str] = []
    for q in queries:
        try:
            urls = run_brave_search(q, api_key=brave_api_key, top_k=search_top_k)
            all_urls.extend(urls)
        except Exception as exc:
            logger.debug("Brave search failed for query '%s': %s", q, exc)

    pprint(all_urls)
    # Deduplicate while preserving order
    seen = set()
    unique_urls: list[str] = []
    for u in all_urls:
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)
    logger.debug("Fetched %s unique URLs.", len(unique_urls))

    docs = fetch_docs(unique_urls, api_key=jina_api_key)
    logger.debug(
        "Retrieved %s/%s documents with non-empty content.",
        sum(bool(d) for d in docs),
        len(docs),
    )
    return docs


def get_base_rates(
    clarified_question: str,
    ref_classes: list[ReferenceClassItem],
    *,
    verbose: bool = False,
) -> list[BaseRate]:
    """
    Implements Step 2.2 — measure the historical base‑rate for each candidate
    reference class using an LLM.

    The function does **one LLM call per reference class**. Each prompt asks
    the model to provide either:

      • numerator / denominator / frequency (binary or proportion) OR
      • lambda                                (Poisson count) OR
      • distribution object {median, p5, p95} (continuous variable)

    The result is parsed into a BaseRate dataclass list.

    Args
    ----
    clarified_question:
        The fully clarified forecasting question (Step 1 output).
    ref_classes:
        Candidate reference classes from Step 2.1.
    verbose:
        If *True*, pretty‑print each BaseRate.

    Returns
    -------
    list[BaseRate]
    """

    system_prompt_template = """
    You are an expert super‑forecaster performing **Step 2.2 – Measure the base rate**.
    For the reference class below, return **ONLY valid JSON** with:

    • reasoning        – 2‑4 sentences on how you located / computed the numbers
                         and why they are reliable. Include source titles (no URLs).
    • numerator        – int   (omit if not a proportion)
    • denominator      – int   (omit if not a proportion)
    • frequency        – float (0–1)  (omit if not a proportion)
    • lambda           – float (Poisson annual rate, omit if N/A)
    • distribution     – object { "median": float, "p5": float, "p95": float }
                         (omit if not continuous)
    • quality_score    – float 0–1 heuristic confidence

    ### Few‑shot examples

    1. Binary – Trump assassination
    {{
      "reasoning": "Counted 4 assassinations among 46 completed U.S. presidencies (Lincoln, Garfield, McKinley, Kennedy) using Wikipedia lists; gives an 8.7 % frequency.",
      "numerator": 4,
      "denominator": 46,
      "frequency": 0.087,
      "quality_score": 0.9
    }}

    2. Continuous – S&P 500 year‑end
    {{
      "reasoning": "Pulled year‑end closes 1974‑2023 from multpl.com dataset, computed median 4570, p5 = 1600, p95 = 7900.",
      "distribution": {{ "median": 4570, "p5": 1600, "p95": 7900 }},
      "quality_score": 0.85
    }}

    3. Count – Cat‑5 hurricanes
    {{
      "reasoning": "NOAA Best Track data list 42 Category‑5 Atlantic storms in 101 seasons 1924‑2024 ⇒ λ ≈ 0.42 per season.",
      "numerator": 42,
      "denominator": 101,
      "frequency": 0.416,
      "lambda": 0.42,
      "quality_score": 0.8
    }}

    ### Instructions
    1. Think step‑by‑step internally but output **only** JSON.
    2. Provide whichever fields are appropriate; omit the others (do NOT output null).
    """

    base_rates: list[BaseRate] = []

    for ref_item in ref_classes:
        user_prompt = (
            f"Clarified question: ```{clarified_question}```\n\n"
            f"Reference class description: ```{ref_item.reference_class}```\n\n"
            f"Please measure the base rate for this class."
        )

        data = _call_llm_json(system_prompt_template, user_prompt)

        base_rate = BaseRate(
            reference_class=ref_item.reference_class,
            reasoning=data.get("reasoning", ""),
            numerator=data.get("numerator"),
            denominator=data.get("denominator"),
            frequency=data.get("frequency"),
            distribution=data.get("distribution"),
            lambda_=data.get("lambda"),
            quality_score=data.get("quality_score"),
        )
        base_rates.append(base_rate)
        logger.debug("%s", base_rate)
        logger.debug("-" * 80)

    return base_rates


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    reference_classes = [
        ReferenceClassItem(
            reasoning="The target event is about forecasting OpenAI's "
            "annual revenue in a specific future year. A "
            "good reference class would be all public "
            "companies that have experienced similar growth "
            "stages or market conditions to capture "
            "analogous financial development patterns.",
            reference_class="Annual revenues of AI-focused technology companies during their high-growth phase",
        ),
        ReferenceClassItem(
            reasoning="To account for the broader economic context, we "
            "can consider major tech company annual revenues "
            "over time as they reflect industry-wide trends "
            "and cyclical factors that might influence "
            "OpenAI's performance.",
            reference_class="Annual revenues of leading tech companies (e.g., Google, Microsoft) from 2015 to present",
        ),
        ReferenceClassItem(
            reasoning="Since the target is revenue growth in a "
            "specific fiscal year, looking at historical "
            "data for AI-related businesses or startups that "
            "have reached significant scale can provide "
            "insights into scaling patterns.",
            reference_class="Annual revenues of companies with similar "
            "market capitalization trajectory to "
            "OpenAI over the last decade",
        ),
    ]

    all_queries: list[str] = []
    for ref_item in reference_classes:
        docs = gather_documents_for_reference(
            clarified_question="What will OpenAI's revenue be for 2027?", ref_item=ref_item
        )
        logger.debug(docs)
