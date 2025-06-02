# src/forecasting/ingestion/fetchers.py
from abc import ABC, abstractmethod

import feedparser
import requests
from pydantic import HttpUrl

from forecasting.ingestion.models import RawDocument


class BaseFetcher(ABC):
    @abstractmethod
    def fetch_latest(self, limit: int = 5) -> list[RawDocument]:
        """Fetches the latest 'limit' documents from the source."""
        pass


class SecFilingsFetcher(BaseFetcher):
    # SEC EDGAR requires a User-Agent string.
    # Format: "Sample Company Name AdminContact@example.com"
    USER_AGENT = "ForecastingApp EDOps@example.com"  # Replace with your app name/email

    def __init__(self, user_agent: str = USER_AGENT):
        self.user_agent = user_agent
        # Base URL for recent filings submissions (JSON format)
        # See https://www.sec.gov/os/accessing-edgar-data
        self.submissions_url = "https://data.sec.gov/submissions/"
        # For specific filing types, you might query their RSS feeds or specific APIs if available
        # For simplicity, we'll fetch recent general submissions and filter later,
        # or you can target specific RSS feeds for 10-K, 10-Q etc.
        # Example: 10-K RSS: https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=10-K&count=10&output=atom

    def fetch_latest(
        self, limit: int = 5, form_type: str = "10-K"
    ) -> list[RawDocument]:
        """
        Fetches latest filings. This is a simplified example.
        A more robust implementation would:
        1. Get a list of recent CIKs or use an RSS feed for specific form types.
        2. For each CIK, get their recent filings.
        3. Download the primary document for each filing.
        For now, let's simulate fetching a few known 10-K HTML documents by constructing URLs.
        This is highly simplified and not robust for production.
        A better way: use the SEC's RSS feeds or specific APIs.
        Example 10-K RSS feed:
        https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&CIK=&type=10-K&owner=include&count=10&output=atom
        """
        print(f"Fetching latest {limit} SEC filings of type {form_type} (simulated)...")
        raw_docs: list[RawDocument] = []

        # Example: Using an RSS feed for 10-Ks
        rss_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type={form_type}&count={limit * 2}&output=atom"  # Fetch more to find unique ones
        headers = {"User-Agent": self.user_agent}

        try:
            feed_response = requests.get(rss_url, headers=headers, timeout=10)
            feed_response.raise_for_status()
            feed = feedparser.parse(feed_response.content)

            for entry in feed.entries:
                if len(raw_docs) >= limit:
                    break

                # Each entry usually has multiple links (filing details, interactive data, etc.)
                # We need the primary document, often an .htm file.
                # The 'filing-href' is usually the landing page for the filing.
                filing_landing_page_url = entry.link

                # To get the actual document (e.g., the 10-K HTML file):
                # You would typically need to parse the filing_landing_page_url to find the link
                # to the primary document (e.g., a file ending in .htm that's not an exhibit).
                # This is complex. For MVP, we'll assume `entry.link` can lead us to content
                # or use a placeholder if direct content isn't easily found.

                # Let's try to find a primary document link (this is heuristic)
                primary_doc_url = None
                if hasattr(entry, "links"):
                    for link_info in entry.links:
                        if (
                            link_info.get("type") == "text/html"
                            and ".htm" in link_info.href
                            and not any(
                                ex in link_info.href.lower()
                                for ex in ["-index.html", "exhibit"]
                            )
                        ):
                            if link_info.href.startswith(
                                "http"
                            ):  # Ensure it's a full URL
                                primary_doc_url = link_info.href
                                break

                if (
                    not primary_doc_url
                ):  # Fallback, or a simpler approach if direct docs are hard
                    # This might be the filing detail page, not the raw doc itself.
                    # For a real system, more sophisticated parsing of this page is needed.
                    # print(f"Could not find direct primary document for {entry.title}, using landing page: {filing_landing_page_url}")
                    # For now, let's skip if we can't find a direct doc to simplify.
                    # Or, as a placeholder:
                    # primary_doc_url = filing_landing_page_url
                    continue

                print(f"Fetching content from: {primary_doc_url}")
                try:
                    doc_response = requests.get(
                        primary_doc_url, headers=headers, timeout=15
                    )
                    doc_response.raise_for_status()
                    content = doc_response.content  # bytes

                    raw_docs.append(
                        RawDocument(
                            source_name=f"SEC EDGAR - {form_type}",
                            identifier=primary_doc_url,  # Use the document URL as identifier
                            content=content,
                            content_type="text/html",  # Assuming it's HTML
                            metadata={
                                "title": entry.title,
                                "published_date": entry.get("published"),
                                "form_type": form_type,
                            },
                        )
                    )
                except requests.RequestException as e:
                    print(f"Failed to fetch content from {primary_doc_url}: {e}")

        except requests.RequestException as e:
            print(f"Failed to fetch SEC RSS feed {rss_url}: {e}")

        print(f"Fetched {len(raw_docs)} SEC documents.")
        return raw_docs


class RssFetcher(BaseFetcher):
    def __init__(self, feed_urls: list[HttpUrl]):
        self.feed_urls = feed_urls

    def fetch_latest(self, limit_per_feed: int = 3) -> list[RawDocument]:
        print(f"Fetching latest {limit_per_feed} items per RSS feed...")
        raw_docs: list[RawDocument] = []
        for url in self.feed_urls:
            print(f"Fetching from RSS feed: {str(url)}")
            try:
                # Some feeds might require a user-agent
                headers = {
                    "User-Agent": SecFilingsFetcher.USER_AGENT
                }  # Borrowing user agent
                feed_data = feedparser.parse(str(url), agent=headers.get("User-Agent"))

                if feed_data.bozo:  # feedparser sets bozo if there's an error parsing
                    print(
                        f"Warning: Error parsing feed {url}. Bozo exception: {feed_data.bozo_exception}"
                    )
                    # continue # Or try to process entries anyway if some are available

                entries_processed = 0
                for entry in feed_data.entries:
                    if entries_processed >= limit_per_feed:
                        break

                    content = ""
                    # Try to get full content, fallback to summary
                    if hasattr(entry, "content") and entry.content:
                        content = (
                            entry.content[0].value
                            if isinstance(entry.content, list)
                            else entry.content.get("value", "")
                        )
                    if not content and hasattr(entry, "summary"):
                        content = entry.summary

                    if not content:
                        print(
                            f"Skipping entry '{entry.get('title', 'Unknown Title')}' from {url} due to missing content."
                        )
                        continue

                    # Ensure identifier is unique, link is good, fallback to id
                    identifier = (
                        entry.link
                        if hasattr(entry, "link") and entry.link
                        else entry.get("id")
                    )
                    if not identifier:
                        print(
                            f"Skipping entry '{entry.get('title', 'Unknown Title')}' from {url} due to missing link/id."
                        )
                        continue

                    raw_docs.append(
                        RawDocument(
                            source_name=(
                                feed_data.feed.title
                                if hasattr(feed_data, "feed")
                                and hasattr(feed_data.feed, "title")
                                else str(url)
                            ),
                            identifier=identifier,
                            content=content,  # This is likely HTML
                            content_type="text/html",  # Assume HTML from RSS feeds
                            metadata={
                                "title": (
                                    entry.title if hasattr(entry, "title") else "N/A"
                                ),
                                "published_date": (
                                    entry.published
                                    if hasattr(entry, "published")
                                    else None
                                ),
                                "link": entry.link if hasattr(entry, "link") else None,
                            },
                        )
                    )
                    entries_processed += 1
            except Exception as e:
                print(f"Error fetching or parsing RSS feed {url}: {e}")
        print(f"Fetched {len(raw_docs)} total documents from RSS feeds.")
        return raw_docs


# You can add more fetchers, e.g., for local files, APIs, etc.
