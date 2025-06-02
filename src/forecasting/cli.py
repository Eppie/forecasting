import typer
from rich.console import Console
from rich.table import Table

from forecasting.db.models import create_db_and_tables
from forecasting.ingestion.chunker import NaiveTextChunker
from forecasting.ingestion.fetchers import RssFetcher, SecFilingsFetcher
from forecasting.ingestion.parser import DocumentParser
from forecasting.ingestion.pipeline import IngestionPipeline
from main import embedder, run_forecast_pipeline, vector_store

app = typer.Typer()
console = Console()


@app.command()
def ingest_data(
    run_once: bool = typer.Option(True, "--once", help="Run ingestion once and exit."),
    # Add fetcher_name option later if you want to run specific fetchers
):
    """
    Run the data ingestion pipeline to fetch, process, and store documents.
    """
    if (
        not run_once
    ):  # For now, only --once is implemented directly. Scheduler is separate.
        console.print(
            "Scheduled ingestion not yet fully implemented via this command. Use --once."
        )
        return

    console.print("Starting data ingestion pipeline (run once)...")

    # Initialize components for the pipeline
    # You might want to make RSS URLs configurable via settings.py
    # For example: settings.RSS_NEWS_URLS (a list of strings)
    rss_news_urls_str = [
        "http://feeds.reuters.com/Reuters/businessNews",  # Example Reuters Business
        "https://feeds.a.dj.com/rss/RSSWorldNews.xml",  # Example WSJ World News
        # Add more relevant, high-quality news RSS feeds
    ]
    # Convert string URLs to HttpUrl if your RssFetcher expects them
    from pydantic import HttpUrl as PydanticHttpUrl  # Alias to avoid confusion

    rss_http_urls = [PydanticHttpUrl(url) for url in rss_news_urls_str]

    fetchers = [
        SecFilingsFetcher(),
        RssFetcher(feed_urls=rss_http_urls),
    ]
    parser = DocumentParser()
    # Adjust chunker_params if settings.CHUNK_SIZE implies tokens
    chunker = NaiveTextChunker(chunk_size_chars=1000, chunk_overlap_chars=150)

    # The embedder and vector_store are already initialized globally in main.py
    # and imported here.

    pipeline = IngestionPipeline(
        fetchers=fetchers,
        parser=parser,
        chunker=chunker,
        embedder=embedder,  # Using the global instance from main.py
        vector_store=vector_store,  # Using the global instance from main.py
    )

    try:
        pipeline.run_all()
        console.print("[green]Data ingestion pipeline completed successfully.[/green]")
    except Exception as e:
        console.print(f"[bold red]Error during data ingestion:[/bold red] {e}")
        import traceback

        traceback.print_exc()  # Print full traceback for debugging


@app.command()
def forecast(
    question: str = typer.Argument(..., help="The forecasting question to ask.")
):
    """
    Ask a forecasting question and get a probabilistic answer with citations.
    """
    console.print(f'Processing question: "{question}"...')
    try:
        result = run_forecast_pipeline(question)
        print(result)
        if result.probability_float is not None:
            console.print(
                f"\n[bold green]Forecast Probability:[/bold green] {result.probability_float:.2%}"
            )
        else:
            console.print(
                "\n[bold green]Forecast Probability:[/bold green] Not explicitly found in LLM output."
            )
        console.print("\n[bold blue]Rationale:[/bold blue]")
        console.print(result.rationale)

        if result.citations:
            console.print("\n[bold yellow]Citations:[/bold yellow]")
            table = Table(
                "Label", "Source Snippet (first 50 chars)", "Full Source (if available)"
            )
            for label, (snippet, full_source_ref) in result.citations.items():
                table.add_row(label, snippet[:50] + "...", full_source_ref or "N/A")
            console.print(table)
        console.print(
            f"\n[dim]Forecast ID {result.forecast_id} logged to database.[/dim]"
        )

    except Exception as e:
        console.print(f"[bold red]Error processing forecast:[/bold red] {e}")
        # Add more detailed error log here if needed


@app.command()
def init_db():
    """Initialize the database and create tables."""
    try:
        create_db_and_tables()
        console.print("[green]Database initialized successfully.[/green]")
    except Exception as e:
        console.print(f"[bold red]Error initializing database:[/bold red] {e}")


if __name__ == "__main__":
    app()
