"""Command line interface for the forecasting package."""

import logging

import typer  # type: ignore

from workflow import run_workflow


def _setup_logging(verbose: bool) -> None:
    """Configure basic logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level)


app = typer.Typer(help="CLI for forecasting questions")


@app.command()
def forecast(
    question: str,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Pose a forecasting question."""
    _setup_logging(verbose)
    logger = logging.getLogger(__name__)
    logger.debug("Forecasting question: %s", question)

    result = run_workflow(question, verbose)
    typer.echo(str(result))


if __name__ == "__main__":
    app()
