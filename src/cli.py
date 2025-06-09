from pprint import pprint

import typer  # type: ignore
from workflow import run_workflow

app = typer.Typer(help="CLI for forecasting questions")


@app.command()
def forecast(
    question: str,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Pose a forecasting question"""
    if verbose:
        typer.echo(f"Forecasting question: {question}")

    result = run_workflow(question)
    pprint(result)


if __name__ == "__main__":
    app()
