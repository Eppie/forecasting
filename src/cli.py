import typer

app = typer.Typer(help="CLI for forecasting questions")

@app.command()
def forecast(
    question: str,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Pose a forecasting question"""
    if verbose:
        typer.echo(f"Forecasting question: {question}")
    else:
        typer.echo("Forecasting: " + question)
    # TODO: integrate forecasting backend


if __name__ == "__main__":
    app()
