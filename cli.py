"""
ForecastGuard Command Line Interface.
"""

import sys
import json
from pathlib import Path
import yaml
import typer
import dataclasses
from rich.console import Console
from rich.table import Table

from data.m5_loader import load_m5, generate_synthetic_m5
from profiler.profiler import SeriesDNA
from profiler.router import route

app = typer.Typer(name="forecastguard")
console = Console()


def _get_checkmark() -> str:
    """Returns a green checkmark if supported by stdout, otherwise 'v'."""
    try:
        "✔".encode(sys.stdout.encoding or "utf-8")
        return "✔"
    except Exception:
        return "v"


@app.command()
def profile(
    config: Path = typer.Option(
        Path("config/fg.yaml"),
        "--config",
        "-c",
        help="Path to the fg.yaml configuration file."
    )
) -> None:
    """
    Profiles demand series, routes forecasting models and metrics, and outputs results.
    """
    # 1. Load config from YAML
    if not config.exists():
        console.print(f"[red]Error:[/red] Config file '{config}' not found.")
        raise typer.Exit(code=1)

    try:
        with open(config, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to parse config file '{config}': {e}")
        raise typer.Exit(code=1)

    dataset_path_str = cfg.get("dataset_path", "data/sample_m5.csv") if cfg else "data/sample_m5.csv"
    dataset_path = Path(dataset_path_str)

    # 2. Load data
    skus = {}
    if dataset_path.exists():
        try:
            skus = load_m5(str(dataset_path))
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Failed to load data from '{dataset_path}': {e}")
            console.print("[yellow]Falling back to synthetic data generation...[/yellow]")
            skus = generate_synthetic_m5()
    else:
        console.print(f"[yellow]Warning:[/yellow] Dataset file '{dataset_path}' not found.")
        console.print("[yellow]Generating synthetic data for profiling...[/yellow]")
        skus = generate_synthetic_m5()

    # 3. Profile & Route each SKU
    dna_calculator = SeriesDNA()
    results = {}
    
    intermittent_count = 0
    high_cv_count = 0
    stable_count = 0

    table = Table(title="Demand Profiler Results")
    table.add_column("SKU ID")
    table.add_column("CV")
    table.add_column("IR")
    table.add_column("Trend")
    table.add_column("Seasonality")
    table.add_column("Mode")
    table.add_column("Primary Metric")
    table.add_column("Recommended Models")

    for sku_id, series in skus.items():
        try:
            dna = dna_calculator.compute(sku_id, series)
            routing = route(dna)
            
            # Save combined metrics
            results[sku_id] = {
                "dna": dna,
                "routing": dataclasses.asdict(routing)
            }
            
            # Set row color based on cv_mode
            mode = routing.cv_mode
            if mode == "intermittent":
                row_style = "yellow"
                intermittent_count += 1
            elif mode == "high_cv":
                row_style = "red"
                high_cv_count += 1
            else:
                row_style = "green"
                stable_count += 1

            # Format recommended models list
            models_str = ", ".join(routing.recommended_models)

            table.add_row(
                sku_id,
                f"{dna['cv']:.4f}",
                f"{dna['intermittency_ratio']:.4f}",
                f"{dna['trend_strength']:.4f}",
                f"{dna['seasonality_index']:.4f}",
                mode,
                routing.primary_metric,
                models_str,
                style=row_style
            )
        except Exception as e:
            console.print(f"[red]Error profiling SKU {sku_id}: {e}[/red]")

    # Print the table
    console.print(table)

    # 4. Print summary
    total_skus = len(skus)
    console.print(
        f"\nProfiled [bold]{total_skus}[/bold] SKUs — "
        f"[yellow]{intermittent_count} intermittent[/yellow], "
        f"[red]{high_cv_count} high-CV[/red], "
        f"[green]{stable_count} stable[/green]"
    )

    # 5. Save results to JSON
    output_path = Path("profiler_output.json")
    try:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"\n[green]{_get_checkmark()}[/green] Results saved to [bold]{output_path}[/bold]")
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to write output JSON to {output_path}: {e}")


@app.command()
def version() -> None:
    """
    Prints the version of ForecastGuard.
    """
    console.print("ForecastGuard Version 1.0.0")


if __name__ == "__main__":
    app()
