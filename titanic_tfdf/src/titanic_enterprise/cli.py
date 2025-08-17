"""Command-line interface for the Titanic Enterprise ML pipeline."""

import click
from rich.console import Console
from rich.table import Table

from titanic_enterprise import __version__
from titanic_enterprise.utils.config import load_config
from titanic_enterprise.utils.logging import setup_logging

console = Console()


@click.group()
@click.version_option(version=__version__)
@click.option(
    "--config-dir",
    default=None,
    help="Directory containing configuration files",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Logging level",
)
@click.pass_context
def cli(ctx, config_dir, log_level):
    """Titanic Enterprise ML Pipeline CLI."""
    ctx.ensure_object(dict)
    ctx.obj["config_dir"] = config_dir
    ctx.obj["log_level"] = log_level
    
    # Setup logging
    setup_logging(level=log_level)
    
    console.print(f"[bold blue]Titanic Enterprise ML Pipeline v{__version__}[/bold blue]")


@cli.command()
@click.option(
    "--config-name",
    default="config",
    help="Name of the configuration file",
)
@click.option(
    "--overrides",
    multiple=True,
    help="Configuration overrides in Hydra format",
)
@click.pass_context
def train(ctx, config_name, overrides):
    """Train the ML model."""
    console.print("[bold green]Starting model training...[/bold green]")
    
    try:
        # Load configuration
        cfg = load_config(
            config_name=config_name,
            config_dir=ctx.obj["config_dir"],
            overrides=list(overrides),
        )
        
        console.print(f"Loaded configuration: {config_name}")
        console.print(f"Experiment: {cfg.experiment.name}")
        console.print(f"Model: {cfg.model.name}")
        
        # TODO: Implement training logic in future tasks
        console.print("[yellow]Training logic will be implemented in future tasks[/yellow]")
        
    except Exception as e:
        console.print(f"[bold red]Training failed: {e}[/bold red]")
        raise click.ClickException(str(e))


@cli.command()
@click.option(
    "--config-name",
    default="config",
    help="Name of the configuration file",
)
@click.option(
    "--model-path",
    required=True,
    help="Path to the trained model",
)
@click.option(
    "--input-file",
    required=True,
    help="Path to input data file",
)
@click.option(
    "--output-file",
    default="outputs/predictions.csv",
    help="Path to save predictions",
)
@click.pass_context
def predict(ctx, config_name, model_path, input_file, output_file):
    """Generate predictions using a trained model."""
    console.print("[bold green]Starting prediction...[/bold green]")
    
    try:
        # Load configuration
        cfg = load_config(
            config_name=config_name,
            config_dir=ctx.obj["config_dir"],
        )
        
        console.print(f"Model path: {model_path}")
        console.print(f"Input file: {input_file}")
        console.print(f"Output file: {output_file}")
        
        # TODO: Implement prediction logic in future tasks
        console.print("[yellow]Prediction logic will be implemented in future tasks[/yellow]")
        
    except Exception as e:
        console.print(f"[bold red]Prediction failed: {e}[/bold red]")
        raise click.ClickException(str(e))


@cli.command()
@click.option(
    "--config-name",
    default="config",
    help="Name of the configuration file",
)
@click.option(
    "--stage",
    type=click.Choice(["preprocess", "train", "evaluate", "predict", "all"]),
    default="all",
    help="Pipeline stage to run",
)
@click.pass_context
def pipeline(ctx, config_name, stage):
    """Run the complete ML pipeline."""
    console.print("[bold green]Starting ML pipeline...[/bold green]")
    
    try:
        # Load configuration
        cfg = load_config(
            config_name=config_name,
            config_dir=ctx.obj["config_dir"],
        )
        
        console.print(f"Pipeline stage: {stage}")
        console.print(f"Configuration: {config_name}")
        
        # Create pipeline status table
        table = Table(title="Pipeline Status")
        table.add_column("Stage", style="cyan")
        table.add_column("Status", style="green")
        
        stages = ["preprocess", "train", "evaluate", "predict"] if stage == "all" else [stage]
        
        for pipeline_stage in stages:
            table.add_row(pipeline_stage, "Pending")
        
        console.print(table)
        
        # TODO: Implement pipeline logic in future tasks
        console.print("[yellow]Pipeline logic will be implemented in future tasks[/yellow]")
        
    except Exception as e:
        console.print(f"[bold red]Pipeline failed: {e}[/bold red]")
        raise click.ClickException(str(e))


@cli.command()
def info():
    """Display system and configuration information."""
    console.print("[bold blue]System Information[/bold blue]")
    
    # Create info table
    table = Table()
    table.add_column("Component", style="cyan")
    table.add_column("Version/Status", style="green")
    
    table.add_row("Titanic Enterprise", __version__)
    
    try:
        import tensorflow as tf
        table.add_row("TensorFlow", tf.__version__)
    except ImportError:
        table.add_row("TensorFlow", "Not installed")
    
    try:
        import tensorflow_decision_forests as tfdf
        table.add_row("TF Decision Forests", tfdf.__version__)
    except ImportError:
        table.add_row("TF Decision Forests", "Not installed")
    
    try:
        import mlflow
        table.add_row("MLflow", mlflow.__version__)
    except ImportError:
        table.add_row("MLflow", "Not installed")
    
    console.print(table)


if __name__ == "__main__":
    cli()