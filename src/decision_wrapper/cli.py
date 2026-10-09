from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from .backend import create_backend
from .benchmark.report import summarize, to_markdown, write_csv
from .benchmark.runner import RunConfig, load_dataset, run_benchmark
from .engine import decide
from .schema import DecisionRequest, DecisionResponse

app = typer.Typer(add_completion=False, help="Jev-style decision API wrapper on top of the Copilot CLI.")
console = Console()


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8099,
    reload: bool = False,
    warm: bool = typer.Option(False, "--warm", help="Keep a Copilot CLI process warm across requests."),
) -> None:
    """Run the FastAPI decision server."""
    import uvicorn

    if warm:
        os.environ["DECISION_WRAPPER_WARM"] = "1"

    uvicorn.run("decision_wrapper.app:app", host=host, port=port, reload=reload)


@app.command()
def decide_cmd(
    request_file: Path = typer.Argument(..., help="JSON file matching the DecisionRequest schema."),
    samples: int = typer.Option(3, help="Independent CLI samples per question for calibration."),
    concurrency: int = typer.Option(4, help="Max concurrent `copilot` subprocesses."),
    warm: bool = typer.Option(False, "--warm", help="Reuse a Copilot CLI process with fresh sample sessions."),
) -> None:
    """Evaluate a single DecisionRequest JSON file and print the response."""
    payload = json.loads(request_file.read_text())
    request = DecisionRequest.model_validate(payload)

    async def run() -> DecisionResponse:
        async with create_backend(warm=warm, max_concurrency=concurrency) as backend:
            return await decide(request, backend, samples=samples)

    response = asyncio.run(run())
    console.print_json(response.model_dump_json())


app.command(name="decide")(decide_cmd)


@app.command()
def benchmark(
    dataset: Path = typer.Argument(..., help="JSONL dataset: {state, questions, expected} per line."),
    models: str = typer.Option(
        "gpt-5.4-mini,gpt-6-luna,mai-code-1.1-flash", help="Comma-separated model IDs."
    ),
    reasoning_efforts: str = typer.Option(
        "low,medium,high", help="Comma-separated reasoning-effort levels to sweep."
    ),
    samples: int = typer.Option(3, help="Independent CLI samples per question."),
    concurrency: int = typer.Option(4, help="Max concurrent `copilot` subprocesses."),
    timeout_s: float = typer.Option(90.0, help="Per-call timeout in seconds."),
    out_dir: Optional[Path] = typer.Option(None, help="Directory to write report.csv/report.md."),
    warm: bool = typer.Option(False, "--warm", help="Reuse a Copilot CLI process across the benchmark."),
) -> None:
    """Sweep models x reasoning-effort modes over a labeled dataset and
    report accuracy, latency, and premium-request cost per combination."""

    items = load_dataset(dataset)
    config = RunConfig(
        models=[m.strip() for m in models.split(",") if m.strip()],
        reasoning_efforts=[e.strip() for e in reasoning_efforts.split(",") if e.strip()],  # type: ignore[arg-type]
        samples=samples,
        max_concurrency=concurrency,
        timeout_s=timeout_s,
        warm=warm,
    )
    console.print(
        f"Running {len(items)} item(s) x {len(config.models)} model(s) x "
        f"{len(config.reasoning_efforts)} reasoning-effort mode(s), "
        f"{samples} sample(s)/question..."
    )
    results = asyncio.run(run_benchmark(items, config))
    rows = summarize(results)
    if any(row["total_ai_credits"] is None for row in rows):
        console.print("[yellow]AI-credit accounting is unavailable for some calls; costs are unknown, not zero.[/yellow]")

    table = Table(title="Decision benchmark")
    for col in rows[0].keys() if rows else []:
        table.add_column(col)
    for row in rows:
        table.add_row(*("unknown" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v) for v in row.values()))
    console.print(table)

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(rows, out_dir / "report.csv")
        (out_dir / "report.md").write_text(to_markdown(rows))
        console.print(f"Wrote {out_dir / 'report.csv'} and {out_dir / 'report.md'}")


@app.command()
def fetch_rewardbench(
    out: Path = typer.Option(
        Path("datasets/rewardbench_sample.jsonl"), help="Output JSONL path."
    ),
    n: int = typer.Option(30, help="Stratified sample size across RewardBench's 23 subsets."),
    seed: int = typer.Option(42, help="Random seed for sampling and position randomization."),
) -> None:
    """Download RewardBench (allenai/reward-bench) and convert a stratified
    sample into this repo's dataset format -- one of the actual published
    Jev evaluation benchmarks, not a toy dataset."""
    from .benchmark.sources.rewardbench import build_sample

    items = build_sample(out, n=n, seed=seed)
    console.print(f"Wrote {len(items)} RewardBench item(s) to {out}")


if __name__ == "__main__":
    app()
