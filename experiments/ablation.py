"""
experiments/ablation.py

Ablation Study: A1 → A4

Runs all four distillation configurations and saves a comparison table.

A1: TinyPhoBERT — No distillation (CE only)
A2: TinyPhoBERT + Logit KD (CE + α·KL)
A3: TinyPhoBERT + Logit + Hidden KD (CE + α·KL + β·MSE_hidden)
A4: TinyPhoBERT + Full (CE + α·KL + β·MSE_hidden + γ·MSE_att)

Usage:
    python experiments/ablation.py
    python experiments/ablation.py --config configs/distillation_config.yaml
    python experiments/ablation.py --skip_training  # Only aggregate existing results
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_student import train as train_student
from utils.seed import set_seed

console = Console()
RESULTS_DIR = Path("results")
ABLATION_DIR = RESULTS_DIR / "ablation"
PLOTS_DIR = ABLATION_DIR / "plots"

ABLATION_CONFIGS = {
    "A1_no_distill": {
        "name": "TinyPhoBERT\n(No Distill)",
        "use_logit_kd": False,
        "use_hidden_kd": False,
        "use_attention_kd": False,
        "alpha": 0.0,
        "beta": 0.0,
        "gamma": 0.0,
    },
    "A2_logit_kd": {
        "name": "TinyPhoBERT\n+ Logit KD",
        "use_logit_kd": True,
        "use_hidden_kd": False,
        "use_attention_kd": False,
        "alpha": 0.5,
        "beta": 0.0,
        "gamma": 0.0,
    },
    "A3_logit_hidden": {
        "name": "TinyPhoBERT\n+ Logit+Hidden",
        "use_logit_kd": True,
        "use_hidden_kd": True,
        "use_attention_kd": False,
        "alpha": 0.5,
        "beta": 0.1,
        "gamma": 0.0,
    },
    "A4_full": {
        "name": "TinyPhoBERT\n+ Full KD",
        "use_logit_kd": True,
        "use_hidden_kd": True,
        "use_attention_kd": True,
        "alpha": 0.5,
        "beta": 0.1,
        "gamma": 0.1,
    },
}


def run_ablation(base_config: dict, run_id: str, ablation_cfg: dict) -> dict:
    """Run a single ablation configuration and return results."""
    config = copy.deepcopy(base_config)

    # Override distillation settings
    for key in ["use_logit_kd", "use_hidden_kd", "use_attention_kd", "alpha", "beta", "gamma"]:
        if key in ablation_cfg:
            config["distillation"][key] = ablation_cfg[key]

    config["logging"]["run_name"] = run_id
    config["training"]["output_dir"] = f"checkpoints/ablation"

    console.print(f"\n{'='*60}")
    console.print(f"[bold yellow]Running Ablation: {run_id}[/bold yellow]")
    console.print(f"  KD={ablation_cfg['use_logit_kd']} | "
                  f"Hidden={ablation_cfg['use_hidden_kd']} | "
                  f"Att={ablation_cfg['use_attention_kd']}")
    console.print(f"  α={ablation_cfg['alpha']} β={ablation_cfg['beta']} γ={ablation_cfg['gamma']}")
    console.print(f"{'='*60}\n")

    set_seed(base_config["data"]["seed"])
    metrics = train_student(config, run_name=run_id)
    metrics["run_id"] = run_id
    metrics["display_name"] = ablation_cfg["name"]
    return metrics


def plot_ablation_results(results: dict, save_dir: str) -> None:
    """Generate ablation study plots for the paper."""
    os.makedirs(save_dir, exist_ok=True)

    run_ids = list(results.keys())
    display_names = [results[r].get("display_name", r).replace("\n", "\n") for r in run_ids]
    f1_scores = [results[r].get("macro_f1", 0) * 100 for r in run_ids]
    acc_scores = [results[r].get("accuracy", 0) * 100 for r in run_ids]

    # ── Plot 1: Macro-F1 bar chart ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    bars = ax.bar(range(len(run_ids)), f1_scores, color=colors, width=0.6, alpha=0.9,
                  edgecolor="white", linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(range(len(run_ids)))
    ax.set_xticklabels(display_names, fontsize=10)
    ax.set_ylabel("Macro-F1 (%)", fontsize=12)
    ax.set_title("Ablation Study: Effect of Each Distillation Level", fontsize=13, fontweight="bold")
    ax.set_ylim(max(0, min(f1_scores) - 5), max(f1_scores) + 3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "ablation_f1.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    console.print(f"  Saved: {save_path}")
    plt.close()

    # ── Plot 2: Incremental gain ──────────────────────────────────────────
    if len(f1_scores) >= 2:
        gains = [f1_scores[0]] + [f1_scores[i] - f1_scores[i-1] for i in range(1, len(f1_scores))]
        gain_labels = ["Base", "+Logit KD", "+Hidden KD", "+Attention KD"][:len(gains)]

        fig, ax = plt.subplots(figsize=(8, 5))
        bar_colors = ["#4C72B0"] + ["#55A868"] * (len(gains) - 1)
        bars = ax.bar(range(len(gains)), gains, color=bar_colors, width=0.5, alpha=0.9,
                      edgecolor="white", linewidth=1.5)

        for bar, val in zip(bars, gains):
            label = f"+{val:.1f}%" if val > 0 else f"{val:.1f}%"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    label, ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_xticks(range(len(gains)))
        ax.set_xticklabels(gain_labels, fontsize=10)
        ax.set_ylabel("F1 Gain / Base F1 (%)", fontsize=11)
        ax.set_title("Incremental Contribution of Each KD Level", fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(save_dir, "ablation_gain.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        console.print(f"  Saved: {save_path}")
        plt.close()


def print_ablation_table(results: dict) -> None:
    """Print a rich table of ablation results."""
    table = Table(title="Ablation Study Results (A1 → A4)", border_style="bold")
    table.add_column("Config", style="cyan", min_width=20)
    table.add_column("Logit KD", style="yellow", justify="center")
    table.add_column("Hidden KD", style="yellow", justify="center")
    table.add_column("Att KD", style="yellow", justify="center")
    table.add_column("Macro-F1", style="bold green", justify="right")
    table.add_column("Accuracy", style="green", justify="right")
    table.add_column("Δ F1", style="bold", justify="right")

    base_f1 = None
    for run_id, m in results.items():
        abl_cfg = ABLATION_CONFIGS.get(run_id, {})
        f1 = m.get("macro_f1", 0)
        if base_f1 is None:
            base_f1 = f1
            delta_str = "—"
        else:
            delta = (f1 - base_f1) * 100
            delta_str = f"[green]+{delta:.1f}%[/green]" if delta >= 0 else f"[red]{delta:.1f}%[/red]"

        table.add_row(
            run_id,
            "✓" if abl_cfg.get("use_logit_kd") else "✗",
            "✓" if abl_cfg.get("use_hidden_kd") else "✗",
            "✓" if abl_cfg.get("use_attention_kd") else "✗",
            f"{f1:.4f}",
            f"{m.get('accuracy', 0):.4f}",
            delta_str,
        )

    console.print(table)


def aggregate_existing_results() -> dict:
    """Load existing ablation result files."""
    results = {}
    for run_id in ABLATION_CONFIGS:
        result_file = RESULTS_DIR / f"{run_id}_results.json"
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
            data["display_name"] = ABLATION_CONFIGS[run_id]["name"]
            results[run_id] = data
            console.print(f"  Loaded: {result_file}")
        else:
            console.print(f"  [yellow]Missing: {result_file}[/yellow]")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run ablation study A1→A4.")
    parser.add_argument("--config", type=str, default="configs/distillation_config.yaml")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training, only aggregate existing results.")
    parser.add_argument("--ablations", nargs="+",
                        choices=list(ABLATION_CONFIGS.keys()),
                        default=list(ABLATION_CONFIGS.keys()),
                        help="Which ablation configs to run.")
    args = parser.parse_args()

    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    results = {}

    if args.skip_training:
        console.print("[bold cyan]Loading existing ablation results...[/bold cyan]")
        results = aggregate_existing_results()
    else:
        for run_id in args.ablations:
            ablation_cfg = ABLATION_CONFIGS[run_id]
            metrics = run_ablation(base_config, run_id, ablation_cfg)
            results[run_id] = metrics

            # Save individual result
            save_path = RESULTS_DIR / f"{run_id}_results.json"
            with open(save_path, "w") as f:
                json.dump(metrics, f, indent=2)

    if results:
        # Print summary table
        print_ablation_table(results)

        # Generate plots
        console.print("\n[bold cyan]Generating ablation plots...[/bold cyan]")
        plot_ablation_results(results, str(PLOTS_DIR))

        # Save combined results
        combined_path = ABLATION_DIR / "ablation_results.json"
        with open(combined_path, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"\n[green]Ablation results saved to: {combined_path}[/green]")
        console.print(f"[green]Plots saved to: {PLOTS_DIR}/[/green]")
    else:
        console.print("[red]No results found. Run training first.[/red]")


if __name__ == "__main__":
    main()
