"""
notebooks/03_demo.py

Interactive demo for TinyPhoBERT.
Loads the trained student model and classifies Vietnamese text.

Usage:
    python notebooks/03_demo.py
    python notebooks/03_demo.py --text "Bình luận cần phân tích"
    python notebooks/03_demo.py --interactive
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.student import build_student_from_config
from models.teacher import get_teacher_tokenizer
from utils.seed import get_device

console = Console()

LABEL_NAMES = {0: "CLEAN ✅", 1: "OFFENSIVE ⚠️", 2: "HATE 🚫"}
LABEL_COLORS = {0: "green", 1: "yellow", 2: "red"}

DEMO_TEXTS = [
    "Bài hát này hay quá, mình nghe mãi không chán.",
    "Trò chơi này thật sự rất vui và thú vị.",
    "Phim dở ẹc, đạo diễn này không biết làm phim gì cả.",
    "Thật là ngu ngốc khi tin vào điều này.",
    "Đồ súc vật, mày có biết mày đang làm gì không?",
    "Tao ghét mày, mày là thứ vô dụng nhất thế giới.",
]


def classify(model, tokenizer, text: str, device, max_length=128) -> dict:
    """Classify a single Vietnamese text."""
    model.eval()
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    latency_ms = (time.perf_counter() - t0) * 1000

    logits = outputs["logits"].squeeze(0)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    pred = int(probs.argmax())

    return {
        "text": text,
        "prediction": pred,
        "label": LABEL_NAMES[pred],
        "confidence": float(probs[pred]),
        "probabilities": {
            "CLEAN": float(probs[0]),
            "OFFENSIVE": float(probs[1]),
            "HATE": float(probs[2]),
        },
        "latency_ms": round(latency_ms, 2),
    }


def display_result(result: dict) -> None:
    """Display a single classification result."""
    color = LABEL_COLORS[result["prediction"]]

    console.print(Panel(
        f"[bold]{result['text']}[/bold]\n\n"
        f"Prediction: [{color}]{result['label']}[/{color}] "
        f"(confidence: {result['confidence']*100:.1f}%)\n"
        f"Latency: {result['latency_ms']:.1f} ms",
        title="[bold blue]TinyPhoBERT Classification[/bold blue]",
        border_style="blue",
    ))

    table = Table(title="Class Probabilities")
    table.add_column("Class", style="cyan")
    table.add_column("Probability", style="green")
    table.add_column("Bar")
    for cls, prob in result["probabilities"].items():
        bar = "█" * int(prob * 30)
        table.add_row(cls, f"{prob*100:.1f}%", f"[green]{bar}[/green]")
    console.print(table)


def run_demo(model, tokenizer, device) -> None:
    """Run demo on predefined examples."""
    console.print("\n[bold cyan]TinyPhoBERT Demo — Vietnamese Hate Speech Detection[/bold cyan]\n")

    results = []
    for text in DEMO_TEXTS:
        result = classify(model, tokenizer, text, device)
        results.append(result)

    # Summary table
    table = Table(title="Demo Results", border_style="bold")
    table.add_column("Text", max_width=45, overflow="fold")
    table.add_column("Prediction", style="bold")
    table.add_column("Confidence", justify="right")
    table.add_column("Latency")

    for r in results:
        color = LABEL_COLORS[r["prediction"]]
        table.add_row(
            r["text"][:50] + "..." if len(r["text"]) > 50 else r["text"],
            f"[{color}]{r['label']}[/{color}]",
            f"{r['confidence']*100:.1f}%",
            f"{r['latency_ms']:.1f} ms",
        )
    console.print(table)

    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    console.print(f"\n[green]Average latency: {avg_latency:.1f} ms/sample[/green]")


def interactive_mode(model, tokenizer, device) -> None:
    """Interactive classification from terminal input."""
    console.print("\n[bold cyan]TinyPhoBERT — Interactive Mode[/bold cyan]")
    console.print("Nhập văn bản tiếng Việt để phân tích (gõ 'quit' để thoát)\n")

    while True:
        text = console.input("[bold]> [/bold]").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue
        result = classify(model, tokenizer, text, device)
        display_result(result)
        console.print()


def main():
    parser = argparse.ArgumentParser(description="TinyPhoBERT Demo")
    parser.add_argument("--model_path", type=str,
                        default="checkpoints/distillation/TinyPhoBERT_full/best_model.pt")
    parser.add_argument("--config_path", type=str, default="configs/student_config.yaml")
    parser.add_argument("--text", type=str, default=None, help="Single text to classify.")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode.")
    args = parser.parse_args()

    import yaml
    device = get_device()
    tokenizer = get_teacher_tokenizer("vinai/phobert-base")

    with open(args.config_path) as f:
        cfg = yaml.safe_load(f)
    model = build_student_from_config(cfg)

    if Path(args.model_path).exists():
        ckpt = torch.load(args.model_path, map_location=device)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd)
        console.print(f"[green]✓ Model loaded from: {args.model_path}[/green]")
    else:
        console.print(f"[yellow]Warning: Checkpoint not found. Using random weights.[/yellow]")
    model = model.to(device)

    if args.text:
        result = classify(model, tokenizer, args.text, device)
        display_result(result)
    elif args.interactive:
        interactive_mode(model, tokenizer, device)
    else:
        run_demo(model, tokenizer, device)


if __name__ == "__main__":
    main()
