"""
evaluation/benchmark.py

Efficiency benchmarking: parameters, model size, and inference latency.

Measures:
    - Total parameters
    - Model size (MB)
    - Inference latency (ms/sample)
    - Throughput (samples/sec)
    - GPU memory usage (if CUDA)

Usage:
    python evaluation/benchmark.py
    python evaluation/benchmark.py --model_path checkpoints/distillation/best_model.pt
    python evaluation/benchmark.py --compare_all
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.teacher import PhoBERTTeacher, get_teacher_tokenizer
from models.student import TinyPhoBERT, build_student_from_config
from utils.seed import get_device

console = Console()


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def model_size_mb(model: torch.nn.Module) -> float:
    """Estimate model size in MB (float32 weights)."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def measure_latency(
    model: torch.nn.Module,
    device: torch.device,
    tokenizer,
    seq_len: int = 128,
    batch_size: int = 1,
    num_warmup: int = 20,
    num_runs: int = 100,
    model_type: str = "student",
) -> Dict[str, float]:
    """
    Measure inference latency using synthetic inputs.

    Args:
        model: The model to benchmark.
        device: Target device.
        tokenizer: Tokenizer for vocabulary size reference.
        seq_len: Sequence length.
        batch_size: Batch size.
        num_warmup: Warmup runs (not measured).
        num_runs: Timed runs.
        model_type: 'student' or 'teacher'.

    Returns:
        Dictionary with latency stats.
    """
    model.eval()
    model = model.to(device)

    # Create synthetic input
    vocab_size = getattr(tokenizer, "vocab_size", 64001)
    dummy_ids = torch.randint(
        1, min(vocab_size, 64000), (batch_size, seq_len), device=device
    )
    dummy_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_ids, dummy_mask)

    # Synchronize CUDA before timing
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(dummy_ids, dummy_mask)
                end.record()
                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))  # ms
            else:
                t0 = time.perf_counter()
                _ = model(dummy_ids, dummy_mask)
                latencies.append((time.perf_counter() - t0) * 1000)  # ms

    latencies = np.array(latencies)
    per_sample_ms = latencies / batch_size

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "mean_ms_per_sample": round(float(per_sample_ms.mean()), 3),
        "std_ms_per_sample": round(float(per_sample_ms.std()), 3),
        "p50_ms": round(float(np.percentile(per_sample_ms, 50)), 3),
        "p95_ms": round(float(np.percentile(per_sample_ms, 95)), 3),
        "p99_ms": round(float(np.percentile(per_sample_ms, 99)), 3),
        "throughput_samples_per_sec": round(float(1000 / per_sample_ms.mean()), 1),
    }


def gpu_memory_mb(model: torch.nn.Module, device: torch.device) -> float:
    """Measure peak GPU memory during a forward pass."""
    if device.type != "cuda":
        return 0.0
    torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    dummy_ids = torch.randint(1, 64000, (1, 128), device=device)
    dummy_mask = torch.ones(1, 128, device=device, dtype=torch.long)
    with torch.no_grad():
        _ = model(dummy_ids, dummy_mask)
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def benchmark_model(
    model: torch.nn.Module,
    model_name: str,
    device: torch.device,
    tokenizer,
    model_type: str = "student",
) -> Dict:
    """Full benchmark for one model."""
    console.print(f"\n[bold cyan]Benchmarking: {model_name}[/bold cyan]")

    params = count_parameters(model)
    size = model_size_mb(model)

    latency_bs1 = measure_latency(model, device, tokenizer, batch_size=1, model_type=model_type)
    latency_bs32 = measure_latency(model, device, tokenizer, batch_size=32, model_type=model_type)

    gpu_mem = gpu_memory_mb(model, device)

    result = {
        "model_name": model_name,
        "total_params": params["total"],
        "trainable_params": params["trainable"],
        "size_mb": round(size, 2),
        "latency_bs1_ms": latency_bs1["mean_ms_per_sample"],
        "latency_bs1_p95": latency_bs1["p95_ms"],
        "latency_bs32_ms": latency_bs32["mean_ms_per_sample"],
        "throughput_bs32": latency_bs32["throughput_samples_per_sec"],
        "gpu_memory_mb": round(gpu_mem, 1),
        "device": str(device),
    }

    table = Table(title=model_name)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Parameters", f"{params['total']:,}")
    table.add_row("Model Size", f"{size:.1f} MB")
    table.add_row("Latency (BS=1)", f"{latency_bs1['mean_ms_per_sample']:.2f} ms/sample")
    table.add_row("Latency P95 (BS=1)", f"{latency_bs1['p95_ms']:.2f} ms/sample")
    table.add_row("Latency (BS=32)", f"{latency_bs32['mean_ms_per_sample']:.2f} ms/sample")
    table.add_row("Throughput (BS=32)", f"{latency_bs32['throughput_samples_per_sec']:.0f} samples/s")
    if gpu_mem > 0:
        table.add_row("GPU Memory", f"{gpu_mem:.0f} MB")
    console.print(table)

    return result


def compare_speedup(teacher_result: dict, student_result: dict) -> None:
    """Print speedup comparison between teacher and student."""
    if not teacher_result or not student_result:
        return

    param_ratio = teacher_result["total_params"] / student_result["total_params"]
    size_ratio = teacher_result["size_mb"] / student_result["size_mb"]
    speedup_bs1 = teacher_result["latency_bs1_ms"] / student_result["latency_bs1_ms"]
    speedup_bs32 = teacher_result["latency_bs32_ms"] / student_result["latency_bs32_ms"]

    table = Table(title="Teacher vs TinyPhoBERT Comparison", border_style="bold green")
    table.add_column("Metric", style="cyan")
    table.add_column("Teacher", style="red")
    table.add_column("TinyPhoBERT", style="green")
    table.add_column("Ratio", style="bold yellow")

    table.add_row(
        "Parameters",
        f"{teacher_result['total_params']:,}",
        f"{student_result['total_params']:,}",
        f"{param_ratio:.1f}× fewer",
    )
    table.add_row(
        "Size (MB)",
        f"{teacher_result['size_mb']:.1f}",
        f"{student_result['size_mb']:.1f}",
        f"{size_ratio:.1f}× smaller",
    )
    table.add_row(
        "Latency BS=1 (ms)",
        f"{teacher_result['latency_bs1_ms']:.2f}",
        f"{student_result['latency_bs1_ms']:.2f}",
        f"{speedup_bs1:.1f}× faster",
    )
    table.add_row(
        "Latency BS=32 (ms)",
        f"{teacher_result['latency_bs32_ms']:.2f}",
        f"{student_result['latency_bs32_ms']:.2f}",
        f"{speedup_bs32:.1f}× faster",
    )
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Benchmark model efficiency.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Student checkpoint path.")
    parser.add_argument("--teacher_path", type=str, default=None,
                        help="Teacher checkpoint path.")
    parser.add_argument("--config_path", type=str, default="configs/student_config.yaml")
    parser.add_argument("--compare_all", action="store_true",
                        help="Benchmark both teacher and student.")
    parser.add_argument("--seq_len", type=int, default=128)
    args = parser.parse_args()

    device = get_device()
    tokenizer = get_teacher_tokenizer("vinai/phobert-base")

    os.makedirs("results", exist_ok=True)
    all_benchmark = {}

    # ── Teacher ───────────────────────────────────────────────────────────────
    if args.compare_all or args.teacher_path:
        console.print("[bold]Loading Teacher (PhoBERT-base)...[/bold]")
        teacher = PhoBERTTeacher("vinai/phobert-base")
        if args.teacher_path and os.path.isfile(args.teacher_path):
            ckpt = torch.load(args.teacher_path, map_location="cpu")
            sd = ckpt.get("model_state_dict", ckpt)
            teacher.load_state_dict(sd)
        teacher_result = benchmark_model(teacher, "PhoBERT-base (Teacher)", device, tokenizer, "teacher")
        all_benchmark["teacher"] = teacher_result
        del teacher

    # ── Student ───────────────────────────────────────────────────────────────
    if args.compare_all or args.model_path:
        import yaml
        with open(args.config_path) as f:
            cfg = yaml.safe_load(f)
        student = build_student_from_config(cfg)
        if args.model_path and os.path.isfile(args.model_path):
            ckpt = torch.load(args.model_path, map_location="cpu")
            sd = ckpt.get("model_state_dict", ckpt)
            student.load_state_dict(sd)
        student_result = benchmark_model(student, "TinyPhoBERT (Student)", device, tokenizer, "student")
        all_benchmark["student"] = student_result
        del student

    # ── Comparison ────────────────────────────────────────────────────────────
    if "teacher" in all_benchmark and "student" in all_benchmark:
        compare_speedup(all_benchmark["teacher"], all_benchmark["student"])

    # ── Default: just print student stats without checkpoint ─────────────────
    if not all_benchmark:
        import yaml
        console.print("[bold]Benchmarking default TinyPhoBERT architecture...[/bold]")
        with open(args.config_path) as f:
            cfg = yaml.safe_load(f)
        student = build_student_from_config(cfg)
        student_result = benchmark_model(student, "TinyPhoBERT (Student)", device, tokenizer, "student")
        all_benchmark["student"] = student_result

        console.print("\n[bold]Benchmarking PhoBERT-base for comparison...[/bold]")
        teacher = PhoBERTTeacher("vinai/phobert-base")
        teacher_result = benchmark_model(teacher, "PhoBERT-base (Teacher)", device, tokenizer, "teacher")
        all_benchmark["teacher"] = teacher_result
        compare_speedup(teacher_result, student_result)

    with open("results/benchmark.json", "w") as f:
        json.dump(all_benchmark, f, indent=2)
    console.print("\n[green]Benchmark results saved to results/benchmark.json[/green]")


if __name__ == "__main__":
    main()
