"""
training/train_fusion.py

Train LLM-Fused PhoBERT cho Vietnamese Hate Speech Detection.

Yêu cầu: đã chạy data/extract_llm_embeddings.py để có:
    data/llm_embeddings/train_llm_embeddings.npy
    data/llm_embeddings/val_llm_embeddings.npy
    data/llm_embeddings/test_llm_embeddings.npy

Điểm khác biệt chính so với train_teacher.py:
    - DataLoader trả thêm llm_embedding tensor mỗi batch
    - Model forward nhận thêm llm_embeddings argument
    - PhoBERT backbone dùng LLRD (layer-wise LR decay)
    - LLM projection head và FusionMLP dùng LR cao hơn backbone
      (vì chúng train từ đầu, không phải pretrained)

Usage:
    python training/train_fusion.py --config configs/fusion_config.yaml
    python training/train_fusion.py --config configs/fusion_config.yaml --fp16
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.fusion_model import LLMFusedPhoBERT, LLMEmbeddingDataset, build_fusion_model_from_config
from models.teacher import get_teacher_tokenizer
from utils.data_utils import get_class_weights, get_weighted_sampler
from utils.metrics import compute_metrics, print_classification_report
from utils.seed import set_seed, get_device
from utils.logging_utils import ExperimentLogger

console = Console()


class EarlyStopping:
    def __init__(self, patience: int = 6, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = None

    def __call__(self, score: float) -> bool:
        if self.best is None or score > self.best + self.min_delta:
            self.best = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_epoch(model, dataloader, optimizer, scheduler, device, grad_clip=1.0, fp16=False, scaler=None):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(dataloader, desc="  Training", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        llm_embeddings = batch["llm_embedding"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        if fp16 and scaler is not None:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model(input_ids, attention_mask, llm_embeddings, labels)
                loss = outputs["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids, attention_mask, llm_embeddings, labels)
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

        preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(dataloader)
    return metrics


@torch.no_grad()
def evaluate(model, dataloader, device, split_name="Val"):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc=f"  {split_name}", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        llm_embeddings = batch["llm_embedding"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask, llm_embeddings, labels)
        total_loss += outputs["loss"].item()

        preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(dataloader)
    return metrics, all_labels, all_preds


def build_optimizer_with_llrd(model: LLMFusedPhoBERT, config: dict) -> torch.optim.AdamW:
    """
    Layer-wise LR Decay cho PhoBERT backbone + LR cao hơn cho fusion heads.

    Lý do dùng LR khác nhau:
        - PhoBERT backbone: pretrained → LR nhỏ, decay theo layer
        - LLM projection + FusionMLP: train từ đầu → LR lớn hơn (~10x backbone)
    """
    training_cfg = config["training"]
    base_lr = training_cfg["learning_rate"]
    llrd_factor = training_cfg.get("llrd_factor", 0.9)
    fusion_lr_mult = training_cfg.get("fusion_lr_multiplier", 10.0)
    weight_decay = training_cfg["weight_decay"]
    no_decay = ["bias", "LayerNorm.weight"]

    num_layers = model.num_layers  # 24 for phobert-large
    groups = []

    # Embeddings (lowest LR)
    embed_lr = base_lr * (llrd_factor ** num_layers)
    for decay in [False, True]:
        params = [
            p for n, p in model.backbone.embeddings.named_parameters()
            if (any(nd in n for nd in no_decay)) == decay
        ]
        groups.append({
            "params": params,
            "lr": embed_lr,
            "weight_decay": 0.0 if decay else weight_decay,
        })

    # Transformer layers (increasing LR from bottom to top)
    for i, layer in enumerate(model.backbone.encoder.layer):
        layer_lr = base_lr * (llrd_factor ** (num_layers - i))
        for decay in [False, True]:
            params = [
                p for n, p in layer.named_parameters()
                if (any(nd in n for nd in no_decay)) == decay
            ]
            groups.append({
                "params": params,
                "lr": layer_lr,
                "weight_decay": 0.0 if decay else weight_decay,
            })

    # Fusion heads (LLM projection + FusionMLP) — LR lớn hơn nhiều
    fusion_lr = base_lr * fusion_lr_mult
    # Fusion heads (LLM projection + FusionMLP) dùng weight_decay CAO HƠN backbone
    # vì chúng train từ đầu (không pretrained) → dễ overfit hơn, cần regularize mạnh.
    # Backbone đã có inductive bias từ pretraining nên weight_decay thấp hơn là đủ.
    fusion_weight_decay = training_cfg.get("fusion_weight_decay", weight_decay * 2)
    fusion_params = list(model.llm_proj.parameters()) + list(model.fusion_head.parameters())
    groups.append({"params": fusion_params, "lr": fusion_lr, "weight_decay": fusion_weight_decay})

    console.print(
        f"  [yellow]LLRD: backbone base LR={base_lr} | "
        f"fusion LR={fusion_lr:.1e} (×{fusion_lr_mult}) | "
        f"fusion weight_decay={fusion_weight_decay}[/yellow]"
    )
    return torch.optim.AdamW(groups)


def train(config: dict, config_path: str = "") -> None:
    training_cfg = config["training"]
    data_cfg = config["data"]
    fusion_cfg = config.get("fusion", {})
    log_cfg = config.get("logging", {})

    # ── Banner xác nhận config đang dùng — tránh lặp lại lỗi "kết quả y hệt
    # lần trước" do nhầm chạy code/config cũ. In rõ các giá trị quan trọng
    # ngay từ đầu để kiểm tra bằng mắt trước khi tốn thời gian train.
    console.print("\n" + "=" * 70)
    console.print("[bold cyan]CONFIG FINGERPRINT — kiểm tra kỹ trước khi train[/bold cyan]")
    console.print("=" * 70)
    console.print(f"  Config file:             {config_path or '(không rõ đường dẫn)'}")
    console.print(f"  num_epochs:               {training_cfg.get('num_epochs')}")
    console.print(f"  batch_size:               {training_cfg.get('batch_size')}")
    console.print(f"  use_weighted_sampler:     {training_cfg.get('use_weighted_sampler')}")
    console.print(f"  warmup_ratio:             {training_cfg.get('warmup_ratio')}")
    console.print(f"  weight_decay:             {training_cfg.get('weight_decay')}")
    console.print(f"  fusion_weight_decay:      {training_cfg.get('fusion_weight_decay', '(default = weight_decay × 2)')}")
    console.print(f"  early_stopping_patience:  {training_cfg.get('early_stopping_patience')}")
    console.print("=" * 70)

    # Cảnh báo nếu checkpoint cũ tồn tại — best_model.pt cũ KHÔNG bị xóa tự
    # động, nếu training crash sớm hoặc bị nhầm chạy nhánh code cũ, kết quả
    # cuối có thể vô tình load lại checkpoint cũ thay vì checkpoint mới.
    ckpt_path = os.path.join(training_cfg["output_dir"], "best_model.pt")
    if os.path.isfile(ckpt_path):
        import time
        mtime = os.path.getmtime(ckpt_path)
        age_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
        console.print(
            f"[bold yellow]⚠ Checkpoint cũ đã tồn tại: {ckpt_path}\n"
            f"  Thời gian tạo: {age_str}\n"
            f"  Checkpoint này sẽ bị GHI ĐÈ nếu epoch mới đạt best_f1 cao hơn.\n"
            f"  Nếu muốn chắc chắn train từ đầu sạch, hãy xóa file này trước khi chạy.[/bold yellow]"
        )
        response_hint = (
            f"  Gợi ý: mv {ckpt_path} {ckpt_path}.bak_{int(mtime)}"
        )
        console.print(f"[dim]{response_hint}[/dim]\n")

    set_seed(data_cfg["seed"])
    device = get_device()
    os.makedirs(training_cfg["output_dir"], exist_ok=True)

    logger = ExperimentLogger(
        project_name=log_cfg.get("project_name", "TinyPhoBERT"),
        run_name=log_cfg.get("run_name", "llm-fusion"),
        log_dir=log_cfg.get("log_dir", "logs/fusion"),
        use_wandb=log_cfg.get("use_wandb", False),
        use_tensorboard=log_cfg.get("use_tensorboard", True),
        config=config,
    )

    # ── Tokenizer & Data ──────────────────────────────────────────────────────
    console.print("[bold cyan]Loading tokenizer and data...[/bold cyan]")
    tokenizer = get_teacher_tokenizer(config["model"]["name"])

    embedding_dir = fusion_cfg.get("embedding_dir", "data/llm_embeddings")
    text_col = data_cfg["text_col"]
    label_col = data_cfg["label_col"]

    def load_split(split: str):
        csv_path = data_cfg[f"{split}_file"]
        emb_path = os.path.join(embedding_dir, f"{split}_llm_embeddings.npy")

        if not os.path.isfile(emb_path):
            console.print(
                f"[bold red]✗ Missing LLM embeddings: {emb_path}\n"
                f"Run: python data/extract_llm_embeddings.py --config configs/fusion_config.yaml[/bold red]"
            )
            sys.exit(1)

        df = pd.read_csv(csv_path)
        texts = df[text_col].astype(str).tolist()
        labels = df[label_col].astype(int).tolist()
        dataset = LLMEmbeddingDataset(
            texts, labels, tokenizer, emb_path,
            max_length=training_cfg["max_seq_length"],
        )
        console.print(f"  {split}: {len(dataset):,} samples | dist={dict(zip(*np.unique(labels, return_counts=True)))}")
        return dataset, labels

    import numpy as np
    train_ds, train_labels = load_split("train")
    val_ds, _ = load_split("val")
    test_ds, _ = load_split("test")

    # ── Class weights + Weighted Sampler ─────────────────────────────────────
    class_weights = get_class_weights(train_labels, num_classes=config["model"]["num_labels"])
    class_weights_device = class_weights.to(device)

    use_sampler = training_cfg.get("use_weighted_sampler", True)
    train_loader = DataLoader(
        train_ds,
        batch_size=training_cfg["batch_size"],
        sampler=get_weighted_sampler(train_labels, strength=training_cfg.get("sampler_strength", 0.5)) if use_sampler else None,
        shuffle=not use_sampler,
        num_workers=training_cfg.get("dataloader_num_workers", 4),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(val_ds, batch_size=training_cfg["batch_size"] * 2, shuffle=False,
                            num_workers=training_cfg.get("dataloader_num_workers", 4))
    test_loader = DataLoader(test_ds, batch_size=training_cfg["batch_size"] * 2, shuffle=False,
                             num_workers=training_cfg.get("dataloader_num_workers", 4))

    # ── Model ─────────────────────────────────────────────────────────────────
    console.print(f"\n[bold cyan]Building LLMFusedPhoBERT...[/bold cyan]")
    model = build_fusion_model_from_config(config)
    model.class_weights = class_weights_device
    model = model.to(device)

    param_info = model.count_parameters()
    console.print(f"  Total params:     [bold green]{param_info['total']:,}[/bold green]")
    console.print(f"  PhoBERT backbone: {param_info['phobert_backbone']:,}")
    console.print(f"  Fusion heads:     {param_info['fusion_heads']:,} (llm_proj + fusion_mlp)")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    use_llrd = training_cfg.get("use_llrd", True)
    if use_llrd:
        optimizer = build_optimizer_with_llrd(model, config)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_cfg["learning_rate"],
            weight_decay=training_cfg["weight_decay"],
        )

    num_steps = len(train_loader) * training_cfg["num_epochs"]
    num_warmup = int(num_steps * training_cfg["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup, num_steps)
    console.print(f"  Schedule: Cosine | warmup={num_warmup} steps")

    fp16 = training_cfg.get("fp16", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if fp16 else None
    if fp16:
        console.print("  [yellow]FP16 enabled[/yellow]")

    patience = training_cfg.get("early_stopping_patience", 6)
    early_stopping = EarlyStopping(patience=patience)
    console.print(f"  Early stopping: patience={patience}")

    # ── Training Loop ─────────────────────────────────────────────────────────
    best_f1 = 0.0
    best_epoch = 0
    global_step = 0
    history = []

    console.print(f"\n[bold cyan]Training LLMFusedPhoBERT ({training_cfg['num_epochs']} epochs max)...[/bold cyan]\n")

    for epoch in range(1, training_cfg["num_epochs"] + 1):
        current_lr = scheduler.get_last_lr()[0]
        console.print(f"[bold]Epoch {epoch}/{training_cfg['num_epochs']}[/bold]  (LR={current_lr:.2e})")

        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            grad_clip=training_cfg["max_grad_norm"], fp16=fp16, scaler=scaler,
        )
        global_step += len(train_loader)
        val_metrics, _, _ = evaluate(model, val_loader, device, "Val")

        f1_off = val_metrics.get("f1_offensive", 0)
        f1_hate = val_metrics.get("f1_hate", 0)

        log_dict = {
            "train/loss": train_metrics["loss"], "train/f1_macro": train_metrics["macro_f1"],
            "val/loss": val_metrics["loss"], "val/f1_macro": val_metrics["macro_f1"],
            "val/f1_offensive": f1_off, "val/f1_hate": f1_hate,
        }
        logger.log(log_dict, step=global_step)

        console.print(
            f"  Train: loss={train_metrics['loss']:.4f} | f1_macro={train_metrics['macro_f1']:.4f}\n"
            f"  Val  : loss={val_metrics['loss']:.4f}   | f1_macro={val_metrics['macro_f1']:.4f}\n"
            f"         f1_off={f1_off:.4f} | f1_hate={f1_hate:.4f}"
        )
        history.append({"epoch": epoch, **log_dict})

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            ckpt_path = os.path.join(training_cfg["output_dir"], "best_model.pt")
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "val_f1": best_f1, "config": config},
                ckpt_path,
            )
            console.print(f"  [bold green]✓ Best model saved (Macro-F1={best_f1:.4f} | F1_OFF={f1_off:.4f} | F1_HATE={f1_hate:.4f})[/bold green]")

        if early_stopping(val_metrics["macro_f1"]):
            console.print(f"\n[bold yellow]Early stopping at epoch {epoch}.[/bold yellow]")
            break

    console.print(f"\n[bold green]Training complete! Best Val Macro-F1={best_f1:.4f} at epoch {best_epoch}[/bold green]")
    with open(os.path.join(training_cfg["output_dir"], "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ── Final Test Evaluation ──────────────────────────────────────────────────
    console.print("\n[bold cyan]Final Test Evaluation...[/bold cyan]")
    ckpt = torch.load(
        os.path.join(training_cfg["output_dir"], "best_model.pt"),
        map_location=device, weights_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics, test_labels, test_preds = evaluate(model, test_loader, device, "Test")

    print_classification_report(test_labels, test_preds)

    table = Table(title="LLMFusedPhoBERT Test Results")
    table.add_column("Metric", style="cyan"); table.add_column("Value", style="green")
    for k, v in test_metrics.items():
        table.add_row(k, f"{v:.4f}")
    console.print(table)

    os.makedirs("results", exist_ok=True)
    import time
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_payload = {
        **test_metrics,
        "best_epoch": best_epoch,
        "model": "LLMFusedPhoBERT",
        "run_timestamp": run_timestamp,
        "config_num_epochs": training_cfg.get("num_epochs"),
        "config_use_weighted_sampler": training_cfg.get("use_weighted_sampler"),
    }

    # Lưu file timestamped — tránh nhầm lẫn kết quả run cũ/mới khi xem lại
    timestamped_path = f"results/fusion_results_{run_timestamp}.json"
    with open(timestamped_path, "w") as f:
        json.dump(result_payload, f, indent=2)

    # Lưu thêm bản "latest" cố định tên để dễ tham chiếu trong script khác
    with open("results/fusion_results.json", "w") as f:
        json.dump(result_payload, f, indent=2)

    console.print(f"Results saved to {timestamped_path}")
    console.print("(và results/fusion_results.json — bản mới nhất)")
    logger.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fusion_config.yaml")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--force_fresh", action="store_true",
        help="Tự động backup (đổi tên) best_model.pt cũ trước khi train, "
             "đảm bảo không nhầm lẫn kết quả cũ/mới.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.fp16:
        config["training"]["fp16"] = True

    if args.force_fresh:
        import time
        ckpt_path = os.path.join(config["training"]["output_dir"], "best_model.pt")
        if os.path.isfile(ckpt_path):
            backup_path = f"{ckpt_path}.bak_{int(time.time())}"
            os.rename(ckpt_path, backup_path)
            print(f"[force_fresh] Backed up old checkpoint to: {backup_path}")

    train(config, config_path=args.config)


if __name__ == "__main__":
    main()