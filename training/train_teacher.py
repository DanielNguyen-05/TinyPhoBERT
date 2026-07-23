"""
training/train_teacher.py  [BẢN KHÔI PHỤC — đạt Macro-F1=65.38%]

Fine-tune PhoBERT-large (Teacher) trên dữ liệu đã chuẩn bị sẵn bởi
data/prepare_data.py. KHÔNG có SupCon (SupCon là nguyên nhân gây NaN
và giảm hiệu suất trong các thử nghiệm sau này).

Kết quả tham chiếu đã đạt được với bản này:
    accuracy=0.8728 | macro_f1=0.6538 | f1_offensive=0.4176 | f1_hate=0.6068
    best_epoch=22

Quy trình bắt buộc TRƯỚC khi chạy file này:
    python data/prepare_data.py --config configs/teacher_config.yaml

Usage:
    python training/train_teacher.py --config configs/teacher_config.yaml
    python training/train_teacher.py --config configs/teacher_config.yaml --fp16
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.teacher import PhoBERTTeacher, get_teacher_tokenizer
from utils.data_utils import (
    get_class_weights,
    get_weighted_sampler,
    HateSpeechDataset,
)
from utils.metrics import compute_metrics, print_classification_report
from utils.seed import set_seed, get_device
from utils.logging_utils import ExperimentLogger

console = Console()


class EarlyStopping:
    """Dừng training nếu val metric không cải thiện sau `patience` epochs."""
    def __init__(self, patience: int = 4, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False


def train_epoch(model, dataloader, optimizer, scheduler, device, grad_clip=1.0, fp16=False, scaler=None, fgm=None):
    """
    Args:
        fgm: FGM instance (từ models/fgm.py) hoặc None để tắt adversarial training.
             Khi bật: sau backward() loss gốc, perturb embedding theo hướng
             gradient, forward+backward lại trên embedding nhiễu, rồi restore.
             Gradient của cả 2 lần backward() tích lũy vào cùng .grad trước
             khi optimizer.step() — model học từ cả input gốc lẫn "khó nhất".
    """
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(dataloader, desc="  Training", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        if fp16 and scaler is not None:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs["loss"]
            scaler.scale(loss).backward()

            if fgm is not None:
                # QUAN TRỌNG: KHÔNG gọi scaler.unscale_() trước đây — attack()
                # dùng norm của gradient để tính hướng nhiễu, scale factor tự
                # triệt tiêu trong phép chia (scaled_grad/norm(scaled_grad) =
                # real_grad/norm(real_grad)), nên an toàn dùng gradient đã scale.
                fgm.attack()
                with autocast():
                    outputs_adv = model(input_ids, attention_mask, labels)
                    loss_adv = outputs_adv["loss"]
                # Cùng scale factor với lần backward() đầu (scaler chỉ update()
                # 1 lần ở cuối step) → 2 gradient tích lũy nhất quán, an toàn.
                scaler.scale(loss_adv).backward()
                fgm.restore()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs["loss"]
            loss.backward()

            if fgm is not None:
                fgm.attack()
                outputs_adv = model(input_ids, attention_mask, labels)
                loss_adv = outputs_adv["loss"]
                loss_adv.backward()
                fgm.restore()

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

    pbar = tqdm(dataloader, desc=f"  {split_name}", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask, labels)
        loss = outputs["loss"]
        total_loss += loss.item()

        preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(dataloader)
    return metrics, all_labels, all_preds


def load_prepared_data(config: dict):
    """Đọc dữ liệu ĐÃ ĐƯỢC CHUẨN BỊ SẴN từ data/prepare_data.py."""
    import pandas as pd

    data_cfg = config["data"]
    aug_dir = Path(data_cfg.get("augmented_dir", "data/augmented"))

    train_path = aug_dir / "train.csv"
    val_path = aug_dir / "val.csv"
    test_path = aug_dir / "test.csv"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        console.print(
            f"\n[bold red]✗ Không tìm thấy dữ liệu đã chuẩn bị tại {aug_dir}/[/bold red]\n"
            f"[yellow]Hãy chạy bước chuẩn bị dữ liệu trước:[/yellow]\n"
            f"  python data/prepare_data.py --config {config.get('_config_path', 'configs/teacher_config.yaml')}\n"
        )
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    console.print(f"[cyan]Loaded prepared data from {aug_dir}/[/cyan]")
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = df[data_cfg["label_col"]].value_counts().to_dict()
        console.print(f"  {name}: {len(df):,} samples | dist={dist}")

    return train_df, val_df, test_df


def train(config: dict) -> None:
    set_seed(config["data"]["seed"])
    device = get_device()
    os.makedirs(config["training"]["output_dir"], exist_ok=True)

    log_cfg = config.get("logging", {})
    logger = ExperimentLogger(
        project_name=log_cfg.get("project_name", "TinyPhoBERT"),
        run_name=log_cfg.get("run_name", "teacher"),
        log_dir=log_cfg.get("log_dir", "logs/teacher"),
        use_wandb=log_cfg.get("use_wandb", False),
        use_tensorboard=log_cfg.get("use_tensorboard", True),
        config=config,
    )

    console.print("[bold cyan]Loading tokenizer...[/bold cyan]")
    tokenizer = get_teacher_tokenizer(config["model"]["name"])

    train_df, val_df, test_df = load_prepared_data(config)

    text_col = config["data"]["text_col"]
    label_col = config["data"]["label_col"]

    train_labels = train_df[label_col].astype(int).tolist()

    train_ds = HateSpeechDataset(
        train_df[text_col].astype(str).tolist(), train_labels,
        tokenizer, max_length=config["training"]["max_seq_length"],
    )
    val_ds = HateSpeechDataset(
        val_df[text_col].astype(str).tolist(),
        val_df[label_col].astype(int).tolist(),
        tokenizer, max_length=config["training"]["max_seq_length"],
    )
    test_ds = HateSpeechDataset(
        test_df[text_col].astype(str).tolist(),
        test_df[label_col].astype(int).tolist(),
        tokenizer, max_length=config["training"]["max_seq_length"],
    )

    # ── Class Weights + Weighted Sampler ─────────────────────────────────────
    use_weighted_sampler = config["training"].get("use_weighted_sampler", True)
    class_weights = get_class_weights(train_labels, num_classes=config["model"]["num_labels"])
    class_weights = class_weights.to(device)
    console.print(
        f"\n  [yellow]Class weights:[/yellow] "
        f"CLEAN={class_weights[0]:.3f} | OFFENSIVE={class_weights[1]:.3f} | HATE={class_weights[2]:.3f}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        sampler=get_weighted_sampler(
            train_labels, strength=config["training"].get("sampler_strength", 0.5)
        ) if use_weighted_sampler else None,
        shuffle=False if use_weighted_sampler else True,
        num_workers=config["training"].get("dataloader_num_workers", 4),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["training"]["batch_size"] * 2, shuffle=False,
        num_workers=config["training"].get("dataloader_num_workers", 4),
    )
    test_loader = DataLoader(
        test_ds, batch_size=config["training"]["batch_size"] * 2, shuffle=False,
        num_workers=config["training"].get("dataloader_num_workers", 4),
    )

    # ── Model — KHÔNG có SupCon ───────────────────────────────────────────────
    console.print(f"\n[bold cyan]Building Teacher model ({config['model']['name']})...[/bold cyan]")
    use_focal = config["training"].get("use_focal_loss", True)
    focal_gamma = config["training"].get("focal_gamma", 2.0)
    label_smoothing = config["training"].get("label_smoothing", 0.1)

    model = PhoBERTTeacher(
        model_name=config["model"]["name"],
        num_labels=config["model"]["num_labels"],
        dropout=config["model"]["dropout"],
        class_weights=class_weights if use_focal else None,
        use_focal_loss=use_focal,
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing,
    ).to(device)

    console.print(f"  Teacher params: [bold green]{model.count_parameters():,}[/bold green]")
    console.print(f"  Loss: {'Focal Loss' if use_focal else 'Weighted CE'} | γ={focal_gamma}")

    # ── FGM Adversarial Training (tùy chọn) ───────────────────────────────────
    use_fgm = config["training"].get("use_fgm", False)
    fgm = None
    if use_fgm:
        from models.fgm import FGM
        fgm_epsilon = config["training"].get("fgm_epsilon", 1.0)
        fgm = FGM(model, epsilon=fgm_epsilon, emb_name="word_embeddings")
        console.print(f"  [yellow]FGM Adversarial Training: ON | epsilon={fgm_epsilon}[/yellow]")
        console.print(
            "  [dim]Mỗi batch sẽ tốn ~2x thời gian forward/backward "
            "(clean pass + adversarial pass)[/dim]"
        )

    # ── Optimizer (LLRD) & Scheduler ──────────────────────────────────────────
    use_llrd = config["training"].get("use_llrd", False)
    llrd_factor = config["training"].get("llrd_factor", 0.9)

    if use_llrd:
        base_lr = config["training"]["learning_rate"]
        num_layers = model.num_layers
        no_decay = ["bias", "LayerNorm.weight"]
        groups = []

        embed_lr = base_lr * (llrd_factor ** num_layers)
        groups.append({"params": [p for n, p in model.backbone.embeddings.named_parameters()
                                   if not any(nd in n for nd in no_decay)],
                        "weight_decay": config["training"]["weight_decay"], "lr": embed_lr})
        groups.append({"params": [p for n, p in model.backbone.embeddings.named_parameters()
                                   if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0, "lr": embed_lr})

        for i, layer in enumerate(model.backbone.encoder.layer):
            layer_lr = base_lr * (llrd_factor ** (num_layers - i))
            groups.append({"params": [p for n, p in layer.named_parameters()
                                       if not any(nd in n for nd in no_decay)],
                            "weight_decay": config["training"]["weight_decay"], "lr": layer_lr})
            groups.append({"params": [p for n, p in layer.named_parameters()
                                       if any(nd in n for nd in no_decay)],
                            "weight_decay": 0.0, "lr": layer_lr})

        groups.append({"params": [p for n, p in model.classifier.named_parameters()
                                   if not any(nd in n for nd in no_decay)],
                        "weight_decay": config["training"]["weight_decay"], "lr": base_lr})
        groups.append({"params": [p for n, p in model.classifier.named_parameters()
                                   if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0, "lr": base_lr})

        optimizer = torch.optim.AdamW(groups)
        console.print(f"  [yellow]LLRD enabled: factor={llrd_factor}[/yellow]")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

    num_training_steps = len(train_loader) * config["training"]["num_epochs"]
    num_warmup_steps = int(num_training_steps * config["training"]["warmup_ratio"])

    if config["training"].get("use_cosine_schedule", True):
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        console.print("  [yellow]Schedule: Cosine with warmup[/yellow]")
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    fp16 = config["training"].get("fp16", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if fp16 else None
    if fp16:
        console.print("  [yellow]FP16 mixed precision enabled.[/yellow]")

    patience = config["training"].get("early_stopping_patience", 4)
    early_stopping = EarlyStopping(patience=patience, mode="max")
    console.print(f"  [yellow]Early stopping: patience={patience} epochs[/yellow]")

    # ── Training Loop ─────────────────────────────────────────────────────────
    best_f1 = 0.0
    best_epoch = 0
    global_step = 0
    history = []

    console.print(f"\n[bold cyan]Starting training ({config['training']['num_epochs']} epochs max)...[/bold cyan]\n")

    for epoch in range(1, config["training"]["num_epochs"] + 1):
        console.print(f"[bold]Epoch {epoch}/{config['training']['num_epochs']}[/bold]")

        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            grad_clip=config["training"]["max_grad_norm"], fp16=fp16, scaler=scaler,
            fgm=fgm,
        )
        global_step += len(train_loader)

        val_metrics, val_labels, val_preds = evaluate(model, val_loader, device, "Val")
        f1_clean = val_metrics.get("f1_clean", 0)
        f1_off = val_metrics.get("f1_offensive", 0)
        f1_hate = val_metrics.get("f1_hate", 0)

        log_dict = {
            "train/loss": train_metrics["loss"], "train/f1_macro": train_metrics["macro_f1"],
            "val/loss": val_metrics["loss"], "val/f1_macro": val_metrics["macro_f1"],
            "val/accuracy": val_metrics["accuracy"],
            "val/f1_clean": f1_clean, "val/f1_offensive": f1_off, "val/f1_hate": f1_hate,
        }
        logger.log(log_dict, step=global_step)

        console.print(
            f"  Train: loss={train_metrics['loss']:.4f} | f1_macro={train_metrics['macro_f1']:.4f}\n"
            f"  Val  : loss={val_metrics['loss']:.4f}   | f1_macro={val_metrics['macro_f1']:.4f}\n"
            f"         f1_clean={f1_clean:.4f} | f1_off={f1_off:.4f} | f1_hate={f1_hate:.4f}"
        )
        history.append({"epoch": epoch, **log_dict})

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            ckpt_dir = config["training"]["output_dir"]
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "val_f1": best_f1, "val_metrics": val_metrics, "config": config},
                os.path.join(ckpt_dir, "best_model.pt"),
            )
            model.backbone.save_pretrained(os.path.join(ckpt_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(ckpt_dir, "best_model"))
            console.print(
                f"  [bold green]✓ Best model saved "
                f"(Macro-F1={best_f1:.4f} | F1_OFF={f1_off:.4f} | F1_HATE={f1_hate:.4f})[/bold green]"
            )

        if early_stopping(val_metrics["macro_f1"]):
            console.print(f"\n[bold yellow]Early stopping at epoch {epoch}.[/bold yellow]")
            break

    console.print(f"\n[bold green]Training complete! Best Val Macro-F1={best_f1:.4f} at epoch {best_epoch}[/bold green]")

    with open(os.path.join(config["training"]["output_dir"], "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ── Final Test Evaluation ──────────────────────────────────────────────────
    console.print("\n[bold cyan]Final Test Evaluation...[/bold cyan]")
    ckpt = torch.load(os.path.join(config["training"]["output_dir"], "best_model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics, test_labels, test_preds = evaluate(model, test_loader, device, "Test")
    console.print("\n[bold]Test Results:[/bold]")
    print_classification_report(test_labels, test_preds)

    table = Table(title=f"Teacher ({config['model']['name']}) Test Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in test_metrics.items():
        table.add_row(k, f"{v:.4f}")
    console.print(table)

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "teacher_results.json"), "w") as f:
        json.dump({
            **test_metrics, "best_epoch": best_epoch, "model_name": config["model"]["name"],
            "augmentation": True, "preprocessing": True,
        }, f, indent=2)
    console.print(f"Results saved to {results_dir}/teacher_results.json")

    logger.finish()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune PhoBERT Teacher on prepared ViHSD data.")
    parser.add_argument("--config", type=str, default="configs/teacher_config.yaml")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config["_config_path"] = args.config

    if args.fp16:
        config["training"]["fp16"] = True

    train(config)


if __name__ == "__main__":
    main()