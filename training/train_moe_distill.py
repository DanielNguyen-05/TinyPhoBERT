"""Distill fused heterogeneous teachers into DAMS-TinyPhoBERT.

Usage:
    python training/train_moe_distill.py \\
        --moe_teacher_dir checkpoints/class_aware_ensemble \\
        --config configs/comparison_distillation_config.yaml \\
        --student_config configs/comparison_student_config.yaml \\
        --init_teacher_checkpoint checkpoints/teacher_strong/best_model.pt \\
        --output_dir checkpoints/dams_multiteacher
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.student import build_student_from_config
from utils.seed import set_seed

console = Console()
LABEL_NAMES = ["CLEAN", "OFFENSIVE", "HATE"]


def compute_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_precision = precision_score(
        y_true, y_pred, average=None, zero_division=0
    )
    per_class_recall = recall_score(
        y_true, y_pred, average=None, zero_division=0
    )
    for i, name in enumerate(LABEL_NAMES):
        if i < len(per_class_f1):
            metrics[f"f1_{name.lower()}"] = per_class_f1[i]
            metrics[f"precision_{name.lower()}"] = per_class_precision[i]
            metrics[f"recall_{name.lower()}"] = per_class_recall[i]
    return metrics


class DistillDataset(Dataset):
    """Text + hard label + soft label (MoE teacher probs) cho từng sample."""
    def __init__(self, texts, hard_labels, teacher_probs, tokenizer, max_length=128):
        self.texts = texts
        self.hard_labels = hard_labels
        self.teacher_probs = teacher_probs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.hard_labels[idx], dtype=torch.long),
            "teacher_probs": torch.tensor(self.teacher_probs[idx], dtype=torch.float32),
        }


def distill_loss(
    student_logits,
    teacher_probs,
    hard_labels,
    temperature=4.0,
    alpha=0.7,
    class_weights=None,
    confidence_floor=0.0,
    focal_gamma=0.0,
    label_smoothing=0.0,
):
    """
    Loss = alpha * KD(student, teacher_probs) + (1-alpha) * CE(student, hard_labels)
    KD dùng KL divergence với temperature scaling (Hinton et al. 2015).
    """
    student_log_probs_T = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs_T = teacher_probs.clamp(min=1e-8)
    teacher_probs_T = teacher_probs_T ** (1.0 / temperature)
    teacher_probs_T = teacher_probs_T / teacher_probs_T.sum(dim=-1, keepdim=True)

    kd_per_class = F.kl_div(
        student_log_probs_T, teacher_probs_T, reduction="none"
    ).sum(dim=-1)
    if confidence_floor > 0:
        # Normalized confidence is 0 for uniform and 1 for a one-hot teacher.
        n_classes = teacher_probs.shape[-1]
        original_teacher = teacher_probs / teacher_probs.sum(
            dim=-1, keepdim=True
        )
        entropy = -(
            original_teacher * original_teacher.clamp_min(1e-8).log()
        ).sum(-1)
        confidence = 1.0 - entropy / np.log(n_classes)
        confidence = confidence_floor + (1.0 - confidence_floor) * confidence
        kd_loss = (
            kd_per_class * confidence.detach()
        ).sum() / confidence.sum().clamp_min(1e-8)
    else:
        kd_loss = kd_per_class.mean()
    kd_loss = kd_loss * (temperature ** 2)
    ce_per_sample = F.cross_entropy(
        student_logits, hard_labels, reduction="none",
        label_smoothing=label_smoothing,
    )
    if class_weights is not None:
        sample_multiplier = class_weights[hard_labels]
    else:
        sample_multiplier = torch.ones_like(ce_per_sample)
    if focal_gamma > 0:
        target_probability = F.softmax(
            student_logits, dim=-1
        ).gather(1, hard_labels[:, None]).squeeze(1)
        sample_multiplier = sample_multiplier * (
            1.0 - target_probability
        ).pow(focal_gamma)
    ce_loss = (
        ce_per_sample * sample_multiplier
    ).sum() / sample_multiplier.sum().clamp_min(1e-8)

    total = alpha * kd_loss + (1 - alpha) * ce_loss
    return total, kd_loss.detach(), ce_loss.detach()


def anchor_hidden_loss(
    student_projected,
    teacher_hidden,
    attention_mask,
    layer_mapping,
):
    """Cosine KD from one representation anchor, complementary to fused logits."""
    mask = attention_mask.unsqueeze(-1).to(student_projected[0].dtype)
    denominator = mask.sum(dim=1).clamp_min(1.0)
    losses = []
    for student_idx, teacher_idx in layer_mapping.items():
        student_layer = student_idx + 1  # hidden tuple includes embeddings
        teacher_layer = teacher_idx + 1
        if (
            student_layer >= len(student_projected)
            or teacher_layer >= len(teacher_hidden)
        ):
            continue
        student_mean = (
            student_projected[student_layer] * mask
        ).sum(dim=1) / denominator
        teacher_mean = (
            teacher_hidden[teacher_layer].detach() * mask
        ).sum(dim=1) / denominator
        losses.append(
            (1.0 - F.cosine_similarity(
                student_mean, teacher_mean, dim=-1
            )).mean()
        )
    if not losses:
        raise ValueError("No valid student/anchor-teacher layer mappings")
    return torch.stack(losses).mean()


class EarlyStopping:
    def __init__(self, patience=6, min_delta=1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.counter, self.best = 0, None

    def __call__(self, score):
        if self.best is None or score > self.best + self.min_delta:
            self.best, self.counter = score, 0
            return False
        self.counter += 1
        return self.counter >= self.patience


@torch.no_grad()
def predict_probabilities(model, dataloader, device):
    model.eval()
    probabilities, labels = [], []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids, attention_mask)
        probabilities.append(
            F.softmax(outputs["logits"].float(), dim=-1).cpu().numpy()
        )
        labels.append(batch["labels"].numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def train(args, config):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    metadata_path = os.path.join(args.moe_teacher_dir, "ensemble_metadata.json")
    if os.path.isfile(metadata_path):
        with open(metadata_path) as metadata_file:
            teacher_metadata = json.load(metadata_file)
        if not teacher_metadata.get("train_targets_oof", False):
            console.print(
                "[bold yellow]Note: ensemble train targets are in-sample. "
                "This is valid for standard KD, but OOF targets are preferable "
                "for leakage-free stacking and matched confidence.[/bold yellow]"
            )

    tokenizer_name = args.tokenizer_name or config.get("teacher", {}).get(
        "model_name", "vinai/phobert-large"
    )
    console.print(
        "[cyan]Resolved distillation settings:[/cyan] "
        f"epochs={args.num_epochs}, batch={args.batch_size}, "
        f"lr={args.lr:.2e}, warmup={args.warmup_ratio:.2f}, "
        f"T={args.temperature:.2f}, alpha={args.alpha:.2f}, "
        f"focal_gamma={args.focal_gamma:.2f}, "
        f"class_weights={args.use_class_weights}"
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    text_col = config["data"]["text_col"]
    label_col = config["data"]["label_col"]

    data_config = config["data"]
    if all(data_config.get(f"{split}_file") for split in ["train", "val", "test"]):
        dfs = {
            split: pd.read_csv(data_config[f"{split}_file"])
            for split in ["train", "val", "test"]
        }
    else:
        aug_dir = Path(data_config.get("augmented_dir", "data/augmented"))
        dfs = {
            split: pd.read_csv(aug_dir / f"{split}.csv")
            for split in ["train", "val", "test"]
        }

    teacher_probs = {}
    teacher_labels = {}
    for s in ["train", "val", "test"]:
        teacher_probs[s] = np.load(os.path.join(args.moe_teacher_dir, f"{s}_probs.npy"))
        teacher_labels[s] = np.load(os.path.join(args.moe_teacher_dir, f"{s}_labels.npy"))
        id_path = os.path.join(
            args.moe_teacher_dir, f"{s}_sample_ids.npy"
        )
        if os.path.isfile(id_path) and "sample_id" in dfs[s].columns:
            target_ids = np.load(id_path).astype(str)
            frame_ids = dfs[s]["sample_id"].astype(str)
            if frame_ids.duplicated().any() or len(set(target_ids.tolist())) != len(target_ids):
                raise ValueError(f"Duplicate sample IDs in distillation split={s}")
            if set(frame_ids.tolist()) != set(target_ids.tolist()):
                raise ValueError(
                    f"CSV and teacher target sample IDs differ for split={s}"
                )
            dfs[s] = (
                dfs[s].assign(sample_id=frame_ids)
                .set_index("sample_id", drop=False)
                .loc[target_ids.tolist()]
                .reset_index(drop=True)
            )
        elif os.path.isfile(id_path) or "sample_id" in dfs[s].columns:
            console.print(
                f"[yellow]{s}: sample IDs exist on only one side; using "
                "legacy positional alignment.[/yellow]"
            )
        if teacher_probs[s].shape != (len(dfs[s]), 3):
            raise ValueError(
                f"{s}_probs shape {teacher_probs[s].shape} does not match "
                f"{len(dfs[s])} rows and 3 labels"
            )
        if not np.isfinite(teacher_probs[s]).all() or (teacher_probs[s] < 0).any():
            raise ValueError(f"{s}_probs contains invalid probabilities")
        probability_sums = teacher_probs[s].sum(axis=1, keepdims=True)
        if (probability_sums <= 0).any():
            raise ValueError(f"{s}_probs contains a zero-sum row")
        teacher_probs[s] = teacher_probs[s] / probability_sums
        df_labels = dfs[s][label_col].astype(int).values
        assert np.array_equal(df_labels, teacher_labels[s]), (
            f"Label mismatch giữa {s}.csv và {s}_labels.npy — kiểm tra lại "
            f"data/augmented có khớp với lúc train experts không."
        )
    datasets = {}
    for s in ["train", "val", "test"]:
        datasets[s] = DistillDataset(
            dfs[s][text_col].astype(str).tolist(),
            dfs[s][label_col].astype(int).tolist(),
            teacher_probs[s], tokenizer, max_length=args.max_seq_length,
        )

    train_loader = DataLoader(
        datasets["train"], batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        datasets["val"], batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        datasets["test"], batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers,
    )

    console.print(f"[bold cyan]Building TinyPhoBERT student...[/bold cyan]")
    with open(args.student_config) as f:
        student_config = yaml.safe_load(f)
    model = build_student_from_config(student_config).to(device)
    console.print(f"  Student params: {model.count_parameters():,}")

    anchor_teacher = None
    layer_mapping = {
        int(student_idx): int(teacher_idx)
        for student_idx, teacher_idx in student_config.get(
            "model", {}
        ).get("layer_mapping", {}).items()
    }
    teacher_checkpoint = (
        args.anchor_teacher_checkpoint or args.init_teacher_checkpoint
    )
    if args.init_teacher_checkpoint:
        from models.teacher import PhoBERTTeacher

        console.print(
            f"[bold cyan]Initializing DAMS backbone from {args.init_teacher_checkpoint}[/bold cyan]"
        )
        initialization_teacher = PhoBERTTeacher.from_pretrained_checkpoint(
            args.init_teacher_checkpoint,
            output_attentions=False,
        )
        model.init_from_teacher(
            initialization_teacher, layer_mapping=layer_mapping
        )
        if (
            args.hidden_kd_weight > 0
            and teacher_checkpoint == args.init_teacher_checkpoint
        ):
            anchor_teacher = initialization_teacher
        else:
            del initialization_teacher

    if args.hidden_kd_weight > 0 and anchor_teacher is None:
        if not teacher_checkpoint:
            raise ValueError(
                "--hidden_kd_weight > 0 requires --anchor_teacher_checkpoint "
                "or --init_teacher_checkpoint"
            )
        from models.teacher import PhoBERTTeacher

        anchor_teacher = PhoBERTTeacher.from_pretrained_checkpoint(
            teacher_checkpoint,
            output_attentions=False,
        )
    if anchor_teacher is not None:
        if model.teacher_hidden_size != anchor_teacher.hidden_size:
            raise ValueError(
                "Student hidden projection size "
                f"{model.teacher_hidden_size} does not match anchor teacher "
                f"hidden size {anchor_teacher.hidden_size}"
            )
        anchor_teacher.freeze()
        anchor_teacher.to(device).eval()
        console.print(
            f"  Anchor hidden-state KD: weight={args.hidden_kd_weight:.3f}"
        )

    class_weights = None
    if args.use_class_weights:
        counts = np.bincount(
            dfs["train"][label_col].astype(int).values, minlength=3
        ).astype(np.float32)
        weights = (
            len(dfs["train"]) / (3.0 * np.maximum(counts, 1.0))
        ) ** args.class_weight_power
        weights /= weights.mean()
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        console.print(f"  Hard-label class weights: {weights.tolist()}")

    # The multi-scale classification head is randomly initialized while the
    # sliced backbone already contains pretrained weights.  A single LR made
    # the backbone drift early and left the head under-trained.
    head_prefixes = ("multiscale_head.", "classifier.")
    backbone_parameters, head_parameters = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(head_prefixes):
            head_parameters.append(parameter)
        elif name.startswith("hidden_projection.") and anchor_teacher is None:
            # Logit-only multi-teacher KD has no hidden-state teacher target.
            parameter.requires_grad_(False)
        elif name.startswith("hidden_projection."):
            head_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.lr},
            {
                "params": head_parameters,
                "lr": args.lr * args.head_lr_multiplier,
            },
        ],
        weight_decay=args.weight_decay,
    )
    console.print(
        f"  Learning rates: backbone={args.lr:.2e} | "
        f"head={args.lr * args.head_lr_multiplier:.2e}"
    )
    num_steps = len(train_loader) * args.num_epochs
    num_warmup = int(num_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup, num_steps)

    early_stopping = EarlyStopping(patience=args.patience)
    best_f1, best_epoch = 0.0, 0
    history = []

    console.print(f"\n[bold cyan]Distilling ({args.num_epochs} epochs max)...[/bold cyan]\n")

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss = total_kd = total_ce = total_hidden = 0.0
        for batch in tqdm(train_loader, desc=f"  Epoch {epoch}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            teacher_p = batch["teacher_probs"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids, attention_mask,
                return_distill_outputs=anchor_teacher is not None,
                return_attentions=False,
            )
            loss, kd_loss, ce_loss = distill_loss(
                outputs["logits"], teacher_p, labels,
                temperature=args.temperature, alpha=args.alpha,
                class_weights=class_weights,
                confidence_floor=args.confidence_floor,
                focal_gamma=args.focal_gamma,
                label_smoothing=args.label_smoothing,
            )
            hidden_loss = torch.tensor(0.0, device=device)
            if anchor_teacher is not None:
                with torch.no_grad():
                    anchor_outputs = anchor_teacher(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                hidden_loss = anchor_hidden_loss(
                    outputs["projected_hidden"],
                    anchor_outputs["hidden_states"],
                    attention_mask,
                    layer_mapping,
                )
                loss = loss + args.hidden_kd_weight * hidden_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            total_kd += kd_loss.item()
            total_ce += ce_loss.item()
            total_hidden += hidden_loss.detach().item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids, attention_mask)
                preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(batch["labels"].numpy().tolist())

        val_metrics = compute_metrics(all_labels, all_preds)
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            "train_kd_loss": total_kd / len(train_loader),
            "train_ce_loss": total_ce / len(train_loader),
            "train_hidden_loss": total_hidden / len(train_loader),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        console.print(
            f"Epoch {epoch:3d} | loss={total_loss/len(train_loader):.4f} "
            f"(KD={total_kd/len(train_loader):.4f}, CE={total_ce/len(train_loader):.4f}) | "
            f"hidden={total_hidden/len(train_loader):.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | val_f1_off={val_metrics.get('f1_offensive',0):.4f} | "
            f"val_f1_hate={val_metrics.get('f1_hate',0):.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = val_metrics["macro_f1"], epoch
            torch.save({"model_state_dict": model.state_dict(), "val_f1": best_f1,
                        "epoch": epoch, "student_config": student_config,
                        "distillation_args": vars(args)},
                       os.path.join(args.output_dir, "best_model.pt"))
            console.print(f"  [bold green]✓ Best model saved (Macro-F1={best_f1:.4f})[/bold green]")

        if early_stopping(val_metrics["macro_f1"]):
            console.print(f"\n[bold yellow]Early stopping tại epoch {epoch}.[/bold yellow]")
            break

    console.print(f"\n[bold green]Distillation complete! Best Val Macro-F1={best_f1:.4f} tại epoch {best_epoch}[/bold green]")
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    ckpt = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    val_student_probs, val_student_labels = predict_probabilities(
        model, val_loader, device
    )
    test_student_probs, test_student_labels = predict_probabilities(
        model, test_loader, device
    )
    np.save(os.path.join(args.output_dir, "val_probs.npy"), val_student_probs)
    np.save(os.path.join(args.output_dir, "val_labels.npy"), val_student_labels)
    np.save(os.path.join(args.output_dir, "test_probs.npy"), test_student_probs)
    np.save(os.path.join(args.output_dir, "test_labels.npy"), test_student_labels)

    test_metrics = compute_metrics(
        test_student_labels.tolist(),
        test_student_probs.argmax(axis=1).tolist(),
    )
    table = Table(title="TinyPhoBERT (MoE-Distilled) Test Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in test_metrics.items():
        table.add_row(k, f"{v:.4f}")
    console.print(table)

    os.makedirs("results", exist_ok=True)
    with open("results/moe_distill_results.json", "w") as f:
        json.dump({
            **test_metrics,
            "best_val_macro_f1": best_f1,
            "best_epoch": best_epoch,
            "student_params": model.count_parameters(),
            "teacher_artifact": args.moe_teacher_dir,
            "temperature": args.temperature,
            "alpha": args.alpha,
            "confidence_floor": args.confidence_floor,
            "use_class_weights": args.use_class_weights,
            "class_weight_power": args.class_weight_power,
            "focal_gamma": args.focal_gamma,
            "label_smoothing": args.label_smoothing,
            "backbone_lr": args.lr,
            "head_lr": args.lr * args.head_lr_multiplier,
            "hidden_kd_weight": args.hidden_kd_weight,
            "anchor_teacher_checkpoint": teacher_checkpoint,
        }, f, indent=2)
    console.print("Results saved to results/moe_distill_results.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moe_teacher_dir", type=str, required=True)
    parser.add_argument(
        "--config", type=str,
        default="configs/comparison_distillation_config.yaml",
    )
    parser.add_argument(
        "--student_config", type=str,
        default="configs/comparison_student_config.yaml",
    )
    parser.add_argument("--output_dir", type=str, default="checkpoints/dams_multiteacher")
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--head_lr_multiplier", type=float, default=5.0)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None, help="Trọng số KD loss vs CE loss")
    parser.add_argument(
        "--confidence_floor", type=float, default=1.0,
        help="Minimum KD weight; uncertain teacher targets receive less weight.",
    )
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument(
        "--class_weight_power", type=float, default=0.5,
        help="0=no reweighting, 0.5=sqrt inverse frequency, 1=full inverse.",
    )
    parser.add_argument("--focal_gamma", type=float, default=None)
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument("--init_teacher_checkpoint", type=str, default=None)
    parser.add_argument("--anchor_teacher_checkpoint", type=str, default=None)
    parser.add_argument(
        "--hidden_kd_weight", type=float, default=0.0,
        help="Optional cosine hidden-state KD from one PhoBERT anchor teacher.",
    )
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    training_config = config.get("training", {})
    distillation_config = config.get("distillation", {})
    resolved_defaults = {
        "num_epochs": training_config.get("num_epochs", 40),
        "batch_size": training_config.get("batch_size", 32),
        "max_seq_length": training_config.get("max_seq_length", 128),
        "lr": training_config.get("learning_rate", 2e-5),
        "weight_decay": training_config.get("weight_decay", 0.01),
        "warmup_ratio": training_config.get("warmup_ratio", 0.1),
        "num_workers": training_config.get("dataloader_num_workers", 4),
        "patience": training_config.get("early_stopping_patience", 6),
        "temperature": distillation_config.get("temperature", 2.0),
        "alpha": distillation_config.get("alpha", 0.5),
        "focal_gamma": distillation_config.get("focal_gamma", 1.0),
        "label_smoothing": distillation_config.get("label_smoothing", 0.05),
    }
    for argument, default_value in resolved_defaults.items():
        if getattr(args, argument) is None:
            setattr(args, argument, default_value)

    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be in [0, 1]")
    if not 0.0 <= args.confidence_floor <= 1.0:
        parser.error("--confidence_floor must be in [0, 1]")
    if not 0.0 <= args.class_weight_power <= 1.0:
        parser.error("--class_weight_power must be in [0, 1]")
    if args.focal_gamma < 0:
        parser.error("--focal_gamma must be non-negative")
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error("--label_smoothing must be in [0, 1)")
    if args.head_lr_multiplier <= 0:
        parser.error("--head_lr_multiplier must be positive")
    if args.hidden_kd_weight < 0:
        parser.error("--hidden_kd_weight must be non-negative")
    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error("--warmup_ratio must be in [0, 1)")

    train(args, config)


if __name__ == "__main__":
    main()
