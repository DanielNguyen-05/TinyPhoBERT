"""
training/train_gating.py

Train Gating Network — kết hợp N expert bằng trọng số học được (input-dependent),
thay cho weighted-average tĩnh.

Usage:
    python training/train_gating.py \\
        --model_dirs checkpoints/teacher_large checkpoints/phobert_v2_fgm_noaug checkpoints/visobert_noaug checkpoints/qwen_classifier checkpoints/vibert_noaug \\
        --output_dir checkpoints/gating_network
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from rich.console import Console
from rich.table import Table
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.gating_network import GatingNetwork

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
    for i, name in enumerate(LABEL_NAMES):
        if i < len(per_class_f1):
            metrics[f"f1_{name.lower()}"] = per_class_f1[i]
    return metrics


def load_split(model_dirs, split):
    probs_list, labels_ref = [], None
    for d in model_dirs:
        p = np.load(os.path.join(d, f"{split}_probs.npy"))
        l = np.load(os.path.join(d, f"{split}_labels.npy"))
        if labels_ref is None:
            labels_ref = l
        else:
            assert np.array_equal(labels_ref, l), f"Labels mismatch in {d}/{split}_labels.npy"
        probs_list.append(p)
    stacked = np.stack(probs_list, axis=1)  # (N_samples, n_experts, n_classes)
    return stacked, labels_ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/gating_network")
    parser.add_argument("--hidden_dim", type=int, default=16,
                         help="Giảm từ 64→16: train trên VAL set nhỏ hơn nhiều (2672 samples), "
                              "cần regularize mạnh hơn để tránh overfit.")
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                         help="Tăng từ 1e-4→1e-2: chống overfit trên tập nhỏ.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--gate_val_split", type=float, default=0.2,
                         help="Tỷ lệ tách từ VAL set làm holdout cho early stopping của gating.")
    parser.add_argument(
        "--export_moe_teacher", action="store_true",
        help="Also combine in-sample train probabilities for distillation. "
             "Requires every expert export to use the same canonical train rows.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_experts = len(args.model_dirs)
    console.print(f"[bold cyan]Loading probabilities từ {n_experts} experts...[/bold cyan]")

    val_probs_full, val_labels_full = load_split(args.model_dirs, "val")
    test_probs, test_labels = load_split(args.model_dirs, "test")

    train_probs_leaky = train_labels_leaky = None
    if args.export_moe_teacher:
        train_probs_leaky, train_labels_leaky = load_split(args.model_dirs, "train")
        console.print(
            f"  train (in-sample; distillation export only): {train_probs_leaky.shape}"
        )
    console.print(f"  val (dùng để TRAIN gating — expert chưa từng thấy): {val_probs_full.shape}")
    console.print(f"  test: {test_probs.shape}")

    # ── QUAN TRỌNG: train Gating Network trên VAL set, KHÔNG dùng TRAIN set ───
    # Lý do: mỗi expert đã được train trực tiếp trên train set → dự đoán của
    # chúng trên CHÍNH train set cực kỳ tự tin/thiên lệch (gần như đã "học
    # thuộc"), không phản ánh đúng độ tin cậy thật khi gặp dữ liệu mới. Nếu
    # gating network học từ tín hiệu này, nó học sai lệch ("tin expert nào
    # overfit train mạnh nhất" chứ không phải "tin expert nào tổng quát tốt
    # nhất") → kết quả tệ hơn cả weighted-average tĩnh (tune trên val) như
    # đã thấy (67.61% < 69.17%).
    # Val set (2672 samples, expert CHƯA từng thấy) mới là tín hiệu đúng để
    # gating học "khi nào nên tin ai". Tách val thành gate_train/gate_val để
    # vẫn có early stopping hợp lệ mà không cần đụng đến train/test.
    n_val = len(val_labels_full)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_val)
    n_gate_val = int(n_val * args.gate_val_split)
    gate_val_idx = perm[:n_gate_val]
    gate_train_idx = perm[n_gate_val:]

    train_probs = val_probs_full[gate_train_idx]
    train_labels = val_labels_full[gate_train_idx]
    val_probs = val_probs_full[gate_val_idx]
    val_labels = val_labels_full[gate_val_idx]

    console.print(
        f"  → Tách VAL: gate_train={len(train_labels)} | gate_val={len(val_labels)} "
        f"(dùng cho early stopping của gating network)"
    )

    n_classes = train_probs.shape[-1]

    # Class weights cho Focal Loss
    from collections import Counter
    counter = Counter(train_labels.tolist())
    n_total = len(train_labels)
    class_weights = torch.tensor(
        [n_total / (n_classes * counter.get(c, 1)) for c in range(n_classes)],
        dtype=torch.float32,
    )
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.to(device)
    console.print(f"  Class weights: {class_weights.tolist()}")

    train_ds = TensorDataset(torch.tensor(train_probs, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(val_probs, dtype=torch.float32), torch.tensor(val_labels, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(test_probs, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 4, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 4, shuffle=False)

    model = GatingNetwork(
        n_experts=n_experts, n_classes=n_classes,
        hidden_dim=args.hidden_dim, dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []

    console.print(f"\n[bold cyan]Training Gating Network ({args.num_epochs} epochs max)...[/bold cyan]\n")

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_probs, batch_labels in train_loader:
            batch_probs, batch_labels = batch_probs.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            out = model.compute_loss(
                batch_probs, batch_labels, class_weights=class_weights,
                focal_gamma=args.focal_gamma, label_smoothing=args.label_smoothing,
            )
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Eval
        model.eval()
        all_preds, all_labels_eval = [], []
        with torch.no_grad():
            for batch_probs, batch_labels in val_loader:
                batch_probs = batch_probs.to(device)
                out = model(batch_probs)
                preds = out["final_probs"].argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels_eval.extend(batch_labels.numpy().tolist())

        val_metrics = compute_metrics(all_labels_eval, all_preds)
        scheduler.step(val_metrics["macro_f1"])

        history.append({"epoch": epoch, "train_loss": total_loss / len(train_loader), **{f"val_{k}": v for k, v in val_metrics.items()}})

        if epoch % 5 == 0 or epoch == 1:
            console.print(
                f"Epoch {epoch:3d} | train_loss={total_loss/len(train_loader):.4f} | "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} | val_acc={val_metrics['accuracy']:.4f}"
            )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict(), "val_f1": best_f1, "epoch": epoch,
                        "n_experts": n_experts, "n_classes": n_classes,
                        "hidden_dim": args.hidden_dim, "dropout": args.dropout,
                        "model_dirs": args.model_dirs},
                       os.path.join(args.output_dir, "best_gating.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                console.print(f"\n[bold yellow]Early stopping tại epoch {epoch}.[/bold yellow]")
                break

    console.print(f"\n[bold green]Training complete! Best Val Macro-F1={best_f1:.4f} tại epoch {best_epoch}[/bold green]")

    with open(os.path.join(args.output_dir, "gating_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Final test eval với best checkpoint
    ckpt = torch.load(os.path.join(args.output_dir, "best_gating.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds, all_labels_test, all_gate_weights = [], [], []
    with torch.no_grad():
        for batch_probs, batch_labels in test_loader:
            batch_probs = batch_probs.to(device)
            out = model(batch_probs)
            preds = out["final_probs"].argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels_test.extend(batch_labels.numpy().tolist())
            all_gate_weights.append(out["gate_weights"].cpu().numpy())

    test_metrics = compute_metrics(all_labels_test, all_preds)
    gate_weights_all = np.concatenate(all_gate_weights, axis=0)
    avg_gate_weights = gate_weights_all.mean(axis=0)

    table = Table(title="Gating Network — Test Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in test_metrics.items():
        table.add_row(k, f"{v:.4f}")
    console.print(table)

    console.print(f"\n[bold]Trọng số gate trung bình trên test set:[/bold]")
    for d, w in zip(args.model_dirs, avg_gate_weights):
        console.print(f"  {os.path.basename(d)}: {w:.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/gating_results.json", "w") as f:
        json.dump({
            **test_metrics, "best_epoch": best_epoch,
            "avg_gate_weights": dict(zip([os.path.basename(d) for d in args.model_dirs], avg_gate_weights.tolist())),
        }, f, indent=2)
    console.print("\nResults saved to results/gating_results.json")

    # Lưu final probs của MoE (train+val+test) — dùng làm teacher signal cho distillation.
    # QUAN TRỌNG: dùng train_probs_leaky/val_probs_full (dữ liệu ĐẦY ĐỦ gốc),
    # KHÔNG dùng train_probs/val_probs (chỉ là tập con gate_train/gate_val
    # dùng để TRAIN gating network ở trên) — nếu không, moe_teacher_probs sẽ
    # có kích thước sai, gây lệch với data/augmented/train.csv thật khi
    # train_moe_distill.py load lại.
    if args.export_moe_teacher:
        moe_dir = os.path.join(args.output_dir, "moe_teacher_probs")
        os.makedirs(moe_dir, exist_ok=True)
        with torch.no_grad():
            for split, probs_arr, labels_arr in [
                ("train", train_probs_leaky, train_labels_leaky),
                ("val", val_probs_full, val_labels_full),
                ("test", test_probs, test_labels),
            ]:
                probs_t = torch.tensor(probs_arr, dtype=torch.float32).to(device)
                out = model(probs_t)
                final_probs = out["final_probs"].cpu().numpy()
                np.save(os.path.join(moe_dir, f"{split}_probs.npy"), final_probs)
                np.save(os.path.join(moe_dir, f"{split}_labels.npy"), labels_arr)
        console.print(
            f"MoE teacher probs (train={len(train_labels_leaky)}, "
            f"val={len(val_labels_full)}, test={len(test_labels)}) "
            f"saved to {moe_dir}/"
        )


if __name__ == "__main__":
    main()
