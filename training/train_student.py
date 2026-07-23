"""
training/train_student.py

Train TinyPhoBERT Student with Multi-Level Knowledge Distillation.

Usage:
    # Full distillation (A4)
    python training/train_student.py --config configs/distillation_config.yaml

    # No distillation (A1 baseline)
    python training/train_student.py --config configs/distillation_config.yaml --no_kd

    # Custom weights
    python training/train_student.py --alpha 0.7 --beta 0.2 --gamma 0.1
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.teacher import PhoBERTTeacher, get_teacher_tokenizer
from models.student import TinyPhoBERT, build_student_from_config
from models.distillation import (
    MultiLevelDistillationLoss,
    DistillationTrainer,
    build_distillation_loss_from_config,
)
from utils.data_utils import load_vihsd_from_csv, build_datasets, get_class_weights, get_weighted_sampler
from utils.metrics import compute_metrics, print_classification_report
from utils.seed import set_seed, get_device
from utils.logging_utils import ExperimentLogger

console = Console()


def train_epoch(
    distill_trainer: DistillationTrainer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    grad_clip: float = 1.0,
    fp16: bool = False,
    scaler=None,
) -> dict:
    distill_trainer.student.train()
    total_losses = {
        "loss": 0.0, "loss_ce": 0.0,
        "loss_kd": 0.0, "loss_hidden": 0.0, "loss_att": 0.0,
    }
    all_preds, all_labels = [], []

    pbar = tqdm(dataloader, desc="  Training", leave=False)
    for batch in pbar:
        optimizer.zero_grad()

        if fp16 and scaler is not None:
            from torch.cuda.amp import autocast
            with autocast():
                losses = distill_trainer.distill_step(batch)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(distill_trainer.student.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses = distill_trainer.distill_step(batch)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(distill_trainer.student.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()

        for k in total_losses:
            total_losses[k] += losses[k].item()

        # Get predictions from student
        with torch.no_grad():
            labels = batch["labels"].to(distill_trainer.device)
            input_ids = batch["input_ids"].to(distill_trainer.device)
            attention_mask = batch["attention_mask"].to(distill_trainer.device)
            out = distill_trainer.student(input_ids, attention_mask)
            preds = torch.argmax(out["logits"], dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        pbar.set_postfix({"loss": f"{losses['loss'].item():.4f}"})

    n = len(dataloader)
    metrics = {k: v / n for k, v in total_losses.items()}
    task_metrics = compute_metrics(all_labels, all_preds)
    metrics.update(task_metrics)
    return metrics


@torch.no_grad()
def evaluate(
    student: TinyPhoBERT,
    dataloader: DataLoader,
    device: torch.device,
    split_name: str = "Val",
) -> tuple:
    student.eval()
    all_preds, all_labels = [], []

    pbar = tqdm(dataloader, desc=f"  {split_name}", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = student(input_ids, attention_mask)
        preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    return metrics, all_labels, all_preds


def train(config: dict, run_name: Optional[str] = None) -> dict:
    """
    Main distillation training function.

    Args:
        config: Full distillation config dictionary.
        run_name: Optional override for the experiment run name.

    Returns:
        Test metrics dictionary.
    """
    training_cfg = config["training"]
    data_cfg = config["data"]
    log_cfg = config.get("logging", {})
    distill_cfg = config["distillation"]

    # QUAN TRỌNG: load student_config.yaml (chứa layer_mapping thực tế cho
    # PhoBERT-large 24-layer Teacher). Nếu không, MultiLevelDistillationLoss
    # sẽ rơi vào default {0:1,1:3,...,5:11} — mapping này được thiết kế cho
    # teacher 12-layer (phobert-base), SAI hoàn toàn với phobert-large 24-layer
    # đang dùng. Hậu quả: hidden-state KD và attention KD ép student học theo
    # các teacher layer giữa thay vì layer cuối (gần classifier, mang thông
    # tin task-specific quan trọng nhất cho OFFENSIVE/HATE).
    student_cfg_path = config.get("student", {}).get("config_path")
    if student_cfg_path and os.path.isfile(student_cfg_path):
        with open(student_cfg_path, "r") as f:
            student_full_cfg = yaml.safe_load(f)
        config["student"]["model"] = student_full_cfg.get("model", {})
        # training/data trong student_config.yaml không được dùng ở đây —
        # distillation_config.yaml là nguồn sự thật cho training/data.
    else:
        console.print(
            f"[bold red]✗ Cảnh báo: không tìm thấy student.config_path "
            f"('{student_cfg_path}'). layer_mapping sẽ dùng giá trị mặc định "
            f"SAI cho teacher 24-layer.[/bold red]"
        )

    set_seed(data_cfg["seed"])
    device = get_device()

    actual_run_name = run_name or log_cfg.get("run_name", "distillation")
    output_dir = os.path.join(training_cfg["output_dir"], actual_run_name)
    os.makedirs(output_dir, exist_ok=True)

    logger = ExperimentLogger(
        project_name=log_cfg.get("project_name", "TinyPhoBERT"),
        run_name=actual_run_name,
        log_dir=os.path.join(log_cfg.get("log_dir", "logs"), actual_run_name),
        use_wandb=log_cfg.get("use_wandb", False),
        use_tensorboard=log_cfg.get("use_tensorboard", True),
        config=config,
    )

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    console.print("[bold cyan]Loading tokenizer...[/bold cyan]")
    # QUAN TRỌNG: dùng đúng tokenizer của Teacher (phobert-large), không phải
    # phobert-base. Dù 2 model share BPE vocab nên hiện tại không gây lỗi rõ
    # rệt, dùng sai tên model ở đây là nguồn lỗi tiềm ẩn nếu vocab thay đổi.
    tokenizer = get_teacher_tokenizer(config["teacher"].get("model_name", "vinai/phobert-large"))

    # ── Data ───────────────────────────────────────────────────────────────────
    console.print("[bold cyan]Loading ViHSD dataset...[/bold cyan]")
    train_df, val_df, test_df = load_vihsd_from_csv(
        data_cfg["train_file"], data_cfg["val_file"], data_cfg["test_file"],
        text_col=data_cfg["text_col"], label_col=data_cfg["label_col"],
    )
    train_ds, val_ds, test_ds = build_datasets(
        train_df, val_df, test_df, tokenizer,
        text_col=data_cfg["text_col"],
        label_col=data_cfg["label_col"],
        max_length=training_cfg["max_seq_length"],
    )

    train_labels = train_df[data_cfg["label_col"]].astype(int).tolist()

    # ── Class Weights + Weighted Sampler ────────────────────────────────────
    use_weighted_sampler = training_cfg.get("use_weighted_sampler", True)
    class_weights = get_class_weights(train_labels, num_classes=3)
    class_weights_dev = class_weights.to(device)
    console.print(
        f"  [yellow]Class weights:[/yellow] "
        f"CLEAN={class_weights[0]:.3f} | "
        f"OFFENSIVE={class_weights[1]:.3f} | "
        f"HATE={class_weights[2]:.3f}"
    )

    nw = training_cfg.get("dataloader_num_workers", 4)
    bs = training_cfg["batch_size"]
    sampler_strength = training_cfg.get("sampler_strength", 0.5)  # soft rebalancing
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        sampler=get_weighted_sampler(train_labels, strength=sampler_strength) if use_weighted_sampler else None,
        shuffle=False if use_weighted_sampler else True,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
    )
    if use_weighted_sampler:
        console.print(f"  [yellow]WeightedRandomSampler: ON[/yellow] strength={sampler_strength}")
    val_loader = DataLoader(val_ds, batch_size=bs * 2, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs * 2, shuffle=False, num_workers=nw)

    # ── Teacher ────────────────────────────────────────────────────────────────
    teacher_path = config["teacher"]["model_path"]
    console.print(f"[bold cyan]Loading Teacher from: {teacher_path}[/bold cyan]")
    checkpoint_candidates = [
        teacher_path,
        teacher_path + ".pt",
        os.path.join(teacher_path, "best_model.pt"),
        os.path.join(os.path.dirname(teacher_path), "best_model.pt"),
    ]
    ckpt_file = next((p for p in checkpoint_candidates if os.path.isfile(p)), None)
    if ckpt_file is not None:
        teacher = PhoBERTTeacher.from_pretrained_checkpoint(ckpt_file)
    elif os.path.isdir(teacher_path):
        # HuggingFace format
        from transformers import AutoModelForSequenceClassification
        teacher = PhoBERTTeacher(model_name=teacher_path)
    else:
        console.print(
            f"[yellow]Warning: Teacher checkpoint not found at '{teacher_path}'. "
            "Loading fresh PhoBERT-base (results will be suboptimal). "
            "Please run training/train_teacher.py first.[/yellow]"
        )
        teacher = PhoBERTTeacher("vinai/phobert-base")

    teacher.freeze()
    teacher_params = teacher.count_parameters()
    console.print(f"  Teacher params: [bold]{teacher_params:,}[/bold]")

    # ── Student ────────────────────────────────────────────────────────────────
    console.print("[bold cyan]Building TinyPhoBERT Student...[/bold cyan]")
    student = build_student_from_config(config.get("student", {}))
    student_params = student.count_parameters()
    student_mb = student.model_size_mb()
    student.print_summary()

    # ── Init student from teacher (weight slicing) ─────────────────────────
    init_from_teacher_flag = config.get("student", {}).get("init_from_teacher", True)
    if init_from_teacher_flag:
        console.print("[bold cyan]Initializing student from teacher weights (weight slicing)...[/bold cyan]")
        init_mapping = config.get("student", {}).get("model", {}).get("layer_mapping")
        if init_mapping:
            init_mapping = {int(k): int(v) for k, v in init_mapping.items()}
        student.init_from_teacher(teacher, layer_mapping=init_mapping)
        console.print("  [bold green]✓ Student initialized from teacher layers[/bold green]")

    # ── Distillation Loss ─────────────────────────────────────────────────
    # Dùng factory build_distillation_loss_from_config thay vì khởi tạo trực
    # tiếp MultiLevelDistillationLoss — factory này đọc đúng layer_mapping
    # từ config["student"]["model"]["layer_mapping"] (đã load ở trên).
    # Khởi tạo trực tiếp trước đây bỏ qua tham số layer_mapping → luôn rơi
    # vào default {0:1,...,5:11}, sai với teacher 24-layer.
    distill_loss = build_distillation_loss_from_config(config, class_weights=class_weights_dev)
    console.print(
        f"  [yellow]Layer mapping (student→teacher):[/yellow] {distill_loss.layer_mapping}"
    )
    console.print(
        f"  Distillation: KD={distill_cfg['use_logit_kd']} | "
        f"Hidden={distill_cfg['use_hidden_kd']} | "
        f"Att={distill_cfg['use_attention_kd']}\n"
        f"  Weights: α={distill_cfg['alpha']} β={distill_cfg['beta']} γ={distill_cfg['gamma']} "
        f"T={distill_cfg['temperature']}\n"
        f"  Loss: {'Focal' if distill_cfg.get('use_focal_loss', True) else 'WeightedCE'} "
        f"| hidden='{distill_cfg.get('hidden_loss_type', 'cosine')}' "
        f"| label_smooth={distill_cfg.get('label_smoothing', 0.1)}"
    )

    distill_trainer = DistillationTrainer(teacher, student, distill_loss, device)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    # CHẨN ĐOÁN: best_epoch=5/60 ở lần chạy trước cho thấy model đạt đỉnh
    # ngay khi warmup_ratio=0.1 (~6 epoch) còn chưa kết thúc — tức là LR
    # chưa đạt full strength (5e-5) lúc đạt best, rồi LR tiếp tục tăng và
    # phá hỏng model đã tốt. Đây là dấu hiệu LR quá cao cho student 35M
    # tham số đang tối ưu đồng thời 4 loss term (CE + logit KD + hidden KD
    # + attention KD), đặc biệt khi init_from_teacher=true (model đã ở gần
    # một điểm tốt, dễ bị "đá văng" bởi bước gradient lớn).
    #
    # Sửa: (1) giảm LR mặc định, (2) rút ngắn warmup để đạt đỉnh sớm hơn rồi
    # giảm dần ngay (cosine), tránh giữ LR cao kéo dài, (3) log LR mỗi epoch
    # để quan sát trực tiếp quan hệ giữa LR và val F1.
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )
    num_steps = len(train_loader) * training_cfg["num_epochs"]
    num_warmup = int(num_steps * training_cfg["warmup_ratio"])
    use_cosine = training_cfg.get("use_cosine_schedule", True)
    if use_cosine:
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup, num_steps)
        console.print(f"  [yellow]Schedule: Cosine | warmup_steps={num_warmup}/{num_steps}[/yellow]")
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup, num_steps)
        console.print(f"  [yellow]Schedule: Linear | warmup_steps={num_warmup}/{num_steps}[/yellow]")
    console.print(f"  [yellow]Base LR: {training_cfg['learning_rate']}[/yellow]")

    fp16 = training_cfg.get("fp16", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if fp16 else None

    # ── Training Loop ────────────────────────────────────────────────────────────
    best_f1 = 0.0
    best_epoch = 0
    global_step = 0

    # Early stopping — tránh lãng phí compute và giúp chẩn đoán: nếu student
    # dừng sớm (best_epoch << num_epochs), đó là tín hiệu model đã bão hòa/
    # overfit sớm, hữu ích để so sánh với Teacher (dừng ở epoch 22/60).
    patience = training_cfg.get("early_stopping_patience", 8)
    es_counter = 0
    es_best = 0.0

    console.print(
        f"\n[bold cyan]Starting Distillation for {training_cfg['num_epochs']} epochs...[/bold cyan]\n"
    )

    for epoch in range(1, training_cfg["num_epochs"] + 1):
        current_lr = scheduler.get_last_lr()[0]
        console.print(f"[bold]Epoch {epoch}/{training_cfg['num_epochs']}[/bold]  (LR={current_lr:.2e})")

        train_metrics = train_epoch(
            distill_trainer, train_loader, optimizer, scheduler,
            grad_clip=training_cfg["max_grad_norm"], fp16=fp16, scaler=scaler,
        )
        global_step += len(train_loader)

        val_metrics, val_labels, val_preds = evaluate(student, val_loader, device, "Val")

        logger.log({
            "train/loss": train_metrics["loss"],
            "train/loss_ce": train_metrics["loss_ce"],
            "train/loss_kd": train_metrics["loss_kd"],
            "train/loss_hidden": train_metrics["loss_hidden"],
            "train/loss_att": train_metrics["loss_att"],
            "train/f1_macro": train_metrics["macro_f1"],
            "val/f1_macro": val_metrics["macro_f1"],
            "val/accuracy": val_metrics["accuracy"],
        }, step=global_step)

        console.print(
            f"  Train: loss={train_metrics['loss']:.4f} "
            f"(ce={train_metrics['loss_ce']:.4f} "
            f"kd={train_metrics['loss_kd']:.4f} "
            f"hid={train_metrics['loss_hidden']:.4f} "
            f"att={train_metrics['loss_att']:.4f})\n"
            f"  Val  : f1={val_metrics['macro_f1']:.4f} acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": student.state_dict(),
                    "val_f1": best_f1,
                    "run_name": actual_run_name,
                    "config": config,
                },
                os.path.join(output_dir, "best_model.pt"),
            )
            console.print(f"  [bold green]✓ Best model saved (F1={best_f1:.4f})[/bold green]")

        # Early stopping check
        if val_metrics["macro_f1"] > es_best + 1e-4:
            es_best = val_metrics["macro_f1"]
            es_counter = 0
        else:
            es_counter += 1
            if es_counter >= patience:
                console.print(
                    f"\n[bold yellow]Early stopping tại epoch {epoch} "
                    f"(val Macro-F1 không cải thiện sau {patience} epochs).[/bold yellow]"
                )
                break

    console.print(
        f"\n[bold green]Distillation complete! Best Val F1={best_f1:.4f} at epoch {best_epoch}[/bold green]"
    )

    # ── Final Test Evaluation ─────────────────────────────────────────────────
    console.print("\n[bold cyan]Final Test Evaluation...[/bold cyan]")
    ckpt = torch.load(
        os.path.join(output_dir, "best_model.pt"), 
        map_location=device,
        weights_only=False,)
    student.load_state_dict(ckpt["model_state_dict"])

    test_metrics, test_labels, test_preds = evaluate(student, test_loader, device, "Test")
    print_classification_report(test_labels, test_preds)

    table = Table(title=f"TinyPhoBERT [{actual_run_name}] Test Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in test_metrics.items():
        table.add_row(k, f"{v:.4f}")
    console.print(table)

    # Save results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    test_metrics["run_name"] = actual_run_name
    test_metrics["params"] = student_params
    test_metrics["size_mb"] = round(student_mb, 2)
    with open(os.path.join(results_dir, f"{actual_run_name}_results.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    logger.finish()
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train TinyPhoBERT Student with Multi-Level KD.")
    parser.add_argument("--config", type=str, default="configs/distillation_config.yaml")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--no_kd", action="store_true", help="Disable distillation (A1 baseline).")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--no_init_from_teacher",
        action="store_true",
        help="Không khởi tạo student từ teacher weights (train từ đầu).",
    )
    parser.add_argument(
        "--two_stage",
        action="store_true",
        help=(
            "Two-stage training: Stage 1 = Focal CE only (không distill), "
            "Stage 2 = Full distillation. "
            "Sử dụng num_epochs từ config, chia đều cho 2 stages."
        ),
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.no_kd:
        config["distillation"]["use_logit_kd"] = False
        config["distillation"]["use_hidden_kd"] = False
        config["distillation"]["use_attention_kd"] = False
        config["distillation"]["alpha"] = 0.0
        config["distillation"]["beta"] = 0.0
        config["distillation"]["gamma"] = 0.0

    if args.alpha is not None:
        config["distillation"]["alpha"] = args.alpha
    if args.beta is not None:
        config["distillation"]["beta"] = args.beta
    if args.gamma is not None:
        config["distillation"]["gamma"] = args.gamma
    if args.fp16:
        config["training"]["fp16"] = True
    if args.no_init_from_teacher:
        config.setdefault("student", {})["init_from_teacher"] = False

    if args.two_stage:
        # Two-stage training:
        # Stage 1: Pure task learning (Focal CE, no KD)
        # Stage 2: Full distillation (KD + hidden + attention)
        total_epochs = config["training"]["num_epochs"]
        stage1_epochs = max(1, total_epochs // 3)      # ~1/3 thời gian
        stage2_epochs = total_epochs - stage1_epochs   # ~2/3 thời gian

        console.print(f"\n[bold cyan]=== TWO-STAGE TRAINING ===[/bold cyan]")
        console.print(f"  Stage 1: {stage1_epochs} epochs (Focal CE, no KD)")
        console.print(f"  Stage 2: {stage2_epochs} epochs (Full distillation)")

        # ─ Stage 1: Pure task learning ─
        console.print("\n[bold yellow]--- STAGE 1: Task Learning (no distillation) ---[/bold yellow]")
        cfg_s1 = yaml.safe_load(yaml.dump(config))  # deep copy
        cfg_s1["training"]["num_epochs"] = stage1_epochs
        cfg_s1["training"]["learning_rate"] = config["training"].get("learning_rate", 5e-5)
        cfg_s1["distillation"]["use_logit_kd"] = False
        cfg_s1["distillation"]["use_hidden_kd"] = False
        cfg_s1["distillation"]["use_attention_kd"] = False
        cfg_s1["distillation"]["alpha"] = 0.0
        cfg_s1["distillation"]["beta"] = 0.0
        cfg_s1["distillation"]["gamma"] = 0.0
        cfg_s1["logging"]["run_name"] = (args.run_name or "distillation-full") + "-stage1"
        train(cfg_s1, run_name=cfg_s1["logging"]["run_name"])

        # ─ Stage 2: Full distillation (load from stage 1 checkpoint) ─
        console.print("\n[bold yellow]--- STAGE 2: Full Distillation ---[/bold yellow]")
        s1_run = cfg_s1["logging"]["run_name"]
        s1_ckpt = os.path.join(config["training"]["output_dir"], s1_run, "best_model.pt")

        cfg_s2 = yaml.safe_load(yaml.dump(config))  # deep copy
        cfg_s2["training"]["num_epochs"] = stage2_epochs
        cfg_s2["training"]["learning_rate"] = config["training"].get("learning_rate", 5e-5) * 0.4  # lower LR
        cfg_s2["student"]["init_from_teacher"] = False  # đã init ở stage 1
        cfg_s2["student"]["pretrained_checkpoint"] = s1_ckpt  # load stage 1
        cfg_s2["logging"]["run_name"] = (args.run_name or "distillation-full") + "-stage2"
        train(cfg_s2, run_name=cfg_s2["logging"]["run_name"])
    else:
        train(config, run_name=args.run_name)


if __name__ == "__main__":
    main()
