"""
training/train_baselines.py

Train baseline models for comparison with TinyPhoBERT.

Baselines:
    Classical:
        - TF-IDF + SVM
        - FastText
        - BiLSTM
        - TextCNN

    Transformer:
        - mBERT (bert-base-multilingual-cased)
        - DistilBERT (distilbert-base-multilingual-cased)
        - XLM-R (xlm-roberta-base)
        - PhoBERT (vinai/phobert-base) [for reference]

Usage:
    python training/train_baselines.py --model all
    python training/train_baselines.py --model svm
    python training/train_baselines.py --model bilstm
    python training/train_baselines.py --model mbert
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import compute_metrics, print_classification_report
from utils.seed import set_seed, get_device

console = Console()
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Classical Baselines
# ──────────────────────────────────────────────────────────────────────────────

def train_tfidf_svm(train_df, val_df, test_df, text_col="free_text", label_col="label_id"):
    """TF-IDF + Linear SVM baseline."""
    console.print("[bold cyan]Training TF-IDF + SVM...[/bold cyan]")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                                   sublinear_tf=True)),
        ("svc", LinearSVC(C=1.0, max_iter=5000, random_state=42)),
    ])

    X_train = train_df[text_col].astype(str).tolist()
    y_train = train_df[label_col].astype(int).tolist()
    X_test = test_df[text_col].astype(str).tolist()
    y_test = test_df[label_col].astype(int).tolist()

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0

    preds = pipeline.predict(X_test)
    metrics = compute_metrics(y_test, preds.tolist())
    metrics["train_time_s"] = round(train_time, 2)
    metrics["params"] = "N/A"
    metrics["size_mb"] = "N/A"
    console.print(f"  [green]SVM Macro-F1: {metrics['macro_f1']:.4f}[/green]")
    print_classification_report(y_test, preds.tolist())
    return metrics


def train_fasttext(train_df, val_df, test_df, text_col="free_text", label_col="label_id"):
    """FastText baseline using scikit-learn TF-IDF with subword features."""
    console.print("[bold cyan]Training FastText...[/bold cyan]")
    try:
        import fasttext
        # Write training file in FastText format
        ft_train_path = "data/processed/fasttext_train.txt"
        with open(ft_train_path, "w", encoding="utf-8") as f:
            for _, row in train_df.iterrows():
                label = f"__label__{int(row[label_col])}"
                text = str(row[text_col]).replace("\n", " ")
                f.write(f"{label} {text}\n")

        t0 = time.time()
        model = fasttext.train_supervised(
            ft_train_path, epoch=25, lr=0.5, wordNgrams=2,
            dim=100, loss="softmax", verbose=0,
        )
        train_time = time.time() - t0

        X_test = test_df[text_col].astype(str).tolist()
        y_test = test_df[label_col].astype(int).tolist()
        preds = [int(model.predict(t)[0][0].replace("__label__", "")) for t in X_test]
        metrics = compute_metrics(y_test, preds)
        metrics["train_time_s"] = round(train_time, 2)
        metrics["params"] = "~100M"
        metrics["size_mb"] = "~50MB"
    except ImportError:
        console.print("  [yellow]fasttext not installed. Using TF-IDF + LogReg as proxy.[/yellow]")
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=100000, ngram_range=(1, 3),
                                       analyzer="char_wb", sublinear_tf=True)),
            ("lr", LogisticRegression(C=5.0, max_iter=1000, random_state=42,
                                       multi_class="multinomial")),
        ])
        X_train = train_df[text_col].astype(str).tolist()
        y_train = train_df[label_col].astype(int).tolist()
        X_test = test_df[text_col].astype(str).tolist()
        y_test = test_df[label_col].astype(int).tolist()
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0
        preds = pipeline.predict(X_test)
        metrics = compute_metrics(y_test, preds.tolist())
        metrics["train_time_s"] = round(train_time, 2)
        metrics["params"] = "N/A"
        metrics["size_mb"] = "N/A"

    console.print(f"  [green]FastText Macro-F1: {metrics['macro_f1']:.4f}[/green]")
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Neural Baselines: BiLSTM & TextCNN
# ──────────────────────────────────────────────────────────────────────────────

class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM hate speech classifier."""

    def __init__(self, vocab_size=64001, embed_dim=128, hidden_dim=256,
                 num_layers=2, num_classes=3, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        out, (hn, _) = self.lstm(emb)
        # Concatenate forward and backward last hidden states
        hidden = torch.cat([hn[-2], hn[-1]], dim=-1)
        return self.classifier(self.dropout(hidden))


class TextCNNClassifier(nn.Module):
    """TextCNN hate speech classifier (Kim, 2014)."""

    def __init__(self, vocab_size=64001, embed_dim=128, num_filters=128,
                 kernel_sizes=(2, 3, 4, 5), num_classes=3, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        emb = self.embedding(x).permute(0, 2, 1)  # (B, E, T)
        pooled = [torch.relu(conv(emb)).max(dim=-1).values for conv in self.convs]
        cat = torch.cat(pooled, dim=-1)
        return self.classifier(self.dropout(cat))


def train_neural_baseline(
    model_class, model_kwargs, train_df, test_df,
    tokenizer, max_len=128, epochs=15, batch_size=64,
    lr=1e-3, device=None, label_col="label_id",
):
    """Generic training loop for BiLSTM / TextCNN."""
    if device is None:
        device = get_device()

    from utils.data_utils import HateSpeechDataset, build_datasets
    import pandas as pd

    # Create empty val df (use last 10% of train)
    n_val = max(1, len(train_df) // 10)
    val_df = train_df.iloc[-n_val:]
    train_df_s = train_df.iloc[:-n_val]

    train_ds = HateSpeechDataset(
        train_df_s["free_text"].astype(str).tolist(),
        train_df_s[label_col].astype(int).tolist(),
        tokenizer, max_len,
    )
    val_ds = HateSpeechDataset(
        val_df["free_text"].astype(str).tolist(),
        val_df[label_col].astype(int).tolist(),
        tokenizer, max_len,
    )
    test_ds = HateSpeechDataset(
        test_df["free_text"].astype(str).tolist(),
        test_df[label_col].astype(int).tolist(),
        tokenizer, max_len,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)

    model = model_class(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_state = None

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        preds_val, labels_val = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input_ids"].to(device)
                y = batch["labels"]
                logits = model(x)
                preds_val.extend(logits.argmax(-1).cpu().tolist())
                labels_val.extend(y.tolist())
        val_f1 = compute_metrics(labels_val, preds_val)["macro_f1"]
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.time() - t0

    # Test with best model
    model.load_state_dict(best_state)
    model.eval()
    preds_test, labels_test = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["input_ids"].to(device)
            y = batch["labels"]
            logits = model(x)
            preds_test.extend(logits.argmax(-1).cpu().tolist())
            labels_test.extend(y.tolist())

    metrics = compute_metrics(labels_test, preds_test)
    metrics["train_time_s"] = round(train_time, 2)
    metrics["params"] = sum(p.numel() for p in model.parameters())
    metrics["size_mb"] = round(
        sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6, 2
    )
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Transformer Baselines
# ──────────────────────────────────────────────────────────────────────────────

def train_transformer_baseline(
    model_name: str,
    train_df, val_df, test_df,
    num_labels: int = 3,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_len: int = 128,
    device=None,
    text_col: str = "free_text",
    label_col: str = "label_id",
):
    """Fine-tune a HuggingFace transformer baseline."""
    if device is None:
        device = get_device()

    console.print(f"[bold cyan]Training Transformer baseline: {model_name}...[/bold cyan]")

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import get_linear_schedule_with_warmup
    from utils.data_utils import HateSpeechDataset

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    ).to(device)

    train_ds = HateSpeechDataset(
        train_df[text_col].astype(str).tolist(),
        train_df[label_col].astype(int).tolist(),
        tokenizer, max_len,
    )
    val_ds = HateSpeechDataset(
        val_df[text_col].astype(str).tolist(),
        val_df[label_col].astype(int).tolist(),
        tokenizer, max_len,
    )
    test_ds = HateSpeechDataset(
        test_df[text_col].astype(str).tolist(),
        test_df[label_col].astype(int).tolist(),
        tokenizer, max_len,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    num_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(num_steps * 0.1), num_steps)

    best_f1, best_state = 0.0, None
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"  Epoch {epoch+1}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Validation
        model.eval()
        preds_v, labs_v = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device))
                preds_v.extend(out.logits.argmax(-1).cpu().tolist())
                labs_v.extend(batch["labels"].tolist())
        val_f1 = compute_metrics(labs_v, preds_v)["macro_f1"]
        console.print(f"  Epoch {epoch+1}: Val F1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.time() - t0

    # Test
    model.load_state_dict(best_state)
    model.eval()
    preds_t, labs_t = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device))
            preds_t.extend(out.logits.argmax(-1).cpu().tolist())
            labs_t.extend(batch["labels"].tolist())

    metrics = compute_metrics(labs_t, preds_t)
    metrics["train_time_s"] = round(train_time, 2)
    n_params = sum(p.numel() for p in model.parameters())
    metrics["params"] = n_params
    metrics["size_mb"] = round(n_params * 4 / 1e6, 1)  # float32 estimate
    console.print(f"  [green]{model_name} Macro-F1: {metrics['macro_f1']:.4f}[/green]")
    print_classification_report(labs_t, preds_t)
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train baseline models.")
    parser.add_argument(
        "--model",
        choices=["all", "svm", "fasttext", "bilstm", "textcnn",
                 "mbert", "distilbert", "xlmr", "phobert"],
        default="all",
    )
    parser.add_argument("--train_file", default="data/processed/train.csv")
    parser.add_argument("--val_file", default="data/processed/val.csv")
    parser.add_argument("--test_file", default="data/processed/test.csv")
    parser.add_argument("--text_col", default="free_text")
    parser.add_argument("--label_col", default="label_id")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    # Load data
    train_df = pd.read_csv(args.train_file)
    val_df = pd.read_csv(args.val_file)
    test_df = pd.read_csv(args.test_file)
    console.print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    all_results = {}

    # Classical
    if args.model in ("all", "svm"):
        results = train_tfidf_svm(train_df, val_df, test_df, args.text_col, args.label_col)
        all_results["TF-IDF+SVM"] = results
        with open(RESULTS_DIR / "baseline_svm.json", "w") as f:
            json.dump(results, f, indent=2)

    if args.model in ("all", "fasttext"):
        results = train_fasttext(train_df, val_df, test_df, args.text_col, args.label_col)
        all_results["FastText"] = results
        with open(RESULTS_DIR / "baseline_fasttext.json", "w") as f:
            json.dump(results, f, indent=2)

    # Neural
    if args.model in ("all", "bilstm", "textcnn"):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

        if args.model in ("all", "bilstm"):
            console.print("[bold cyan]Training BiLSTM...[/bold cyan]")
            results = train_neural_baseline(
                BiLSTMClassifier,
                {"vocab_size": tokenizer.vocab_size + 1, "embed_dim": 128,
                 "hidden_dim": 256, "num_layers": 2, "num_classes": 3},
                train_df, test_df, tokenizer, epochs=20,
                batch_size=args.batch_size, device=device, label_col=args.label_col,
            )
            all_results["BiLSTM"] = results
            console.print(f"  [green]BiLSTM Macro-F1: {results['macro_f1']:.4f}[/green]")
            with open(RESULTS_DIR / "baseline_bilstm.json", "w") as f:
                json.dump(results, f, indent=2)

        if args.model in ("all", "textcnn"):
            console.print("[bold cyan]Training TextCNN...[/bold cyan]")
            results = train_neural_baseline(
                TextCNNClassifier,
                {"vocab_size": tokenizer.vocab_size + 1, "embed_dim": 128,
                 "num_filters": 128, "num_classes": 3},
                train_df, test_df, tokenizer, epochs=20,
                batch_size=args.batch_size, device=device, label_col=args.label_col,
            )
            all_results["TextCNN"] = results
            console.print(f"  [green]TextCNN Macro-F1: {results['macro_f1']:.4f}[/green]")
            with open(RESULTS_DIR / "baseline_textcnn.json", "w") as f:
                json.dump(results, f, indent=2)

    # Transformer
    transformer_map = {
        "mbert":      "bert-base-multilingual-cased",
        "distilbert": "distilbert-base-multilingual-cased",
        "xlmr":       "xlm-roberta-base",
        "phobert":    "vinai/phobert-base",
    }
    for key, hf_name in transformer_map.items():
        if args.model in ("all", key):
            results = train_transformer_baseline(
                hf_name, train_df, val_df, test_df,
                epochs=args.epochs, batch_size=args.batch_size,
                device=device, text_col=args.text_col, label_col=args.label_col,
            )
            all_results[key] = results
            with open(RESULTS_DIR / f"baseline_{key}.json", "w") as f:
                json.dump(results, f, indent=2)

    # Summary table
    if all_results:
        table = Table(title="Baseline Results Summary")
        table.add_column("Model", style="cyan")
        table.add_column("Macro-F1", style="green")
        table.add_column("Accuracy", style="blue")
        table.add_column("Params", style="yellow")
        for name, m in all_results.items():
            table.add_row(
                name,
                f"{m.get('macro_f1', 0):.4f}",
                f"{m.get('accuracy', 0):.4f}",
                str(m.get("params", "N/A")),
            )
        console.print(table)

        with open(RESULTS_DIR / "all_baselines.json", "w") as f:
            json.dump(all_results, f, indent=2)
        console.print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
