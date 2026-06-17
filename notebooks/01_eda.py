"""
notebooks/01_eda.py

Exploratory Data Analysis for ViHSD dataset.
Run as a script or convert to notebook:
    jupyter nbconvert --to notebook --execute 01_eda.py

Analysis:
    1. Dataset statistics
    2. Label distribution
    3. Text length distribution
    4. Sample examples per class
    5. Vietnamese character/word analysis
"""

# %% [markdown]
# # ViHSD — Exploratory Data Analysis
# Vietnamese Hate Speech Dataset Analysis

# %%
import sys
sys.path.insert(0, "..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from collections import Counter

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"CLEAN": "#2ecc71", "OFFENSIVE": "#f39c12", "HATE": "#e74c3c"}
FIGSIZE = (12, 6)

# %%
# Load data
train_df = pd.read_csv("../data/processed/train.csv")
val_df = pd.read_csv("../data/processed/val.csv")
test_df = pd.read_csv("../data/processed/test.csv")
full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
print(f"Total: {len(full_df):,}")
print(f"\nColumns: {list(full_df.columns)}")
print(f"\nLabel Distribution:")
print(full_df["label_name"].value_counts())

# %% [markdown]
# ## 1. Label Distribution

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (df, name) in enumerate([(train_df, "Train"), (val_df, "Val"), (test_df, "Test")]):
    counts = df["label_name"].value_counts()
    colors = [COLORS.get(l, "#666") for l in counts.index]
    axes[i].bar(counts.index, counts.values, color=colors, alpha=0.85, edgecolor="white")
    axes[i].set_title(f"{name} Set (n={len(df):,})", fontsize=13, fontweight="bold")
    axes[i].set_ylabel("Count")
    for j, (label, val) in enumerate(zip(counts.index, counts.values)):
        axes[i].text(j, val + 50, f"{val:,}\n({val/len(df)*100:.1f}%)",
                     ha="center", fontsize=9)

plt.suptitle("Label Distribution Across Splits", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("../results/plots/label_distribution.png", dpi=150)
plt.show()

# %% [markdown]
# ## 2. Text Length Distribution

# %%
full_df["word_count"] = full_df["free_text"].str.split().str.len()
full_df["char_count"] = full_df["free_text"].str.len()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Word count histogram
for label, color in COLORS.items():
    subset = full_df[full_df["label_name"] == label]["word_count"]
    axes[0, 0].hist(subset, bins=50, alpha=0.6, color=color, label=label, density=True)
axes[0, 0].set_title("Word Count Distribution by Label", fontweight="bold")
axes[0, 0].set_xlabel("Word Count")
axes[0, 0].legend()
axes[0, 0].set_xlim(0, 150)

# Char count histogram
for label, color in COLORS.items():
    subset = full_df[full_df["label_name"] == label]["char_count"]
    axes[0, 1].hist(subset, bins=50, alpha=0.6, color=color, label=label, density=True)
axes[0, 1].set_title("Character Count Distribution by Label", fontweight="bold")
axes[0, 1].set_xlabel("Character Count")
axes[0, 1].legend()
axes[0, 1].set_xlim(0, 500)

# Box plots
data = [full_df[full_df["label_name"] == l]["word_count"].values for l in COLORS]
bp = axes[1, 0].boxplot(data, labels=list(COLORS.keys()), patch_artist=True)
for patch, (label, color) in zip(bp["boxes"], COLORS.items()):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1, 0].set_title("Word Count Box Plot by Label", fontweight="bold")
axes[1, 0].set_ylabel("Word Count")

# Stats table
stats = full_df.groupby("label_name")["word_count"].agg(["mean", "median", "std", "max"])
axes[1, 1].axis("off")
table = axes[1, 1].table(
    cellText=[[f"{v:.1f}" for v in row] for row in stats.values],
    rowLabels=stats.index.tolist(),
    colLabels=["Mean", "Median", "Std", "Max"],
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(True)
table.scale(1, 2)
axes[1, 1].set_title("Word Count Statistics", fontweight="bold", pad=20)

plt.suptitle("Text Length Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("../results/plots/text_length_analysis.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3. Sample Examples

# %%
for label in ["CLEAN", "OFFENSIVE", "HATE"]:
    samples = full_df[full_df["label_name"] == label]["free_text"].sample(3, random_state=42)
    print(f"\n{'='*60}")
    print(f"[{label}] Examples:")
    print(f"{'='*60}")
    for i, s in enumerate(samples, 1):
        print(f"{i}. {s}")

# %% [markdown]
# ## 4. Class Imbalance Summary

# %%
label_counts = full_df["label_name"].value_counts()
total = len(full_df)
print("\nClass Imbalance Summary:")
print(f"{'Label':<15} {'Count':>10} {'%':>8}")
print("-" * 35)
for label, count in label_counts.items():
    print(f"{label:<15} {count:>10,} {count/total*100:>7.1f}%")
print(f"\nImbalance ratio (max/min): {label_counts.max() / label_counts.min():.1f}x")

print("\n✅ EDA complete! Plots saved to results/plots/")
