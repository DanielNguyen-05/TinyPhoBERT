"""
notebooks/02_results_analysis.py

Results analysis and visualization for TinyPhoBERT paper.

Generates:
    1. Full comparison table (all models)
    2. Params vs F1 scatter plot
    3. Ablation study bar chart
    4. Speed vs F1 Pareto frontier plot
    5. Confusion matrix comparison
"""

# %%
import sys, os
sys.path.insert(0, "..")
os.makedirs("../results/plots", exist_ok=True)

import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from rich.console import Console
from rich.table import Table

plt.style.use("seaborn-v0_8-whitegrid")
console = Console()

# %% [markdown]
# ## Load All Results

# %%
def load_all_results(results_dir="../results"):
    results = {}
    for f in glob.glob(f"{results_dir}/*.json"):
        if "benchmark" in f:
            continue
        with open(f) as fp:
            data = json.load(fp)
        name = data.get("run_name", os.path.basename(f).replace(".json", ""))
        results[name] = data
    return results

results = load_all_results()
print(f"Loaded {len(results)} result files")

# %% [markdown]
# ## 1. Full Comparison Table

# %%
# Define expected models for a complete paper table
PAPER_MODELS = {
    "TF-IDF+SVM":        {"category": "Classical"},
    "FastText":          {"category": "Classical"},
    "BiLSTM":            {"category": "Neural"},
    "TextCNN":           {"category": "Neural"},
    "mBERT":             {"category": "Transformer"},
    "DistilBERT":        {"category": "Transformer"},
    "XLM-R":             {"category": "Transformer"},
    "PhoBERT-base":      {"category": "Transformer"},
    "A1_no_distill":     {"category": "Ours"},
    "A2_logit_kd":       {"category": "Ours"},
    "A3_logit_hidden":   {"category": "Ours"},
    "A4_full":           {"category": "Ours (Full)"},
}

rows = []
for name, info in PAPER_MODELS.items():
    m = results.get(name, {})
    rows.append({
        "Model": name,
        "Category": info["category"],
        "Macro-F1": m.get("macro_f1", None),
        "Accuracy": m.get("accuracy", None),
        "Macro-P": m.get("macro_precision", None),
        "Macro-R": m.get("macro_recall", None),
        "Params": m.get("params", None),
        "Size(MB)": m.get("size_mb", None),
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))

# %%
# Save to CSV for LaTeX
df.to_csv("../results/comparison_table.csv", index=False)
print("Saved: ../results/comparison_table.csv")

# %% [markdown]
# ## 2. Params vs F1 Scatter Plot

# %%
fig, ax = plt.subplots(figsize=(12, 7))

CATEGORY_COLORS = {
    "Classical":  "#95a5a6",
    "Neural":     "#3498db",
    "Transformer":"#e74c3c",
    "Ours":       "#f39c12",
    "Ours (Full)":"#2ecc71",
}

for _, row in df.iterrows():
    if pd.isna(row["Macro-F1"]) or pd.isna(row["Params"]):
        continue
    params_m = row["Params"] / 1e6 if isinstance(row["Params"], (int, float)) else None
    if params_m is None:
        continue
    color = CATEGORY_COLORS.get(row["Category"], "#666")
    size = 150 if "Ours" in row["Category"] else 80
    marker = "*" if "Ours" in row["Category"] else "o"
    ax.scatter(params_m, row["Macro-F1"] * 100, c=color, s=size, marker=marker,
               zorder=5, edgecolors="white", linewidth=0.8)
    ax.annotate(
        row["Model"].replace("_", " "),
        (params_m, row["Macro-F1"] * 100),
        textcoords="offset points",
        xytext=(8, 3),
        fontsize=9,
    )

# Legend
patches = [mpatches.Patch(color=c, label=k) for k, c in CATEGORY_COLORS.items()]
ax.legend(handles=patches, loc="lower right", fontsize=10)

ax.set_xlabel("Parameters (Millions)", fontsize=12)
ax.set_ylabel("Macro-F1 (%)", fontsize=12)
ax.set_title("Efficiency–Performance Trade-off: Parameters vs. Macro-F1",
             fontsize=13, fontweight="bold")
ax.set_xscale("log")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))

plt.tight_layout()
plt.savefig("../results/plots/params_vs_f1.png", dpi=200, bbox_inches="tight")
plt.show()
print("Saved: ../results/plots/params_vs_f1.png")

# %% [markdown]
# ## 3. Ablation Study Chart

# %%
ablation_runs = ["A1_no_distill", "A2_logit_kd", "A3_logit_hidden", "A4_full"]
ablation_labels = [
    "No Distill\n(A1)",
    "+ Logit KD\n(A2)",
    "+ Hidden KD\n(A3)",
    "+ Attention KD\n(A4, Full)",
]
ablation_f1 = [results.get(r, {}).get("macro_f1", 0) * 100 for r in ablation_runs]

fig, ax = plt.subplots(figsize=(10, 6))
colors_bar = ["#95a5a6", "#3498db", "#f39c12", "#2ecc71"]
bars = ax.bar(range(len(ablation_runs)), ablation_f1, color=colors_bar,
              width=0.55, alpha=0.9, edgecolor="white", linewidth=1.5)

# Annotations
for i, (bar, val) in enumerate(zip(bars, ablation_f1)):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    if i > 0 and ablation_f1[i-1] > 0:
        gain = val - ablation_f1[i-1]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f"+{gain:.1f}%", ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")

ax.set_xticks(range(len(ablation_runs)))
ax.set_xticklabels(ablation_labels, fontsize=11)
ax.set_ylabel("Macro-F1 (%)", fontsize=12)
ax.set_title("Ablation Study: Contribution of Each KD Level", fontsize=13, fontweight="bold")
ax.set_ylim(max(0, min(ablation_f1) - 5) if ablation_f1[0] > 0 else 0,
            max(ablation_f1) + 3 if ablation_f1 else 100)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("../results/plots/ablation_study.png", dpi=200, bbox_inches="tight")
plt.show()
print("Saved: ../results/plots/ablation_study.png")

# %% [markdown]
# ## 4. LaTeX Table Generation

# %%
def to_latex_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table for the paper."""
    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\caption{Comparison of Models on ViHSD Test Set}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{l r r r r r r}",
        r"\toprule",
        r"Model & Params & Size & Acc & Macro-P & Macro-R & \textbf{Macro-F1} \\",
        r"\midrule",
    ]

    categories = df["Category"].unique()
    for cat in categories:
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{cat}}}}} \\\\")
        for _, row in df[df["Category"] == cat].iterrows():
            params_str = f"{row['Params']/1e6:.0f}M" if isinstance(row["Params"], (int, float)) else "—"
            size_str = f"{row['Size(MB)']:.0f}" if isinstance(row["Size(MB)"], (int, float)) else "—"
            f1_str = f"\\textbf{{{row['Macro-F1']*100:.1f}}}" if "Ours (Full)" in row["Category"] else (
                f"{row['Macro-F1']*100:.1f}" if isinstance(row["Macro-F1"], float) else "—"
            )
            lines.append(
                f"~~{row['Model']} & {params_str} & {size_str} MB & "
                f"{row['Accuracy']*100:.1f} & {row['Macro-P']*100:.1f} & "
                f"{row['Macro-R']*100:.1f} & {f1_str} \\\\"
                if all(isinstance(row[c], float) for c in ["Accuracy", "Macro-P", "Macro-R"])
                else f"~~{row['Model']} & {params_str} & {size_str} MB & — & — & — & — \\\\"
            )
        lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)

latex = to_latex_table(df)
with open("../results/latex_table.tex", "w") as f:
    f.write(latex)
print("LaTeX table saved to: ../results/latex_table.tex")
print("\nPreview:")
print(latex[:500] + "...")

print("\n✅ Results analysis complete!")
