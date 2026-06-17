#!/usr/bin/env bash
# experiments/run_all.sh
# Run the complete TinyPhoBERT experiment pipeline.
#
# Usage:
#   bash experiments/run_all.sh
#   bash experiments/run_all.sh --skip_teacher   # Skip teacher fine-tuning
#   bash experiments/run_all.sh --skip_baselines # Skip baseline training
#
# Requirements:
#   pip install -r requirements.txt
#   python data/download_data.py   # Must be run first

set -e  # Exit on error

# ── Parse args ──────────────────────────────────────────────────────────────
SKIP_TEACHER=false
SKIP_BASELINES=false
SKIP_ABLATION=false

for arg in "$@"; do
  case $arg in
    --skip_teacher)    SKIP_TEACHER=true ;;
    --skip_baselines)  SKIP_BASELINES=true ;;
    --skip_ablation)   SKIP_ABLATION=true ;;
  esac
done

echo "╔════════════════════════════════════════════════════╗"
echo "║        TinyPhoBERT — Full Experiment Pipeline      ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# ── Step 0: Download data (if not already done) ─────────────────────────────
if [ ! -f "data/processed/train.csv" ]; then
  echo "▶ Step 0: Downloading ViHSD dataset..."
  python data/download_data.py --source github
else
  echo "✓ Step 0: Data already downloaded."
fi

# ── Step 1: Fine-tune Teacher ───────────────────────────────────────────────
if [ "$SKIP_TEACHER" = false ]; then
  echo ""
  echo "▶ Step 1: Fine-tuning PhoBERT Teacher..."
  python training/train_teacher.py --config configs/teacher_config.yaml
  echo "✓ Teacher fine-tuning complete."
else
  echo "⏭ Step 1: Skipping teacher fine-tuning."
fi

# ── Step 2: Train Baselines ─────────────────────────────────────────────────
if [ "$SKIP_BASELINES" = false ]; then
  echo ""
  echo "▶ Step 2: Training baseline models..."
  python training/train_baselines.py --model svm
  python training/train_baselines.py --model fasttext
  python training/train_baselines.py --model bilstm
  python training/train_baselines.py --model textcnn
  python training/train_baselines.py --model mbert
  python training/train_baselines.py --model distilbert
  python training/train_baselines.py --model xlmr
  echo "✓ Baseline training complete."
else
  echo "⏭ Step 2: Skipping baseline training."
fi

# ── Step 3: Ablation Study (A1 → A4) ────────────────────────────────────────
if [ "$SKIP_ABLATION" = false ]; then
  echo ""
  echo "▶ Step 3: Running Ablation Study (A1-A4)..."
  python experiments/ablation.py --config configs/distillation_config.yaml
  echo "✓ Ablation study complete."
else
  echo "⏭ Step 3: Skipping ablation study."
fi

# ── Step 4: Full Distillation Training ─────────────────────────────────────
echo ""
echo "▶ Step 4: Training TinyPhoBERT (Full Distillation)..."
python training/train_student.py \
  --config configs/distillation_config.yaml \
  --run_name "TinyPhoBERT_full"
echo "✓ Distillation training complete."

# ── Step 5: Evaluation ─────────────────────────────────────────────────────
echo ""
echo "▶ Step 5: Evaluating all models..."
python evaluation/evaluate.py --compare_all
echo "✓ Evaluation complete."

# ── Step 6: Benchmark ──────────────────────────────────────────────────────
echo ""
echo "▶ Step 6: Running efficiency benchmark..."
python evaluation/benchmark.py --compare_all \
  --model_path "checkpoints/distillation/TinyPhoBERT_full/best_model.pt" \
  --teacher_path "checkpoints/teacher/best_model.pt"
echo "✓ Benchmark complete."

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║              All Experiments Complete!             ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""
echo "Results:"
echo "  📊 Metrics:    results/"
echo "  📈 Plots:      results/ablation/plots/"
echo "  💾 Checkpoints: checkpoints/"
echo ""
echo "Next steps:"
echo "  jupyter lab notebooks/02_results_analysis.ipynb"
