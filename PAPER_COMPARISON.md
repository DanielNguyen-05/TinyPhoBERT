# Paper comparison and proposed model

## What the three papers establish

| Paper | Benchmark | Core idea | Reported primary result |
|---|---|---|---:|
| Tran et al., *Vietnamese Hate and Offensive Detection using PhoBERT-CNN* (2022) | ViHSD, official train/dev/test | PhoBERT-large + parallel TextCNN; preprocessing and EDA | 67.46 macro-F1, 87.76 accuracy |
| Nguyen, *VIHATET5* (ACL Findings 2024) | ViHSD, official train/dev/test | T5 pre-trained on 10.8M domain-specific social comments | 68.67 macro-F1, 88.76 accuracy |
| Pham et al., *Improving RoBERTa-Based Vietnamese HSD* (2021) | HSD-VLSP 2019, stratified 10-fold CV | Domain-adaptive MLM, upper-layer concatenation, block-wise LR, label smoothing | 72.11 macro-F1 |

The HSD-VLSP result must not be placed in the same ranking as ViHSD. It uses a
different dataset and 10-fold cross-validation. The abstract reports 72.21,
while the paper's ablation table reports 72.11 for the combined method; both
values should be reproduced before choosing one for a final publication table.

## Proposed architecture: DAMS-TinyPhoBERT

**D**omain-**A**dapted **M**ulti-**S**cale TinyPhoBERT is designed to answer a
different research question from the three papers: how much of their accuracy
can be retained in a compact deployable model?

```text
Vietnamese comment
       |
PhoBERT tokenizer
       |
6-layer / 384-hidden TinyPhoBERT backbone (~40M)
       |
learned scalar mix of the final 4 student layers
       |-----------------------|
  CLS + masked mean       Conv1D k={1,3,5} + max pool
       |-----------------------|
             gated fusion
                  |
          CLEAN / OFFENSIVE / HATE

Training only:
domain-adapted PhoBERT-large teacher
  -> logit KD + hidden-state KD + attention KD
```

The head incorporates the local n-gram idea from PhoBERT-CNN and the upper
layer finding from Pham et al. Domain knowledge is supplied by continued MLM
pre-training of the teacher and transferred into the student, reflecting the
central finding of VIHATET5 without requiring a 223M-parameter generator at
inference. Distillation and efficiency are this model's differentiating claims.

## Required experiment protocol

Use the exact published ViHSD train/dev/test split. Never augment validation or
test data. Report mean and standard deviation over seeds 13, 21, 42, 87, and
100, with macro-F1 as the model-selection metric.

Run these ablations using identical data and seeds:

| ID | Model |
|---|---|
| A0 | PhoBERT-large classifier |
| A1 | Original linear-head TinyPhoBERT, no KD |
| A2 | DAMS head, no KD |
| A3 | DAMS + logit KD |
| A4 | DAMS + logit and hidden KD |
| A5 | DAMS + full logit/hidden/attention KD |
| A6 | A5 without EDA |
| A7 | A5 with a non-domain-adapted teacher |

For every run report macro-F1, accuracy, weighted-F1, per-class F1, parameter
count, checkpoint size, latency, throughput, and peak memory. Statistical
comparison should use paired bootstrap confidence intervals on the fixed test
predictions; do not select hyperparameters on the test set.

The directly comparable ViHSD reference lines are 67.46 macro-F1 for
PhoBERT-CNN and 68.67 for VIHATET5. Reproduce their preprocessing/split rules
or label a comparison as "reported" rather than "reproduced".

## Commands

Train the stronger teacher baseline first:

```bash
python data/prepare_data.py --config configs/teacher_strong_config.yaml --no_augment
python training/train_teacher.py --config configs/teacher_strong_config.yaml --fp16
python training/test_teacher.py \
  --config configs/teacher_strong_config.yaml \
  --checkpoint checkpoints/teacher_strong/best_model.pt
```

First continue MLM pre-training of PhoBERT-large on unlabeled Vietnamese social
comments, then fine-tune that checkpoint as the teacher. The repository does
not yet contain an MLM-training command, so this is a required experiment stage
rather than an already implemented result.

Train the proposed distilled classifier:

```bash
python training/train_student.py \
  --config configs/comparison_distillation_config.yaml \
  --run_name DAMS-TinyPhoBERT
```

Evaluate only the saved best-validation checkpoint on the untouched test set:

```bash
python evaluation/evaluate.py \
  --model_type student \
  --model_path checkpoints/comparison/DAMS-TinyPhoBERT/best_model.pt \
  --config_path configs/comparison_student_config.yaml \
  --test_file data/augmented/test.csv
```
