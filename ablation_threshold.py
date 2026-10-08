#!/usr/bin/env python3
"""
ablation_threshold.py
Margin-threshold sensitivity analysis for confidence-aware refinement.
Run AFTER the ablation loop has produced ABLATION_thr_*.csv and
ABLATION_stats_thr_*.csv files (see the archiving loop).
"""
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, recall_score

LABEL_DIR = Path("results/label_prediction")
STATS_DIR = Path("results/concept_prediction")
THRESHOLDS = ["0.1", "0.2", "0.3"]

print(f"{'thr':>5} {'BAcc':>7} {'Sens':>7} {'Spec':>7} {'AvgIter':>8}")
print("-" * 40)

for thr in THRESHOLDS:
    label_path = LABEL_DIR / f"ABLATION_thr_{thr}.csv"
    stats_path = STATS_DIR / f"ABLATION_stats_thr_{thr}.csv"

    if not label_path.exists():
        print(f"{thr:>5}   (missing {label_path.name})")
        continue

    df = pd.read_csv(label_path)
    yt = df["gt_response"].str.lower().str.strip()
    yp = df["llm_response"].str.lower().str.strip()
    bacc = balanced_accuracy_score(yt, yp) * 100
    sens = recall_score(yt, yp, pos_label="melanoma", zero_division=0) * 100
    spec = recall_score(yt, yp, pos_label="nevus",    zero_division=0) * 100

    # AvgIter = how often refinement actually fired (trigger-rate proxy).
    # Should rise with the threshold as more concepts get flagged uncertain.
    if stats_path.exists():
        avg_iter = pd.read_csv(stats_path)["iterations"].mean()
        avg_iter_str = f"{avg_iter:>8.3f}"
    else:
        avg_iter_str = f"{'—':>8}"

    print(f"{thr:>5} {bacc:>7.2f} {sens:>7.2f} {spec:>7.2f} {avg_iter_str}")
