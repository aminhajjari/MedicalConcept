#!/usr/bin/env python3
"""
evaluate_results.py  —  Compare 3 self-refine configs across all datasets.

Configs compared:
  (a) ExpLICD + MMed refiner  + MMed LLM
  (b) ExpLICD + Mistral refiner + Mistral LLM
  (c) ExpLICD + Rule-based refiner + MMed LLM
"""

import pandas as pd
import os
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, recall_score, confusion_matrix
import glob

# ── paths & constants ──────────────────────────────────────────────────────────
RESULTS_DIR = Path("results/label_prediction")
N_SPLITS    = 5

# The 6 configs to compare
CONFIGS = [
{
        "label"  : "ExpLICD+MMed+Mistral",
        "llm"    : "Mistral",
        "refiner": "mmed",
    },
    # ── new classifiers: MedGemma ──────────────────────────────
    {
        "label"  : "ExpLICD+Rule+MedGemma",
        "llm"    : "MedGemma",
        "refiner": "rule",
    },
    {
        "label"  : "ExpLICD+Mistral+MedGemma",
        "llm"    : "MedGemma",
        "refiner": "mistral",
    },
    {
        "label"  : "ExpLICD+MMed+MedGemma",
        "llm"    : "MedGemma",
        "refiner": "mmed",
    },
    # ── new classifiers: Qwen ──────────────────────────────────
    {
        "label"  : "ExpLICD+Rule+Qwen",
        "llm"    : "Qwen",
        "refiner": "rule",
    },
    {
        "label"  : "ExpLICD+Mistral+Qwen",
        "llm"    : "Qwen",
        "refiner": "mistral",
    },
    {
        "label"  : "ExpLICD+MMed+Qwen",
        "llm"    : "Qwen",
        "refiner": "mmed",
    },
]

PAPER_BACC = {"PH2": 78.07, "Derm7pt": 78.56, "HAM10000": 76.00}

def csv_path(cfg, dataset, split=None):
    llm     = cfg["llm"]
    refiner = cfg["refiner"]
    suffix  = (f"raw_values_False_gt_concepts_False"
               f"_model_extractor_Explicd_n_demos_0_refiner_{refiner}_retrieval_rices")
    if split is not None:
        name = f"{dataset}_split_{split}_{llm}_diagnostic_report_validation_{suffix}.csv"
    else:
        name = f"{dataset}_{llm}_diagnostic_report_validation_{suffix}.csv"
    return RESULTS_DIR / name


# ── metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(path):
    if not Path(path).exists():
        return None
    df     = pd.read_csv(path)
    y_true = df["gt_response"].str.lower().str.strip()
    y_pred = df["llm_response"].str.lower().str.strip()
    bacc   = balanced_accuracy_score(y_true, y_pred) * 100
    sens   = recall_score(y_true, y_pred, pos_label="melanoma", zero_division=0) * 100
    spec   = recall_score(y_true, y_pred, pos_label="nevus",    zero_division=0) * 100
    return {"bacc": bacc, "sens": sens, "spec": spec}

def avg_ph2(cfg):
    results = [compute_metrics(csv_path(cfg, "PH2", s)) for s in range(N_SPLITS)]
    valid   = [r for r in results if r]
    if not valid:
        return None
    return {k: np.mean([r[k] for r in valid]) for k in ("bacc","sens","spec")}

# ── table printer ──────────────────────────────────────────────────────────────
def pct(v):
    return f"{v:.2f}%" if v is not None else "MISSING"

def delta(our, paper):
    if our is None:
        return "N/A"
    d = our - paper
    return f"{'+' if d>=0 else ''}{d:.2f}%"

def print_comparison_table(dataset, rows):
    """rows = list of (config_label, bacc, sens, spec)"""
    paper = PAPER_BACC[dataset]
    sep   = "=" * 78
    print(f"\n{sep}")
    print(f"  DATASET: {dataset}   (paper 0-shot BAcc = {paper:.2f}%)")
    print(sep)
    header = f"  {'Config':<22}  {'BAcc':>8}  {'Sens':>8}  {'Spec':>8}  {'Δ vs Paper':>10}"
    print(header)
    print("  " + "-"*70)
    for label, m in rows:
        if m:
            print(f"  {label:<22}  {pct(m['bacc']):>8}  {pct(m['sens']):>8}  {pct(m['spec']):>8}  {delta(m['bacc'], paper):>10}")
        else:
            print(f"  {label:<22}  {'MISSING':>8}  {'MISSING':>8}  {'MISSING':>8}  {'N/A':>10}")
    print(sep)



def evaluate_concepts():
    
    print("\n" + "="*78)
    print("  CONCEPT-LEVEL EVALUATION — violation reduction per refiner")
    print("="*78)
    rows = []
    for f in sorted(glob.glob("results/concept_prediction/*refinement_stats*.csv")):
        d = pd.read_csv(f)
        aff = d[d.initial_violations > 0]
        name = f.split("/")[-1]
        init, final = d.initial_violations.mean(), d.final_violations.mean()
        reduced = int((d.final_violations < d.initial_violations).sum())
        print(f"  {name}")
        print(f"     init {init:.3f} -> final {final:.3f} | "
              f"reduced on {reduced} | affected {len(aff)}/{len(d)}")
        rows.append({"file": name, "init_viol": round(init,3),
                     "final_viol": round(final,3), "reduced_imgs": reduced,
                     "affected_imgs": len(aff), "total_imgs": len(d)})
    os.makedirs("results/tables", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/tables/concept_violation_stats.csv", index=False)
    print("\n  Saved: results/tables/concept_violation_stats.csv")


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*78)
    print("  SELF-REFINE COMPARISON  —  6 Configs  x  3 Datasets")
    print("="*78)

    for dataset in ["PH2", "Derm7pt", "HAM10000"]:
        rows = []
        for cfg in CONFIGS:
            if dataset == "PH2":
                m = avg_ph2(cfg)
            else:
                m = compute_metrics(csv_path(cfg, dataset))
            rows.append((cfg["label"], m))
        print_comparison_table(dataset, rows)
        
    print()
evaluate_concepts()
if __name__ == "__main__":
    main()
