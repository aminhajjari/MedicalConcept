#!/usr/bin/env python3
"""
evaluate_fewshot_results.py
Compare all 6 configs across 0/1/2/4/8 shots for all 3 datasets.
Also compares RICES retrieval vs Random demo selection (1-shot and 2-shot).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, recall_score

REPORTED_CLASSIFIERS = ["MMed", "Mistral", "Qwen"]   # MedGemma: ensemble member only
RESULTS_DIR = Path("results/label_prediction")
N_SPLITS    = 5

CONFIGS = [
    {"label": "Rule+MMed",       "llm": "MMed",     "refiner": "rule"},
    {"label": "Rule+Mistral",    "llm": "Mistral",  "refiner": "rule"},
    {"label": "Mistral+Mistral", "llm": "Mistral",  "refiner": "mistral"},
    {"label": "Mistral+MMed",    "llm": "MMed",     "refiner": "mistral"},
    {"label": "MMed+MMed",       "llm": "MMed",     "refiner": "mmed"},
    {"label": "MMed+Mistral",    "llm": "Mistral",  "refiner": "mmed"},
    # --- new classifiers ---
    {"label": "Rule+MedGemma",    "llm": "MedGemma", "refiner": "rule"},
    {"label": "Mistral+MedGemma", "llm": "MedGemma", "refiner": "mistral"},
    {"label": "MMed+MedGemma",    "llm": "MedGemma", "refiner": "mmed"},
    {"label": "Rule+Qwen",        "llm": "Qwen",     "refiner": "rule"},
    {"label": "Mistral+Qwen",     "llm": "Qwen",     "refiner": "mistral"},
    {"label": "MMed+Qwen",        "llm": "Qwen",     "refiner": "mmed"},
]

PAPER_BACC = {"PH2": 78.07, "Derm7pt": 78.56, "HAM10000": 76.00}
N_SHOTS    = [0, 1, 2, 4, 8]


# ── path helpers ───────────────────────────────────────────────────────────────

def csv_path(cfg, dataset, n_demos, split=None, retrieval="rices"):
    llm     = cfg["llm"]
    refiner = cfg["refiner"]
    suffix  = (f"raw_values_False_gt_concepts_False"
               f"_model_extractor_Explicd_n_demos_{n_demos}_refiner_{refiner}_retrieval_{retrieval}")
    if split is not None:
        name = f"{dataset}_split_{split}_{llm}_diagnostic_report_validation_{suffix}.csv"
    else:
        name = f"{dataset}_{llm}_diagnostic_report_validation_{suffix}.csv"
    return RESULTS_DIR / name


# ── metric helpers ─────────────────────────────────────────────────────────────

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

def avg_ph2(cfg, n_demos, retrieval="rices"):
    results = [
        compute_metrics(csv_path(cfg, "PH2", n_demos, s, retrieval=retrieval))
        for s in range(N_SPLITS)
    ]
    valid = [r for r in results if r]
    if not valid:
        return None
    return {k: np.mean([r[k] for r in valid]) for k in ("bacc", "sens", "spec")}

def get_metrics(cfg, dataset, n_demos, retrieval="rices"):
    """Unified helper: returns metrics for any dataset/retrieval combo."""
    if dataset == "PH2":
        return avg_ph2(cfg, n_demos, retrieval=retrieval)
    return compute_metrics(csv_path(cfg, dataset, n_demos, retrieval=retrieval))


# ── formatters ─────────────────────────────────────────────────────────────────

def pct(v):
    return f"{v:.2f}" if v is not None else "  —  "

def delta(v, baseline):
    if v is None:
        return "  —  "
    d = v - baseline
    return f"{'+' if d >= 0 else ''}{d:.2f}"

def delta_ab(a, b):
    """Delta between two metric dicts (a - b), or '—' if missing."""
    if a is None or b is None:
        return "  —  "
    d = a["bacc"] - b["bacc"]
    return f"{'+' if d >= 0 else ''}{d:.2f}"


# ── TABLE 1: few-shot comparison (RICES only) ──────────────────────────────────

def print_fewshot_table():
    print("\n" + "="*110)
    print("  FEW-SHOT COMPARISON — All 6 Configs × All Datasets  [RICES retrieval]")
    print("="*110)

    for dataset in ["PH2", "Derm7pt", "HAM10000"]:
        paper = PAPER_BACC[dataset]
        shots = [0, 1, 2] if dataset == "HAM10000" else N_SHOTS

        print(f"\n{'='*110}")
        print(f"  DATASET: {dataset}   (paper baseline = {paper:.2f}%)")
        print(f"{'='*110}")

        shots_header = "".join(f"  {n:>5}-shot" for n in shots)
        print(f"  {'Config':<20}{shots_header}")
        print(f"  {'(BAcc %)':<20}" + "  " + "-"*90)

        for cfg in CONFIGS:
            row = f"  {cfg['label']:<20}"
            best_bacc = 0
            for n in shots:
                m = get_metrics(cfg, dataset, n, retrieval="rices")
                if m:
                    row += f"  {pct(m['bacc']):>8}"
                    best_bacc = max(best_bacc, m['bacc'])
                else:
                    row += f"  {'MISSING':>8}"

            best_delta = delta(best_bacc, paper) if best_bacc > 0 else "N/A"
            row += f"   (best Δ vs paper: {best_delta:>6}%)"
            print(row)

        print(f"  {'Paper baseline':<20}" + f"  {paper:>8.2f}" * len(shots))
        print(f"{'='*110}")


# ── TABLE 2: detailed BAcc/Sens/Spec per n-shot (RICES only) ──────────────────

def print_detailed_table():
    print("\n\n" + "="*110)
    print("  DETAILED VIEW — BAcc / Sens / Spec per n-shot  [RICES retrieval]")
    print("="*110)

    for dataset in ["PH2", "Derm7pt", "HAM10000"]:
        paper = PAPER_BACC[dataset]
        shots = [0, 1, 2] if dataset == "HAM10000" else N_SHOTS

        print(f"\n{'='*110}")
        print(f"  DATASET: {dataset}")
        print(f"{'='*110}")
        print(f"  {'Config':<20}  {'n':>4}  {'BAcc':>7}  {'Sens':>7}  {'Spec':>7}  {'Δ Paper':>8}")
        print(f"  {'-'*60}")

        for cfg in CONFIGS:
            for n in shots:
                m = get_metrics(cfg, dataset, n, retrieval="rices")
                label = cfg['label'] if n == shots[0] else ""
                if m:
                    print(f"  {label:<20}  {n:>4}  "
                          f"{pct(m['bacc']):>7}  {pct(m['sens']):>7}  "
                          f"{pct(m['spec']):>7}  {delta(m['bacc'], paper):>8}%")
                else:
                    print(f"  {label:<20}  {n:>4}  "
                          f"{'MISSING':>7}  {'MISSING':>7}  {'MISSING':>7}  {'N/A':>8}")
            print()


# ── TABLE 3: RICES vs Random — side-by-side ───────────────────────────────────

def print_rices_vs_random_table():
    """
    For each dataset × config × n_shots in [1, 2]:
      BAcc(RICES) | BAcc(Random) | Δ(RICES−Random) | Sens | Spec
    Only configs/shots where at least one retrieval method has a file are shown.
    """
    SHOTS_TO_COMPARE = [1, 2]

    print("\n\n" + "="*130)
    print("  RICES vs RANDOM DEMO SELECTION — Side-by-Side Comparison")
    print("  (Only 1-shot and 2-shot; HAM10000 capped at 2-shot)")
    print("="*130)

    sep = "="*130

    for dataset in ["PH2", "Derm7pt", "HAM10000"]:
        paper = PAPER_BACC[dataset]
        shots = [s for s in SHOTS_TO_COMPARE if not (dataset == "HAM10000" and s > 2)]

        print(f"\n{sep}")
        print(f"  DATASET: {dataset}   (paper baseline = {paper:.2f}%)")
        print(f"{sep}")
        print(
            f"  {'Config':<20}  {'n':>4}  "
            f"{'BAcc RICES':>11}  {'BAcc RND':>9}  {'Δ(R−Rnd)':>9}  "
            f"{'Sens RICES':>11}  {'Sens RND':>9}  "
            f"{'Spec RICES':>11}  {'Spec RND':>9}  "
            f"{'Winner':>8}"
        )
        print(f"  {'-'*120}")

        any_row_printed = False
        for cfg in CONFIGS:
            first_row = True
            for n in shots:
                mr = get_metrics(cfg, dataset, n, retrieval="rices")
                mn = get_metrics(cfg, dataset, n, retrieval="random")

                # Skip if both missing
                if mr is None and mn is None:
                    continue

                any_row_printed = True
                label = cfg['label'] if first_row else ""
                first_row = False

                # Case: only RICES exists, random was never run
                if mn is None:
                    print(
                        f"  {label:<20}  {n:>4}  "
                        f"{pct(mr['bacc']):>11}  {'NOT RUN':>9}  {'N/A':>9}  "
                        f"{pct(mr['sens']):>11}  {'N/A':>9}  "
                        f"{pct(mr['spec']):>11}  {'N/A':>9}  {'N/A':>8}"
                    )
                    continue

                # Case: only Random exists (unusual but handle it)
                if mr is None:
                    print(
                        f"  {label:<20}  {n:>4}  "
                        f"{'NOT RUN':>11}  {pct(mn['bacc']):>9}  {'N/A':>9}  "
                        f"{'N/A':>11}  {pct(mn['sens']):>9}  "
                        f"{'N/A':>11}  {pct(mn['spec']):>9}  {'N/A':>8}"
                    )
                    continue

                # Both exist — full comparison
                # FIX: assign bacc_r and bacc_n BEFORE using them
                bacc_r = pct(mr['bacc'])
                bacc_n = pct(mn['bacc'])
                sens_r = pct(mr['sens'])
                sens_n = pct(mn['sens'])
                spec_r = pct(mr['spec'])
                spec_n = pct(mn['spec'])
                d_str  = delta_ab(mr, mn)

                if mr['bacc'] > mn['bacc'] + 0.1:
                    winner = "RICES ▲"
                elif mn['bacc'] > mr['bacc'] + 0.1:
                    winner = "Random ▲"
                else:
                    winner = "  tie  "

                print(
                    f"  {label:<20}  {n:>4}  "
                    f"{bacc_r:>11}  {bacc_n:>9}  {d_str:>9}  "
                    f"{sens_r:>11}  {sens_n:>9}  "
                    f"{spec_r:>11}  {spec_n:>9}  "
                    f"{winner:>8}"
                )

            if not first_row:
                print()  # blank line between configs

        if not any_row_printed:
            print(f"  *** No result files found for {dataset}. ***")
            print(f"  *** Run the pipeline with --random_demos to generate random baselines. ***")

        print(sep)


# ── TABLE 4: RICES vs Random — best-config summary ────────────────────────────

def print_rices_vs_random_summary():
    """
    One-row-per-dataset summary: best BAcc under RICES vs best BAcc under Random,
    across all configs and shots [1, 2].
    """
    SHOTS_TO_COMPARE = [1, 2]

    print("\n\n" + "="*90)
    print("  RICES vs RANDOM — Best-Config Summary  (best BAcc across all configs & shots)")
    print("="*90)
    print(
        f"  {'Dataset':<12}  {'Best RICES':>11}  {'Best Config (RICES)':>22}  "
        f"{'Best Random':>12}  {'Best Config (Rnd)':>22}  {'Δ':>7}"
    )
    print(f"  {'-'*85}")

    for dataset in ["PH2", "Derm7pt", "HAM10000"]:
        shots = [s for s in SHOTS_TO_COMPARE if not (dataset == "HAM10000" and s > 2)]

        best_r_bacc, best_r_label = 0.0, "—"
        best_n_bacc, best_n_label = 0.0, "—"

        for cfg in CONFIGS:
            for n in shots:
                mr = get_metrics(cfg, dataset, n, retrieval="rices")
                mn = get_metrics(cfg, dataset, n, retrieval="random")

                if mr and mr['bacc'] > best_r_bacc:
                    best_r_bacc  = mr['bacc']
                    best_r_label = f"{cfg['label']} ({n}-shot)"

                if mn and mn['bacc'] > best_n_bacc:
                    best_n_bacc  = mn['bacc']
                    best_n_label = f"{cfg['label']} ({n}-shot)"

        r_str = f"{best_r_bacc:.2f}" if best_r_bacc > 0 else "MISSING"
        n_str = f"{best_n_bacc:.2f}" if best_n_bacc > 0 else "MISSING"

        if best_r_bacc > 0 and best_n_bacc > 0:
            d = best_r_bacc - best_n_bacc
            d_str = f"{'+' if d >= 0 else ''}{d:.2f}"
        else:
            d_str = "  —  "

        print(
            f"  {dataset:<12}  {r_str:>11}  {best_r_label:>22}  "
            f"{n_str:>12}  {best_n_label:>22}  {d_str:>7}"
        )

    print("="*90)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print_fewshot_table()
    print_detailed_table()
    print_rices_vs_random_table()
    print_rices_vs_random_summary()
    print()


if __name__ == "__main__":
    main()
