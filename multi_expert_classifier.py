
import os
import re
import glob
import argparse
import itertools
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score, recall_score, accuracy_score,
    f1_score, roc_auc_score,
)

PRED_DIR   = "results/label_prediction"
ENS_DIR    = "results/multi_expert"
TABLE_DIR  = "results/tables"

# Every classifier / refiner the pipeline can emit. Extend here if you add more.
KNOWN_CLASSIFIERS = ["MMed", "Mistral", "MedGemma", "Qwen"]
KNOWN_REFINERS    = ["rule", "mmed", "mistral", "none"]

POS_LABEL = "melanoma"   # positive class for sensitivity
NEG_LABEL = "nevus"


# --------------------------------------------------------------------------- #
# Filename parsing
# --------------------------------------------------------------------------- #
def build_regex():
    clf = "|".join(map(re.escape, KNOWN_CLASSIFIERS))
    ref = "|".join(map(re.escape, KNOWN_REFINERS))
    # raw_values / gt_concepts / extractor are captured (not hardcoded) so the
    # parser survives config changes.
    return re.compile(
        r"(?P<dataset>PH2|Derm7pt|HAM10000)(?:_split_(?P<split>\d+))?_"
        rf"(?P<clf>{clf})_diagnostic_report_validation_"
        r"raw_values_(?P<raw>True|False)_"
        r"gt_concepts_(?P<gt>True|False)_"
        r"model_extractor_(?P<extractor>[A-Za-z0-9]+)_"
        r"n_demos_(?P<demos>\d+)_"
        rf"refiner_(?P<refiner>{ref})_"
        r"retrieval_(?P<ret>rices|random)\.csv$"
    )


NAME_RE = build_regex()


def parse_name(fn):
    m = NAME_RE.match(fn)
    return m.groupdict() if m else None


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def metric_row(y_true, y_pred, y_score=None):
    """All metrics as percentages, using the repo's melanoma=positive convention."""
    row = {
        "N":       len(y_true),
        "Acc":     accuracy_score(y_true, y_pred) * 100,
        "BAcc":    balanced_accuracy_score(y_true, y_pred) * 100,
        "Sens":    recall_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0) * 100,
        "Spec":    recall_score(y_true, y_pred, pos_label=NEG_LABEL, zero_division=0) * 100,
        "MacroF1": f1_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "AUC":     np.nan,
    }
    if y_score is not None:
        y_bin = np.array([1 if t == POS_LABEL else 0 for t in y_true])
        if len(np.unique(y_bin)) == 2:      # AUC undefined for a single class
            row["AUC"] = roc_auc_score(y_bin, y_score) * 100
    return row


# --------------------------------------------------------------------------- #
# Voting
# --------------------------------------------------------------------------- #
def hard_vote(labels, tie):
    counts = Counter(labels).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:      # tie
        if tie == "abstain":
            return None
        return POS_LABEL if tie == "melanoma" else NEG_LABEL
    return counts[0][0]


def soft_vote(probs):
    """probs = list of P(melanoma). Returns (label, mean_prob)."""
    p = float(np.mean(probs))
    return (POS_LABEL if p >= 0.5 else NEG_LABEL), p


# --------------------------------------------------------------------------- #
# Core: one (n_demos, retrieval) setting
# --------------------------------------------------------------------------- #
def run_setting(experts, n_demos, retrieval, tie, prob_col):
    """Return a long-form DataFrame with one row per (dataset,split,refiner,method)."""
    groups = {}
    for path in glob.glob(f"{PRED_DIR}/*.csv"):
        info = parse_name(os.path.basename(path))
        if not info:
            continue
        if info["demos"] != str(n_demos) or info["ret"] != retrieval:
            continue
        key = (info["dataset"], info["split"], info["refiner"])
        groups.setdefault(key, {})[info["clf"]] = path

    rows = []
    for (dataset, split, refiner), files in sorted(groups.items()):
        present = [e for e in experts if e in files]
        if len(present) < 2:            # nothing to ensemble
            continue

        dfs = {e: pd.read_csv(files[e]).set_index("image_id") for e in present}
        ids = sorted(set.intersection(*[set(d.index) for d in dfs.values()]))
        if not ids:
            continue

        use_soft = all(prob_col in dfs[e].columns for e in present)

        gt = [dfs[present[0]].loc[i, "gt_response"] for i in ids]

        # --- individual members ---
        for e in present:
            y_pred = [dfs[e].loc[i, "llm_response"] for i in ids]
            y_score = ([float(dfs[e].loc[i, prob_col]) for i in ids]
                       if prob_col in dfs[e].columns else None)
            r = metric_row(gt, y_pred, y_score)
            r.update({"dataset": dataset, "split": split, "refiner": refiner,
                      "method": e, "kind": "member"})
            rows.append(r)

        # --- ensemble ---
        ens_pred, ens_score = [], []
        for i in ids:
            if use_soft:
                lab, p = soft_vote([float(dfs[e].loc[i, prob_col]) for e in present])
                ens_pred.append(lab); ens_score.append(p)
            else:
                ens_pred.append(hard_vote([dfs[e].loc[i, "llm_response"]
                                           for e in present], tie))
        keep = [j for j, p in enumerate(ens_pred) if p is not None]   # drop abstains
        gt_k   = [gt[j] for j in keep]
        pred_k = [ens_pred[j] for j in keep]
        score_k = [ens_score[j] for j in keep] if use_soft else None

        r = metric_row(gt_k, pred_k, score_k)
        r.update({"dataset": dataset, "split": split, "refiner": refiner,
                  "method": "+".join(present), "kind": "ensemble"})
        rows.append(r)

        # persist the per-image ensemble decisions
        os.makedirs(ENS_DIR, exist_ok=True)
        out = pd.DataFrame({"image_id": ids, "gt_response": gt,
                            "ensemble_response": ens_pred})
        for e in present:
            out[f"{e}_response"] = [dfs[e].loc[i, "llm_response"] for i in ids]
        tag = dataset + (f"_split_{split}" if split else "")
        out.to_csv(f"{ENS_DIR}/{tag}_ENSEMBLE_refiner_{refiner}"
                   f"_ndemos_{n_demos}_{retrieval}.csv", index=False)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["n_demos"] = n_demos
    df["retrieval"] = retrieval
    return df


# --------------------------------------------------------------------------- #
# Aggregation + table export
# --------------------------------------------------------------------------- #
METRIC_COLS = ["N", "Acc", "BAcc", "Sens", "Spec", "MacroF1", "AUC"]


def aggregate(df):
    """Average PH2 over its 5 splits; single value for Derm7pt/HAM10000."""
    keys = ["dataset", "refiner", "method", "kind", "n_demos", "retrieval"]
    agg = df.groupby(keys, as_index=False)[METRIC_COLS].mean()
    return agg


def add_ensemble_delta(agg):
    """For each (dataset,refiner,shot), delta = ensemble.BAcc - best member.BAcc."""
    agg = agg.copy()
    agg["delta_vs_best"] = np.nan
    grp = ["dataset", "refiner", "n_demos", "retrieval"]
    for _, g in agg.groupby(grp):
        members = g[g["kind"] == "member"]
        ens = g[g["kind"] == "ensemble"]
        if members.empty or ens.empty:
            continue
        best = members["BAcc"].max()
        agg.loc[ens.index, "delta_vs_best"] = ens["BAcc"].values - best
    return agg


def export_tables(agg, stem):
    os.makedirs(TABLE_DIR, exist_ok=True)
    disp = agg.copy()
    for c in METRIC_COLS + ["delta_vs_best"]:
        if c in disp.columns and c != "N":
            disp[c] = disp[c].round(2)
    disp["N"] = disp["N"].round(0).astype(int)

    csv_path = f"{TABLE_DIR}/{stem}.csv"
    md_path  = f"{TABLE_DIR}/{stem}.md"
    tex_path = f"{TABLE_DIR}/{stem}.tex"

    disp.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write(disp.to_markdown(index=False))
    with open(tex_path, "w") as f:
        f.write(disp.to_latex(index=False, float_format="%.2f",
                              caption="Multi-expert ensemble vs. individual "
                                      "classifiers per refiner.",
                              label=f"tab:{stem}"))
    return csv_path, md_path, tex_path


def pretty_print(agg):
    for (ds, ref, nd, ret), g in agg.groupby(
            ["dataset", "refiner", "n_demos", "retrieval"]):
        print(f"\n{'='*74}")
        print(f"  {ds}  |  refiner={ref}  |  n_demos={nd}  |  {ret}")
        print('='*74)
        print(f"  {'method':<26}{'N':>5}{'Acc':>8}{'BAcc':>8}"
              f"{'Sens':>8}{'Spec':>8}{'F1':>8}")
        print("  " + "-"*70)
        for _, r in g.sort_values("kind").iterrows():
            star = " *" if r["kind"] == "ensemble" else ""
            print(f"  {r['method'][:24]:<26}{int(r['N']):>5}{r['Acc']:>8.2f}"
                  f"{r['BAcc']:>8.2f}{r['Sens']:>8.2f}{r['Spec']:>8.2f}"
                  f"{r['MacroF1']:>8.2f}{star}")
        ens = g[g["kind"] == "ensemble"]
        if not ens.empty and not np.isnan(ens["delta_vs_best"].iloc[0]):
            d = ens["delta_vs_best"].iloc[0]
            sign = "+" if d >= 0 else ""
            print(f"  -> ensemble BAcc {sign}{d:.2f} vs best single expert")


# --------------------------------------------------------------------------- #
def parse_sweep(items):
    out = []
    for it in items:
        nd, ret = it.split(":")
        out.append((int(nd), ret))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", nargs="+", default=["MMed", "Mistral", "MedGemma", "Qwen"],
                    help="Committee members (order irrelevant). Odd sizes avoid ties.")
    ap.add_argument("--n_demos", default="0")
    ap.add_argument("--retrieval", default="rices", choices=["rices", "random"])
    ap.add_argument("--sweep", nargs="*", default=None,
                    help='Override n_demos/retrieval with pairs like "0:rices" "4:random".')
    ap.add_argument("--tie", default="melanoma",
                    choices=["melanoma", "nevus", "abstain"],
                    help="Tie policy for even committees.")
    ap.add_argument("--prob_col", default="p_melanoma",
                    help="If this column exists in every member CSV, soft-vote + AUC.")
    args = ap.parse_args()

    for e in args.experts:
        if e not in KNOWN_CLASSIFIERS:
            raise SystemExit(f"Unknown expert '{e}'. Known: {KNOWN_CLASSIFIERS}")

    settings = (parse_sweep(args.sweep) if args.sweep
                else [(int(args.n_demos), args.retrieval)])

    all_parts = []
    for nd, ret in settings:
        part = run_setting(args.experts, nd, ret, args.tie, args.prob_col)
        if not part.empty:
            all_parts.append(part)

    if not all_parts:
        print(f"No ensembles built. Need >=2 of {args.experts} sharing the same "
              f"(dataset, split, refiner) for the requested setting(s).")
        return

    raw = pd.concat(all_parts, ignore_index=True)
    agg = add_ensemble_delta(aggregate(raw))

    committee = "-".join(args.experts)
    shots = "_".join(f"{nd}{ret}" for nd, ret in settings)
    stem = f"ensemble_{committee}_{shots}"

    pretty_print(agg)
    paths = export_tables(agg, stem)
    print(f"\nSaved tables:\n  " + "\n  ".join(paths))


if __name__ == "__main__":
    main()
