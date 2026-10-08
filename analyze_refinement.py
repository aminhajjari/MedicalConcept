
import pandas as pd
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results/concept_prediction")
DATASETS = ["Derm7pt", "HAM10000"]
REFINERS = ["rule", "mistral", "mmed"]
N_SPLITS = 5  # for PH2

def load_stats(dataset, refiner, split=None):
    if split is not None:
        path = RESULTS_DIR / f"{dataset}_split_{split}_refinement_stats_Explicd_refiner_{refiner}.csv"
    else:
        path = RESULTS_DIR / f"{dataset}_refinement_stats_Explicd_refiner_{refiner}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

def analyze(dataset, refiner):
    if dataset == "PH2":
        dfs = [load_stats("PH2", refiner, s) for s in range(N_SPLITS)]
        dfs = [d for d in dfs if d is not None]
        if not dfs:
            return None
        df = pd.concat(dfs)
    else:
        df = load_stats(dataset, refiner)
        if df is None:
            return None
    
    total = len(df)
    had_violations = (df['initial_violations'] > 0).sum()
    fully_resolved = ((df['initial_violations'] > 0) & 
                       (df['final_violations'] == 0)).sum()
    converged = df['converged'].sum()
    
    return {
        'dataset': dataset,
        'refiner': refiner,
        'total_images': total,
        'pct_with_violations': had_violations / total * 100,
        'mean_initial_violations': df['initial_violations'].mean(),
        'mean_final_violations': df['final_violations'].mean(),
        'violation_reduction_pct': (
            (df['initial_violations'].mean() - df['final_violations'].mean()) 
            / max(df['initial_violations'].mean(), 0.001) * 100
        ),
        'full_convergence_rate': fully_resolved / max(had_violations, 1) * 100,
        'mean_iterations': df['iterations'].mean(),
    }

def main():
    rows = []
    for dataset in ["PH2", "Derm7pt", "HAM10000"]:
        for refiner in REFINERS:
            result = analyze(dataset, refiner)
            if result:
                rows.append(result)
    
    df = pd.DataFrame(rows)
    
    print("\n" + "="*90)
    print("  REFINEMENT ANALYSIS TABLE")
    print("="*90)
    print(f"  {'Dataset':<12} {'Refiner':<10} {'%w/Viol':>8} "
          f"{'Init Viol':>10} {'Final Viol':>11} {'Reduction%':>11} "
          f"{'Conv%':>7} {'Avg Iter':>9}")
    print("  " + "-"*80)
    
    for _, row in df.iterrows():
        print(f"  {row['dataset']:<12} {row['refiner']:<10} "
              f"{row['pct_with_violations']:>8.1f} "
              f"{row['mean_initial_violations']:>10.2f} "
              f"{row['mean_final_violations']:>11.2f} "
              f"{row['violation_reduction_pct']:>11.1f} "
              f"{row['full_convergence_rate']:>7.1f} "
              f"{row['mean_iterations']:>9.2f}")
    
    print("="*90)
    df.to_csv("results/refinement_analysis.csv", index=False)
    print("\nSaved to results/refinement_analysis.csv")

if __name__ == "__main__":
    main()
