"""SHAPBoost vs every baseline on the same outer splits, with uncertainty.

For each dataset and baseline, the paired per-split differences (baseline minus
SHAPBoost) are summarized with the Nadeau-Bengio corrected resampled t-test,
which accounts for overlapping training sets in repeated cross-validation:
  * difference in test score of the selected subsets (95% CI, p-value);
  * difference in number of selected features (95% CI, p-value);
p-values are Holm-adjusted over the baselines within each dataset.
A positive score difference means the baseline is better (for MAE: worse).

Outputs results/tables/shapboost_vs_baselines_<task>_<evaluator>_<metric>.csv and a
readable summary on screen.

Usage: python compare_to_shapboost.py [--reg-metric R2|MAE] [--evaluators ...] [--datasets ...]
"""
import argparse

import numpy as np
import pandas as pd

import report_utils as ru
from report_ablation import corrected_ttest, test_train_ratio

ALL_EVALUATORS = {"reg": ["LinearRegression", "GradientBoosting"],
                  "surv": ["CoxPH", "RSF", "XGBoost-Cox"]}


def holm(p):
    """Holm step-down adjustment; NaNs are kept."""
    p = np.asarray(p, float)
    ok = np.flatnonzero(np.isfinite(p))
    out = np.full_like(p, np.nan)
    order = ok[np.argsort(p[ok])]
    m, running = len(order), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def compare(task, evaluator, datasets, metric):
    ref = ru.MAIN[task]
    rows = []
    for ds in datasets:
        res = ru.load_results(ds, evaluator, task)
        if res is None:
            continue
        _, curves = res
        op = ru.operating_points(curves, metric).set_index(["method", "split_id"])
        methods = op.index.get_level_values(0).unique()
        if ref not in methods:
            continue
        ratio = test_train_ratio(ds)
        b = op.loc[ref]
        ds_rows = []
        for m in methods:
            if ru.is_proposed(m):
                continue
            a = op.loc[m]
            common = a.index.intersection(b.index)
            d_s = (a.loc[common, "score"] - b.loc[common, "score"]).values
            d_k = (a.loc[common, "n_selected"] - b.loc[common, "n_selected"]).values
            ms, los, his, ps = corrected_ttest(d_s, ratio)
            mk, lok, hik, pk = corrected_ttest(d_k, ratio)
            ds_rows.append(dict(
                dataset=ru.DS_NAME[ds], baseline=m, n_splits=len(common),
                shapboost_score=b.loc[common, "score"].mean(),
                baseline_score=a.loc[common, "score"].mean(),
                d_score=ms, d_score_lo=los, d_score_hi=his, p_score=ps,
                shapboost_k=b.loc[common, "n_selected"].median(),
                baseline_k=a.loc[common, "n_selected"].median(),
                d_k=mk, d_k_lo=lok, d_k_hi=hik, p_k=pk))
        if ds_rows:
            t = pd.DataFrame(ds_rows)
            t["p_score_holm"] = holm(t.p_score)
            t["p_k_holm"] = holm(t.p_k)
            rows.append(t)
    if not rows:
        print(f"no results for {task}/{evaluator}")
        return
    tab = pd.concat(rows, ignore_index=True)
    ru.save_table(tab.set_index(["dataset", "baseline"]),
                  f"results/tables/shapboost_vs_baselines_{task}_{evaluator}_{metric}.csv")

    print(f"\n=== {evaluator} ({metric}); difference = baseline - SHAPBoost, 95% CI, Holm p")
    for _, r in tab.iterrows():
        print(f"{r.dataset:18s} {r.baseline:22s} "
              f"score {r.baseline_score:6.3f} vs {r.shapboost_score:6.3f}: "
              f"{r.d_score:+.3f} [{r.d_score_lo:+.3f}, {r.d_score_hi:+.3f}] p={r.p_score_holm:.3f} | "
              f"k {r.baseline_k:g} vs {r.shapboost_k:g}: "
              f"{r.d_k:+.1f} [{r.d_k_lo:+.1f}, {r.d_k_hi:+.1f}] p={r.p_k_holm:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reg-metric", choices=["R2", "MAE"], default="R2")
    ap.add_argument("--evaluators", nargs="*", default=["LinearRegression", "CoxPH"])
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()
    for task in ("reg", "surv"):
        datasets = ru.REGRESSION_DATASETS if task == "reg" else ru.SURVIVAL_DATASETS
        if args.datasets:
            datasets = [d for d in datasets if d in args.datasets]
        for ev in ALL_EVALUATORS[task]:
            if datasets and ev in args.evaluators:
                compare(task, ev, datasets, args.reg_metric if task == "reg" else "C-index")


if __name__ == "__main__":
    main()
