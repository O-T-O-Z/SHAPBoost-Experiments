"""Evaluate every selection record on the test part of the SAME outer split.

Replaces perform_{regression,survival}_experiment.py. Differences:
  * the split is looked up by split_id from splits/<dataset>.json (no re-seeding
    with random_state=42, no positional idx), and its test_hash is verified;
  * preprocessing is refitted on that split's training part only;
  * every fold of every method is reported; failed selections stay in the table
    as NaN with their status, so all methods are compared on the same 30 splits;
  * output: per-fold long table + per-method summary incl. subset-size spread;
  * per-fold curves (performance on the first 1..k selected features) are written
    to <dataset>_curves.csv for plot_curves.py. Nothing is trimmed: plot_curves.py
    carries each fold's last value forward ("feature budget" curve). k = 0 is the
    no-feature baseline, so folds with an empty or failed selection still count.

Usage: python run_evaluation.py -d diabetes -t reg [--dense 25 --step 5 | --no-curves]
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

from cv_protocol import get_splits, split_arrays
from dataloading import load_regression_dataset, load_survival_dataset
from fs_methods import to_surv
from xgb_survival_regressor import XGBSurvivalRegressor

# as in the original perform_survival_experiment.py
XGB_COX_PARAMS = {"objective": "survival:cox", "eval_metric": "cox-nloglik",
                    "learning_rate": 0.05, "max_depth": 3, "grow_policy": "lossguide",
                    "lambda": 0.01, "alpha": 0.02, "n_jobs": 1}

EVALUATORS = {
    "reg": {
        "LinearRegression": lambda seed: LinearRegression(),
        "GradientBoosting": lambda seed: GradientBoostingRegressor(
            learning_rate=0.01, max_depth=4, n_iter_no_change=10, random_state=seed),
    },
    "surv": {
        "CoxPH": lambda seed: CoxPHSurvivalAnalysis(alpha=0.1),
        "RSF": lambda seed: RandomSurvivalForest(n_estimators=200, min_samples_leaf=15,
                                                    n_jobs=-1, random_state=seed),
        "XGBoost-Cox": lambda seed: XGBSurvivalRegressor(**XGB_COX_PARAMS, random_state=seed),
    },
}


def score(task, model, X_tr, y_tr, X_te, y_te):
    if task == "reg":
        model.fit(X_tr, y_tr.ravel())
        p = model.predict(X_te)
        return {"MAE": mean_absolute_error(y_te, p), "R2": r2_score(y_te, p)}
    # XGBSurvivalRegressor takes [lower, upper] bounds; sksurv models a structured array.
    # Both predict a risk score (higher = earlier event).
    model.fit(X_tr, y_tr if isinstance(model, XGBSurvivalRegressor) else to_surv(y_tr))
    s = to_surv(y_te)
    return {"C-index": concordance_index_censored(s["event"], s["time"], model.predict(X_te))[0]}


def null_score(task, y_tr, y_te):
    """k = 0: model without features (training mean / no risk ordering)."""
    if task == "reg":
        p = np.full(len(y_te), y_tr.mean())
        return {"MAE": mean_absolute_error(y_te, p), "R2": r2_score(y_te, p)}
    return {"C-index": 0.5}


def curve_grid(n: int, dense: int, step: int) -> list:
    """Every k up to `dense`, then every `step`-th k; the full subset is always included."""
    ks = list(range(1, min(n, dense) + 1)) + list(range(dense + step, n, step))
    return sorted(set(ks + [n])) if n else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True)
    ap.add_argument("-t", "--task", choices=["reg", "surv"], required=True)
    ap.add_argument("--dense", type=int, default=25, help="evaluate every k up to this")
    ap.add_argument("--step", type=int, default=5, help="then every step-th k")
    ap.add_argument("--no-curves", action="store_true", help="only score the full subset")
    args = ap.parse_args()

    loader = load_regression_dataset if args.task == "reg" else load_survival_dataset
    X, y = loader(args.dataset)
    X, y = X.values.astype(float), y.values.astype(float)
    splits = {s["split_id"]: s for s in get_splits(args.dataset, y, args.task)}

    rows, curves = [], []
    for path in sorted(glob.glob(f"results/selection/{args.dataset}/*/*.json")):
        rec = json.load(open(path))
        split = splits[rec["split_id"]]
        if split["test_hash"] != rec["test_hash"]:
            raise RuntimeError(f"{path}: selection was run on a different split.")
        X_tr, X_te, y_tr, y_te = split_arrays(X, y, split)
        base = {k: rec[k] for k in ("method", "split_id", "repeat", "fold", "status",
                                    "n_selected", "wall_seconds", "cpu_seconds")}
        f = rec["features"] if rec["status"] == "ok" else []
        ks = [len(f)] if args.no_curves else curve_grid(len(f), args.dense, args.step)
        for ev_name, factory in EVALUATORS[args.task].items():
            row = {**base, "evaluator": ev_name}
            key = {"method": rec["method"], "evaluator": ev_name, "split_id": rec["split_id"],
                    "repeat": rec["repeat"], "fold": rec["fold"], "n_selected": len(f)}
            curves.append({**key, "k": 0, **null_score(args.task, y_tr, y_te)})
            for k in ks:
                try:
                    res = score(args.task, factory(rec["seed"]), X_tr[:, f[:k]], y_tr,
                                X_te[:, f[:k]], y_te)
                except Exception as e:
                    row.update(status=f"eval_failed at k={k}: {e!r}")
                    break
                curves.append({**key, "k": k, **res})
                if k == len(f):
                    row.update(res)  # the method's actual selected subset
            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs("results/evaluation", exist_ok=True)
    df.to_csv(f"results/evaluation/{args.dataset}_per_fold.csv", index=False)
    pd.DataFrame(curves).to_csv(f"results/evaluation/{args.dataset}_curves.csv", index=False)

    metric = "MAE" if args.task == "reg" else "C-index"
    summ = df.groupby(["evaluator", "method"]).agg(
        n_splits=("split_id", "nunique"),
        n_ok=("status", lambda s: int((s == "ok").sum())),
        metric_mean=(metric, "mean"), metric_sd=(metric, "std"),
        k_median=("n_selected", "median"),
        k_q25=("n_selected", lambda s: s.quantile(0.25)),
        k_q75=("n_selected", lambda s: s.quantile(0.75)),
        k_min=("n_selected", "min"), k_max=("n_selected", "max"),
        wall_s_median=("wall_seconds", "median"),
    ).reset_index().rename(columns={"metric_mean": f"{metric}_mean", "metric_sd": f"{metric}_sd"})
    summ.to_csv(f"results/evaluation/{args.dataset}_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
