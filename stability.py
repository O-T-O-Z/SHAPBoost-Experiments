"""Stability and runtime measurements (reviewer point 4).

Two stability analyses, both on selections obtained from DIFFERENT data:
  1. across the 30 outer training folds already produced by run_selection.py;
  2. a dedicated subsampling experiment: B random half-samples of the data
     (without replacement), each with its own inner-CV seed.
Reported: Nogueira et al. (2018, JMLR 18:174) stability index with 95% CI,
and mean pairwise Jaccard. Runtime: median/IQR of wall and CPU seconds.

Usage:
  python stability.py -d diabetes -t reg                 # analysis 1 + runtime
  python stability.py -d diabetes -t reg --subsample 50  # adds analysis 2
"""
import argparse
import glob
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

from cv_protocol import fit_preprocessor
from dataloading import load_regression_dataset, load_survival_dataset
from fs_methods import METHODS


def nogueira(Z: np.ndarray):
    """Nogueira et al. (2018) stability of a (M runs x p features) 0/1 matrix.

    Returns (phi, ci_low, ci_high); follows the authors' reference implementation
    (github.com/nogueirs/JMLR2018), including the asymptotic variance.
    """
    M, p = Z.shape
    pf = Z.mean(0)
    k_bar = pf.sum()
    k = Z.sum(1)
    denom = (k_bar / p) * (1 - k_bar / p)
    if M < 2 or denom == 0:
        return np.nan, np.nan, np.nan
    phi = 1 - M / (M - 1) * np.mean(pf * (1 - pf)) / denom
    phi_i = (1 / denom) * (
        (Z * pf).mean(1) - k * k_bar / p ** 2
        + phi / 2 * (2 * k * k_bar / p ** 2 - k / p - k_bar / p + 1)
    )
    var = 4 / M ** 2 * ((phi_i - phi_i.mean()) ** 2).sum()
    half = 1.96 * np.sqrt(var)
    return float(phi), float(phi - half), float(phi + half)


def jaccard_mean(sets):
    vals = [len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2) if a | b]
    return float(np.mean(vals)) if vals else np.nan


def to_matrix(feature_lists, p):
    Z = np.zeros((len(feature_lists), p), dtype=int)
    for i, f in enumerate(feature_lists):
        Z[i, f] = 1
    return Z


def summarize(dataset, p, records, label):
    out = []
    for method, recs in records.items():
        feats = [r["features"] for r in recs if r["status"] == "ok"]
        if len(feats) < 2:
            continue
        phi, lo, hi = nogueira(to_matrix(feats, p))
        wall = np.array([r["wall_seconds"] for r in recs])
        cpu = np.array([r["cpu_seconds"] for r in recs])
        out.append(dict(dataset=dataset, analysis=label, method=method, runs=len(feats),
                        nogueira=phi, ci_low=lo, ci_high=hi,
                        jaccard=jaccard_mean([set(f) for f in feats]),
                        k_median=np.median([len(f) for f in feats]),
                        wall_s_median=np.median(wall), wall_s_q75=np.quantile(wall, .75),
                        cpu_s_median=np.median(cpu)))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True)
    ap.add_argument("-t", "--task", choices=["reg", "surv"], required=True)
    ap.add_argument("-m", "--methods", nargs="*", default=None,
                    help="methods for the half-sample analysis (default: all); the "
                         "outer-fold analysis always covers every method")
    ap.add_argument("--subsample", type=int, default=0, help="B half-samples")
    args = ap.parse_args()

    loader = load_regression_dataset if args.task == "reg" else load_survival_dataset
    X, y = loader(args.dataset)
    X, y = X.values.astype(float), y.values.astype(float)
    p = X.shape[1]

    recs = {}
    for path in glob.glob(f"results/selection/{args.dataset}/*/*.json"):
        r = json.load(open(path))
        recs.setdefault(r["method"], []).append(r)  # outer folds: always all methods
    os.makedirs("results/stability", exist_ok=True)
    out = f"results/stability/{args.dataset}.csv"
    tables = [summarize(args.dataset, p, recs, "outer folds")]

    def save():
        """Written after every method, so a time-out keeps what was computed."""
        res = pd.concat(tables)
        res.to_csv(out, index=False)
        return res

    save()  # the cheap outer-fold analysis is on disk before the slow part starts

    if args.subsample:
        # method by method (not half-sample by half-sample): every finished method
        # is saved, so a time-out costs only the method that was running.
        for m in args.methods or METHODS:
            rng = np.random.default_rng(2024)  # same half-samples for every method
            runs = []
            for b in range(args.subsample):
                idx = rng.choice(len(y), size=len(y) // 2, replace=False)
                Xb = fit_preprocessor(X[idx]).transform(X[idx])
                t0, c0 = time.perf_counter(), time.process_time()
                try:
                    f, st = METHODS[m](Xb, y[idx], args.task, 10_000 + b), "ok"
                except Exception:
                    f, st = [], "failed"
                runs.append(dict(features=list(map(int, f)), status=st,
                                 wall_seconds=time.perf_counter() - t0,
                                 cpu_seconds=time.process_time() - c0))
            tables.append(summarize(args.dataset, p, {m: runs},
                                    f"{args.subsample} half-samples"))
            save()
            print(f"{m}: {args.subsample} half-samples done", flush=True)

    print(save().round(3).to_string(index=False))


if __name__ == "__main__":
    main()