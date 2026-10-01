"""Contribution of SHAP and reweighting, stability, runtime.

1. Ablation (same search, stopping rule and splits; only the importance measure
   and/or the reweighting step differ). Each variant is compared with full
   SHAPBoost on the SAME outer splits:
     - difference in test score of the selected subsets, with the Nadeau-Bengio
       corrected resampled t-test (repeated-CV folds overlap, so the naive
       paired t-test is anti-conservative);
     - difference in number of selected features (same test);
     - Jaccard overlap between the two selected subsets, per split.
2. Stability: Nogueira index (95% CI) and mean pairwise Jaccard from stability.py
   (outer folds and, if run with --subsample, random half-samples).
3. Runtime: CPU seconds per selection from the selection records (use
   run_selection.py, i.e. sequential, for the timing runs).

Outputs
  plots/ablation_<task>_<evaluator>.pdf
  results/tables/ablation_<task>_<evaluator>_<metric>.csv
  plots/stability_runtime_<task>.pdf
  results/tables/stability_runtime.csv

Usage: python report_ablation.py [--reg-metric R2|MAE] [--evaluators ...] [--datasets ...]
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

import report_utils as ru  # noqa: E402

VARIANTS = ["SHAPBoost-noRW", "GainBoost", "GainBoost-noRW"]
VARIANT_LABEL = {"SHAPBoost-noRW": "SHAP, no reweighting",
                 "GainBoost": "Gain importance, reweighting",
                 "GainBoost-noRW": "Gain importance, no reweighting"}
VARIANT_COLOR = {"SHAPBoost-noRW": "#1f77b4", "GainBoost": "#d62728",
                 "GainBoost-noRW": "#9467bd"}
ALL_EVALUATORS = {"reg": ["LinearRegression", "GradientBoosting"],
                  "surv": ["CoxPH", "RSF", "XGBoost-Cox"]}


def corrected_ttest(d: np.ndarray, test_train_ratio: float):
    """Nadeau & Bengio (2003) corrected resampled t-test for J paired differences.

    Returns (mean, ci_low, ci_high, p_value); variance inflated by (1/J + n_test/n_train).
    """
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    J = len(d)
    if J < 2:
        return (float(d.mean()) if J else np.nan), np.nan, np.nan, np.nan
    se = np.sqrt((1 / J + test_train_ratio) * d.var(ddof=1))
    mean = d.mean()
    if se == 0:
        return float(mean), float(mean), float(mean), (1.0 if mean == 0 else 0.0)
    t = sps.t.ppf(0.975, J - 1)
    p = 2 * sps.t.sf(abs(mean / se), J - 1)
    return float(mean), float(mean - t * se), float(mean + t * se), float(p)


def test_train_ratio(dataset: str) -> float:
    with open(f"splits/{dataset}.json") as f:
        s = json.load(f)[0]
    return len(s["test"]) / len(s["train"])


def selected_sets(dataset: str, method: str) -> dict:
    out = {}
    for p in glob.glob(f"results/selection/{dataset}/{method}/*.json"):
        r = json.load(open(p))
        out[r["split_id"]] = set(r["features"])
    return out


def ablation_rows(ds, task, evaluator, metric):
    res = ru.load_results(ds, evaluator, task)
    if res is None:
        return []
    _, curves = res
    main = ru.MAIN[task]
    op = ru.operating_points(curves, metric).set_index(["method", "split_id"])
    if main not in op.index.get_level_values(0):
        return []
    ratio = test_train_ratio(ds)
    base_sets = selected_sets(ds, "SHAPBoost")
    rows = []
    for v in VARIANTS:
        if v not in op.index.get_level_values(0):
            continue
        a, b = op.loc[v], op.loc[main]
        common = a.index.intersection(b.index)
        d_score = (a.loc[common, "score"] - b.loc[common, "score"]).values
        d_k = (a.loc[common, "n_selected"] - b.loc[common, "n_selected"]).values
        v_sets = selected_sets(ds, v)
        jac = [len(v_sets[s] & base_sets[s]) / max(len(v_sets[s] | base_sets[s]), 1)
               for s in common if s in v_sets and s in base_sets]
        m, lo, hi, p = corrected_ttest(d_score, ratio)
        mk, lok, hik, pk = corrected_ttest(d_k, ratio)
        rows.append(dict(dataset=ru.DS_NAME[ds], variant=v, n_splits=len(common),
                         d_score=m, d_score_lo=lo, d_score_hi=hi, p_score=p,
                         d_k=mk, d_k_lo=lok, d_k_hi=hik, p_k=pk,
                         jaccard_vs_shapboost=float(np.mean(jac)) if jac else np.nan,
                         identical_subsets=int(sum(j == 1 for j in jac))))
    return rows


def plot_ablation(task, evaluator, datasets, metric):
    rows = [r for ds in datasets for r in ablation_rows(ds, task, evaluator, metric)]
    if not rows:
        print(f"no ablation results for {task}/{evaluator}")
        return
    tab = pd.DataFrame(rows)
    names = [n for n in (ru.DS_NAME[d] for d in datasets) if n in set(tab.dataset)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 0.55 * len(names) * 3 + 1.8),
                                   sharey=True)
    for j, v in enumerate(VARIANTS):
        t = tab[tab.variant == v]
        y = np.array([names.index(n) for n in t.dataset]) * 4 + j
        for ax, c in ((ax1, "d_score"), (ax2, "d_k")):
            ax.errorbar(t[c], y, xerr=[t[c] - t[f"{c}_lo"], t[f"{c}_hi"] - t[c]], fmt="o",
                        color=VARIANT_COLOR[v], capsize=3, label=VARIANT_LABEL[v])
    for ax in (ax1, ax2):
        ax.axvline(0, color="black", lw=1)
        ax.grid(alpha=0.3, axis="x")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax1.set_yticks(np.arange(len(names)) * 4 + 1, names)
    ax1.invert_yaxis()
    better = "higher" if ru.higher_is_better(metric) else "lower"
    ax1.set_xlabel(f"Δ {ru.METRIC_LABEL[metric]} vs SHAPBoost ({better} = variant better)")
    ax2.set_xlabel("Δ number of selected features vs SHAPBoost")
    ax1.legend(fontsize=9, loc="best")
    fig.suptitle(f"Ablation ({evaluator}): mean paired difference over the same outer "
                 "splits, 95% CI from the Nadeau–Bengio corrected t-test", fontsize=10)
    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    fig.savefig(f"plots/ablation_{task}_{evaluator}.pdf")
    plt.close(fig)
    ru.save_table(tab.set_index(["dataset", "variant"]),
                  f"results/tables/ablation_{task}_{evaluator}_{metric}.csv")
    print(f"saved plots/ablation_{task}_{evaluator}.pdf")


def stability_runtime(datasets_by_task):
    rows = []
    for task, datasets in datasets_by_task.items():
        for ds in datasets:
            recs = [json.load(open(p)) for p in glob.glob(f"results/selection/{ds}/*/*.json")]
            if not recs:
                continue
            rt = pd.DataFrame(recs).groupby("method").agg(
                cpu_s_median=("cpu_seconds", "median"),
                cpu_s_q25=("cpu_seconds", lambda s: s.quantile(.25)),
                cpu_s_q75=("cpu_seconds", lambda s: s.quantile(.75)),
                wall_s_median=("wall_seconds", "median"), n_runs=("split_id", "count"))
            path = f"results/stability/{ds}.csv"
            if os.path.exists(path):
                st = pd.read_csv(path)
                st = st.pivot_table(index="method", columns="analysis",
                                    values=["nogueira", "ci_low", "ci_high", "jaccard"])
                st.columns = [f"{v} ({a})" for v, a in st.columns]
                rt = rt.join(st, how="left")
            rt = rt.reset_index()
            rt["method"] = rt["method"].replace(ru._DISPLAY[task])
            rows.append(rt.assign(task=task, dataset=ru.DS_NAME[ds]))
    if not rows:
        print("no selection records found")
        return
    tab = pd.concat(rows, ignore_index=True)
    ru.save_table(tab.set_index(["task", "dataset", "method"]),
                  "results/tables/stability_runtime.csv")

    for task in tab.task.unique():
        t = tab[tab.task == task]
        names = list(dict.fromkeys(t.dataset))
        methods = sorted(t.method.unique())
        cmap = plt.get_cmap("tab20")
        stab_col = next((c for c in t.columns if c.startswith("nogueira (outer")), None)
        ncols = 2 if stab_col else 1
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 0.45 * len(names) * 4 + 2),
                                 sharey=True, squeeze=False)
        for i, m in enumerate(methods):
            r = t[t.method == m]
            y = np.array([names.index(n) for n in r.dataset]) + (i - len(methods) / 2) * 0.05
            axes[0, 0].errorbar(r.cpu_s_median, y, xerr=[r.cpu_s_median - r.cpu_s_q25,
                                r.cpu_s_q75 - r.cpu_s_median], fmt="o", color=cmap(i % 20),
                                label=m, capsize=2, ms=5)
            if stab_col:
                lo = r[stab_col.replace("nogueira", "ci_low")]
                hi = r[stab_col.replace("nogueira", "ci_high")]
                axes[0, 1].errorbar(r[stab_col], y, xerr=[r[stab_col] - lo, hi - r[stab_col]],
                                    fmt="o", color=cmap(i % 20), capsize=2, ms=5)
        axes[0, 0].set_xscale("log")
        axes[0, 0].set_xlabel("CPU seconds per selection (median, IQR)")
        axes[0, 0].set_yticks(range(len(names)), names)
        axes[0, 0].invert_yaxis()
        if stab_col:
            axes[0, 1].set_xlabel(f"Nogueira stability, {stab_col[10:-1]} (95% CI)")
            axes[0, 1].set_xlim(-0.1, 1.05)
        for ax in axes[0]:
            ax.grid(alpha=0.3, axis="x")
        axes[0, 0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.08), ncol=4)
        fig.tight_layout()
        fig.savefig(f"plots/stability_runtime_{task}.pdf")
        plt.close(fig)
        print(f"saved plots/stability_runtime_{task}.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reg-metric", choices=["R2", "MAE"], default="R2")
    ap.add_argument("--evaluators", nargs="*", default=["LinearRegression", "CoxPH"])
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()
    by_task = {"reg": ru.REGRESSION_DATASETS, "surv": ru.SURVIVAL_DATASETS}
    if args.datasets:
        by_task = {t: [d for d in ds if d in args.datasets] for t, ds in by_task.items()}
    for task, datasets in by_task.items():
        for ev in ALL_EVALUATORS[task]:
            if datasets and ev in args.evaluators:
                plot_ablation(task, ev, datasets, args.reg_metric if task == "reg" else "C-index")
    stability_runtime(by_task)


if __name__ == "__main__":
    main()
