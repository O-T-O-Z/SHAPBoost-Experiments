"""Review point 3: all methods under comparable (inner-CV) stopping rules.

One panel per dataset. Every method is drawn at its operating point:
x = median number of selected features (bar: IQR over splits),
y = mean test score of the selected subsets (bar: ±1 SE over splits).
Methods on the Pareto front (no other method is both smaller and better) are
connected, so "similar performance with fewer features" can be read directly.

Marker shape shows the method family, so the new baselines requested by the
reviewer (lasso / elastic net / penalized Cox, C-index boosting with stability
selection) and the CV-stopped stepwise methods are easy to find.

Outputs
  plots/point3_operating_points_<task>_<evaluator>.pdf
  results/tables/point3_<task>_<evaluator>_<metric>.csv   (incl. pareto flag)

Usage: python plot_operating_points.py [--reg-metric R2|MAE] [--evaluators ...] [--datasets ...]
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import report_utils as ru  # noqa: E402

FAMILIES = [  # (label, marker, test on method name)
    ("SHAPBoost (proposed)", "*", lambda m: m.startswith("SHAPBoost")),
    ("Ablation variants", "P", lambda m: m.startswith("GainBoost")),
    ("Penalized (lasso / elastic net / Cox)", "s",
     lambda m: m.startswith(("Lasso", "ElasticNet", "Elastic net"))),
    ("Stepwise, inner-CV stopping", "^", lambda m: m.startswith(("Forward", "Backward"))),
    ("Boosting + stability selection [B]", "D", lambda m: m.startswith(("CIndexBoost", "C-index boosting"))),
    ("Rankers, size by inner CV", "o", lambda m: True),
]
ALL_EVALUATORS = {"reg": ["LinearRegression", "GradientBoosting"],
                  "surv": ["CoxPH", "RSF", "XGBoost-Cox"]}


def family(method):
    return next((lbl, mk) for lbl, mk, test in FAMILIES if test(method))


def pareto_front(stats: pd.DataFrame, metric: str) -> pd.Series:
    """True for methods not dominated in (smaller median k, better mean score)."""
    better = (lambda a, b: a > b) if ru.higher_is_better(metric) else (lambda a, b: a < b)
    flags = []
    for _, r in stats.iterrows():
        dominated = any(
            (o.k_median <= r.k_median and not better(r["mean"], o["mean"]))
            and (o.k_median < r.k_median or better(o["mean"], r["mean"]))
            for _, o in stats.iterrows() if o.method != r.method
        )
        flags.append(not dominated)
    return pd.Series(flags, index=stats.index)


def plot(task, evaluator, datasets, metric, out_pdf):
    # pass 1: statistics for every dataset, so colors are fixed across panels
    loaded = {}
    for ds in datasets:
        res = ru.load_results(ds, evaluator, task)
        if res is not None:
            per_fold, curves = res
            stats = ru.method_stats(per_fold, curves, metric).dropna(subset=["mean"])
            stats["pareto"] = pareto_front(stats, metric).values
            loaded[ds] = stats
    methods = sorted({m for st in loaded.values() for m in st.method})
    cmap = plt.get_cmap("tab20")
    color = {m: cmap(i % 20) for i, m in enumerate(methods)}

    # pass 2: panels
    fig, axes, legend_ax = ru.make_grid(len(datasets))
    rows = []
    for ax, ds in zip(axes, datasets):
        if ds not in loaded:
            ax.set_title(f"{ru.DS_NAME[ds]} (not run yet)")
            continue
        stats = loaded[ds]
        rows.append(stats.assign(dataset=ru.DS_NAME[ds]))
        for _, r in stats.iterrows():
            _, marker = family(r.method)
            ax.errorbar(max(r.k_median, 0.8), r["mean"], yerr=r.se,
                        xerr=[[r.k_median - r.k_q25], [r.k_q75 - r.k_median]],
                        fmt=marker, ms=13 if marker == "*" else 8, color=color[r.method],
                        mec="black", mew=0.6, capsize=2, elinewidth=1, alpha=0.9,
                        zorder=5 if r.pareto else 3)
        front = stats[stats.pareto].sort_values("k_median")
        ax.plot(front.k_median.clip(lower=0.8), front["mean"], color="grey", lw=1, ls=":",
                zorder=1)
        n, p = ru.dataset_shape(ds, task)
        ax.set_xscale("log")
        ax.set_title(f"{ru.DS_NAME[ds]}, $\\mathbf{{p={p}}}$, $\\mathbf{{n={n}}}$",
                     fontdict={"weight": "bold"})
        ax.set_xlabel("Median number of selected features (bar: IQR)")
        ax.set_ylabel(f"{ru.METRIC_LABEL[metric]} of selected subset (±1 SE)")
        ax.grid(alpha=0.3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    handles = [Line2D([0], [0], marker=family(m)[1], ls="", color=color[m], mec="black",
                      ms=10 if family(m)[1] == "*" else 7) for m in methods]
    handles += [Line2D([0], [0], color="grey", ls=":", lw=1)]
    legend_ax.legend(handles, methods + ["Pareto front (fewer features & better score)"],
                     loc="center", fontsize=10, ncol=2)
    fig.text(0.01, 0.003, "Markers: star SHAPBoost, plus ablation, square penalized, triangle "
             "stepwise with inner-CV stopping, diamond C-index boosting + stability selection, "
             "circle rankers (size chosen by inner CV).", fontsize=8)
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)
    if rows:
        tab = pd.concat(rows, ignore_index=True)
        tab["family"] = tab.method.map(lambda m: family(m)[0])
        ru.save_table(tab.set_index(["dataset", "method"]),
                      f"results/tables/point3_{task}_{evaluator}_{metric}.csv")
    print(f"saved {out_pdf}")


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
                metric = args.reg_metric if task == "reg" else "C-index"
                plot(task, ev, datasets, metric,
                     f"plots/point3_operating_points_{task}_{ev}.pdf")


if __name__ == "__main__":
    main()
