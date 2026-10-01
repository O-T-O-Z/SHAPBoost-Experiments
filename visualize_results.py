"""Main comparison: SHAPBoost vs. the strongest baselines on every dataset.

Per dataset, the figure shows SHAPBoost, SHAPBoost + collinearity check, and two
baselines picked among the non-SHAPBoost methods:
  * best performance: best mean score at the selected subset
    (ties -> smaller median subset size);
  * least features: smallest median subset size (ties -> better mean score).
The table lists every method (baselines, SHAPBoost variants and ablations).

Adapted to the revised protocol (review point 2): reads results/evaluation/*.csv
from run_evaluation.py; no folds are trimmed or dropped (see report_utils). The
original tie-breaking picked the LARGEST subset among equally good baselines;
the intended rule (smaller subset) is used here.

Outputs
  plots/Figure_<i>.pdf
  results/tables/<task>_<evaluator>_<metric>.csv        mean ± sd (median k [IQR])
  results/tables/<task>_<evaluator>_<metric>_long.csv   all statistics, numeric

Usage: python visualize_results.py [--reg-metric R2|MAE] [--evaluators ...] [--datasets ...]
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import report_utils as ru  # noqa: E402

COLORS = {
    "SHAPBoost": "#1b1b3a",
    "SHAPBoost-C": "#3a7a6a",
    "best": "#e3735e",
    "fewest": "#a3395f",
}
PLAN = [("reg", "LinearRegression"), ("surv", "CoxPH")]  # Figure 2 and 3 of the paper
ALL_EVALUATORS = {"reg": ["LinearRegression", "GradientBoosting"],
                  "surv": ["CoxPH", "RSF", "XGBoost-Cox"]}


def pick_baselines(stats: pd.DataFrame, metric: str):
    """Return (best-performing, fewest-features) baseline method names."""
    base = stats[~stats.method.map(ru.is_proposed)].dropna(subset=["mean"])
    if base.empty:
        return None, None
    sign = -1 if ru.higher_is_better(metric) else 1
    base = base.assign(_score=sign * base["mean"])
    best = base.sort_values(["_score", "k_median"]).iloc[0].method
    fewest = base.sort_values(["k_median", "_score"]).iloc[0].method
    return best, fewest


def plot_comparison(task, evaluator, datasets, metric, save_path):
    main = ru.MAIN[task]
    fig, axes, legend_ax = ru.make_grid(len(datasets))
    rows = []
    for ax, ds in zip(axes, datasets):
        res = ru.load_results(ds, evaluator, task)
        if res is None:
            ax.set_title(f"{ru.DS_NAME[ds]} (not run yet)")
            continue
        per_fold, curves = res
        stats = ru.method_stats(per_fold, curves, metric)
        best, fewest = pick_baselines(stats, metric)

        lines = [(main, main, COLORS["SHAPBoost"], "-"),
                 ("SHAPBoost-C", "SHAPBoost-C", COLORS["SHAPBoost-C"], "--")]
        if best is not None and best == fewest:
            lines.append((f"{best} (best & least)", best, COLORS["best"], "-."))
        elif best is not None:
            lines += [(best, best, COLORS["best"], "-"), (fewest, fewest, COLORS["fewest"], "-")]
        lines = [ln for ln in lines if ln[1] in set(stats.method)]

        n, p = ru.dataset_shape(ds, task)
        title = f"{ru.DS_NAME[ds]}, $\\mathbf{{p={p}}}$, $\\mathbf{{n={n}}}$"
        ru.draw_panel(ax, curves, stats, lines, metric, title, guide=main)
        # per-panel legend names the actual baselines, as in the original figure
        panel = [Line2D([0], [0], color=c, lw=2.5, ls=ls) for lbl, m, c, ls in lines
                 if not ru.is_proposed(m)]
        ax.legend(panel, [lbl for lbl, m, _, _ in lines if not ru.is_proposed(m)],
                  loc="best", fontsize=10)
        rows.append(stats.assign(dataset=ru.DS_NAME[ds], best_baseline=best,
                                 fewest_baseline=fewest))

    legend_ax.legend(
        [Line2D([0], [0], color=COLORS[k], lw=2.5, ls="--" if k == "SHAPBoost-C" else "-")
         for k in COLORS],
        ["SHAPBoost", "SHAPBoost + collinearity check", "Best performance (SOTA)",
         "Least features selected (SOTA)"],
        loc="center", fontsize=14, handlelength=2.5)
    fig.text(0.01, 0.003, ru.CAPTION, fontsize=8)
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)

    if rows:
        long = pd.concat(rows, ignore_index=True)
        long["cell"] = long.apply(ru.format_cell, axis=1, metric=metric)
        wide = long.pivot(index="dataset", columns="method", values="cell")
        order = [ru.DS_NAME[d] for d in datasets]
        wide = wide.reindex([o for o in order if o in wide.index])
        first = [c for c in (main, "SHAPBoost-C") if c in wide.columns]
        proposed = sorted(c for c in wide.columns if ru.is_proposed(c) and c not in first)
        baselines = sorted(c for c in wide.columns if not ru.is_proposed(c))
        wide = wide[first + proposed + baselines]
        base = f"results/tables/{task}_{evaluator}_{metric}"
        ru.save_table(wide, f"{base}.csv")
        ru.save_table(long.drop(columns="cell").set_index(["dataset", "method"]),
                      f"{base}_long.csv")
    print(f"saved {save_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reg-metric", choices=["R2", "MAE"], default="R2",
                    help="regression metric (the original figures used R2)")
    ap.add_argument("--evaluators", nargs="*", default=None,
                    help="default: LinearRegression (reg) and CoxPH (surv), as in the paper; "
                         "'all' for every evaluator")
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()

    plan = PLAN
    if args.evaluators == ["all"]:
        plan = [(t, e) for t in ("reg", "surv") for e in ALL_EVALUATORS[t]]
    elif args.evaluators:
        plan = [(t, e) for t in ("reg", "surv") for e in ALL_EVALUATORS[t]
                if e in args.evaluators]
    for i, (task, evaluator) in enumerate(plan):
        datasets = ru.REGRESSION_DATASETS if task == "reg" else ru.SURVIVAL_DATASETS
        if args.datasets:
            datasets = [d for d in datasets if d in args.datasets]
        if not datasets:
            continue
        metric = args.reg_metric if task == "reg" else "C-index"
        name = f"Figure_{i + 2}" if plan is PLAN else f"Figure_comparison_{evaluator}"
        plot_comparison(task, evaluator, datasets, metric, f"plots/{name}.pdf")


if __name__ == "__main__":
    main()
