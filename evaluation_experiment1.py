import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import report_utils as ru  # noqa: E402

COLORS = {"linear": "#1b1b3a", "tree": "#c0527a"}  # dark / rose, as the cubehelix pair

PLAN = [  # (task, evaluator) in the order of the original figures
    ("reg", "LinearRegression"),
    ("reg", "GradientBoosting"),
    ("surv", "CoxPH"),
    ("surv", "RSF"),
    ("surv", "XGBoost-Cox"),
]


def plot_experiment1(task, evaluator, datasets, metric, save_path):
    main, tree = ru.MAIN[task], ru.TREE[task]
    lines = [(main, main, COLORS["linear"], "-"), (tree, tree, COLORS["tree"], "-")]
    fig, axes, legend_ax = ru.make_grid(len(datasets))
    rows = []
    for ax, ds in zip(axes, datasets):
        res = ru.load_results(ds, evaluator, task)
        if res is None:
            ax.set_title(f"{ru.DS_NAME[ds]} (not run yet)")
            continue
        per_fold, curves = res
        stats = ru.method_stats(per_fold, curves, metric)
        n, p = ru.dataset_shape(ds, task)
        title = f"{ru.DS_NAME[ds]}, $\\mathbf{{p={p}}}$, $\\mathbf{{n={n}}}$"
        ru.draw_panel(ax, curves, stats, lines, metric, title, guide=main)
        stats = stats[stats.method.isin([main, tree])]
        rows.append(stats.assign(dataset=ru.DS_NAME[ds]))

    legend_ax.legend([Line2D([0], [0], color=c, lw=2.5, ls=ls) for _, _, c, ls in lines],
                        [lbl for lbl, _, _, _ in lines], loc="center", fontsize=14,
                        handlelength=2.5)
    fig.text(0.01, 0.003, ru.CAPTION, fontsize=8)
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)

    if rows:
        long = pd.concat(rows, ignore_index=True)
        order = [ru.DS_NAME[d] for d in datasets]
        long["cell"] = long.apply(ru.format_cell, axis=1, metric=metric)
        wide = long.pivot(index="dataset", columns="method", values="cell")
        wide = wide.reindex([o for o in order if o in wide.index])
        wide = wide[[m for m in (main, tree) if m in wide.columns]]
        base = f"results/tables/exp1_{task}_{evaluator}_{metric}"
        ru.save_table(wide, f"{base}.csv")
        ru.save_table(long.drop(columns="cell").set_index(["dataset", "method"]),
                        f"{base}_long.csv")
    print(f"saved {save_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reg-metric", choices=["R2", "MAE"], default="R2",
                    help="regression metric (the original figures used R2)")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="restrict to these datasets (default: all 14)")
    args = ap.parse_args()
    for i, (task, evaluator) in enumerate(PLAN):
        datasets = ru.REGRESSION_DATASETS if task == "reg" else ru.SURVIVAL_DATASETS
        if args.datasets:
            datasets = [d for d in datasets if d in args.datasets]
        if not datasets:
            continue
        metric = args.reg_metric if task == "reg" else "C-index"
        plot_experiment1(task, evaluator, datasets, metric,
                            f"plots/Figure_{i + 2}_ex1_{evaluator}.pdf")


if __name__ == "__main__":
    main()
