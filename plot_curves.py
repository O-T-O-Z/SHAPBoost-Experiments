"""Performance vs. number of features, using every fold (no trimming).

Replaces the "trim to the modal subset size" approach. For each fold, the curve
holds the test performance on the first 1..k selected features. When a fold
stopped earlier (k_i < k), its last value is carried forward: with a budget of k
features that method would use its k_i-feature subset. k = 0 is the no-feature
baseline, so folds with an empty or failed selection still contribute. Every
fold therefore counts at every k, and each method is averaged over the same splits.

Top panel: mean +/- 1 SE across splits (the budget curve), plus each method's
operating point: median selected size (horizontal bar = IQR) vs mean test score
of the subsets actually selected. Bottom panel: distribution of selected sizes.

The curves use outer test folds and are descriptive only; subset sizes are
chosen by inner CV during selection, never from these curves.

Usage: python plot_curves.py -d eyedata [-e LinearRegression] [--kmax 50] [--methods ...]
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from report_utils import budget_curves  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--dataset", required=True)
    ap.add_argument("-e", "--evaluators", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--kmax", type=int, default=None, help="default: largest selected size")
    ap.add_argument("--xscale", choices=["linear", "symlog"], default=None)
    args = ap.parse_args()

    base = f"results/evaluation/{args.dataset}"
    cur = pd.read_csv(f"{base}_curves.csv")
    per_fold = pd.read_csv(f"{base}_per_fold.csv")
    metric = "MAE" if "MAE" in cur.columns else "C-index"
    if args.methods:
        cur = cur[cur.method.isin(args.methods)]
        per_fold = per_fold[per_fold.method.isin(args.methods)]
    evaluators = args.evaluators or sorted(cur.evaluator.unique())
    methods = sorted(cur.method.unique())
    palette = list(plt.cm.tab10.colors) + list(plt.cm.Dark2.colors) + list(plt.cm.Set1.colors)
    palette = [c for c in palette if c != plt.cm.tab10.colors[7]]  # drop grey
    colors = {m: palette[i % len(palette)] for i, m in enumerate(methods)}

    n_splits = per_fold.groupby("method").split_id.nunique()
    if n_splits.nunique() > 1:
        print("WARNING: methods were evaluated on different numbers of splits:\n"
              f"{n_splits.to_string()}\nRun the missing selections before comparing.")

    kmax = args.kmax or int(per_fold.n_selected.max())
    grid = np.array(sorted({0, *[k for k in cur.k.unique() if k <= kmax], kmax}))
    xscale = args.xscale or ("symlog" if kmax > 30 else "linear")

    fig, axes = plt.subplots(2, len(evaluators), figsize=(6.5 * len(evaluators), 8),
                             sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1.3]})
    tables = []
    for c, ev in enumerate(evaluators):
        ax, axk = axes[0, c], axes[1, c]
        bc = budget_curves(cur[cur.evaluator == ev], metric, grid)
        agg = bc.groupby(["method", "k"])[metric].agg(["mean", "sem", "count"]).reset_index()
        agg.insert(0, "evaluator", ev)
        tables.append(agg)
        pf = per_fold[per_fold.evaluator == ev]
        for m in methods:
            a = agg[agg.method == m]
            ax.plot(a.k, a["mean"], color=colors[m], lw=1.6, label=m)
            ax.fill_between(a.k, a["mean"] - a["sem"], a["mean"] + a["sem"],
                            color=colors[m], alpha=0.10, lw=0)
            sel = pf[pf.method == m]
            if len(sel):
                q25, q50, q75 = sel.n_selected.quantile([.25, .5, .75])
                ax.errorbar(q50, sel[metric].mean(), xerr=[[q50 - q25], [q75 - q50]],
                            fmt="o", color=colors[m], ms=6, mec="black", mew=0.6,
                            capsize=3, zorder=5)
        ax.set_title(f"{args.dataset} - {ev}")
        ax.set_ylabel(f"test {metric} ({'lower' if metric == 'MAE' else 'higher'} is better)")
        ax.grid(alpha=0.3)

        data = [pf[pf.method == m].n_selected.values for m in methods]
        bp = axk.boxplot(data, vert=False, patch_artist=True, widths=0.6,
                         flierprops={"ms": 3})
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(colors[m])
            patch.set_alpha(0.6)
        axk.set_yticks(range(1, len(methods) + 1), methods, fontsize=7)
        axk.set_xlabel("number of features (budget k)")
        if xscale == "symlog":
            axk.set_xscale("symlog", linthresh=1, linscale=0.4)
            ticks = [t for t in (0, 1, 2, 5, 10, 20, 50, 100, 200, 500) if t <= kmax]
            axk.set_xticks(ticks, [str(t) for t in ticks])
        axk.set_xlim(-0.2 if xscale == "symlog" else -0.5, kmax * 1.05)
        axk.grid(alpha=0.3, axis="x")
    axes[0, -1].legend(fontsize=7, loc="best", ncol=2)
    fig.text(0.01, 0.005, f"Mean ±1 SE over {int(n_splits.max())} outer splits; folds that "
             "stopped early carry their last value forward; dots = median selected size "
             "(bar: IQR) vs mean score of the selected subsets.", fontsize=7)
    fig.tight_layout(rect=(0, 0.02, 1, 1))

    os.makedirs("results/figures", exist_ok=True)
    fig.savefig(f"results/figures/{args.dataset}_budget_curves.png", dpi=200)
    fig.savefig(f"results/figures/{args.dataset}_budget_curves.pdf")
    pd.concat(tables).to_csv(f"{base}_budget_curves.csv", index=False)
    print(f"saved results/figures/{args.dataset}_budget_curves.png/.pdf "
          f"and {base}_budget_curves.csv")


if __name__ == "__main__":
    main()
