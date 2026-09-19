"""Shared helpers for plot_curves.py, evaluation_experiment1.py and visualize_results.py.

Reads the outputs of run_evaluation.py:
  results/evaluation/<dataset>_per_fold.csv  one row per (method, evaluator, split)
  results/evaluation/<dataset>_curves.csv    per-fold score on the first 1..k features

Nothing is trimmed. For every fold, the curve is carried forward after the fold's
last evaluated k (the subset that method would use under a budget of k features),
and k = 0 is the no-feature baseline. Every method is therefore summarized over
all of its splits at every k, and the "operating point" of a fold is its score
on the subset it actually selected.
"""
import os
import warnings

import numpy as np
import pandas as pd

EVAL_DIR = "results/evaluation"

REGRESSION_DATASETS = ["metabric_regression", "eyedata", "crime", "msd", "parkinsons",
                       "diabetes", "housing"]
SURVIVAL_DATASETS = ["metabric_full", "breast_cancer", "nhanes", "support", "nacd",
                     "aids", "whas500"]
DS_NAME = {
    "metabric_full": "METABRIC", "metabric_regression": "METABRIC", "eyedata": "Eye Data",
    "crime": "Crime", "msd": "MSD", "parkinsons": "Parkinson's", "diabetes": "Diabetes",
    "housing": "California Housing", "breast_cancer": "Breast Cancer", "nhanes": "NHANES",
    "support": "SUPPORT", "nacd": "NACD", "aids": "AIDS", "whas500": "WHAS",
}
METRIC_LABEL = {"R2": "R$^2$", "MAE": "MAE", "C-index": "C-index"}
DEFAULT_YLIM = {"R2": (0.0, 1.0), "C-index": (0.45, 0.9)}  # as in the original figures

# Point 3/4 method names -> the names used in the paper. Point-2 names already match.
_DISPLAY = {
    "reg": {"SHAPBoost": "SHAPBoost (LR)", "SHAPBoost (GBR/RSF)": "SHAPBoost (GBR)"},
    "surv": {"SHAPBoost": "SHAPBoost (CoxPH)", "SHAPBoost (GBR/RSF)": "SHAPBoost (RSF)"},
}
MAIN = {"reg": "SHAPBoost (LR)", "surv": "SHAPBoost (CoxPH)"}
TREE = {"reg": "SHAPBoost (GBR)", "surv": "SHAPBoost (RSF)"}


def higher_is_better(metric: str) -> bool:
    return metric != "MAE"


def is_proposed(method: str) -> bool:
    """SHAPBoost variants and the ablation variants (not baselines)."""
    return method.startswith(("SHAPBoost", "GainBoost"))


def load_results(dataset: str, evaluator: str, task: str):
    """Return (per_fold, curves) for one evaluator with paper method names, or None."""
    paths = [f"{EVAL_DIR}/{dataset}_per_fold.csv", f"{EVAL_DIR}/{dataset}_curves.csv"]
    if not all(os.path.exists(p) for p in paths):
        warnings.warn(f"{dataset}: no evaluation results (run run_evaluation.py first)")
        return None
    per_fold, curves = (pd.read_csv(p) for p in paths)
    per_fold = per_fold[per_fold.evaluator == evaluator].copy()
    curves = curves[curves.evaluator == evaluator].copy()
    if per_fold.empty:
        warnings.warn(f"{dataset}: no results for evaluator {evaluator}")
        return None
    for df in (per_fold, curves):
        df["method"] = df["method"].replace(_DISPLAY[task])
    n_splits = per_fold.groupby("method").split_id.nunique()
    if n_splits.nunique() > 1:
        warnings.warn(f"{dataset}/{evaluator}: methods have different numbers of splits:\n"
                      f"{n_splits.to_string()}\nRun the missing selections before comparing.")
    return per_fold, curves


def budget_curves(curves: pd.DataFrame, metric: str, grid) -> pd.DataFrame:
    """One row per (method, split, k in grid), carrying each fold's last value forward."""
    grid = np.asarray(sorted(set(grid)))
    out = []
    for (method, split), g in curves.groupby(["method", "split_id"]):
        s = g.set_index("k")[metric].sort_index()
        s = s[~s.index.duplicated()]
        vals = s.reindex(sorted(set(s.index) | set(grid))).ffill().loc[grid]
        out.append(pd.DataFrame({"method": method, "split_id": split, "k": grid,
                                 metric: vals.values}))
    return pd.concat(out, ignore_index=True)


def aggregate_curves(curves: pd.DataFrame, metric: str, grid) -> pd.DataFrame:
    """Mean, SE and number of splits per (method, k); every split counts at every k."""
    bc = budget_curves(curves, metric, grid)
    return bc.groupby(["method", "k"])[metric].agg(["mean", "sem", "count"]).reset_index()


def operating_points(curves: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per (method, split): score of the subset the method actually selected.

    Taken from the curve at the fold's largest evaluated k, so an empty selection
    gets the k = 0 baseline instead of being dropped.
    """
    last = curves.sort_values("k").groupby(["method", "split_id"]).tail(1)
    return last[["method", "split_id", "n_selected", "k", metric]].rename(
        columns={metric: "score", "k": "k_evaluated"})


def method_stats(per_fold: pd.DataFrame, curves: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per method: score at the selected subset and subset-size spread."""
    op = operating_points(curves, metric)
    stats = op.groupby("method").agg(
        n_splits=("split_id", "nunique"), mean=("score", "mean"), sd=("score", "std"),
        se=("score", "sem"), k_median=("n_selected", "median"),
        k_q25=("n_selected", lambda s: s.quantile(0.25)),
        k_q75=("n_selected", lambda s: s.quantile(0.75)),
        k_min=("n_selected", "min"), k_max=("n_selected", "max"),
    )
    status = per_fold.groupby("method").status
    stats["n_failed"] = status.apply(lambda s: int((s != "ok").sum()))
    stats["n_empty"] = op.groupby("method").n_selected.apply(lambda s: int((s == 0).sum()))
    return stats.reset_index()


def format_cell(row, metric: str) -> str:
    """'mean ± sd (median k [q25–q75])'."""
    fmt = (lambda v: f"{v:.3g}") if metric == "MAE" else (lambda v: f"{v:.2f}")
    k = f"{row.k_median:g} [{row.k_q25:g}–{row.k_q75:g}]"
    return f"{fmt(row['mean'])} ± {fmt(row.sd)} ({k})"


def dataset_shape(dataset: str, task: str):
    from dataloading import load_regression_dataset, load_survival_dataset
    X, _ = (load_regression_dataset if task == "reg" else load_survival_dataset)(dataset)
    return X.shape  # (n, p)


# --------------------------------------------------------------------------- #
# plotting helpers (shared by evaluation_experiment1.py and visualize_results.py)
# --------------------------------------------------------------------------- #
def make_grid(n_panels: int):
    """Two-column grid with slot 1 reserved for the legend, as in the original figures."""
    import matplotlib.pyplot as plt

    rows = (n_panels + 2) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(16, 4.4 * rows))
    axes = list(axes.flatten())
    legend_ax = axes.pop(1)
    legend_ax.axis("off")
    for ax in axes[n_panels:]:
        ax.axis("off")
    return fig, axes[:n_panels], legend_ax


def draw_panel(ax, curves, stats, lines, metric, title, guide=None):
    """Budget curves (mean ± 1 SE over all splits) plus operating points.

    lines: list of (label, method, color, linestyle). The operating point of a
    method is its median selected size (bar: IQR) vs mean score at that subset.
    guide: method whose operating point gets dashed guide lines.
    """
    import matplotlib.pyplot as plt

    methods = [m for _, m, _, _ in lines]
    st = stats.set_index("method")
    kmax = int(max([st.loc[m, "k_max"] for m in methods if m in st.index] + [1]))
    cur = curves[curves.method.isin(methods)]
    grid = {0, kmax, *[int(k) for k in cur.k.unique() if k <= kmax]}
    agg = aggregate_curves(cur, metric, grid)

    ys = []
    for label, m, color, ls in lines:
        a = agg[agg.method == m]
        if a.empty:
            continue
        ax.plot(a.k, a["mean"], color=color, ls=ls, lw=2.2, label=label)
        ax.fill_between(a.k, a["mean"] - a["sem"], a["mean"] + a["sem"], color=color,
                        alpha=0.12, lw=0)
        r = st.loc[m]
        ax.errorbar(r.k_median, r["mean"], xerr=[[r.k_median - r.k_q25], [r.k_q75 - r.k_median]],
                    fmt="o", color=color, ms=7, mec="black", mew=0.7, capsize=3, zorder=10)
        ys.append(r["mean"])
    if guide is not None and guide in st.index:
        r = st.loc[guide]
        color = next(c for _, m, c, _ in lines if m == guide)
        ax.axvline(r.k_median, color=color, lw=2.5, ls="--", alpha=0.25, zorder=0)
        ax.axhline(r["mean"], color=color, lw=2.5, ls="--", alpha=0.25, zorder=0)

    if metric in DEFAULT_YLIM and ys:
        lo, hi = DEFAULT_YLIM[metric]
        lo, hi = min(lo, min(ys) - 0.03), max(hi, max(ys) + 0.03)
        ax.set_ylim(lo, hi)
        inner = agg[agg.k > 0]["mean"]  # the k = 0 baseline may sit just below R2 = 0
        if (inner < lo - 0.01).any() or (inner > hi + 0.01).any():
            ax.text(0.99, 0.01, "curve continues beyond axis", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color="grey")
    if kmax > 30:
        ax.set_xscale("symlog", linthresh=1, linscale=0.4)
        ticks = [t for t in (0, 1, 2, 5, 10, 20, 50, 100, 200, 500) if t <= kmax]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.set_xlim(-0.1, kmax * 1.08)
    else:
        ax.set_xlim(-0.3, kmax + 0.5)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_title(title, fontdict={"weight": "bold"})
    ax.set_ylabel(METRIC_LABEL.get(metric, metric), fontsize=12)
    ax.set_xlabel("Number of features (budget)", fontsize=12)
    ax.tick_params(labelsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(2)


CAPTION = ("Lines: mean ± 1 SE over all outer splits; a fold that stopped at fewer features "
           "keeps its last value (budget curve), k = 0 is the no-feature baseline. "
           "Dots: median selected size (bar: IQR) vs mean score of the selected subsets.")


def save_table(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep=";")
