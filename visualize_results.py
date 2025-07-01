import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection

from dataloading import load_regression_dataset, load_survival_dataset

pd.set_option("display.max_columns", None)


def trim_results(feature_selectors: dict, metric: str) -> dict:
    """
    Remove the results that have shorter sets than the mode and calculate the mean of the metric.

    :param feature_selectors: feature selectors dictionary.
    :param metric: metric to calculate the mean.
    :return: feature selectors dictionary with the mean of the metric.
    """
    for selector, results in feature_selectors.items():
        if not results:
            feature_selectors[selector] = []
            continue
        lengths = [len(lst) for lst in results[f"{metric}_per_fold"]]
        if lengths == []:
            feature_selectors[selector] = []
            continue
        mode = max(set(lengths), key=lengths.count)
        for inner_key, lst_of_lsts in results.items():
            if not isinstance(lst_of_lsts[0], list):
                continue
            equal_or_greater_than_mode = [
                lst for lst in lst_of_lsts if len(lst) >= mode
            ]
            results[inner_key] = [lst[:mode] for lst in equal_or_greater_than_mode]
        feature_selectors[selector][f"mean_{metric}"] = list(
            np.mean(results[f"{metric}_per_fold"], axis=0)
        )

    return feature_selectors


def reformat_dict(cell: dict) -> str:
    """
    Reformat the dictionary to a string with rate ± std.

    :param cell: dictionary with rate and std.
    :return: string with rate ± std.
    """
    if isinstance(cell, dict) and "rate" in cell and "std" in cell:
        rounded_rate = round(cell["rate"] * 1000, 4)
        rounded_sd = round(cell["std"] * 1000, 4)
        return f"{rounded_rate:.2f} ± {rounded_sd:.2f}"
    return cell


def load_dataset_regression(dataset: str, clf: str) -> tuple:
    """
    Load the dataset and the results of the feature selection algorithms.

    :param dataset: dataset name.
    :param clf: classifier name.
    :return: dataset and feature selectors.
    """
    X, y = load_regression_dataset(dataset)
    with open(f"regression_features/{dataset}_{clf}.json") as f:
        data = json.load(f)
    return X, trim_results(data, "r2")


def load_dataset_survival(dataset: str, clf: str) -> tuple:
    """
    Load the dataset and the results of the feature selection algorithms.

    :param dataset: dataset name.
    :param clf: classifier name.
    :return: dataset and feature selectors.
    """
    X, y = load_survival_dataset(dataset)
    with open(f"survival_features/{dataset}_{clf}.json") as f:
        data = json.load(f)
    return X, trim_results(data, "cindex")


def format_results_dataframe(
    results_df: pd.DataFrame, metric: str, datasets: list, ds_name_sub: dict
) -> pd.DataFrame:
    """
    Format the results dataframe with performance values and feature counts.

    :param results_df: DataFrame containing performance results
    :param metric: Metric used for evaluation ('cindex' or 'r2')
    :param datasets: List of dataset names
    :param ds_name_sub: Dictionary mapping dataset names to display names
    :return: Formatted DataFrame
    """
    # Add combined performance and feature count column
    results_df["Value"] = (
        results_df["Performance"].round(2).astype(str)
        + " ("
        + results_df["# Features"].astype(str)
        + ")"
        if metric == "cindex" or metric == "r2"
        else results_df["Performance"].round(1).astype(str)
        + " ("
        + results_df["# Features"].astype(str)
        + ")"
    )

    # Pivot and format DataFrame
    results_df = results_df.pivot(
        index="Dataset", columns="Feature Selector", values="Value"
    ).reset_index()
    results_df = results_df.set_index("Dataset")

    # Sort by datasets
    datasets.pop(1)
    results_df = results_df.reindex([ds_name_sub[ds] for ds in datasets])

    # Reorder SHAPBoost columns to front
    shapboost_cols = [
        "SHAPBoost (LR)",
        "SHAPBoost (CoxPH)",
        "SHAPBoost-C",
    ]
    for col in shapboost_cols:
        if col in results_df.columns:
            results_df = pd.concat(
                [
                    results_df[[col]],
                    results_df.drop(columns=[col]),
                ],
                axis=1,
            )

    # Remove empty SHAPBoost variant columns
    results_df = results_df.dropna(axis=1, how="all")

    return results_df


def find_smallest_and_best_performers(feature_selectors: dict, metric: str) -> list:
    """
    Find the smallest and best performing feature selectors. SHAPBoost is always included.

    If multiple feature selectors have the same performance, the one with the smallest number of features is chosen.
    If multiple feature selectors have the same number of features, the one with the best performance is chosen.

    :param feature_selectors: feature selectors dictionary.
    :return: list of feature selectors.
    :param metric: metric to use for sorting.
    """
    # exclude SHAPBoost variants
    feature_selectors = {
        k: v for k, v in feature_selectors.items() if not k.startswith("SHAPBoost")
    }
    # Create list of tuples with (selector, (size, performance))
    selector_stats = []
    for selector, stats in feature_selectors.items():
        if stats == []:
            continue
        size = len(stats[f"mean_{metric}"])  # Get length of mean_metric list
        performance = stats[f"mean_{metric}"][-1]  # Get last mean metric value
        selector_stats.append((selector, (size, performance)))

    # Sort by size and performance separately
    size_sorted = sorted(selector_stats, key=lambda x: x[1][0])
    perf_sorted = sorted(
        selector_stats,
        key=lambda x: x[1][1],
        reverse=metric == "cindex" or metric == "r2",
    )
    # Get smallest size selector, if multiple have same size pick best performing among those
    smallest_size = size_sorted[0][1][0]
    smallest_size_selectors = [s for s in size_sorted if s[1][0] == smallest_size]
    size_smallest = (
        min(smallest_size_selectors, key=lambda x: x[1][1])
        if metric != "cindex" and metric != "r2"
        else max(smallest_size_selectors, key=lambda x: x[1][1])
    )

    # Get best performing selector, if multiple have same performance pick smallest size among those
    best_performance = perf_sorted[0][1][1]
    best_performance_selectors = [s for s in perf_sorted if s[1][1] == best_performance]
    performance_best = (
        min(best_performance_selectors, key=lambda x: x[1][0])
        if metric != "cindex" and metric != "r2"
        else max(best_performance_selectors, key=lambda x: x[1][0])
    )

    return size_smallest[0], performance_best[0]


def plot_multiplot(
    datasets: list[str],
    type_: str,
    clf: str,
    save_path: str,
    metric: str,
    yticks: dict = None,
) -> None:
    """
    Plot the results of the feature selection algorithms for multiple datasets.

    :param datasets: list of dataset names.
    :param type_: type of the dataset (regression or survival).
    :param clf: classifier name.
    :param save_path: path to save the plot.
    :param metric: metric to plot.
    :param yticks: yticks for the plot.
    """
    datasets.insert(1, "empty")
    if yticks is None:
        yticks = {}
    ds_name_sub = {
        "metabric_full": "METABRIC",
        "metabric_regression": "METABRIC",
        "eyedata": "Eye Data",
        "crime": "Crime",
        "msd": "MSD",
        "parkinsons": "Parkinson's",
        "diabetes": "Diabetes",
        "housing": "California Housing",
        "breast_cancer": "Breast Cancer",
        "nhanes": "NHANES",
        "support": "SUPPORT",
        "nacd": "NACD",
        "aids": "AIDS",
        "whas500": "WHAS",
    }
    colors = {
        "Boruta": sns.color_palette("CMRmap", n_colors=7)[0],
        "SHAPBoost-C": sns.color_palette("cubehelix", n_colors=6)[2],
        "SHAPBoost (LR)": sns.color_palette("cubehelix", n_colors=6)[0],
        "SHAPBoost (GBR)": sns.color_palette("cubehelix", n_colors=6)[0],
        "SHAPBoost (CoxPH)": sns.color_palette("cubehelix", n_colors=6)[1],
        "SHAPBoost (RSF)": sns.color_palette("cubehelix", n_colors=6)[1],
        "Best performance (SOTA)": sns.color_palette("flare", n_colors=6)[0],
        "Least features selected (SOTA)": sns.color_palette("flare", n_colors=6)[2],
    }
    sns.set_style("whitegrid")
    rows = ((len(datasets) - 1) // 2) + 1
    size = (16, 18) if len(datasets) > 4 else (18, 12)
    fig, axes = plt.subplots(rows, 2, figsize=size)
    dataset_loader = (
        load_dataset_regression if type_ == "regression" else load_dataset_survival
    )
    axes = axes.flatten()
    sns.despine()
    get_yticks = yticks == {}

    if get_yticks:
        yticks = [None] * len(datasets)
    results_df = pd.DataFrame(
        columns=["Dataset", "Feature Selector", "Performance", "# Features"]
    )

    for i, ds in enumerate(datasets):
        if ds == "empty":
            continue
        X, feature_selectors = dataset_loader(ds, clf)
        (
            smallest_feature_selector,
            best_feature_selector,
        ) = find_smallest_and_best_performers(feature_selectors, metric)
        all_feature_selectors = feature_selectors.copy()
        feature_selectors = {
            k: v
            for k, v in feature_selectors.items()
            if k
            in [
                "SHAPBoost-C",
                "SHAPBoost (LR)",
                "SHAPBoost (CoxPH)",
                smallest_feature_selector,
                best_feature_selector,
            ]
            and v != []
        }

        for feat_selector, res in feature_selectors.items():
            selector = (
                "Best performance (SOTA)"
                if feat_selector == best_feature_selector
                else (
                    "Least features selected (SOTA)"
                    if feat_selector == smallest_feature_selector
                    else feat_selector
                )
            )

            x_values = np.array(range(1, len(res[f"mean_{metric}"]) + 1))
            y_values = np.array(res[f"mean_{metric}"])
            points = np.array([x_values, y_values]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Create alternating colors

            color = colors[selector]
            if feat_selector == best_feature_selector == smallest_feature_selector:
                color1 = colors["Best performance (SOTA)"]
                color2 = colors["Least features selected (SOTA)"]
                segment_colors = []
                for seg_idx in range(len(segments)):
                    segment_colors.append(color1 if seg_idx % 2 == 0 else color2)

                lc = LineCollection(
                    segments, colors=segment_colors, linewidth=2.5, alpha=1
                )
                axes[i].add_collection(lc)

                # Add to legend
                axes[i].plot([], [], color=color1, label=selector, linewidth=2.5)
                axes[i].plot(
                    [], [], color=color2, label=f"{selector} (alt)", linewidth=2.5
                )
            else:
                axes[i].plot(
                    range(1, len(res[f"mean_{metric}"]) + 1),
                    res[f"mean_{metric}"],
                    label=selector,
                    linewidth=2.5,
                    color=color,
                    alpha=1,
                )
            # plot a dot at the end of the line
            axes[i].plot(
                len(res[f"mean_{metric}"]),
                res[f"mean_{metric}"][-1],
                "o",
                color=color,
                zorder=100 if feat_selector == smallest_feature_selector else 0,
            )
        # Create legend with original feature selector names
        handles = []
        labels = []
        for label, selector in [
            ("Best performance (SOTA)", best_feature_selector),
            ("Least features selected (SOTA)", smallest_feature_selector),
        ]:
            line = plt.Line2D([0], [0], color=colors[label], linewidth=2.5)
            handles.append(line)
            labels.append(selector)
        axes[i].legend(
            handles, labels, loc="upper right", fontsize=12, fancybox=True, shadow=False
        )
        if get_yticks:
            yticks[i] = axes[i].get_yticks()
        old_ticks = axes[i].get_yticks()
        if max(old_ticks) > max(yticks[i]) or min(old_ticks) < min(yticks[i]):
            yticks[i] = old_ticks
        axes[i].set_yticks(yticks[i])

        n, p = X.shape
        label = "SHAPBoost (LR)" if metric == "r2" else "SHAPBoost (CoxPH)"
        axes[i].axvline(
            x=len(feature_selectors[label][f"mean_{metric}"]),
            color=colors[label],
            linewidth=2.5,
            linestyle="--",
            alpha=0.25,
            zorder=0,
        )
        axes[i].axhline(
            y=feature_selectors[label][f"mean_{metric}"][-1],
            color=colors[label],
            linewidth=2.5,
            linestyle="--",
            alpha=0.25,
            zorder=0,
        )

        # make the title bold
        axes[i].set_title(
            f"{ds_name_sub[ds]}, $\\mathbf{{p={p}}}$, $\\mathbf{{n={n}}}$",
            fontdict={"weight": "bold"},
        )

        if metric == "r2":
            label = "R$^2$"
            axes[i].set_yticks(np.arange(0, 1.1, 0.1))
            axes[i].set_ylim(0, 1)
        elif metric == "cindex":
            label = "C-Index"
            axes[i].set_yticks(np.arange(0.45, 1.1, 0.1))
            axes[i].set_ylim(0.45, 0.9)
        axes[i].set_ylabel(label, fontsize=12)
        axes[i].set_xlabel("Number of Features", fontsize=12)
        axes[i].tick_params(axis="y", labelcolor="black", labelsize=12)
        axes[i].tick_params(axis="x", labelcolor="black", labelsize=12)
        lines, labels = axes[i].get_legend_handles_labels()
        # reorder legend such that SHAPBoost is first
        for lab in [la for la in labels if la.startswith("SHAPBoost")]:
            idx = labels.index(lab)
            if lab == "SHAPBoost-C":
                new_lab = "SHAPBoost + Collinearity check"
                labels.pop(idx)
            elif lab.startswith("SHAPBoost"):
                new_lab = "SHAPBoost"
                labels.pop(idx)
            lines.insert(0, lines.pop(idx))
            labels.insert(0, new_lab)

        max_lim = 100 if p > 100 else p
        axes[i].set_xlim(1, max_lim + 1)
        axes[i].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        # Make axis lines thicker to match plot lines
        for spine in axes[i].spines.values():
            spine.set_linewidth(2.5)
        # remove the grid
        axes[i].grid(False)
        for feat_selector, res in all_feature_selectors.items():
            if res == []:
                continue
            results_df = pd.concat(
                [
                    results_df,
                    pd.DataFrame(
                        {
                            "Dataset": [ds_name_sub[ds]],
                            "Feature Selector": [feat_selector],
                            "Performance": res[f"mean_{metric}"][-1],
                            "# Features": len(res[f"mean_{metric}"]),
                        }
                    ),
                ],
                ignore_index=True,
            )
    results_df = format_results_dataframe(results_df, metric, datasets, ds_name_sub)
    results_df.to_csv(f"results/{type_}_{clf}_{metric}.csv", index=True, sep=";")

    # remove empty plot
    axes[1].axis("off")
    axes[1].legend(
        lines,
        labels,
        fancybox=True,
        shadow=False,
        ncol=1,
        fontsize=14,
        prop={"size": 14},
        loc="center",
        handleheight=1.5,
        handlelength=2,
        markerscale=2,
    )
    plt.tight_layout()
    plt.savefig(save_path)
    return yticks


if __name__ == "__main__":
    survival_datasets = [
        "metabric_full",
        "breast_cancer",
        "nhanes",
        "support",
        "nacd",
        "aids",
        "whas500",
    ]
    regression_datasets = [
        "metabric_regression",
        "eyedata",
        "crime",
        "msd",
        "parkinsons",
        "diabetes",
        "housing",
    ]
    datasets = [regression_datasets, survival_datasets]

    for i, to_plot in enumerate(
        [
            ("regression", "LinearRegression"),
            # ("regression", "GradientBoostingRegressor"), # noqa
            ("survival", "CoxPHFitter"),
            # ("survival", "RandomSurvivalForest"), # noqa
            # ("survival", "XGBSurvivalRegressor"), # noqa
        ]
    ):
        type_, clf = to_plot
        datasets = survival_datasets if type_ == "survival" else regression_datasets
        metric = "r2" if type_ == "regression" else "cindex"
        if clf == "GradientBoostingRegressor":
            plot_multiplot(
                datasets, type_, clf, f"plots/Figure_{i+2}.pdf", metric, yticks  # noqa
            )
            yticks = {}
        else:
            yticks = plot_multiplot(
                datasets, type_, clf, f"plots/Figure_{i+2}.pdf", metric
            )
