import argparse
import concurrent.futures
import json
import threading

import numpy as np
import pandas as pd
from lifelines.exceptions import ConvergenceError
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from tqdm import tqdm
from xgboost import XGBRegressor

from dataloading import load_regression_dataset
from test_utils import (
    train_backward_selection,
    train_boruta,
    train_forward_selection,
    train_mrmr,
    train_pvalue,
    train_relief,
    train_shapboost,
    train_shapboost_c,
    train_xgb,
)

FILE_NAME = "_features_selected"


def worker(args: list) -> list:
    """
    Worker function to run the feature selection algorithms.

    :param args: list of arguments.
    :return: list of results.
    """
    fold_idx, train_index, selector, func, xgb_params, X_train, y_train = args
    if func in [train_shapboost, train_shapboost_c]:
        if selector == "SHAPBoost (GBR)":
            eval_model = GradientBoostingRegressor(
                learning_rate=0.01,
                max_depth=4,
                n_iter_no_change=10,
            )
        else:
            eval_model = LinearRegression()
        features, importances = func(
            X_train[train_index],
            y_train[train_index],
            [XGBRegressor(**xgb_params), eval_model],
        )
    elif func == train_mrmr:
        features, importances = func(
            pd.DataFrame(X_train[train_index]),
            pd.DataFrame(y_train[train_index]),
            XGBRegressor(**xgb_params),
        )
    elif func in [train_forward_selection, train_backward_selection]:
        features, importances = func(
            pd.DataFrame(X_train[train_index]),
            pd.DataFrame(y_train[train_index]),
            "reg",
        )
    else:
        features, importances = func(
            X_train[train_index],
            y_train[train_index],
            XGBRegressor(**xgb_params),
        )
    if np.unique(importances).shape[0] != 1:
        # use the mean value of the scores as a lower threshold.
        most_important_ = np.where(importances >= np.mean(importances))[0]
        features = features[: len(most_important_)]
        # get importance of those features
        importances = importances[: len(most_important_)]

    # limit the number of features to 100
    if len(features) > 100:
        features = features[:100]
        importances = importances[:100]

    importances = [float(i) for i in list(importances)]
    features = [int(f) for f in list(features)]
    return (fold_idx, selector, func, features, importances)


def run_cross_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    dataset_name: str,
    feature_selectors: dict,
    t_rand: int,
    e_rand: int,
) -> None:
    """
    Run cross validation for the feature selection algorithms on specific seeds.

    :param X_train: The training features.
    :param y_train: The training labels.
    :param dataset_name: The name of the dataset.
    :param feature_selectors: The dictionary containing the results of the feature selection algorithms.
    :param t_rand: train seed.
    """
    xgb_params = {
        "n_estimators": 100,
        "max_depth": 20,
        "n_jobs": -1,
        "random_state": t_rand,
    }
    folds = KFold(n_splits=10, shuffle=True, random_state=e_rand)
    funcs = [
        ("RReliefF", train_relief),
        ("Boruta", train_boruta),
        ("XGBoost", train_xgb),
        ("MRMR", train_mrmr),
        ("P-value", train_pvalue),
        ("Forward", train_forward_selection),
        ("Backward", train_backward_selection),
        ("SHAPBoost (LR)", train_shapboost),
        ("SHAPBoost-C", train_shapboost_c),
        ("SHAPBoost (GBR)", train_shapboost),
    ]
    funcs = [func for func in funcs if func[0] in feature_selectors]
    # Precompute the fold splits with indices
    folds_split = list(
        enumerate(folds.split(X))
    )  # (fold_idx, (train_index, test_index))

    # Prepare all tasks: each task is a (fold_idx, train_index, selector, func)
    tasks = []
    for fold_idx, (train_index, _) in folds_split:
        for selector, func in funcs:
            tasks.append(
                (fold_idx, train_index, selector, func, xgb_params, X_train, y_train)
            )
    dict_lock = threading.Lock()
    # Process tasks in parallel with a ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(worker, task) for task in tasks]
        with tqdm(total=len(tasks)) as pbar:
            for future in concurrent.futures.as_completed(futures):
                print(future.result())
                try:
                    fold_idx, selector, func, features, _ = future.result()
                    with dict_lock:
                        feature_selectors[selector].append(features)
                        print(f"Updated {selector} to {features}", flush=True)
                    with open(
                        f"regression_features/{dataset_name}{FILE_NAME}.json", "w"
                    ) as f:
                        json.dump(feature_selectors, f)
                except ConvergenceError:
                    print("FAILED")
                pbar.update(1)


def run_experiment(
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    selection_methods: list,
) -> None:
    """
    Run the experiment for the given dataset.

    :param dataset_name: The name of the dataset.
    :param X: Input features.
    :param y: Target values.
    """
    print(f"Running experiment {dataset_name}")
    X_train = np.array(X)
    y_train = np.array(y)
    feature_selectors = {k: [] for k in selection_methods}
    for t_rand, e_rand in tqdm([(42, 84), (55, 110), (875, 1750)]):
        run_cross_validation(
            X_train,
            y_train,
            dataset_name,
            feature_selectors,
            t_rand,
            e_rand,
        )

    for selector, results in feature_selectors.items():
        if not results:
            feature_selectors[selector] = []
            continue
        lengths = [len(lst) for lst in results]
        mode = max(set(lengths), key=lengths.count)
        equal_or_greater_than_mode = [
            lst if len(lst) >= mode else [] for lst in results
        ]
        trimmed = [lst[:mode] for lst in equal_or_greater_than_mode]
        feature_selectors[selector] = trimmed

    with open(f"regression_features/{dataset_name}{FILE_NAME}.json", "w") as f:
        json.dump(feature_selectors, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
    )
    args = parser.parse_args()
    dataset = args.dataset
    X, y = load_regression_dataset(dataset)
    selection_methods = [
        "RReliefF",
        "Boruta",
        "XGBoost",
        "MRMR",
        "P-value",
        "Forward",
        "Backward",
        "SHAPBoost-C",
        "SHAPBoost (LR)",
        "SHAPBoost (GBR)",
    ]
    run_experiment(dataset, X.values, y.values, selection_methods)
