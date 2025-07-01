import argparse
import json

import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

from dataloading import load_survival_dataset
from test_utils import (
    RandomSurvivalForestWrapper,
    train_backward_selection,
    train_forward_selection,
    train_pvalue_survival,
    train_shapboost,
    train_shapboost_c,
    train_xgb_survival,
)
from xgb_survival_regressor import XGBSurvivalRegressor

XGB_PARAMS = {
    "objective": "survival:aft",
    "eval_metric": "aft-nloglik",
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_child_weight": 50,
    "subsample": 1.0,
    "colsample_bynode": 1.0,
    "aft_loss_distribution": "normal",
    "aft_loss_distribution_scale": 1,
    "tree_method": "hist",
    "booster": "gbtree",
    "grow_policy": "lossguide",
    "lambda": 0.01,
    "alpha": 0.02,
    "n_jobs": -1,
}
FILE_NAME = "_features_selected"


def run_cross_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    dataset_name: str,
    feature_selectors: dict,
    params_train: dict,
    e_rand: int,
) -> None:
    """
    Run cross validation for the feature selection algorithms on specific seeds.

    :param X_train: The training features.
    :param y_train: The training labels.
    :param dataset_name: The name of the dataset.
    :param feature_selectors: The dictionary containing the results of the feature selection algorithms.
    :param params_train: train parameters.
    """
    censored = np.array([0 if i[0] == i[1] else 1 for i in y_train])
    folds = KFold(n_splits=10, shuffle=True, random_state=e_rand)
    funcs = [
        ("XGBoost", train_xgb_survival),
        ("P-value", train_pvalue_survival),
        ("Forward", train_forward_selection),
        ("Backward", train_backward_selection),
        ("SHAPBoost-C", train_shapboost_c),
        ("SHAPBoost", train_shapboost),
        ("SHAPBoost (RSF)", train_shapboost),
    ]
    funcs = [func for func in funcs if func[0] in feature_selectors]
    for train_index, _ in tqdm(folds.split(X, censored), total=10):  # type: ignore
        for selector, func in funcs:
            if func in [train_shapboost, train_shapboost_c]:
                if selector == "SHAPBoost (RSF)":
                    eval_model = RandomSurvivalForestWrapper(random_state=42)
                else:
                    eval_model = XGBSurvivalRegressor(**params_train)
                features, importances = func(
                    X_train[train_index],
                    y_train[train_index],
                    [XGBSurvivalRegressor(**params_train), eval_model],
                    metric="c_index",
                )
            elif func in [train_forward_selection, train_backward_selection]:
                features, importances = func(
                    X_train[train_index],
                    y_train[train_index],
                    "surv",
                )
            else:
                features, importances = func(
                    X_train[train_index],
                    y_train[train_index],
                    XGBSurvivalRegressor(**params_train),
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
            feature_selectors[selector].append(features)
            with open(f"survival_features/{dataset_name}{FILE_NAME}.json", "w") as f:
                json.dump(feature_selectors, f)


def run_experiment(
    dataset_name: str, X: np.ndarray, y: np.ndarray, selection_methods: list
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
        xgb_params_train = XGB_PARAMS.copy()
        xgb_params_train["random_state"] = t_rand

        run_cross_validation(
            X_train,
            y_train,
            dataset_name,
            feature_selectors,
            xgb_params_train,
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

    with open(f"survival_features/{dataset_name}{FILE_NAME}.json", "w") as f:
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
    X, y = load_survival_dataset(dataset)
    selection_methods = [
        "XGBoost",
        "P-value",
        "Forward",
        "Backward",
        "SHAPBoost-C",
        "SHAPBoost",
        "SHAPBoost (RSF)",
    ]
    if dataset == "metabric_full":
        selection_methods = [
            s for s in selection_methods if s not in ["Forward", "Backward"]
        ]
    run_experiment(dataset, X.values, y.values, selection_methods)
