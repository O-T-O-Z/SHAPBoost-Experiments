import argparse
import json
from typing import Any
from warnings import simplefilter

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold
from sksurv.ensemble import RandomSurvivalForest
from tqdm import tqdm

from dataloading import load_survival_dataset
from xgb_survival_regressor import XGBSurvivalRegressor

XGB_PARAMS = {
    "objective": "survival:cox",
    "eval_metric": "cox-nloglik",
    "learning_rate": 0.05,
    "max_depth": 3,
    "grow_policy": "lossguide",
    "lambda": 0.01,
    "alpha": 0.02,
    "n_jobs": -1,
}
FILE_END = ""


def run_eval_survival(
    X_train: np.array,
    X_val: np.array,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame,
    all_features: list[int],
    clf: Any,
) -> list[float]:
    """
    Run evaluation for survival data.

    :param X_train: training data.
    :param X_val: validation data.
    :param y_train: training labels.
    :param y_val: validation labels.
    :param all_features: all features.
    :param clf: classifier.
    :return: the C-index per added feature.
    """
    eval_features = []
    cindex_per_added_feature = []
    rsf = clf.__class__.__name__ == "RandomSurvivalForest"

    times = np.quantile(y_val[:, 0], np.linspace(0.1, 0.9, 10))

    # remove all y_val with y_val[:, 0] > max(y_train[:, 0])
    X_val = X_val[y_val[:, 0] < np.max(y_train[:, 0])]
    y_val = y_val[y_val[:, 0] < np.max(y_train[:, 0])]

    df = pd.DataFrame(X_train, columns=list(range(0, X_train.shape[1])))
    df["time"] = y_train[:, 0]
    df["event"] = y_train[:, 2].astype(bool)

    # make sure that the times are after the first event
    min_event_time = np.min(df["time"][df["event"] == 1])
    valid_times = [t for t in times if t >= min_event_time]
    times = np.array(valid_times)

    if rsf:
        y_train = df[["event", "time"]].to_records(index=False)

    for feature in all_features:
        eval_features.append(feature)
        if clf.__class__.__name__ == "CoxPHFitter":
            clf.fit(
                df[["time", "event"] + eval_features],
                duration_col="time",
                event_col="event",
            )
            y_pred = clf.predict_median(X_val[:, eval_features])
        else:
            clf.fit(X_train[:, eval_features], y_train)
            y_pred = clf.predict(X_val[:, eval_features]) * -1

        cindex = concordance_index(
            y_val[:, 0], y_pred, (y_val[:, 0] == y_val[:, 1]).astype(int)
        )
        cindex_per_added_feature.append(cindex)
    return cindex_per_added_feature


if __name__ == "__main__":
    simplefilter("ignore", category=ConvergenceWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
    )
    args = parser.parse_args()
    dataset = args.dataset
    print(f"Running {dataset}")
    for regressor in [
        RandomSurvivalForest,
        XGBSurvivalRegressor,
        CoxPHFitter,
    ]:
        print(f"Running {regressor}")
        X, y = load_survival_dataset(dataset)
        y["event"] = (y.loc[:, "lower_bound"] == y.loc[:, "upper_bound"]).astype(bool)

        with open(f"survival_features/{dataset}_features_selected{FILE_END}.json") as f:
            data = json.load(f)

        X_train = np.array(X.values)
        y_train = np.array(y.values)
        results = {
            k: {"cindex_per_fold": []}
            for k in [
                "XGBoost",
                "P-value",
                "Forward",
                "Backward",
                "SHAPBoost-C",
                "SHAPBoost (CoxPH)",
                "SHAPBoost (RSF)",
            ]
        }
        idx = 0
        for _, e_rand in tqdm([(42, 84), (55, 110), (875, 1750)]):
            print(f"Running random state {e_rand}")
            folds = KFold(n_splits=10, shuffle=True, random_state=42)
            if regressor == XGBSurvivalRegressor:
                evaluator = regressor(random_state=e_rand, **XGB_PARAMS)
            elif regressor == RandomSurvivalForest:
                evaluator = regressor(random_state=e_rand)
            elif regressor == CoxPHFitter:
                evaluator = regressor(penalizer=0.1)
            for train_index, test_index in tqdm(folds.split(X), total=10):  # type: ignore
                for selector in results:
                    print(f"Running {selector}")
                    if selector not in data.keys():
                        continue
                    if len(data[selector]) == 0 or len(data[selector]) <= idx:
                        continue
                    features = data[selector][idx]
                    if len(features) == 0:
                        continue
                    cindex = run_eval_survival(
                        X_train[train_index],
                        X_train[test_index],
                        y_train[train_index],
                        y_train[test_index],
                        features,
                        evaluator,
                    )
                    results[selector]["cindex_per_fold"].append(cindex)
                    with open(
                        f"survival_features/{dataset}_{evaluator.__class__.__name__}{FILE_END}.json",
                        "w",
                    ) as f:
                        json.dump(results, f)
                idx += 1
