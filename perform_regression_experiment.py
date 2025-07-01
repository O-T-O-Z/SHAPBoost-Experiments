import argparse
import json
from warnings import simplefilter

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from tqdm import tqdm

from dataloading import load_regression_dataset
from test_utils import run_eval

FILE_END = ""

if __name__ == "__main__":
    simplefilter("ignore", category=ConvergenceWarning)

    params = {
        "learning_rate": 0.01,
        "max_depth": 4,
        "n_iter_no_change": 10,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
    )
    args = parser.parse_args()
    dataset = args.dataset
    print(f"Running {dataset}")
    for regressor in [LinearRegression, GradientBoostingRegressor]:
        print(f"Running {regressor}")
        X, y = load_regression_dataset(dataset)
        with open(
            f"regression_features/{dataset}_features_selected{FILE_END}.json"
        ) as f:
            data = json.load(f)

        X_train = np.array(X.values)
        y_train = np.array(y.values).ravel()
        results = {
            k: {"mae_per_fold": [], "r2_per_fold": []}
            for k in [
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
        }
        idx = 0
        for _, e_rand in tqdm([(42, 84), (55, 110), (875, 1750)]):
            print(f"Running random state {e_rand}")
            folds = KFold(n_splits=10, shuffle=True, random_state=42)
            if regressor == GradientBoostingRegressor:
                evaluator = regressor(random_state=e_rand, **params)
            else:
                evaluator = regressor()
            for train_index, test_index in tqdm(folds.split(X), total=10):
                for selector in results:
                    print(f"Running {selector}")
                    if selector not in data.keys() or len(data[selector]) == 0:
                        continue
                    features = data[selector][idx]
                    if len(features) == 0:
                        continue
                    mae, r2 = run_eval(
                        X_train[train_index],
                        X_train[test_index],
                        y_train[train_index],
                        y_train[test_index],
                        features,
                        evaluator,
                    )
                    results[selector]["mae_per_fold"].append(mae)
                    results[selector]["r2_per_fold"].append(r2)
                    with open(
                        f"regression_features/{dataset}_{evaluator.__class__.__name__}{FILE_END}.json",
                        "w",
                    ) as f:
                        json.dump(results, f)
                idx += 1
