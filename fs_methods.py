"""Method registry used by run_selection.py.

At this stage the ORIGINAL selection methods and their original size rules
(mean-importance threshold, cap at 100) are kept unchanged; they are only
wrapped so that they run on the training part of the shared, saved splits.
Point 3 replaces the stopping/size rules, point 4 adds the ablations.

Every method: callable(X_train, y_train, task, seed) -> ordered list of feature ids.
"""
import numpy as np
import pandas as pd
from sksurv.util import Surv

MAX_FEATURES = 100

# BorutaPy 0.3 uses aliases removed in NumPy 1.24 (requirements pin numpy 1.26.4).
for _alias, _t in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _t)

XGB_SURV_PARAMS = {
    "objective": "survival:aft", "eval_metric": "aft-nloglik", "learning_rate": 0.05,
    "max_depth": 3, "min_child_weight": 50, "subsample": 1.0, "colsample_bynode": 1.0,
    "aft_loss_distribution": "normal", "aft_loss_distribution_scale": 1,
    "tree_method": "hist", "booster": "gbtree", "grow_policy": "lossguide",
    "lambda": 0.01, "alpha": 0.02, "n_jobs": 1,
}


def to_surv(y):
    return Surv.from_arrays(event=np.isfinite(y[:, 1]), time=y[:, 0])


def _original_size_rule(features, importances):
    """Unchanged from perform_*_selection.py (replaced in point 3)."""
    features, importances = list(features), np.asarray(importances)
    if np.unique(importances).shape[0] != 1:
        k = len(np.where(importances >= np.mean(importances))[0])
        features = features[:k]
    return [int(f) for f in features[:MAX_FEATURES]]


def _original(name):
    def run(X, y, task, seed):
        import test_utils as tu
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression
        from xgboost import XGBRegressor

        from xgb_survival_regressor import XGBSurvivalRegressor

        if task == "reg":
            xgb = lambda: XGBRegressor(n_estimators=100, max_depth=20, n_jobs=1,  # noqa: E731
                                       random_state=seed)
            if name == "SHAPBoost (LR)":
                out = tu.train_shapboost(X, y, [xgb(), LinearRegression()])
            elif name == "SHAPBoost (GBR)":
                # random_state added: with n_iter_no_change the validation split is
                # random, so the original (unseeded) setting was not reproducible.
                out = tu.train_shapboost(X, y, [xgb(), GradientBoostingRegressor(
                    learning_rate=0.01, max_depth=4, n_iter_no_change=10,
                    random_state=seed)])
            elif name == "SHAPBoost-C":
                out = tu.train_shapboost_c(X, y, [xgb(), LinearRegression()])
            elif name in ("Forward", "Backward"):
                fn = tu.train_forward_selection if name == "Forward" else tu.train_backward_selection
                out = fn(pd.DataFrame(X), pd.DataFrame(y), "reg")
            elif name == "MRMR":
                out = tu.train_mrmr(pd.DataFrame(X), pd.DataFrame(y), xgb())
            else:
                fn = {"XGBoost": tu.train_xgb, "P-value": tu.train_pvalue,
                      "RReliefF": tu.train_relief, "Boruta": tu.train_boruta}[name]
                out = fn(X, y, xgb())
        else:
            params = {**XGB_SURV_PARAMS, "random_state": seed}
            if name in ("SHAPBoost (CoxPH)", "SHAPBoost (RSF)", "SHAPBoost-C"):
                # Evaluation model as in the original concurrent script that produced the
                # paper results: penalized CoxPH (RSF for the RSF variant).
                ev = (tu.RandomSurvivalForestWrapper(random_state=42) if name == "SHAPBoost (RSF)"
                      else tu.CoxPHWrapper(penalizer=0.1))
                fn = tu.train_shapboost_c if name == "SHAPBoost-C" else tu.train_shapboost
                out = fn(X, y, [XGBSurvivalRegressor(**params), ev], metric="c_index")
            elif name in ("Forward", "Backward"):
                fn = tu.train_forward_selection if name == "Forward" else tu.train_backward_selection
                out = fn(X, y, "surv")
            else:
                fn = {"XGBoost": tu.train_xgb_survival, "P-value": tu.train_pvalue_survival}[name]
                out = fn(X, y, XGBSurvivalRegressor(**params))
        return _original_size_rule(*out)
    return run


REG_METHODS = ["RReliefF", "Boruta", "XGBoost", "MRMR", "P-value", "Forward", "Backward",
               "SHAPBoost-C", "SHAPBoost (LR)", "SHAPBoost (GBR)"]
SURV_METHODS = ["XGBoost", "P-value", "Forward", "Backward", "SHAPBoost-C",
                "SHAPBoost (CoxPH)", "SHAPBoost (RSF)"]
METHODS = {n: _original(n) for n in dict.fromkeys(REG_METHODS + SURV_METHODS)}


def methods_for(task: str) -> list:
    """Default method list for a task (used by both selection runners)."""
    return list(REG_METHODS if task == "reg" else SURV_METHODS)
