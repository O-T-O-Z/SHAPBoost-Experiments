import numpy as np
import pandas as pd
from boruta import BorutaPy
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from mrmr import mrmr_regression
from shapboost import SHAPBoostRegressor, SHAPBoostSurvivalRegressor
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from skrebate import ReliefF
from sksurv.ensemble import RandomSurvivalForest
from xgboost import XGBRegressor

pd.options.mode.chained_assignment = None


def train_pvalue_survival(X, y, clf):
    df = pd.DataFrame(X, columns=[i for i in range(0, X.shape[1])])
    df["time"] = y[:, 0]
    df["event"] = (y[:, 0] == y[:, 1]).astype(int)
    features = df.columns[:-2]
    significant_features = []

    for feature in features:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(
            df[[feature, "time", "event"]],
            duration_col="time",
            event_col="event",
        )
        p_value = cph.summary.loc[feature, "p"]
        if p_value < 0.05:
            significant_features.append((feature, p_value))

    significant_features.sort(key=lambda x: x[1])
    size = max(100, len(significant_features))
    return [x[0] for x in significant_features[:100]], np.array(
        [1 / size for _ in range(size)]
    )


def train_pvalue(X, y, clf):
    _, p_values = f_regression(X, y)

    # sort p-values
    p_values = pd.Series(
        p_values, index=[i for i in range(0, X.shape[1])]
    ).sort_values()

    significant_features = p_values[p_values < 0.05].index.tolist()
    size = max(100, len(significant_features))
    return significant_features[:100], np.array([1 / size for _ in range(size)])


def train_forward_selection(X, y, clf):
    df = pd.DataFrame(X, columns=[i for i in range(0, X.shape[1])])
    if clf == "surv":
        df["time"] = y[:, 0]
        df["event"] = (y[:, 0] == y[:, 1]).astype(int)
    selected_features = []
    remaining_features = list(df.columns.difference(["time", "event"]))
    best_criterion = np.inf  # Use AIC or BIC (lower is better)
    # Forward selection loop
    while remaining_features:
        best_feature = None
        for feature in remaining_features:
            candidate_features = selected_features + [feature]
            if clf == "surv":
                cph = CoxPHFitter(penalizer=0.1)
                cph.fit(
                    df[["time", "event"] + candidate_features],
                    duration_col="time",
                    event_col="event",
                )
                current_criterion = cph.AIC_partial_  # or cph.BIC_partial_
            elif clf == "reg":
                model = LinearRegression().fit(df[candidate_features], y)
                current_criterion = mean_absolute_error(
                    y, model.predict(df[candidate_features])
                )
            if current_criterion < best_criterion:
                best_criterion = current_criterion
                best_feature = feature
        if best_feature is not None:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
        else:
            break

    selected_features = selected_features[:100]
    size = max(100, len(selected_features))
    return selected_features, np.array([1 / size for _ in range(size)])


def train_backward_selection(X, y, clf):
    df = pd.DataFrame(X, columns=[i for i in range(0, X.shape[1])])
    if clf == "surv":
        df["time"] = y[:, 0]
        df["event"] = (y[:, 0] == y[:, 1]).astype(int)
    remaining_features = list(df.columns.difference(["time", "event"]))
    best_criterion = np.inf  # Use AIC or BIC (lower is better)

    # Backward selection loop
    while remaining_features:
        worst_feature = None
        for feature in remaining_features:
            candidate_features = [f for f in remaining_features if f != feature]
            if clf == "surv":
                cph = CoxPHFitter(penalizer=0.1)
                cph.fit(
                    df[["time", "event"] + candidate_features],
                    duration_col="time",
                    event_col="event",
                )
                current_criterion = cph.AIC_partial_  # or cph.BIC_partial_
            elif clf == "reg":
                model = LinearRegression().fit(df[candidate_features], y)
                current_criterion = mean_absolute_error(
                    y, model.predict(df[candidate_features])
                )

            if current_criterion < best_criterion:
                best_criterion = current_criterion
                worst_feature = feature

        if worst_feature is not None:
            remaining_features.remove(worst_feature)
        else:
            break

    remaining_features = remaining_features[:100]
    size = max(100, len(remaining_features))
    return remaining_features, np.array([1 / size for _ in range(size)])


def train_boruta(X, y, clf):
    boruta_feature_selector = BorutaPy(
        estimator=clf,
        n_estimators="auto",  # type: ignore
        verbose=0,
        random_state=0,
        max_iter=100,
    )
    boruta_feature_selector.fit(X, y)
    features = [x for x in range(X.shape[1])]

    if not any(boruta_feature_selector.support_):
        return features, np.array([1 / X.shape[1] for _ in range(X.shape[1])])
    features = [x for x in range(X.shape[1]) if boruta_feature_selector.support_[x]]

    post_ranker = clf.fit(X[:, boruta_feature_selector.support_], y)

    imp_per_feat = [(k, v) for k, v in zip(features, post_ranker.feature_importances_)]
    imp_per_feat = sorted(imp_per_feat, key=lambda x: x[1], reverse=True)
    features = [x[0] for x in imp_per_feat]
    importances = [x[1] for x in imp_per_feat]
    return features, np.array(importances)


def train_boruta_survival(X, y, clf):
    boruta_feature_selector = BorutaPy(
        estimator=clf,
        n_estimators="auto",  # type: ignore
        verbose=0,
        random_state=0,
        max_iter=100,
    )
    boruta_feature_selector.fit(X, y)
    features = [x for x in range(X.shape[1])]
    if not any(boruta_feature_selector.support_):
        return features, np.array([1 / X.shape[1] for _ in range(X.shape[1])])
    features = [x for x in range(X.shape[1]) if boruta_feature_selector.support_[x]]
    post_ranker = clf.fit(X[:, boruta_feature_selector.support_], y)

    fscore = post_ranker.get_score(importance_type="gain")  # type: ignore
    feature_importance = np.zeros(X.shape[1])
    for k, v in fscore.items():
        feature_importance[int(k[1:])] = v
    imp_per_feat = [(k, v) for k, v in zip(range(X.shape[1]), feature_importance)]
    imp_per_feat = sorted(imp_per_feat, key=lambda x: x[1], reverse=True)
    features = [x[0] for x in imp_per_feat]
    importances = [x[1] for x in imp_per_feat]
    return features, np.array(importances)


def train_relief(X, y, placeholder=None):
    y = y.flatten()
    X = X.astype(np.float64)
    clf = ReliefF(n_features_to_select=X.shape[1], n_neighbors=10, n_jobs=1)
    clf.fit(X, y)
    imp_per_feat = [(k, v) for k, v in zip(range(X.shape[1]), clf.feature_importances_)]
    imp_per_feat = sorted(imp_per_feat, key=lambda x: x[1], reverse=True)
    features = [x[0] for x in imp_per_feat]
    importances = [x[1] for x in imp_per_feat]
    return features, np.array(importances)


def train_shapboost(
    X,
    y,
    estimators=[
        XGBRegressor(n_estimators=100, max_depth=20, n_jobs=-1),
        LinearRegression(),
    ],
    metric="mae",
    use_shap=True,
    collinearity_check=False,
):
    n_features = X.shape[1] if X.shape[1] < 50 else 50
    shapboost_class = (
        SHAPBoostRegressor if metric == "mae" else SHAPBoostSurvivalRegressor
    )
    feature_selector = shapboost_class(
        estimators,
        loss="adaptive",
        metric=metric,
        verbose=0,
        number_of_folds=5,
        siso_ranking_size=n_features,
        max_number_of_features=100,
        siso_order=1,
        num_resets=1,
        epsilon=1e-10,
        use_shap=use_shap,
        collinearity_check=collinearity_check,
    )
    feature_selector.fit(X, y)
    return feature_selector.selected_subset_, np.array(
        [
            1 / len(feature_selector.selected_subset_)
            for _ in range(len(feature_selector.selected_subset_))
        ]
    )


def train_shapboost_c(
    X,
    y,
    estimators=[
        XGBRegressor(n_estimators=100, max_depth=20, n_jobs=-1),
        LinearRegression(),
    ],
    metric="mae",
):
    return train_shapboost(
        X, y, estimators, metric, use_shap=True, collinearity_check=True
    )


def train_xgb(X, y, clf):
    clf.fit(X, y)
    imp_per_feat = [(k, v) for k, v in zip(range(X.shape[1]), clf.feature_importances_)]
    imp_per_feat = sorted(imp_per_feat, key=lambda x: x[1], reverse=True)
    features = [x[0] for x in imp_per_feat]
    importances = [x[1] for x in imp_per_feat]
    return features, np.array(importances)


def train_mrmr(X, y, clf):
    features = mrmr_regression(X, y, K=X.shape[1], n_jobs=1, show_progress=False)
    return features, np.array([1 / X.shape[1] for _ in range(X.shape[1])])


def train_xgb_survival(X, y, clf):
    post_ranker = clf.fit(X, y)

    fscore = post_ranker.get_score(importance_type="gain")  # type: ignore
    feature_importance = np.zeros(X.shape[1])
    for k, v in fscore.items():
        feature_importance[int(k[1:])] = v
    imp_per_feat = [(k, v) for k, v in zip(range(X.shape[1]), feature_importance)]
    imp_per_feat = sorted(imp_per_feat, key=lambda x: x[1], reverse=True)
    features = [x[0] for x in imp_per_feat]
    importances = [x[1] for x in imp_per_feat]
    return features, np.array(importances)


def run_eval(X_train, X_val, y_train, y_val, all_features, evaluator):
    eval_features = []
    mae_per_added_feature = []
    r2_per_added_feature = []
    for feature in all_features:
        eval_features.append(feature)
        evaluator.fit(X_train[:, eval_features], y_train)
        mae = mean_absolute_error(y_val, evaluator.predict(X_val[:, eval_features]))
        r2 = r2_score(y_val, evaluator.predict(X_val[:, eval_features]))
        mae_per_added_feature.append(mae)
        r2_per_added_feature.append(r2)
    return mae_per_added_feature, r2_per_added_feature


def run_eval_survival(X_train, X_val, y_train, y_val, all_features, clf):
    eval_features = []
    cindex_per_added_feature = []
    for feature in all_features:
        eval_features.append(feature)
        clf.fit(X_train[:, eval_features], y_train)
        y_pred = clf.predict(X_val[:, eval_features])
        if clf.get_params()["objective"] == "survival:cox":
            y_pred = -y_pred
        cindex = concordance_index(
            y_val[:, 0], y_pred, (y_val[:, 0] == y_val[:, 1]).astype(int)
        )
        cindex_per_added_feature.append(cindex)
    return cindex_per_added_feature


class RandomSurvivalForestWrapper:
    def __init__(self, **args):
        self.clf = RandomSurvivalForest(**args)

    def fit(self, X, y, **args):
        df = pd.DataFrame(X, columns=[i for i in range(0, X.shape[1])])
        df["time"] = y[:, 0]
        df["event"] = (y[:, 0] == y[:, 1]).astype(bool)
        y = df[["event", "time"]].to_records(index=False)
        self.clf.fit(X, y, **args)

    def predict(self, X):
        return self.clf.predict(X)


class CoxPHWrapper:
    def __init__(self, **args):
        self.clf = CoxPHFitter(**args)

    def fit(self, X, y, **args):
        df = pd.DataFrame(X, columns=[i for i in range(0, X.shape[1])])
        df["time"] = y[:, 0]
        df["event"] = (y[:, 0] == y[:, 1]).astype(bool)
        self.clf.fit(
            df[["time", "event"] + [i for i in range(0, X.shape[1])]],
            duration_col="time",
            event_col="event",
        )

    def predict(self, X):
        return self.clf.predict_median(X) * -1
