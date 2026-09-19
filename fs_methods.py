"""Feature selectors used in the revised experiments.

Every selector receives ONLY the (already preprocessed) training part of an outer
split and returns an ordered list of feature indices. Wherever a subset size has
to be chosen, it is chosen by the same inner 5-fold CV (``inner_cv_score``), with
the inner folds seeded per outer repeat. No selector sees the outer test fold.
"""
import numpy as np
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

MAX_FEATURES = 100

# BorutaPy 0.3 uses aliases removed in NumPy 1.24 (requirements pin numpy 1.26.4).
for _alias, _t in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _t)
N_INNER = 5


# --------------------------------------------------------------------------- #
# shared inner-CV scorer (higher is better: -MAE for regression, C-index for survival)
# --------------------------------------------------------------------------- #
def to_surv(y):
    return Surv.from_arrays(event=np.isfinite(y[:, 1]), time=y[:, 0])


def inner_folds(y, task, seed):
    if task == "surv":
        cv = StratifiedKFold(N_INNER, shuffle=True, random_state=seed)
        return list(cv.split(np.zeros(len(y)), np.isfinite(y[:, 1])))
    return list(KFold(N_INNER, shuffle=True, random_state=seed).split(np.zeros(len(y))))


def _linear_model(task):
    # Same evaluation model family for forward, backward and ranking+CV-k.
    return LinearRegression() if task == "reg" else CoxPHSurvivalAnalysis(alpha=0.1)


def inner_cv_score(X, y, cols, task, folds, model_factory=None):
    if len(cols) == 0:
        return -np.inf
    model_factory = model_factory or (lambda: _linear_model(task))
    scores = []
    for tr, va in folds:
        m = model_factory()
        try:
            if task == "reg":
                m.fit(X[np.ix_(tr, cols)], y[tr].ravel())
                scores.append(-mean_absolute_error(y[va].ravel(), m.predict(X[np.ix_(va, cols)])))
            else:
                m.fit(X[np.ix_(tr, cols)], to_surv(y[tr]))
                s = to_surv(y[va])
                risk = m.predict(X[np.ix_(va, cols)])
                scores.append(concordance_index_censored(s["event"], s["time"], risk)[0])
        except Exception:  # non-convergence etc.: candidate is not usable
            return -np.inf
    return float(np.mean(scores))


# --------------------------------------------------------------------------- #
# Point 3: forward / backward selection with inner-CV stopping
# --------------------------------------------------------------------------- #
def forward_cv(X, y, task, seed, epsilon=0.0, max_features=MAX_FEATURES):
    """Greedy forward selection; stop when the best inner-CV gain is <= epsilon."""
    folds = inner_folds(y, task, seed)
    selected, remaining, best = [], list(range(X.shape[1])), -np.inf
    while remaining and len(selected) < max_features:
        scores = [inner_cv_score(X, y, selected + [f], task, folds) for f in remaining]
        j = int(np.argmax(scores))
        if scores[j] - best <= epsilon:
            break
        best = scores[j]
        selected.append(remaining.pop(j))
    return selected


def backward_cv(X, y, task, seed, epsilon=0.0, max_features=MAX_FEATURES):
    """Backward elimination; stop when every removal lowers inner-CV score by > epsilon.

    The returned order is by importance: features whose removal would hurt
    the inner-CV score most come first (so the evaluation curve is meaningful).
    """
    folds = inner_folds(y, task, seed)
    remaining = list(range(X.shape[1]))
    current = inner_cv_score(X, y, remaining, task, folds)
    while len(remaining) > 1:
        scores = [
            inner_cv_score(X, y, [g for g in remaining if g != f], task, folds)
            for f in remaining
        ]
        j = int(np.argmax(scores))
        if scores[j] < current - epsilon and len(remaining) <= max_features:
            break
        current = scores[j]
        remaining.pop(j)
    drop_cost = [
        current - inner_cv_score(X, y, [g for g in remaining if g != f], task, folds)
        for f in remaining
    ]
    return [remaining[i] for i in np.argsort(drop_cost)[::-1]][:max_features]


def choose_k_by_cv(ranking, X, y, task, seed, max_features=MAX_FEATURES):
    """For pure rankers: pick the prefix size by the same inner CV (replaces the
    'importance >= mean importance' rule)."""
    ranking = list(ranking)[:max_features]
    if not ranking:
        return []
    folds = inner_folds(y, task, seed)
    ks = sorted({k for k in np.unique(np.geomspace(1, len(ranking), 25).astype(int))})
    scores = [inner_cv_score(X, y, ranking[:k], task, folds) for k in ks]
    return ranking[: ks[int(np.argmax(scores))]]


# --------------------------------------------------------------------------- #
# Point 3: penalized baselines (lasso / elastic net / penalized Cox)
# --------------------------------------------------------------------------- #
def elastic_net_select(X, y, task, seed, l1_ratio=1.0, max_features=MAX_FEATURES):
    """l1_ratio=1 -> lasso / lasso-Cox; 0 < l1_ratio < 1 -> elastic net.
    The penalty is chosen by inner 5-fold CV on the training fold."""
    if task == "reg":
        m = ElasticNetCV(l1_ratio=l1_ratio, cv=inner_folds(y, task, seed), n_alphas=100,
                         max_iter=10000, random_state=seed).fit(X, y.ravel())
        coef = m.coef_
    else:
        s = to_surv(y)
        path = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=0.01, n_alphas=50).fit(X, s)
        alphas = path.alphas_
        cv = np.zeros(len(alphas))
        for tr, va in inner_folds(y, task, seed):
            m = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=alphas).fit(X[tr], s[tr])
            for a_i, a in enumerate(alphas):
                risk = m.predict(X[va], alpha=a)
                cv[a_i] += concordance_index_censored(s[va]["event"], s[va]["time"], risk)[0]
        best = alphas[int(np.argmax(cv))]
        coef = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=[best]).fit(X, s).coef_[:, 0]
    nz = np.flatnonzero(np.abs(coef) > 1e-12)
    return nz[np.argsort(-np.abs(coef[nz]))].tolist()[:max_features]


# --------------------------------------------------------------------------- #
# Point 4: SHAPBoost ablations — SHAP vs gain importance, with vs without reweighting
# --------------------------------------------------------------------------- #
def _no_reweight(cls):
    class NoReweight(cls):
        def _update_weights(self, X, Y):
            super()._update_weights(X, Y)  # keep internal bookkeeping identical
            self._global_sample_weights = np.ones_like(self._global_sample_weights)

    NoReweight.__name__ = cls.__name__ + "NoReweight"
    return NoReweight


SHAPBOOST_VARIANTS = {
    # name: (use_shap, reweight)
    "SHAPBoost": (True, True),
    "SHAPBoost-noRW": (True, False),
    "GainBoost": (False, True),
    "GainBoost-noRW": (False, False),
}


def shapboost_select(X, y, task, seed, variant="SHAPBoost", eval_model="linear",
                     collinearity_check=False):
    """Identical search, ranking size, stopping rule and inner folds for every variant;
    only the importance measure and the reweighting step differ."""
    from shapboost import SHAPBoostRegressor, SHAPBoostSurvivalRegressor
    from xgboost import XGBRegressor

    from xgb_survival_regressor import XGBSurvivalRegressor

    use_shap, reweight = SHAPBOOST_VARIANTS[variant]
    if task == "reg":
        cls, metric = SHAPBoostRegressor, "mae"
        ranker = XGBRegressor(n_estimators=100, max_depth=20, n_jobs=1, random_state=seed)
        if eval_model == "linear":
            evaluator = LinearRegression()
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            evaluator = GradientBoostingRegressor(learning_rate=0.01, max_depth=4,
                                                  n_iter_no_change=10, random_state=seed)
    else:
        cls, metric = SHAPBoostSurvivalRegressor, "c_index"
        params = {"objective": "survival:aft", "eval_metric": "aft-nloglik",
                  "learning_rate": 0.05, "max_depth": 3, "min_child_weight": 50,
                  "aft_loss_distribution": "normal", "aft_loss_distribution_scale": 1,
                  "tree_method": "hist", "lambda": 0.01, "alpha": 0.02, "n_jobs": 1,
                  "random_state": seed}
        ranker = XGBSurvivalRegressor(**params)
        # Evaluation models as in the paper: penalized CoxPH ("SHAPBoost (CoxPH)") or
        # RSF ("SHAPBoost (RSF)"). Both wrappers return higher = longer survival, the
        # orientation SHAPBoost's C-index expects (fixed in point 1).
        from test_utils import CoxPHWrapper, RandomSurvivalForestWrapper
        if eval_model in ("rsf", "tree"):
            evaluator = RandomSurvivalForestWrapper(random_state=seed)
        else:
            evaluator = CoxPHWrapper(penalizer=0.1)
    if not reweight:
        cls = _no_reweight(cls)
    sel = cls(
        [ranker, evaluator], loss="adaptive", metric=metric, verbose=0,
        number_of_folds=N_INNER, fold_random_state=seed,
        siso_ranking_size=min(X.shape[1] - 1, 50), max_number_of_features=MAX_FEATURES,
        siso_order=1, num_resets=1, epsilon=1e-10, use_shap=use_shap,
        collinearity_check=collinearity_check,
    )
    sel.fit(X, y)
    return [int(f) for f in sel.selected_subset_]


# --------------------------------------------------------------------------- #
# registry: method name -> callable(X, y, task, seed) -> ordered feature list
# --------------------------------------------------------------------------- #
def _ranker(name):
    def run(X, y, task, seed):
        import test_utils as tu
        from xgboost import XGBRegressor
        if task == "reg":
            xgb = XGBRegressor(n_estimators=100, max_depth=20, n_jobs=1, random_state=seed)
            fn = {"XGBoost": tu.train_xgb, "P-value": tu.train_pvalue, "RReliefF": tu.train_relief,
                  "MRMR": tu.train_mrmr, "Boruta": tu.train_boruta}[name]
            args = (X, y) if name != "MRMR" else (__import__("pandas").DataFrame(X),
                                                   __import__("pandas").Series(y.ravel()))
            ranking, _ = fn(*args, xgb)
        else:
            from xgb_survival_regressor import XGBSurvivalRegressor
            fn = {"XGBoost": tu.train_xgb_survival, "P-value": tu.train_pvalue_survival}[name]
            ranking, _ = fn(X, y, XGBSurvivalRegressor(n_jobs=1, random_state=seed))
        return choose_k_by_cv([int(r) for r in ranking], X, y, task, seed)
    return run


METHODS = {
    "Forward-CV": forward_cv,
    "Backward-CV": backward_cv,
    "Lasso": lambda X, y, t, s: elastic_net_select(X, y, t, s, l1_ratio=1.0),
    "ElasticNet": lambda X, y, t, s: elastic_net_select(X, y, t, s, l1_ratio=0.5),
    **{n: _ranker(n) for n in ["XGBoost", "P-value", "RReliefF", "MRMR", "Boruta"]},
    **{v: (lambda v: lambda X, y, t, s: shapboost_select(X, y, t, s, variant=v))(v)
       for v in SHAPBOOST_VARIANTS},
    "SHAPBoost-C": lambda X, y, t, s: shapboost_select(X, y, t, s, collinearity_check=True),
    "SHAPBoost-tree": lambda X, y, t, s: shapboost_select(X, y, t, s, eval_model="tree"),
}


# Rankers implemented for regression only (test_utils has no survival version).
REG_ONLY = {"RReliefF", "MRMR", "Boruta"}


def methods_for(task: str) -> list:
    """Default method list for a task (used by both selection runners)."""
    return [m for m in METHODS if task == "reg" or m not in REG_ONLY]
