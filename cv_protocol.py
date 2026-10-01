"""Single source of truth for outer cross-validation splits.

Selection and evaluation both call ``get_splits``; the splits are generated once,
written to ``splits/<dataset>.json`` and re-read afterwards, so both stages are
guaranteed to use identical train/test indices. Every result record carries the
split's ``test_hash`` and evaluation refuses to run if they do not match.
"""
import hashlib
import json
import os

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEEDS = (84, 110, 1750)  # one seed per repeat; used for the outer folds only
N_SPLITS = 10
SPLIT_DIR = "splits"


def event_indicator(y: np.ndarray) -> np.ndarray:
    """1 = observed event, 0 = right-censored (upper_bound == inf)."""
    return np.isfinite(y[:, 1]).astype(int)


def _hash(idx: np.ndarray) -> str:
    return hashlib.sha1(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()[:12]


def make_splits(y: np.ndarray, task: str, seeds=SEEDS, n_splits=N_SPLITS) -> list:
    """Repeated K-fold; survival folds are stratified on the event indicator."""
    splits = []
    for repeat, seed in enumerate(seeds):
        if task == "surv":
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            gen = cv.split(np.zeros(len(y)), event_indicator(y))
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            gen = cv.split(np.zeros(len(y)))
        for fold, (tr, te) in enumerate(gen):
            splits.append(
                {
                    "split_id": f"r{repeat}_f{fold}",
                    "repeat": repeat,
                    "seed": seed,
                    "fold": fold,
                    "train": tr.tolist(),
                    "test": te.tolist(),
                    "test_hash": _hash(te),
                }
            )
    return splits


def get_splits(dataset: str, y: np.ndarray, task: str) -> list:
    """Load saved splits, or create and save them on first use."""
    os.makedirs(SPLIT_DIR, exist_ok=True)
    path = os.path.join(SPLIT_DIR, f"{dataset}.json")
    if os.path.exists(path):
        with open(path) as f:
            splits = json.load(f)
        n = max(max(s["train"] + s["test"]) for s in splits) + 1
        if n != len(y):
            raise ValueError(f"{path} was made for {n} rows, data has {len(y)}.")
        return splits
    splits = make_splits(y, task)
    with open(path, "w") as f:
        json.dump(splits, f)
    return splits


def fit_preprocessor(X_train: np.ndarray):
    """Median imputation + scaling, fitted on the training part of a split only."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(
        X_train
    )


def split_arrays(X: np.ndarray, y: np.ndarray, split: dict):
    """Return preprocessed (X_train, X_test, y_train, y_test) for one split."""
    tr, te = np.asarray(split["train"]), np.asarray(split["test"])
    prep = fit_preprocessor(X[tr])
    return prep.transform(X[tr]), prep.transform(X[te]), y[tr], y[te]
