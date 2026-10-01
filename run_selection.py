"""Run feature selection on the training part of every saved outer split.

Replaces perform_{regression,survival}_selection.py:
  * splits come from cv_protocol.get_splits (same indices as evaluation);
  * preprocessing is fitted on the training fold only;
  * one JSON record per (method, split_id) -> results are addressed by key,
    never by list position, so parallel/out-of-order execution is safe;
  * no folds are dropped or truncated to the modal subset size;
  * failures are recorded with their error instead of silently skipped;
  * wall and CPU time are recorded for every selection (single-threaded).

Sequential runner. For the parallel version see run_selection_concurrent.py;
both call select_one() and produce identical records.

Usage: python run_selection.py -d diabetes -t reg [-m METHOD ...] [--splits r0_f0 ...]
"""
import os

# Single-threaded numerics: set before numpy/xgboost are imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

from cv_protocol import get_splits, split_arrays  # noqa: E402
from dataloading import load_regression_dataset, load_survival_dataset  # noqa: E402
from fs_methods import METHODS, methods_for  # noqa: E402

OUT = "results/selection"


def code_version() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def load_data(dataset: str, task: str):
    loader = load_regression_dataset if task == "reg" else load_survival_dataset
    X, y = loader(dataset)
    return X.values.astype(float), y.values.astype(float)


def record_path(dataset: str, method: str, split_id: str) -> str:
    return os.path.join(OUT, dataset, method, f"{split_id}.json")


def write_json_atomic(path: str, obj: dict) -> None:
    """Write to a temp file and rename, so an interrupted run never leaves a
    half-written record that would later be mistaken for a finished one."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def select_one(X, y, dataset, task, split, method, version) -> dict:
    """Run one method on the training part of one split and write its record."""
    X_tr, _, y_tr, _ = split_arrays(X, y, split)  # the test part is never used here
    rec = {k: split[k] for k in ("split_id", "repeat", "fold", "seed", "test_hash")}
    rec.update(dataset=dataset, task=task, method=method, code_version=version)
    t0, c0 = time.perf_counter(), time.process_time()
    try:
        feats = METHODS[method](X_tr, y_tr, task, split["seed"])
        rec.update(status="ok", features=[int(f) for f in feats], n_selected=len(feats))
    except Exception as e:
        rec.update(status="failed", error=repr(e), traceback=traceback.format_exc(),
                    features=[], n_selected=0)
    rec.update(wall_seconds=time.perf_counter() - t0, cpu_seconds=time.process_time() - c0)
    write_json_atomic(record_path(dataset, method, split["split_id"]), rec)
    return rec


def pending_tasks(dataset, task, splits, methods, split_filter=None) -> list:
    """(split, method) pairs without a finished record, i.e. runs are resumable."""
    return [
        (split, m)
        for split in splits
        if not split_filter or split["split_id"] in split_filter
        for m in methods
        if not os.path.exists(record_path(dataset, m, split["split_id"]))
    ]


def parse_args(description=__doc__, extra=None):
    ap = argparse.ArgumentParser(description=description,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--dataset", required=True)
    ap.add_argument("-t", "--task", choices=["reg", "surv"], required=True)
    ap.add_argument("-m", "--methods", nargs="*", default=None,
                    help="default: all methods for the task (fs_methods.methods_for)")
    ap.add_argument("--splits", nargs="*", help="optional subset of split_ids")
    if extra:
        extra(ap)
    args = ap.parse_args()
    args.methods = args.methods or methods_for(args.task)
    unknown = set(args.methods) - set(METHODS)
    if unknown:
        ap.error(f"unknown methods: {sorted(unknown)}")
    return args


def log(rec: dict) -> None:
    print(f"{rec['dataset']} {rec['split_id']} {rec['method']}: {rec['status']} "
            f"k={rec['n_selected']} {rec['wall_seconds']:.1f}s", flush=True)


def main():
    args = parse_args()
    X, y = load_data(args.dataset, args.task)
    splits = get_splits(args.dataset, y, args.task)
    version = code_version()
    for split, method in pending_tasks(args.dataset, args.task, splits, args.methods, args.splits):
        log(select_one(X, y, args.dataset, args.task, split, method, version))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
