"""Build the SLURM task lists (run on the login node by submit_all.sh).

  * checks the survival event coding (fails loudly if the old CSVs are used);
  * creates splits/<dataset>.json for every dataset BEFORE any job starts, so
    array tasks never race to write the same split file;
  * lists only (dataset, method, split) selections WITHOUT a finished record, so
    resubmitting after time-outs or failures reruns only what is missing
    (use --all to list everything).

Writes slurm/tasks/{selection,r_baseline,datasets}.tsv (tab-separated).
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "slurm"))

import numpy as np  # noqa: E402

from cv_protocol import get_splits  # noqa: E402
from dataloading import load_regression_dataset, load_survival_dataset  # noqa: E402
from fs_methods import methods_for  # noqa: E402
from report_utils import REGRESSION_DATASETS, SURVIVAL_DATASETS  # noqa: E402
from run_selection import record_path  # noqa: E402

# Documented exclusions (computational cost measured on one fold; see README).
EXCLUDE = {
    ("metabric_regression", "Backward-CV"),  # ~8 CPU-h per fold
    ("metabric_full", "Backward-CV"),        # far more with Cox models
}
# [B] costs O(n^2) per boosting iteration and refits 100 times for stability selection:
# measured ~1.75 h wall / 6.7 CPU-h per WHAS split (n_train = 450), so AIDS
# (n_train = 1036, ~5x) would need ~1000 CPU-h for 30 splits, and NACD/NHANES/
# SUPPORT/METABRIC far more. Report [B] on the datasets where it is feasible.
R_DATASETS = ["breast_cancer", "whas500"]
R_METHOD = "CIndexBoost-StabSel"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="list tasks even if a record exists")
    ap.add_argument("--datasets", nargs="*", default=None, help="restrict to these datasets")
    ap.add_argument("--include-queued", action="store_true",
                    help="also list tasks that are already queued or running")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="restrict to these methods (incl. CIndexBoost-StabSel for [B])")
    args = ap.parse_args()
    os.makedirs("slurm/tasks", exist_ok=True)

    datasets = [(d, "reg") for d in REGRESSION_DATASETS] + [(d, "surv") for d in SURVIVAL_DATASETS]
    if args.datasets:
        datasets = [(d, t) for d, t in datasets if d in args.datasets]

    # Tasks already in the queue (arrays submitted by this version keep a copy of
    # their task lines in slurm/tasks/submitted/) are skipped, so resubmitting while
    # jobs run never duplicates work.
    queued = {}
    if not args.include_queued:
        from check_status import queued_tasks  # imported here: check_status imports us
        queued = queued_tasks()
        os.chdir(REPO)

    def todo(ds, m, sid):
        return args.all or not (os.path.exists(record_path(ds, m, sid)) or (ds, m, sid) in queued)

    sel, r_tasks = [], []
    for ds, task in datasets:
        X, y = (load_regression_dataset if task == "reg" else load_survival_dataset)(ds)
        splits = get_splits(ds, y.values.astype(float), task)  # creates the file once
        for split in splits:
            sid = split["split_id"]
            for m in methods_for(task):
                if (ds, m) in EXCLUDE or (args.methods and m not in args.methods):
                    continue
                if todo(ds, m, sid):
                    sel.append((ds, task, m, sid))
            if task == "surv" and ds in R_DATASETS and (not args.methods or R_METHOD in args.methods):
                if todo(ds, R_METHOD, sid):
                    r_tasks.append((ds, sid))

    def write(name, rows):
        with open(f"slurm/tasks/{name}.tsv", "w") as f:
            f.writelines("\t".join(r) + "\n" for r in rows)

    write("selection", sel)
    write("r_baseline", r_tasks)
    write("datasets", datasets)
    n_all = len(datasets) * 30
    if queued:
        print(f"skipped {len(queued)} tasks that are already queued or running")
    print(f"selection tasks to run: {len(sel)}   [B] tasks to run: {len(r_tasks)}   "
          f"datasets: {len(datasets)} ({n_all} splits in total)")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
