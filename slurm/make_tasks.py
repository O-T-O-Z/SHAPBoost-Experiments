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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
R_DATASETS = ["breast_cancer", "whas500", "aids"]  # [B] is O(n^2); larger ones infeasible
R_METHOD = "CIndexBoost-StabSel"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="list tasks even if a record exists")
    ap.add_argument("--datasets", nargs="*", default=None, help="restrict to these datasets")
    args = ap.parse_args()
    os.makedirs("slurm/tasks", exist_ok=True)

    datasets = [(d, "reg") for d in REGRESSION_DATASETS] + [(d, "surv") for d in SURVIVAL_DATASETS]
    if args.datasets:
        datasets = [(d, t) for d, t in datasets if d in args.datasets]

    sel, r_tasks = [], []
    for ds, task in datasets:
        X, y = (load_regression_dataset if task == "reg" else load_survival_dataset)(ds)
        splits = get_splits(ds, y.values.astype(float), task)  # creates the file once
        for split in splits:
            sid = split["split_id"]
            for m in methods_for(task):
                if (ds, m) in EXCLUDE:
                    continue
                if args.all or not os.path.exists(record_path(ds, m, sid)):
                    sel.append((ds, task, m, sid))
            if task == "surv" and ds in R_DATASETS:
                if args.all or not os.path.exists(record_path(ds, R_METHOD, sid)):
                    r_tasks.append((ds, sid))

    def write(name, rows):
        with open(f"slurm/tasks/{name}.tsv", "w") as f:
            f.writelines("\t".join(r) + "\n" for r in rows)

    write("selection", sel)
    write("r_baseline", r_tasks)
    write("datasets", datasets)
    n_all = len(datasets) * 30
    print(f"selection tasks to run: {len(sel)}   [B] tasks to run: {len(r_tasks)}   "
          f"datasets: {len(datasets)} ({n_all} splits in total)")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
