"""Parallel version of run_selection.py (replaces perform_*_selection_concurrent.py).

What was wrong with the old concurrent scripts, and how this one avoids it:
  * results were appended to one list per method in *completion* order and the
    fold index was discarded, so "list position" did not identify the fold
    -> every (split, method) task writes its own record file, keyed by split_id
        and carrying the split's test_hash; completion order is irrelevant;
  * a ConvergenceError silently dropped the fold (29/31 entries per method)
    -> failures inside a method are recorded as status="failed"; if a worker
        process itself dies, no record is written and a rerun retries the task;
  * a ThreadPoolExecutor combined with n_jobs=-1 models oversubscribed the CPU
    and made timings meaningless
    -> separate processes, each with single-threaded numerics (n_jobs=1).

Results are identical to the sequential runner: every task is fully determined
by (dataset, split, method, seed) and both runners call the same select_one().
Runs are resumable: finished records are skipped.

Timings: with W workers on a shared machine, wall_seconds includes some
contention. For the runtime comparison in the paper, use cpu_seconds or rerun
the timing subset with the sequential run_selection.py.

Usage:
    python run_selection_concurrent.py -d eyedata -t reg --workers 8
    python run_selection_concurrent.py -d aids -t surv -m SHAPBoost (CoxPH) Forward
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # inherited by the spawned workers

import multiprocessing as mp  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from cv_protocol import get_splits  # noqa: E402
from run_selection import (  # noqa: E402
    code_version,
    load_data,
    log,
    parse_args,
    pending_tasks,
    select_one,
)

# Slow methods are submitted first so the pool does not end on a long tail.
SLOW_FIRST = ("SHAPBoost", "Backward", "Forward", "Boruta", "RReliefF")

_X = _Y = None  # per-worker copies of the data, set once by _init_worker


def _init_worker(dataset: str, task: str) -> None:
    global _X, _Y
    np.seterr(all="ignore")
    _X, _Y = load_data(dataset, task)


def _run(dataset, task, split, method, version) -> dict:
    return select_one(_X, _Y, dataset, task, split, method, version)


def _priority(task_pair) -> int:
    method = task_pair[1]
    for rank, prefix in enumerate(SLOW_FIRST):
        if method.startswith(prefix):
            return rank
    return len(SLOW_FIRST)


def main():
    args = parse_args(
        description=__doc__,
        extra=lambda ap: ap.add_argument(
            "-w", "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)),
    )
    _, y = load_data(args.dataset, args.task)
    # Create/save the splits in the parent BEFORE starting workers (no races on the file).
    splits = get_splits(args.dataset, y, args.task)
    version = code_version()
    tasks = sorted(pending_tasks(args.dataset, args.task, splits, args.methods, args.splits),
                    key=_priority)
    if not tasks:
        print("Nothing to do: all records exist.")
        return
    print(f"{len(tasks)} tasks on {args.workers} workers", flush=True)

    failed_tasks = []
    ctx = mp.get_context("spawn")  # safe with OpenMP/XGBoost; same behaviour on all OSes
    executor = ProcessPoolExecutor(args.workers, mp_context=ctx,
                                    initializer=_init_worker,
                                    initargs=(args.dataset, args.task))
    try:
        futures = {
            executor.submit(_run, args.dataset, args.task, split, method, version):
                (split["split_id"], method)
            for split, method in tasks
        }
        for fut in tqdm(as_completed(futures), total=len(futures)):
            split_id, method = futures[fut]  # identify the task by key, never by order
            try:
                log(fut.result())
            except Exception as e:  # worker crashed (e.g. out of memory): no record written
                failed_tasks.append((split_id, method))
                print(f"{args.dataset} {split_id} {method}: WORKER ERROR {e!r} "
                        "(no record written; rerun to retry)", flush=True)
    except KeyboardInterrupt:
        print("Interrupted: cancelling pending tasks; finished records are kept.")
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown()
    if failed_tasks:
        print(f"{len(failed_tasks)} tasks did not produce a record: {failed_tasks}")


if __name__ == "__main__":
    main()
