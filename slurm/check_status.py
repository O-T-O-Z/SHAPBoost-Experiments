"""Which tasks of the SLURM run finished, failed, or are missing - and why.

Compares the result records on disk with the full list of expected tasks
(the same list submit_all.sh uses) and reads the job logs for missing ones.

  python slurm/check_status.py                    # summary + reasons
  python slurm/check_status.py --details          # list every failed / missing task
  python slurm/check_status.py --datasets aids    # restrict to some datasets
  python slurm/check_status.py --reset-failed     # delete records with status "failed"
                                                  # so submit_all.sh resubmits them

Note: a method that raises an error is saved as a record with status "failed" and
its SLURM job still ends as COMPLETED, so sacct alone does not show these.
Missing records (time limit, out of memory, node failure, still pending) are
resubmitted automatically by running submit_all.sh again; failed records only
after --reset-failed (check the error first: a deterministic error fails again).
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from fs_methods import methods_for  # noqa: E402
from report_utils import REGRESSION_DATASETS, SURVIVAL_DATASETS  # noqa: E402
from run_selection import record_path  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "slurm"))
from make_tasks import EXCLUDE, R_DATASETS, R_METHOD  # noqa: E402

REASONS = [  # (label, regex on the log text), first match wins
    ("time limit", re.compile(r"DUE TO TIME LIMIT|TIME LIMIT", re.I)),
    ("out of memory", re.compile(r"oom[-_ ]kill|out of memory|OUT_OF_MEMORY|MemoryError", re.I)),
    ("node failure", re.compile(r"NODE_FAIL|node failure", re.I)),
    ("cancelled", re.compile(r"CANCELLED", re.I)),
    ("command not found (module not loaded in slurm/env.sh?)",
     re.compile(r"command not found", re.I)),
    ("missing R package", re.compile(r"there is no package called", re.I)),
    ("python error", re.compile(r"^Traceback", re.M)),
    ("R error", re.compile(r"^Error", re.M)),
    ("file not found", re.compile(r"No such file or directory", re.I)),
]


def expected_tasks(datasets):
    for ds, task in datasets:
        path = f"splits/{ds}.json"
        if not os.path.exists(path):
            yield ds, None, None, "no splits file (dataset never submitted)"
            continue
        split_ids = [s["split_id"] for s in json.load(open(path))]
        methods = [m for m in methods_for(task) if (ds, m) not in EXCLUDE]
        if task == "surv" and ds in R_DATASETS:
            methods.append(R_METHOD)
        for m in methods:
            for sid in split_ids:
                yield ds, m, sid, None


def index_logs():
    """(dataset, method, split) -> all its log files, oldest first (from each first line)."""
    idx = defaultdict(list)
    files = glob.glob("slurm/logs/select_*.out") + glob.glob("slurm/logs/rbase_*.out")
    for f in sorted(files, key=os.path.getmtime):
        try:
            with open(f, errors="replace") as fh:
                first = fh.readline()
        except OSError:
            continue
        parts = first.split("] ", 1)[-1].split()
        if f.split("/")[-1].startswith("select_") and len(parts) == 4:
            ds, _task, m, sid = parts
        elif len(parts) == 2:
            (ds, sid), m = parts, R_METHOD
        else:
            continue
        idx[(ds, m, sid)].append(f)
    return idx


def queued_tasks():
    """(dataset, method, split) -> PENDING/RUNNING for array tasks in the queue.

    Uses slurm/tasks/submitted/<array job id>.tsv, written by submit_all.sh for every
    array it submits (not available for arrays submitted by older versions)."""
    try:
        out = subprocess.run(["squeue", "--me", "-r", "-h", "-o", "%i %T"],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return {}
    lists, res = {}, {}
    for line in out.splitlines():
        jid, _, state = line.partition(" ")
        array_id, _, task = jid.partition("_")
        if not task.isdigit():
            continue
        if array_id not in lists:
            path = f"slurm/tasks/submitted/{array_id}.tsv"
            lists[array_id] = open(path).read().splitlines() if os.path.exists(path) else None
        rows = lists[array_id]
        if rows is None or int(task) >= len(rows):
            continue
        parts = rows[int(task)].split("\t")
        key = (parts[0], parts[2], parts[3]) if len(parts) == 4 else (parts[0], R_METHOD, parts[1])
        res[key] = state
    return res


def job_id_from_log(path):
    """slurm/logs/select_<array>_<task>.out -> '<array>_<task>'."""
    m = re.search(r"_(\d+)_(\d+)\.out$", path or "")
    return f"{m.group(1)}_{m.group(2)}" if m else None


def sacct_states(job_ids):
    """SLURM state per job id (RUNNING, COMPLETED, TIMEOUT, ...); {} if sacct fails."""
    states, ids = {}, sorted({j for j in job_ids if j})
    for i in range(0, len(ids), 200):
        try:
            out = subprocess.run(["sacct", "-n", "-X", "-P", "-o", "JobID,State", "-j",
                                  ",".join(ids[i:i + 200])], capture_output=True, text=True,
                                 timeout=60).stdout
        except Exception:
            return states
        for line in out.splitlines():
            jid, _, state = line.partition("|")
            states[jid.strip()] = state.split()[0] if state else ""
    return states


EPILOGUE = re.compile(r"^#{20,}\s*$", re.M)  # cluster job-summary banner (e.g. Habrok)


def program_output(path):
    """Log text without the job-summary banner some clusters append at the end."""
    text = open(path, errors="replace").read()
    m = EPILOGUE.search(text)
    return text[:m.start()] if m else text


def reason_from_log(path, state=None):
    if path is None:
        return "no log (pending, never started, or logs deleted)", ""
    text = program_output(path)
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    last = lines[-1][:160] if lines else "(empty log)"
    for label, rx in REASONS:
        m = rx.search(text)
        if m:  # show the line that matched, not the last line
            start = text.rfind("\n", 0, m.start()) + 1
            end = text.find("\n", m.end())
            return label, text[start:end if end >= 0 else None].strip()[:160]
    if state in ("RUNNING", "PENDING", "REQUEUED", "RESIZING", "SUSPENDED"):
        return f"still {state.lower()}", last
    if state:
        return f"job {state} without writing a record (see last log line)", last
    return "no error in log (running, or ended silently; see last log line)", last


def queued_jobs():
    try:
        out = subprocess.run(["squeue", "--me", "-h", "-o", "%j"], capture_output=True,
                             text=True, timeout=30).stdout.split()
        return Counter(out)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--details", action="store_true", help="list every failed/missing task")
    ap.add_argument("--reset-failed", action="store_true",
                    help="delete failed records so submit_all.sh resubmits them")
    args = ap.parse_args()

    datasets = [(d, "reg") for d in REGRESSION_DATASETS] + [(d, "surv") for d in SURVIVAL_DATASETS]
    if args.datasets:
        datasets = [(d, t) for d, t in datasets if d in args.datasets]

    table = defaultdict(Counter)            # (ds, method) -> Counter(ok/failed/missing)
    failed, missing, not_started = [], [], []
    for ds, m, sid, problem in expected_tasks(datasets):
        if problem:
            not_started.append(ds)
            continue
        path = record_path(ds, m, sid)
        if not os.path.exists(path):
            table[(ds, m)]["missing"] += 1
            missing.append((ds, m, sid))
            continue
        rec = json.load(open(path))
        if rec.get("status") == "ok":
            table[(ds, m)]["ok"] += 1
        else:
            table[(ds, m)]["failed"] += 1
            err = (rec.get("error") or "").strip().splitlines()
            failed.append((ds, m, sid, err[-1][:160] if err else "?", path))

    n_ok = sum(c["ok"] for c in table.values())
    print(f"selection records: {n_ok} ok, {len(failed)} failed, {len(missing)} missing"
          + (f"; not submitted: {', '.join(not_started)}" if not_started else ""))
    q = queued_jobs()
    if q:
        print("still in the queue (squeue): " + ", ".join(f"{n} {k}" for k, n in q.items()))

    bad = {k: c for k, c in table.items() if c["failed"] or c["missing"]}
    if bad:
        print("\ndataset / method with problems (ok / failed / missing):")
        for (ds, m), c in sorted(bad.items()):
            print(f"  {ds:20s} {m:22s} {c['ok']:3d} / {c['failed']:3d} / {c['missing']:3d}")

    if failed:
        print("\nFAILED (method raised an error; SLURM shows these as COMPLETED):")
        for (m, err), n in Counter((m, e) for _, m, _, e, _ in failed).most_common():
            print(f"  {n:4d}x {m}: {err}")
        if args.details:
            for ds, m, sid, err, path in failed:
                print(f"    {ds} {m} {sid}: {path}")

    if missing:
        logs = index_logs()
        queued = queued_tasks()
        states = sacct_states([job_id_from_log(f) for k in missing for f in logs.get(k, [])])
        active = ("RUNNING", "PENDING", "REQUEUED", "RESIZING", "SUSPENDED")
        reasons = defaultdict(list)
        for key in missing:
            files = logs.get(key, [])
            running = [f for f in files if states.get(job_id_from_log(f)) in active]
            if key in queued:
                label, log, last = f"queued ({queued[key].lower()})", (files or [None])[-1], ""
            elif running:  # an earlier or duplicate copy may have a newer log
                log = running[-1]
                label, last = reason_from_log(log, states.get(job_id_from_log(log)))
            else:
                log = files[-1] if files else None
                label, last = reason_from_log(log, states.get(job_id_from_log(log)))
            reasons[label].append((key, log, last))
        print("\nMISSING (no record), by reason found in the job log:")
        for label, items in sorted(reasons.items(), key=lambda x: -len(x[1])):
            by_method = Counter(m for (_, m, _), _, _ in items)
            print(f"  {len(items):4d}x {label}: "
                  + ", ".join(f"{m} ({n})" for m, n in by_method.most_common()))
            for (ds, m, sid), log, last in items[: (len(items) if args.details else 3)]:
                print(f"         {ds} {m} {sid}  {log or ''}  {last}")

    for kind, pattern in (("evaluation", "evaluate"), ("stability", "stability")):
        for f in glob.glob(f"slurm/logs/{pattern}_*.out"):
            text = program_output(f)
            for label, rx in REASONS:
                if rx.search(text):
                    print(f"\n{kind} job problem ({label}): {f}")
                    break

    if args.reset_failed and failed:
        for *_, path in failed:
            os.remove(path)
        print(f"\ndeleted {len(failed)} failed records; run submit_all.sh to resubmit them")

    if failed or missing:
        print("\nnext: missing tasks are resubmitted by running submit_all.sh again; "
              "for failed ones fix the cause, then --reset-failed and submit_all.sh."
              "\nfor time limit / out of memory raise --time / --mem in slurm/select.sbatch "
              "(or r_baseline.sbatch) first.")
    else:
        print("\nall expected selection records are present and ok.")


if __name__ == "__main__":
    main()
