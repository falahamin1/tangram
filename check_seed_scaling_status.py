"""
Run this ON THE CLUSTER, from inside the repo directory you want to check
(tangram-git or tangram-easy), e.g.:

    cd /projects/amfa5003/tangram && python3 check_seed_scaling_status.py
    cd /projects/amfa5003/tangram-easy && python3 check_seed_scaling_status.py

Checks the 10-seed scaling target (seeds 0-9, all 5 methods) against what's
actually on disk: policies/*.pth (done), checkpoints/ progress (in flight),
and the matching *.out log's tail (to tell "still running" apart from
"crashed/timed out and needs resubmitting"). Does not run anything, does not
touch training code -- read-only status check.
"""
import glob
import os
import re

METHODS = ['hrep', 'vrep', 'gnn', 'mlp', 'cnn']
TARGET_SEEDS = list(range(10))

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
POLICY_DIR = os.path.join(REPO_DIR, 'policies')
CKPT_BASE = os.path.join(REPO_DIR, 'checkpoints')

CRASH_MARKERS = ('Traceback (most recent call last)', 'CANCELLED', 'error:', 'Error:')
DONE_MARKER = '] Policy saved'


def latest_checkpoint_episode(method, seed):
    ckpt_dir = os.path.join(CKPT_BASE, method) if seed == 0 else os.path.join(CKPT_BASE, method, f'seed{seed}')
    eps = [int(m.group(1)) for f in glob.glob(os.path.join(ckpt_dir, 'checkpoint_ep*.pth'))
           for m in [re.search(r'ep(\d+)', f)] if m]
    return max(eps) if eps else None


def latest_log_for(method, seed):
    """Most recently modified .out file whose name matches this (method, seed)
    -- covers both the {prefix}-{method}-s{seed}.JOBID.out (single-job) and
    {method}_seed{seed}.JOBID.out naming conventions used across these repos."""
    candidates = (
        glob.glob(os.path.join(REPO_DIR, f'*-{method}-s{seed}.*.out'))
        + glob.glob(os.path.join(REPO_DIR, f'{method}_seed{seed}.*.out'))
        + glob.glob(os.path.join(REPO_DIR, 'logs', f'*-{method}-s{seed}.*.out'))
        + glob.glob(os.path.join(REPO_DIR, 'logs', f'{method}_seed{seed}.*.out'))
    )
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def classify(method, seed):
    pth = os.path.join(POLICY_DIR, f'{method}_seed{seed}.pth')
    if os.path.exists(pth):
        return 'DONE', '-'

    ckpt_ep = latest_checkpoint_episode(method, seed)
    log_path = latest_log_for(method, seed)
    log_text = ''
    if log_path:
        try:
            with open(log_path, errors='ignore') as f:
                log_text = f.read()
        except OSError:
            pass

    if DONE_MARKER in log_text:
        # Log says it finished but no .pth on disk -- inconsistent, flag it.
        return 'INCONSISTENT (log says done, no .pth)', str(ckpt_ep) if ckpt_ep else '-'

    if any(marker in log_text for marker in CRASH_MARKERS):
        return 'NEEDS RERUN (crashed/cancelled)', str(ckpt_ep) if ckpt_ep else '-'

    if ckpt_ep is not None:
        return 'RUNNING (checkpoint growing)', str(ckpt_ep)

    if log_path is not None:
        return 'RUNNING (no checkpoint yet)', '-'

    return 'NOT SUBMITTED / NOT STARTED', '-'


def main():
    print(f"{'method':8s} {'seed':5s} {'status':32s} {'latest_ckpt_ep'}")
    print('-' * 70)
    totals_done = {m: 0 for m in METHODS}
    needs_rerun = []
    for method in METHODS:
        for seed in TARGET_SEEDS:
            status, ckpt_ep = classify(method, seed)
            print(f"{method:8s} {seed:<5d} {status:32s} {ckpt_ep}")
            if status == 'DONE':
                totals_done[method] += 1
            if status.startswith('NEEDS RERUN') or status.startswith('INCONSISTENT'):
                needs_rerun.append((method, seed, status))

    print()
    print("Summary (done / 10 target seeds):")
    for method in METHODS:
        print(f"  {method}: {totals_done[method]}/10")

    if needs_rerun:
        print()
        print("Combos that likely need resubmitting:")
        for method, seed, status in needs_rerun:
            print(f"  {method} seed={seed}: {status}")
    else:
        print()
        print("No combos flagged as crashed/cancelled -- anything not DONE is either still running or not yet submitted.")


if __name__ == '__main__':
    main()
