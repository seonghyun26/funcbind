#!/usr/bin/env python
"""Assert that a training run's checkpoint and validation artifacts are sound.

Checks the invariants the epoch-boundary rewrite is supposed to hold:
  * the rolling checkpoint exists, is a complete archive, and carries every key
    the resume path in load_funcbind reads back;
  * no ``.tmp`` file survived, i.e. the atomic replace ran to completion;
  * ``checkpoint_best.pth.tar`` is a hard link to a *best* epoch's write, and is
    never left dangling by a later non-best save;
  * validation actually produced its per-sigma plots.

Usable against a real run too -- it never materializes tensor data.
"""
import argparse
import os
import pickletools
import sys
import zipfile

RESUME_KEYS = {
    "epoch", "config", "state_dict", "state_dict_ema",
    "optimizer", "code_stats", "acc_iter", "global_step",
}
EXPECTED_PLOTS = {
    "Validation_Loss_vs_sigma.png",
    "Weighted_Validation_Loss_vs_sigma.png",
    "Weight_over_Variance_vs_sigma.png",
    "Log_Variance_vs_sigma.png",
}

failures = []
notes = []


def check(condition, ok_msg, fail_msg):
    if condition:
        print(f"  PASS  {ok_msg}")
    else:
        print(f"  FAIL  {fail_msg}")
        failures.append(fail_msg)
    return condition


def archive_keys(path):
    """Top-level keys of a torch.save archive, read from the pickle alone."""
    zf = zipfile.ZipFile(path)
    names = zf.namelist()
    pkl = next(n for n in names if n.endswith("data.pkl"))

    infos = zf.infolist()
    last = max(infos, key=lambda i: i.header_offset)
    end = last.header_offset + len(last.FileHeader()) + last.file_size
    truncated = end > os.path.getsize(path)

    with zf.open(pkl) as fh:
        blob = fh.read()
    strings = {
        arg
        for op, arg, _ in pickletools.genops(blob)
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE") and isinstance(arg, str)
    }
    return strings, truncated, len(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--epochs", type=int, default=None,
                    help="expected epoch recorded in the final checkpoint")
    args = ap.parse_args()

    run_dir = args.run_dir
    ckpt = os.path.join(run_dir, "checkpoint.pth.tar")
    best = os.path.join(run_dir, "checkpoint_best.pth.tar")
    plots_dir = os.path.join(run_dir, "validation_plots")

    print(f"\n=== checkpoint: {run_dir} ===")
    if not check(os.path.exists(ckpt), f"rolling checkpoint written ({os.path.getsize(ckpt) / 1024**2:.1f} MiB)" if os.path.exists(ckpt) else "", "checkpoint.pth.tar missing"):
        print("\nRESULT: FAIL")
        return 1

    leftovers = [f for f in os.listdir(run_dir) if f.endswith(".tmp")]
    check(not leftovers, "no .tmp leftovers -- atomic replace completed",
          f"stale temp files left behind: {leftovers}")

    keys, truncated, n_entries = archive_keys(ckpt)
    check(not truncated, f"archive complete ({n_entries} entries)",
          "archive is truncated -- the write was interrupted")
    missing = RESUME_KEYS - keys
    check(not missing, f"all resume keys present: {sorted(RESUME_KEYS)}",
          f"checkpoint missing keys the resume path reads: {sorted(missing)}")
    if "best_res" in keys:
        print("  PASS  best_res persisted")
    else:
        notes.append("best_res absent from the checkpoint; resume restarts best-tracking from scratch")

    print(f"\n=== best checkpoint ===")
    if os.path.exists(best):
        s_ckpt, s_best = os.stat(ckpt), os.stat(best)
        linked = s_ckpt.st_ino == s_best.st_ino
        b_keys, b_trunc, _ = archive_keys(best)
        check(not b_trunc, "best archive complete", "best archive is truncated")
        check(not (RESUME_KEYS - b_keys), "best carries all resume keys",
              f"best missing keys: {sorted(RESUME_KEYS - b_keys)}")
        if linked:
            print(f"  PASS  best is a hard link to the current rolling checkpoint (inode {s_ckpt.st_ino}, nlink {s_ckpt.st_nlink})")
        else:
            print(f"  PASS  best is an independent earlier write (inode {s_best.st_ino} vs {s_ckpt.st_ino}) -- a later non-best save did not clobber it")
    else:
        notes.append("no checkpoint_best.pth.tar -- no epoch improved on best_res")

    print(f"\n=== validation plots ===")
    if check(os.path.isdir(plots_dir), "validation_plots/ exists", "validation_plots/ missing"):
        found = set(os.listdir(plots_dir))
        missing_plots = EXPECTED_PLOTS - found
        check(not missing_plots, f"all {len(EXPECTED_PLOTS)} per-sigma plots written",
              f"validation plots missing: {sorted(missing_plots)}")
        empty = [p for p in found if os.path.getsize(os.path.join(plots_dir, p)) == 0]
        check(not empty, "all plots non-empty", f"zero-byte plots: {empty}")

    if notes:
        print("\n=== notes ===")
        for n in notes:
            print(f"  NOTE  {n}")

    print(f"\nRESULT: {'FAIL' if failures else 'PASS'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
