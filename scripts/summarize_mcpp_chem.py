#!/usr/bin/env python
"""Yield + chemistry for an MCP eval tree, without docking.

Vina docking of 48-55 heavy-atom macrocycles ran 8 h for the 2026-08-20 cmp10
comparison, so this reports everything that does NOT need it: how many molecules
each target produced, validity/uniqueness/diversity, QED/SA/logP/Lipinski, heavy
atom count, and similarity to the crystal cyclic peptide.

The numbers come from run_docking_eval.py's own helpers rather than a second
implementation, so they line up with the eval_docking_results.json summaries the
docking run writes later (and with the cmp10 tables).

  /opt/conda/envs/voxdock/bin/python scripts/summarize_mcpp_chem.py <eval_root> \
      [--compare <other_eval_root> ...] [--out summary.json]
"""
import argparse, json, os, sys, glob

sys.path.insert(0, "/home1/irteam/VoxBind/voxbind/exps/frozenenc_probes")
import numpy as np
from rdkit import Chem
from run_docking_eval import (load_connected, n_raw, diversity, ref_similarity,
                              find_ref_ligand, get_chem)


def chem_for_target(tdir):
    sdf = os.path.join(tdir, "samples.sdf")
    valid = load_connected(sdf)
    mols = [m for m, _ in valid]
    smis = [s for _, s in valid]
    ntot = n_raw(sdf)
    if not mols:
        return dict(target=os.path.basename(tdir), n_total=ntot, n_valid=0)
    chem = [get_chem(m) for m in mols]
    entry = dict(
        target=os.path.basename(tdir), n_total=ntot, n_valid=len(mols),
        n_unique=len(set(smis)),
        validity=len(mols) / ntot if ntot else 0.0,
        uniqueness=len(set(smis)) / len(mols),
        diversity=diversity(mols),
        qed=float(np.mean([c["qed"] for c in chem])),
        sa=float(np.mean([c["sa"] for c in chem])),
        logp=float(np.mean([c["logp"] for c in chem])),
        lipinski=float(np.mean([c["lipinski"] for c in chem])),
        n_atoms=float(np.mean([m.GetNumHeavyAtoms() for m in mols])),
    )
    ref = find_ref_ligand(tdir)
    if ref is not None:
        rmol = next(iter(Chem.SDMolSupplier(ref, removeHs=False, sanitize=True)), None)
        if rmol is not None:
            entry.update(ref_similarity(mols, rmol))
    return entry


def summarize(root):
    tdirs = sorted(d for d in glob.glob(os.path.join(root, "target_*")) if os.path.isdir(d))
    per = [chem_for_target(td) for td in tdirs]
    keyed = [e for e in per if e.get("n_valid")]
    # Target means, matching run_docking_eval.py's summary (mean over targets, not
    # over molecules) so the two reports can be read side by side.
    def m(k):
        # NaN, not None, is how a one-molecule target reports diversity (no pair to
        # compare), and one of those poisons the whole mean. run_docking_eval.py's
        # own agg() drops them; match it so the two summaries agree.
        vals = [e[k] for e in keyed if e.get(k) is not None
                and not (isinstance(e[k], float) and np.isnan(e[k]))]
        return float(np.mean(vals)) if vals else None
    summary = dict(
        n_targets=len(per), n_with_mols=len(keyed),
        n_molecules=int(sum(e.get("n_valid", 0) for e in per)),
        **{k: m(k) for k in ("validity", "uniqueness", "diversity", "qed", "sa",
                             "logp", "lipinski", "n_atoms", "ref_tanimoto_mean",
                             "ref_tanimoto_max", "ref_tanimoto_rdk_mean")},
    )
    return summary, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--compare", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Two arms of different runs share their last two path components
    # (cmp10/_eval/finetuned vs cmp10_r10/_eval/finetuned), so a two-component
    # label silently collapses them into one dict entry and the second overwrites
    # the first. Label with the run dir, which is the part that actually differs.
    arms = {}
    for root in [args.root] + args.compare:
        parts = os.path.normpath(os.path.abspath(root)).split(os.sep)
        label = "/".join(parts[-3:])
        while label in arms:
            label += "'"
        arms[label] = summarize(root)

    names = list(arms)
    print("\n=== per-target molecule count ===")
    all_t = sorted({e["target"] for _, per in arms.values() for e in per})
    print(f"{'target':10s}" + "".join(f"{n[-18:]:>20s}" for n in names))
    for t in all_t:
        row = f"{t:10s}"
        for n in names:
            e = next((x for x in arms[n][1] if x["target"] == t), None)
            row += f"{(e['n_valid'] if e else 0):>20d}"
        print(row)
    print(f"{'TOTAL':10s}" + "".join(f"{arms[n][0]['n_molecules']:>20d}" for n in names))

    print("\n=== chemistry (mean over targets that produced molecules) ===")
    keys = ["n_with_mols", "validity", "uniqueness", "diversity", "qed", "sa",
            "logp", "lipinski", "n_atoms", "ref_tanimoto_mean", "ref_tanimoto_max"]
    print(f"{'metric':22s}" + "".join(f"{n[-18:]:>20s}" for n in names))
    for k in keys:
        row = f"{k:22s}"
        for n in names:
            v = arms[n][0].get(k)
            row += f"{'-':>20s}" if v is None else (f"{v:>20d}" if isinstance(v, int) else f"{v:>20.3f}")
        print(row)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({n: dict(summary=s, per_target=p) for n, (s, p) in arms.items()}, fh, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
