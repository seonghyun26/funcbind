#!/usr/bin/env python
"""Assemble a VoxBind-docking-driver eval tree from MCP sampling chunks.

run_docking_eval*.py finds its receptor by globbing "*_pocket10.pdb" in each
target dir and its reference ligand by "*_ref.sdf". MCP targets have neither
name: the receptor is the whole deposited structure (<pdb>-protein.pdb) and the
reference is the crystal cyclic peptide (<pdb>-CP.sdf). So this copies both under
the names the driver expects -- the "pocket10" in the filename is a naming
convention here, not a crop (verified byte-identical to <pdb>-protein.pdb for the
2026-08-20 cmp10 tree).

The target index -> PDB mapping is read from each chunk's own run.log
("| sampling receptor mcpp_dataset/1bm2/1bm2-protein.pdb #0") rather than from a
hardcoded list, so a chunk that sampled a different id set still lands correctly.

  python scripts/build_mcpp_eval_tree.py \
      --chunks artifacts/reproduction/mcpp/cmp10_r10/ft_{a,b,c,d} \
      --out    artifacts/reproduction/mcpp/cmp10_r10/_eval/finetuned
"""
import argparse, os, re, shutil, sys

DATA = os.environ.get("FUNCBIND_ROOT", "/home1/irteam/funcbind") + "/funcbind/dataset/data"
LINE = re.compile(r"sampling receptor\s+(\S+?)/([^/]+)-protein\.pdb\s+#(\d+)")


def mapping_from_log(log_path):
    """{target_index: pdb_id} for one chunk."""
    out = {}
    with open(log_path, errors="replace") as fh:
        for line in fh:
            m = LINE.search(line)
            if m:
                out[int(m.group(3))] = m.group(2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", nargs="+", required=True,
                    help="sampling chunk dirs (each holds run.log and samples/target_XX/)")
    ap.add_argument("--out", required=True, help="eval tree root to create")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n_ok, n_skip = 0, 0
    for chunk in args.chunks:
        log = os.path.join(chunk, "run.log")
        if not os.path.exists(log):
            print(f"!! no run.log in {chunk}", file=sys.stderr)
            continue
        idx2pdb = mapping_from_log(log)
        for idx, pdb in sorted(idx2pdb.items()):
            src_sdf = os.path.join(chunk, "samples", f"target_{idx:02d}", "samples.sdf")
            if not os.path.exists(src_sdf) or os.path.getsize(src_sdf) == 0:
                # A target that yielded nothing writes no samples.sdf; docking it
                # would create an empty result that reads as a failure downstream.
                print(f"-- target_{idx:02d} ({pdb}): no samples, skipped")
                n_skip += 1
                continue
            rec = os.path.join(DATA, "mcpp_dataset", pdb, f"{pdb}-protein.pdb")
            ref = os.path.join(DATA, "mcpp_dataset", pdb, f"{pdb}-CP.sdf")
            for p in (rec, ref):
                if not os.path.exists(p):
                    raise FileNotFoundError(p)
            tdir = os.path.join(args.out, f"target_{idx:02d}")
            os.makedirs(tdir, exist_ok=True)
            shutil.copy2(src_sdf, os.path.join(tdir, "samples.sdf"))
            shutil.copy2(rec, os.path.join(tdir, f"{pdb}_pocket10.pdb"))
            shutil.copy2(ref, os.path.join(tdir, f"{pdb}_ref.sdf"))
            print(f"++ target_{idx:02d} ({pdb}) <- {chunk}")
            n_ok += 1
    print(f"\n{n_ok} targets assembled, {n_skip} skipped -> {args.out}")


if __name__ == "__main__":
    main()
