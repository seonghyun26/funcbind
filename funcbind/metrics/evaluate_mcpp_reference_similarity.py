#!/usr/bin/env python
"""Measure whether generated MCPs copy the ligand visible in a holo density map.

Reports whole-molecule Morgan-fingerprint Tanimoto similarity and, when both inputs are
PDB files, best forward/reverse global residue identity.  This is deliberately separate
from binding/pose metrics: strong density-conditioned generation should improve the
protein-surface interaction without merely reconstructing the deposited CP ligand.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def load_molecule(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
        molecule = next((mol for mol in supplier if mol is not None), None)
    elif suffix == ".pdb":
        molecule = Chem.MolFromPDBFile(str(path), removeHs=False, sanitize=True)
    else:
        molecule = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=True)
    if molecule is None:
        raise ValueError(f"RDKit could not read {path}")
    return molecule


def morgan_tanimoto(reference, generated, radius=2, n_bits=2048) -> float:
    ref_fp = AllChem.GetMorganFingerprintAsBitVect(reference, radius, nBits=n_bits)
    gen_fp = AllChem.GetMorganFingerprintAsBitVect(generated, radius, nBits=n_bits)
    return float(DataStructs.TanimotoSimilarity(ref_fp, gen_fp))


def pdb_residue_sequence(path: Path) -> list[str]:
    sequence = []
    seen = set()
    with path.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            key = (line[21], line[22:26], line[26])
            if key in seen:
                continue
            seen.add(key)
            sequence.append(line[17:20].strip().upper())
    return sequence


def global_residue_identity(a: list[str], b: list[str]) -> float:
    """Needleman-Wunsch identity: matches divided by aligned non-gap length."""
    if not a or not b:
        return float("nan")
    rows, cols = len(a) + 1, len(b) + 1
    score = np.zeros((rows, cols), dtype=np.int32)
    matches = np.zeros((rows, cols), dtype=np.int32)
    aligned = np.zeros((rows, cols), dtype=np.int32)
    score[:, 0] = -np.arange(rows)
    score[0, :] = -np.arange(cols)
    aligned[:, 0] = np.arange(rows)
    aligned[0, :] = np.arange(cols)
    for i in range(1, rows):
        for j in range(1, cols):
            candidates = (
                (score[i - 1, j - 1] + (1 if a[i - 1] == b[j - 1] else -1),
                 matches[i - 1, j - 1] + int(a[i - 1] == b[j - 1]),
                 aligned[i - 1, j - 1] + 1),
                (score[i - 1, j] - 1, matches[i - 1, j], aligned[i - 1, j] + 1),
                (score[i, j - 1] - 1, matches[i, j - 1], aligned[i, j - 1] + 1),
            )
            best = max(candidates, key=lambda item: (item[0], item[1], -item[2]))
            score[i, j], matches[i, j], aligned[i, j] = best
    return float(matches[-1, -1] / max(aligned[-1, -1], 1))


def reference_record(manifest: Path, target_id: str) -> dict:
    records = json.loads(manifest.read_text())
    for record in records:
        if record["target_id"].lower() == target_id.lower():
            return record
    raise KeyError(f"target {target_id!r} is absent from {manifest}")


def evaluate(manifest: Path, target_id: str, generated_paths: list[Path]):
    record = reference_record(manifest, target_id)
    reference_path = Path(record["reference_ligand"])
    reference_pdb = Path(record.get("reference_ligand_pdb", reference_path))
    reference_molecule = load_molecule(reference_path)
    reference_sequence = pdb_residue_sequence(reference_pdb) if reference_pdb.exists() else []

    rows = []
    for generated_path in generated_paths:
        try:
            generated_molecule = load_molecule(generated_path)
            tanimoto = morgan_tanimoto(reference_molecule, generated_molecule)
            identity = float("nan")
            if generated_path.suffix.lower() == ".pdb" and reference_sequence:
                generated_sequence = pdb_residue_sequence(generated_path)
                identity = max(
                    global_residue_identity(reference_sequence, generated_sequence),
                    global_residue_identity(reference_sequence, generated_sequence[::-1]),
                )
            rows.append(dict(
                target_id=target_id.lower(),
                generated_path=str(generated_path),
                reference_path=str(reference_path),
                morgan_tanimoto=tanimoto,
                best_forward_reverse_residue_identity=identity,
                status="ok",
            ))
        except Exception as error:
            rows.append(dict(
                target_id=target_id.lower(),
                generated_path=str(generated_path),
                reference_path=str(reference_path),
                morgan_tanimoto=float("nan"),
                best_forward_reverse_residue_identity=float("nan"),
                status=f"{type(error).__name__}: {error}",
            ))
    valid = [row for row in rows if row["status"] == "ok"]
    values = np.asarray([row["morgan_tanimoto"] for row in valid], dtype=float)
    identities = np.asarray(
        [row["best_forward_reverse_residue_identity"] for row in valid], dtype=float
    )
    summary = dict(
        target_id=target_id.lower(),
        reference_path=str(reference_path),
        n_input=len(rows),
        n_valid=len(valid),
        morgan_tanimoto_mean=float(np.nanmean(values)) if values.size else None,
        morgan_tanimoto_median=float(np.nanmedian(values)) if values.size else None,
        morgan_tanimoto_max=float(np.nanmax(values)) if values.size else None,
        fraction_morgan_tanimoto_lt_0_4=(
            float(np.mean(values < 0.4)) if values.size else None
        ),
        residue_identity_mean=(
            float(np.nanmean(identities)) if np.isfinite(identities).any() else None
        ),
        fraction_residue_identity_lt_0_5=(
            float(np.mean(identities[np.isfinite(identities)] < 0.5))
            if np.isfinite(identities).any() else None
        ),
    )
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--generated", type=Path, nargs="+", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    rows, summary = evaluate(args.manifest, args.target_id, args.generated)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["status"])
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
