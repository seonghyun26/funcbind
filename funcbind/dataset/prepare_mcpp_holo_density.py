#!/usr/bin/env python
"""Prepare receptor-level deposited holo 2Fo-Fc density for FuncBind MCP.

One raw box is cached per MCP target/receptor and shared by every generated MCP
conformer.  The deposited reference ligand is intentionally retained in the map; this
is a holo-density experiment, not an apo-like ligand-erasure experiment.

The script is resumable. It downloads official deposited PDB coordinates from RCSB and
2Fo-Fc CCP4 maps from PDBe, verifies the MCP receptor frame with a heavy-atom Kabsch
fit, and writes a fork-safe float16 memmap plus manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import gemmi
import numpy as np
import scipy.ndimage


DEFAULT_MCPP = Path(__file__).resolve().parent / "data" / "mcpp_dataset"
DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "mcpp_holo_xray_v1"
DEFAULT_RECIPE = Path(os.environ.get("VOXBIND_PYTHON_ROOT", "/home1/irteam/VoxBind")) / (
    "voxbind/dataset/data/pretrain/xray_resample_plinder_v2p1/resample.json"
)
PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
MAP_URL = "https://www.ebi.ac.uk/pdbe/entry-files/{pdb_id}.ccp4"


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def discover_targets(data_root: Path) -> list[str]:
    targets = []
    for path in data_root.iterdir():
        if not path.is_dir() or len(path.name) != 4 or not path.name.isalnum():
            continue
        pdb_id = path.name.lower()
        if (path / f"{pdb_id}-protein.pdb").exists():
            targets.append(pdb_id)
    return sorted(targets)


def resolve_structure_root(data_root: Path) -> tuple[Path, list[str]]:
    """Accept both archive layouts: <root>/<pdb> and <root>/mcpp_dataset/<pdb>."""
    candidates = (data_root, data_root / "mcpp_dataset")
    for candidate in candidates:
        if candidate.is_dir():
            targets = discover_targets(candidate)
            if targets:
                return candidate, targets
    return data_root, []


def _download(url: str, destination: Path) -> tuple[bool, str]:
    if destination.exists() and destination.stat().st_size > 1024:
        return True, "cached"
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "FuncBind-MCP-density/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            with temporary.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
        if temporary.stat().st_size <= 1024:
            temporary.unlink(missing_ok=True)
            return False, "empty_download"
        os.replace(temporary, destination)
        return True, "downloaded"
    except urllib.error.HTTPError as error:
        temporary.unlink(missing_ok=True)
        return False, f"http_{error.code}"
    except Exception as error:
        temporary.unlink(missing_ok=True)
        return False, f"{type(error).__name__}: {error}"


def download_target(pdb_id: str, pdb_dir: Path, map_dir: Path) -> dict:
    pdb_ok, pdb_status = _download(
        PDB_URL.format(pdb_id=pdb_id.upper()), pdb_dir / f"{pdb_id}.pdb"
    )
    if not pdb_ok:
        cif_ok, cif_status = _download(
            CIF_URL.format(pdb_id=pdb_id.upper()), pdb_dir / f"{pdb_id}.cif"
        )
        pdb_ok = cif_ok
        pdb_status = f"pdb:{pdb_status};cif:{cif_status}"
    map_ok, map_status = _download(
        MAP_URL.format(pdb_id=pdb_id), map_dir / f"{pdb_id}.ccp4"
    )
    return dict(
        target_id=pdb_id,
        pdb_ok=pdb_ok,
        map_ok=map_ok,
        pdb_status=pdb_status,
        map_status=map_status,
    )


def download_all(targets: list[str], pdb_dir: Path, map_dir: Path, workers: int) -> dict:
    statuses = {}
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_target, target, pdb_dir, map_dir): target
            for target in targets
        }
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            statuses[result["target_id"]] = result
            if done % 50 == 0 or done == len(futures):
                n_map = sum(item["map_ok"] for item in statuses.values())
                print(
                    f"[download] {done:,}/{len(futures):,}; maps={n_map:,}; "
                    f"elapsed={time.time() - started:.0f}s",
                    flush=True,
                )
    return statuses


def _protein_atoms(path: Path, per_chain: bool = False):
    structure = gemmi.read_structure(str(path))
    result = defaultdict(dict) if per_chain else {}
    for chain in structure[0]:
        for residue in chain:
            if residue.is_water() or residue.het_flag == "H":
                continue
            for atom in residue:
                if atom.is_hydrogen():
                    continue
                key_no_chain = (residue.seqid.num, residue.seqid.icode, atom.name.strip())
                xyz = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64)
                if per_chain:
                    result[chain.name][key_no_chain] = xyz
                else:
                    result[(chain.name, *key_no_chain)] = xyz
    return result


def _heavy_atom_coords(path: Path) -> np.ndarray:
    structure = gemmi.read_structure(str(path))
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.is_water():
                    continue
                for atom in residue:
                    if not atom.is_hydrogen():
                        coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
        break
    if not coords:
        raise ValueError(f"no heavy atoms in {path}")
    return np.asarray(coords, dtype=np.float64)


def kabsch(local: np.ndarray, deposited: np.ndarray):
    local_mean = local.mean(0)
    deposited_mean = deposited.mean(0)
    u, _s, vt = np.linalg.svd((local - local_mean).T @ (deposited - deposited_mean))
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    translation = deposited_mean - local_mean @ rotation.T
    aligned = local @ rotation.T + translation
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - deposited) ** 2, axis=1))))
    return rotation, translation, rmsd


def _protein_residues(path: Path):
    """Protein residues and heavy atoms by chain, independent of residue numbering."""
    structure = gemmi.read_structure(str(path))
    chains = defaultdict(list)
    for chain in structure[0]:
        for residue in chain:
            if residue.is_water() or residue.het_flag == "H":
                continue
            atoms = {
                atom.name.strip(): np.array(
                    [atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64
                )
                for atom in residue
                if not atom.is_hydrogen()
            }
            if atoms:
                chains[chain.name].append((residue.name, atoms))
    return chains


def _align_residue_names(local, deposited):
    """Global sequence alignment returning matching residue-index pairs."""
    n, m = len(local), len(deposited)
    score = np.zeros((n + 1, m + 1), dtype=np.int32)
    trace = np.zeros((n + 1, m + 1), dtype=np.int8)
    score[:, 0] = -2 * np.arange(n + 1)
    score[0, :] = -2 * np.arange(m + 1)
    trace[1:, 0] = 1
    trace[0, 1:] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = (
                score[i - 1, j - 1] + (2 if local[i - 1] == deposited[j - 1] else -1),
                score[i - 1, j] - 2,
                score[i, j - 1] - 2,
            )
            move = int(np.argmax(choices))
            score[i, j] = choices[move]
            trace[i, j] = move
    pairs = []
    i, j = n, m
    while i or j:
        move = int(trace[i, j])
        if i and j and move == 0:
            if local[i - 1] == deposited[j - 1]:
                pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i and (j == 0 or move == 1):
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def _sequence_matched_atoms(local_residues, deposited_residues):
    pairs = _align_residue_names(
        [item[0] for item in local_residues],
        [item[0] for item in deposited_residues],
    )
    local_atoms, deposited_atoms = [], []
    for local_index, deposited_index in pairs:
        local_dict = local_residues[local_index][1]
        deposited_dict = deposited_residues[deposited_index][1]
        for atom_name in sorted(set(local_dict) & set(deposited_dict)):
            local_atoms.append(local_dict[atom_name])
            deposited_atoms.append(deposited_dict[atom_name])
    return np.asarray(local_atoms), np.asarray(deposited_atoms)


def receptor_transform(local_path: Path, deposited_path: Path):
    local = _protein_atoms(local_path, per_chain=False)
    deposited_by_chain = _protein_atoms(deposited_path, per_chain=True)

    exact_local, exact_deposited = [], []
    for key, xyz in local.items():
        chain = key[0]
        no_chain = key[1:]
        if chain in deposited_by_chain and no_chain in deposited_by_chain[chain]:
            exact_local.append(xyz)
            exact_deposited.append(deposited_by_chain[chain][no_chain])
    if len(exact_local) >= 20:
        rotation, translation, rmsd = kabsch(
            np.asarray(exact_local), np.asarray(exact_deposited)
        )
        if rmsd <= 2.0:
            return rotation, translation, rmsd, len(exact_local), "exact_chain_keys"

    # Chain labels can differ. A single well-matched chain is enough to recover the
    # global coordinate-frame transform shared by the complete deposited structure.
    local_by_chain = defaultdict(dict)
    for key, xyz in local.items():
        local_by_chain[key[0]][key[1:]] = xyz
    best = None
    for local_chain, local_atoms in local_by_chain.items():
        for deposited_chain, deposited_atoms in deposited_by_chain.items():
            keys = sorted(set(local_atoms) & set(deposited_atoms))
            if len(keys) < 20:
                continue
            p = np.asarray([local_atoms[key] for key in keys])
            q = np.asarray([deposited_atoms[key] for key in keys])
            rotation, translation, rmsd = kabsch(p, q)
            candidate = (
                rmsd <= 2.0, len(keys), -rmsd, rotation, translation,
                local_chain, deposited_chain,
            )
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    # Biological-assembly files can renumber every residue. Align residue-name
    # sequences, then fit all common heavy atoms from identical aligned residues.
    local_residue_chains = _protein_residues(local_path)
    deposited_residue_chains = _protein_residues(deposited_path)
    for local_chain, local_residues in local_residue_chains.items():
        for deposited_chain, deposited_residues in deposited_residue_chains.items():
            p, q = _sequence_matched_atoms(local_residues, deposited_residues)
            if len(p) < 20:
                continue
            rotation, translation, rmsd = kabsch(p, q)
            candidate = (
                rmsd <= 2.0, len(p), -rmsd, rotation, translation,
                f"seq:{local_chain}", deposited_chain,
            )
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    if best is None:
        raise ValueError("fewer than 20 matched receptor heavy atoms")
    return best[3], best[4], -best[2], best[1], f"chain_pair:{best[5]}->{best[6]}"


def _load_map(path: Path):
    ccp4 = gemmi.read_ccp4_map(str(path))
    ccp4.setup(float("nan"))
    grid = ccp4.grid
    array = np.array(grid, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("map contains non-finite values")
    orthogonal = np.array(grid.unit_cell.orth.mat.tolist(), dtype=np.float64)
    fractional_t = np.linalg.inv(orthogonal).T
    return array, fractional_t


def _sample_map(array: np.ndarray, fractional_t: np.ndarray, cart: np.ndarray):
    shape = np.asarray(array.shape, dtype=np.float64)
    frac = cart @ fractional_t
    indices = frac * shape.reshape(1, 3)
    return scipy.ndimage.map_coordinates(
        array,
        [indices[:, 0] % shape[0], indices[:, 1] % shape[1], indices[:, 2] % shape[2]],
        order=1,
        mode="wrap",
        prefilter=False,
    )


def crop_raw_box(array, fractional_t, center, rotation, translation, g_box, resolution):
    offsets = (np.arange(g_box, dtype=np.float32) - g_box * 0.5) * resolution
    gx, gy, gz = np.meshgrid(offsets, offsets, offsets, indexing="ij")
    local_cart = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    local_cart += np.asarray(center, dtype=np.float32).reshape(1, 3)
    deposited_cart = local_cart @ rotation.T + translation.reshape(1, 3)
    values = _sample_map(array, fractional_t, deposited_cart)
    return values.reshape(g_box, g_box, g_box).astype(np.float32)


def build_target(job):
    (
        index,
        target_id,
        n_targets,
        data_root,
        pdb_dir,
        map_dir,
        box_path,
        g_box,
        resolution,
        max_rmsd,
    ) = job
    target_dir = Path(data_root) / target_id
    local_receptor = target_dir / f"{target_id}-protein.pdb"
    reference_pdb = target_dir / f"{target_id}-CP.pdb"
    reference_sdf = target_dir / f"{target_id}-CP.sdf"
    deposited_pdb = Path(pdb_dir) / f"{target_id}.pdb"
    if not deposited_pdb.exists():
        deposited_pdb = Path(pdb_dir) / f"{target_id}.cif"
    map_path = Path(map_dir) / f"{target_id}.ccp4"
    base = dict(
        target_id=target_id,
        box_index=-1,
        protein=str(local_receptor),
        reference_ligand=str(reference_sdf if reference_sdf.exists() else reference_pdb),
        reference_ligand_pdb=str(reference_pdb),
        deposited_pdb=str(deposited_pdb),
        deposited_2fofc_map=str(map_path),
        density_kind="holo_deposited_2fofc",
    )
    if not map_path.exists():
        return {**base, "reason": "map_unavailable"}
    if not deposited_pdb.exists():
        return {**base, "reason": "deposited_pdb_unavailable"}
    if not reference_pdb.exists():
        return {**base, "reason": "reference_cp_pdb_unavailable"}

    try:
        rotation, translation, rmsd, n_match, match_kind = receptor_transform(
            local_receptor, deposited_pdb
        )
        if rmsd > max_rmsd:
            raise ValueError(f"receptor alignment RMSD {rmsd:.3f} > {max_rmsd:.3f} A")
        reference_atoms = _heavy_atom_coords(reference_pdb)
        reference_center = reference_atoms.mean(0)
        array, fractional_t = _load_map(map_path)
        box = crop_raw_box(
            array,
            fractional_t,
            reference_center,
            rotation,
            translation,
            g_box,
            resolution,
        )
        deposited_reference = reference_atoms @ rotation.T + translation
        ligand_density = _sample_map(array, fractional_t, deposited_reference)

        storage = np.memmap(
            box_path,
            dtype=np.float16,
            mode="r+",
            shape=(n_targets, g_box, g_box, g_box),
        )
        storage[index] = box.astype(np.float16)
        storage.flush()
        del storage
        return {
            **base,
            "box_index": index,
            "reason": "ok",
            "reference_center": reference_center.astype(float).tolist(),
            "local_to_deposited_R": rotation.astype(float).tolist(),
            "local_to_deposited_t": translation.astype(float).tolist(),
            "alignment_rmsd": rmsd,
            "alignment_n_match": n_match,
            "alignment_kind": match_kind,
            "reference_atom_density_mean_raw": float(np.mean(ligand_density)),
            "reference_atom_density_median_raw": float(np.median(ligand_density)),
            "raw_box_mean": float(box.mean()),
            "raw_box_std": float(box.std()),
        }
    except Exception as error:
        return {**base, "reason": f"{type(error).__name__}: {error}"}


def load_normalization(recipe_path: Path) -> dict:
    recipe = json.loads(recipe_path.read_text())
    norm = recipe.get("normalization", recipe)
    required = ("arcsinh_scale", "mu_a", "sigma_a")
    if not all(key in norm for key in required):
        raise ValueError(f"{recipe_path} is not a VoxBind arcsinh normalization recipe")
    return {
        "scheme": norm.get("scheme", "arcsinh_zscore"),
        "arcsinh_scale": float(norm["arcsinh_scale"]),
        "mu_a": float(norm["mu_a"]),
        "sigma_a": float(norm["sigma_a"]),
        "source": str(recipe_path.resolve()),
    }


def build_boxes(args, targets: list[str], statuses: dict, normalization: dict):
    box_path = args.out_dir / "boxes_float16.dat"
    n_targets = len(targets)
    expected_bytes = n_targets * args.g_box ** 3 * np.dtype(np.float16).itemsize
    if not box_path.exists() or box_path.stat().st_size != expected_bytes:
        storage = np.memmap(
            box_path,
            dtype=np.float16,
            mode="w+",
            shape=(n_targets, args.g_box, args.g_box, args.g_box),
        )
        storage.flush()
        del storage

    jobs = [
        (
            index,
            target,
            n_targets,
            str(args.data_root),
            str(args.out_dir / "pdb"),
            str(args.out_dir / "ccp4"),
            str(box_path),
            args.g_box,
            args.resolution,
            args.max_alignment_rmsd,
        )
        for index, target in enumerate(targets)
    ]
    records = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.build_workers) as executor:
        futures = {executor.submit(build_target, job): job[1] for job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            records.append(future.result())
            if done % 25 == 0 or done == len(futures):
                n_ok = sum(record["box_index"] >= 0 for record in records)
                print(
                    f"[boxes] {done:,}/{len(futures):,}; usable={n_ok:,}; "
                    f"elapsed={time.time() - started:.0f}s",
                    flush=True,
                )
    records.sort(key=lambda record: record["target_id"])
    n_ok = sum(record["box_index"] >= 0 for record in records)
    reasons = defaultdict(int)
    for record in records:
        reasons[record["reason"].split(":", 1)[0]] += 1

    _atomic_json(args.out_dir / "manifest.json", records)
    meta = dict(
        format_version=1,
        dataset="MCP receptor-level deposited holo 2Fo-Fc density",
        data_semantics=(
            "Original deposited holo map. The reference-ligand density is retained; "
            "no apo-like masking, ligand erasure, or synthetic density is applied."
        ),
        target_scope="one canonical raw box per MCP receptor target",
        box_file=box_path.name,
        dtype="float16",
        n_boxes=n_targets,
        n_available=n_ok,
        g_box=args.g_box,
        resolution=args.resolution,
        output_crop_grid=64,
        output_crop_resolution=0.25,
        normalization=normalization,
        source_urls=dict(pdb=PDB_URL, mmcif=CIF_URL, map_2fofc=MAP_URL),
        failure_counts=dict(sorted(reasons.items())),
        download_summary={
            "pdb_available": sum(item.get("pdb_ok", False) for item in statuses.values()),
            "map_available": sum(item.get("map_ok", False) for item in statuses.values()),
        },
    )
    _atomic_json(args.out_dir / "meta.json", meta)
    _atomic_json(args.out_dir / ".complete", {"n_available": n_ok, "n_targets": n_targets})
    return meta


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_MCPP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--normalization-recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--g-box", type=int, default=144)
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--max-alignment-rmsd", type=float, default=1.0)
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--build-workers", type=int, default=8)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="process only this four-character PDB target (repeatable)",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.download_only and args.build_only:
        raise ValueError("--download-only and --build-only are mutually exclusive")
    if args.g_box < 64 or args.g_box % 2:
        raise ValueError("--g-box must be an even integer >= 64")
    if abs(args.resolution - 0.25) > 1e-8:
        raise ValueError("the pretrained encoder requires 0.25 A density resolution")

    args.data_root, targets = resolve_structure_root(args.data_root)
    if args.target:
        requested = {target.lower() for target in args.target}
        invalid = sorted(target for target in requested if len(target) != 4 or not target.isalnum())
        if invalid:
            raise ValueError(f"invalid four-character PDB target(s): {invalid}")
        available = set(targets)
        missing = sorted(requested - available)
        if missing:
            raise RuntimeError(f"requested target(s) absent under {args.data_root}: {missing}")
        targets = sorted(requested)
    if args.limit is not None:
        targets = targets[: args.limit]
    if not targets:
        raise RuntimeError(f"no MCP target directories found under {args.data_root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir = args.out_dir / "pdb"
    map_dir = args.out_dir / "ccp4"
    pdb_dir.mkdir(exist_ok=True)
    map_dir.mkdir(exist_ok=True)
    normalization = load_normalization(args.normalization_recipe)
    print(
        f"MCP holo-density: targets={len(targets):,}, g_box={args.g_box}, "
        f"physical_box={args.g_box * args.resolution:.1f} A"
    )

    statuses = {}
    if not args.build_only:
        statuses = download_all(targets, pdb_dir, map_dir, args.download_workers)
        _atomic_json(args.out_dir / "download_status.json", statuses)
    elif (args.out_dir / "download_status.json").exists():
        statuses = json.loads((args.out_dir / "download_status.json").read_text())
    if args.download_only:
        return

    meta = build_boxes(args, targets, statuses, normalization)
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
