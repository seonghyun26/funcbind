#!/usr/bin/env python
"""Download the public FuncBind MCP dataset with a disk-space safety gate.

The prepared split tensors are enough to train/evaluate FuncBind.  Building the
X-ray density cache additionally needs the original structure archive.  The archive
is deliberately opt-in because it is 32.7 GB compressed and substantially larger
after extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

import requests


DEFAULT_REPO_ID = "Willete3/mcpp-dataset"
SPLIT_FILES = ("train_data.pt", "val_data.pt", "test_data.pt")
ARCHIVE = "mcpp_dataset.tar.gz"
EXTRACT_MARKER = ".mcpp_original_extract_complete.json"


def human_size(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def dataset_files(repo_id: str) -> dict[str, dict]:
    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main"
    response = requests.get(
        url,
        params={"recursive": "true", "expand": "true"},
        timeout=60,
    )
    response.raise_for_status()
    result = {}
    for item in response.json():
        if item.get("type") != "file":
            continue
        lfs = item.get("lfs") or {}
        result[item["path"]] = {
            "size": int(item.get("size") or lfs.get("size") or 0),
            "sha256": lfs.get("oid"),
        }
    return result


def existing_ancestor(path: Path) -> Path:
    path = path.resolve()
    while not path.exists():
        path = path.parent
    return path


def remaining_bytes(destination: Path, expected: int) -> int:
    if destination.is_file() and destination.stat().st_size == expected:
        return 0
    partial = destination.with_suffix(destination.suffix + ".part")
    have = partial.stat().st_size if partial.is_file() else 0
    return max(0, expected - min(have, expected))


def sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(repo_id: str, name: str, destination: Path, metadata: dict) -> None:
    expected = int(metadata["size"])
    expected_hash = metadata.get("sha256")
    if destination.is_file() and destination.stat().st_size == expected:
        print(f"[skip] {name}: already present ({human_size(expected)})", flush=True)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{name}"

    for attempt in range(1, 6):
        offset = partial.stat().st_size if partial.is_file() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    print(f"[resume] {name}: server restarted the transfer", flush=True)
                    offset = 0
                mode = "ab" if offset and response.status_code == 206 else "wb"
                written = offset
                started = last_report = time.monotonic()
                with partial.open(mode) as output:
                    for chunk in response.iter_content(chunk_size=16 * 1024 * 1024):
                        if not chunk:
                            continue
                        output.write(chunk)
                        written += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 5:
                            elapsed = max(now - started, 1e-6)
                            rate = (written - offset) / elapsed
                            print(
                                f"[download] {name}: {human_size(written)}/{human_size(expected)} "
                                f"({100.0 * written / expected:.1f}%), {human_size(int(rate))}/s",
                                flush=True,
                            )
                            last_report = now
            if partial.stat().st_size != expected:
                raise RuntimeError(
                    f"size mismatch: got {partial.stat().st_size}, expected {expected}"
                )
            if expected_hash:
                print(f"[verify] {name}: sha256", flush=True)
                actual = sha256(partial)
                if actual != expected_hash:
                    raise RuntimeError(f"sha256 mismatch: got {actual}, expected {expected_hash}")
            os.replace(partial, destination)
            print(f"[done] {name}: {human_size(expected)}", flush=True)
            return
        except Exception as error:  # noqa: BLE001 - retries include transport errors
            if attempt == 5:
                raise
            print(f"[retry {attempt}/5] {name}: {error}", flush=True)
            time.sleep(min(2**attempt, 20))


def archive_unpacked_bytes(path: Path) -> int:
    with tarfile.open(path, "r:gz") as archive:
        return sum(member.size for member in archive if member.isfile())


def extract_archive(path: Path, destination: Path, reserve: int) -> None:
    marker = destination / EXTRACT_MARKER
    if marker.is_file():
        try:
            recorded = json.loads(marker.read_text())
        except (OSError, json.JSONDecodeError):
            recorded = {}
        if recorded.get("archive_size") == path.stat().st_size:
            print(f"[skip] {path.name}: original archive already extracted", flush=True)
            return

    unpacked = archive_unpacked_bytes(path)
    free = shutil.disk_usage(existing_ancestor(destination)).free
    print(
        f"[extract preflight] members={human_size(unpacked)}, free={human_size(free)}, "
        f"reserve={human_size(reserve)}",
        flush=True,
    )
    if free < unpacked + reserve:
        raise RuntimeError(
            "not enough space to extract the original archive safely; "
            f"need at least {human_size(unpacked + reserve)}, have {human_size(free)}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "r:gz") as archive:
        archive.extractall(destination, filter="data")
    temporary = marker.with_suffix(marker.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "archive": path.name,
                "archive_size": path.stat().st_size,
                "unpacked_bytes": unpacked,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    os.replace(temporary, marker)
    print(f"[done] extracted {path.name} into {destination}", flush=True)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dest",
        type=Path,
        default=repo / "funcbind/dataset/data/mcpp_dataset",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--include-original",
        action="store_true",
        help="also download the 32.7 GB original structure archive",
    )
    parser.add_argument(
        "--extract-original",
        action="store_true",
        help="extract the original archive after download (implies --include-original)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-free-gib", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.extract_original:
        args.include_original = True
    metadata = dataset_files(args.repo_id)
    selected = list(SPLIT_FILES)
    if args.include_original:
        selected.append(ARCHIVE)
    missing = [name for name in selected if name not in metadata]
    if missing:
        raise RuntimeError(f"files absent from {args.repo_id}: {missing}")

    reserve = int(args.min_free_gib * 1024**3)
    free = shutil.disk_usage(existing_ancestor(args.dest)).free
    remaining = sum(
        remaining_bytes(args.dest / name, metadata[name]["size"]) for name in selected
    )
    print(f"repo: https://huggingface.co/datasets/{args.repo_id}")
    print(f"destination: {args.dest.resolve()}")
    for name in selected:
        local = args.dest / name
        state = "present" if remaining_bytes(local, metadata[name]["size"]) == 0 else "needed"
        print(f"  {name:24s} {human_size(metadata[name]['size']):>12s}  {state}")
    print(
        f"filesystem free={human_size(free)}; remaining download={human_size(remaining)}; "
        f"required reserve={human_size(reserve)}"
    )
    if free < remaining + reserve:
        print(
            "REFUSED: insufficient disk space. No download was started. "
            f"Need {human_size(remaining + reserve)}, have {human_size(free)}.",
            file=sys.stderr,
        )
        return 2
    if args.dry_run:
        print("DRY RUN PASSED: download fits the configured safety margin")
        return 0

    for name in selected:
        download_file(args.repo_id, name, args.dest / name, metadata[name])
    if args.extract_original:
        extract_archive(args.dest / ARCHIVE, args.dest, reserve)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
