#!/usr/bin/env python3
"""Probe the three SB model URLs without saving or downloading full checkpoints.

Uses only the standard library. A passing result checks the shipped file name,
size, range support and PyTorch ZIP signature, NOT the full SHA-256 or model.
"""

from email.message import Message
import os
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


PROBE_BYTES = 32
ASSETS = (
    ("NF", "model.pt", 1454517523),
    ("FB", "checkpoint.pth.tar", 61681596433),
    ("CDG", "checkpoint_e0025.pth.tar", 1193976342),
)


def download_url(url):
    parts = urlsplit(url)
    if parts.scheme not in ("https", "http") or not parts.hostname:
        raise ValueError("a valid HTTP(S) model URL is required")
    if parts.hostname == "dropbox.com" or parts.hostname.endswith(".dropbox.com"):
        query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != "dl"]
        query.append(("dl", "1"))
        return urlunsplit(parts._replace(query=urlencode(query)))
    return url


def check_asset(name, url, filename, size):
    if not url:
        raise ValueError(f"{name}_MODEL_URL is unset; source scripts/mcp_assets.env.example first")
    request = Request(download_url(url), headers={
        "Range": f"bytes=0-{PROBE_BYTES - 1}",
        "Accept-Encoding": "identity",
        "User-Agent": "FuncBind-asset-link-check/1.0",
    })
    with urlopen(request, timeout=30) as response:
        # Do not consume a full body if the server ignores Range.
        if response.status != 206:
            raise ValueError(f"expected HTTP 206 for a range request, got {response.status}")
        content_range = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes 0-31/(\d+)", content_range)
        if not match or int(match.group(1)) != size:
            raise ValueError("unexpected Content-Range or checkpoint size")
        disposition = Message()
        disposition["Content-Disposition"] = response.headers.get("Content-Disposition", "")
        if disposition.get_filename() != filename:
            raise ValueError("wrong download filename; use a checkpoint file link, not the folder ZIP")
        if "text/" in response.headers.get("Content-Type", "").lower():
            raise ValueError("server returned a text/login page, not a checkpoint")
        prefix = response.read(PROBE_BYTES)
        if len(prefix) != PROBE_BYTES or not prefix.startswith(b"PK\x03\x04"):
            raise ValueError("expected the PyTorch checkpoint ZIP header")
    return f"[PASS] {name}: {filename}, {size:,} bytes, HTTP 206, read {PROBE_BYTES} bytes"


def main():
    failed = False
    for name, filename, size in ASSETS:
        try:
            print(check_asset(name, os.environ.get(f"{name}_MODEL_URL", ""), filename, size), flush=True)
        except HTTPError as exc:
            print(f"[FAIL] {name}: HTTP {exc.code}", flush=True)
            failed = True
        except (ValueError, OSError, URLError) as exc:
            # Do not echo URLs, shared-link keys, or redirect tokens in errors.
            detail = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
            print(f"[FAIL] {name}: {detail}", flush=True)
            failed = True
    print("Link check only: no files saved; full SHA-256/model loading not checked.", flush=True)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
