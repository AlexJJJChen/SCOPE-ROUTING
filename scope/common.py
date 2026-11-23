"""Artifact paths, tabular outputs and content verification."""

import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = Path(os.environ.get("SCOPE_OUTPUT", ROOT / "results")).resolve()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_inputs():
    hashes = json.loads((ROOT / "checksums.json").read_text())
    present = {
        str(path.relative_to(DATA)) for path in DATA.rglob("*") if path.is_file()
    }
    if present != set(hashes):
        raise ValueError("Input file inventory differs from checksums.json")
    for name, expected in hashes.items():
        if digest(DATA / name) != expected:
            raise ValueError(f"Checksum mismatch for {name}")
    return len(hashes)


def table(name, rows):
    rows = list(rows)
    OUTPUT.mkdir(exist_ok=True, parents=True)
    with (OUTPUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(name, value):
    OUTPUT.mkdir(exist_ok=True, parents=True)
    (OUTPUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
