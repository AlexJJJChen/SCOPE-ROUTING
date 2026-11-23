"""Extract and verify the 20 evaluation databases from a pinned Spider archive."""

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://huggingface.co/datasets/HAL-9001/spider-databases/resolve/4a01bbac6520cd35b216db9e1724e5e1ada60aa4/spider_data.zip"
SHA256 = "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--archive",
        type=Path,
        help="existing spider_data.zip; otherwise download the pinned archive",
    )
    p.add_argument("--output", type=Path, default=ROOT / "datasets/databases")
    args = p.parse_args()
    with tempfile.TemporaryDirectory() as stage:
        archive = args.archive
        if archive is None:
            archive = Path(stage) / "spider_data.zip"
            urllib.request.urlretrieve(URL, archive)
        with archive.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != SHA256:
                raise ValueError("Unexpected Spider archive checksum")
        with zipfile.ZipFile(archive) as z:
            for name, expected in json.loads(
                (ROOT / "data/spider/databases.json").read_text()
            ).items():
                payload = z.read(f"spider_data/database/{name}/{name}.sqlite")
                if hashlib.sha256(payload).hexdigest() != expected:
                    raise ValueError(f"Unexpected database checksum: {name}")
                folder = args.output / name
                folder.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=folder, delete=False) as f:
                    f.write(payload)
                os.replace(f.name, folder / (name + ".sqlite"))
    print(f"Verified 20 databases in {args.output}")


if __name__ == "__main__":
    main()
