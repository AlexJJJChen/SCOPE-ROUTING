"""Export scored test graphs from five CAP-DR checkpoints for an offline replay."""

import argparse
import gzip
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path


def read_sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capdr-code", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(
        (
            Path(__file__).resolve().parents[1] / "data/routing/checkpoints.json"
        ).read_text()
    )
    commit = subprocess.check_output(
        ["git", "-C", str(args.capdr_code), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != lock["capdr_commit"]:
        raise ValueError("CAP-DR checkout does not match checkpoints.json")
    sys.path.insert(0, str(args.capdr_code.resolve()))
    import numpy as np
    import torch
    from capdr_core import build_model_from_samples, sample_to_tensors

    if read_sha(args.episodes) != lock["episodes_sha256"]:
        raise ValueError("Episode archive checksum differs from checkpoints.json")
    with gzip.open(args.episodes, "rt") as stream:
        episodes = [json.loads(line) for line in itertools.islice(stream, 20, 24)]
    samples = [s for ep in episodes for s in ep["samples"]]
    assert len(samples) == 64
    fields = (
        "episode_id",
        "time",
        "num_nodes",
        "src",
        "dst",
        "capacity",
        "true_edge_objectives",
        "obs_edge_objectives_proxy",
        "request",
        "oracle_path",
        "oracle_scalar",
    )
    graphs = [{k: s[k] for k in fields} | {"model_scores": []} for s in samples]
    torch.set_num_threads(1)
    for seed in range(5):
        ck = args.checkpoints / f"seed_{seed}" / "capdr_final.pt"
        if read_sha(ck) != lock["weights_sha256"][str(seed)]:
            raise ValueError(f"Weight checksum mismatch for seed {seed}")
        state = torch.load(ck, map_location="cpu", weights_only=True)
        model = build_model_from_samples(samples[:1], state["model_cfg"])
        model.load_state_dict(state["state_dict"], strict=True)
        model.eval()
        with torch.inference_mode():
            for s, g in zip(samples, graphs):
                b = sample_to_tensors(s)
                enc = model.encode(b)
                pred = model.predicted_edge_scalar(
                    b,
                    enc["objective_pred"],
                    enc.get("objective_sigma"),
                    enc.get("proxy_objective"),
                    enc.get("residual_pred"),
                    enc.get("trust_gate"),
                )
                scores = np.asarray(pred).reshape(-1).tolist()
                assert len(scores) == len(s["src"])
                g["model_scores"].append(scores)
    args.output.write_bytes(
        gzip.compress(
            json.dumps(graphs, separators=(",", ":"), allow_nan=False).encode(), mtime=0
        )
    )


if __name__ == "__main__":
    main()
