"""Recover policy rankings and rerank graph paths from frozen CAP-DR edge scores."""

import gzip
import json
from math import log, sqrt

import pandas as pd
from scipy.stats import kendalltau

from .common import DATA, table, write_json
from .paths import (
    enumerate_feasible_paths,
    path_is_feasible,
    path_scalar_cost,
    scalarize_edge_objectives,
)


def guarded(graph, path, costs):
    if path_is_feasible(graph, path):
        return list(path), False
    best = None
    for i, edge in enumerate(path):
        endpoints = (graph["src"][edge], graph["dst"][edge])
        for alt, pair in enumerate(zip(graph["src"], graph["dst"])):
            if alt == edge or pair != endpoints:
                continue
            candidate = path[:i] + [alt] + path[i + 1 :]
            if path_is_feasible(graph, candidate):
                rank = (sum(costs[j] for j in candidate), tuple(candidate))
                if best is None or rank < best:
                    best = rank
    return list(best[1]) if best else list(path), True


def run():
    bundle = DATA / "routing"
    obs = pd.read_csv(bundle / "policy_requests.csv.gz")
    assert (
        len(obs) == 22500
        and obs.request_id.nunique() == 1500
        and obs.policy.nunique() == 15
    )
    rows = []
    for policy, g in obs.groupby("policy"):
        n = len(g)
        fb = g.fallback.astype(float)
        width = fb.std() * sqrt(2 * log(2 / 0.025) / n) + 7 * log(2 / 0.025) / (
            3 * (n - 1)
        )
        lo, hi = fb.mean() - width, fb.mean() + width
        state = "PASS" if hi <= 0.05 else "FAIL" if lo > 0.05 else "UNRESOLVED"
        rows.append(
            dict(
                policy=policy,
                n=n,
                regret=g.regret.mean(),
                fallback=fb.mean(),
                upper=hi,
                gate=state,
            )
        )
    table("routing_policies.csv", rows)
    rank = json.loads((bundle / "rankings.json").read_text())
    a, b = rank["regret_only_ranking"], rank["evidence_licensed_ranking"]
    assert a == [
        r["policy"]
        for r in sorted(rows, key=lambda r: r["regret"])
        if r["policy"] != "DynamicExactOracle"
    ]
    tau = kendalltau(range(len(a)), [b.index(x) for x in a]).statistic
    with gzip.open(bundle / "graphs.json.gz", "rt") as f:
        graphs = json.load(f)
    records = []
    expected = pd.read_csv(bundle / "decisions.csv").set_index(
        ["seed", "episode", "step", "arm"]
    )
    control = json.loads((bundle / "invalid_path.json").read_text())
    fixture = False
    for g in graphs:
        req = g["request"]
        proxy = scalarize_edge_objectives(
            g["obs_edge_objectives_proxy"],
            req["preference"],
            req["scale_latency"],
            req["scale_fee"],
        )
        candidates = [
            list(x["path"])
            for x in enumerate_feasible_paths(g, edge_costs=proxy, max_candidates=4)
        ]
        assert candidates and path_is_feasible(g, g["oracle_path"])
        assert abs(path_scalar_cost(g, g["oracle_path"]) - g["oracle_scalar"]) < 1e-5
        if (g["episode_id"], g["time"]) == (control["episode"], control["step"]):
            bad = control["invalid_path"]
            assert not path_is_feasible(g, bad)
            fixed, used = guarded(g, bad, proxy)
            assert (
                used
                and path_is_feasible(g, fixed)
                and fixed == control["repaired_path"]
            )
            fixture = True
        for seed, model_scores in enumerate(g["model_scores"]):
            for arm, scores in [("base", proxy), ("model", model_scores)]:
                raw = min(candidates, key=lambda p: sum(scores[e] for e in p))
                fixed, invoked = guarded(g, raw, proxy)
                valid = path_is_feasible(g, fixed)
                regret = path_scalar_cost(g, fixed) - g["oracle_scalar"]
                key = (seed, g["episode_id"], g["time"], arm)
                ref = expected.loc[key]
                assert raw == json.loads(ref.raw) and fixed == json.loads(
                    ref.repaired
                ), key
                assert (
                    valid == bool(ref.valid) and abs(regret - ref.rep_regret) < 1e-10
                ), key
                records.append(
                    dict(
                        seed=seed,
                        episode=g["episode_id"],
                        step=g["time"],
                        arm=arm,
                        raw=json.dumps(raw),
                        repaired=json.dumps(fixed),
                        invoked=invoked,
                        valid=valid,
                        regret=regret,
                    )
                )
    assert fixture and len(records) == 640
    table("rg64.csv", records)
    data = pd.DataFrame(records)
    summary = dict(
        rank_tau=float(tau),
        unique_requests=len(graphs),
        checkpoints=5,
        model_observations=int((data.arm == "model").sum()),
        model_regret=data[data.arm == "model"].regret.mean(),
        baseline_regret=data[data.arm == "base"].regret.mean(),
        changed_paths=int((data.raw != data.repaired).sum()),
        invalid_control_repaired=fixture,
    )
    write_json("routing.json", summary)
    return summary
