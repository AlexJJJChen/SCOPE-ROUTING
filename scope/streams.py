import json
import math
from collections import deque
from time import perf_counter_ns

import numpy as np
import pandas as pd
from scipy.stats import norm

from .common import DATA, table
from .contract import CROSS, OBL, TC, bounds, db_engine, kl_upper, sql_frontier


class Incremental:
    def __init__(self, target):
        self.last = {}
        self.target = target
        self.missing = {c: len(gates) for c, gates in OBL.items()}
        self.blocks = {c: 0 for c in OBL}
        self.active = {c for c in OBL if (c, target) in TC or c in CROSS}

    def apply(self, states):
        for gate, state in states.items():
            bit = state == "PASS"
            if self.last.get(gate, False) == bit:
                continue
            self.last[gate] = bit
            for c, obs in OBL.items():
                if gate not in obs or c not in self.active:
                    continue
                was = self.missing[c] == 0
                self.missing[c] += -1 if bit else 1
                live = self.missing[c] == 0
                if live != was:
                    for anc in self.active:
                        if anc != c and (anc, c) in TC:
                            self.blocks[anc] += 1 if live else -1
        eligible = sorted(
            c for c in self.active if self.missing[c] == 0 and self.blocks[c] == 0
        )
        return [c for c in eligible if c not in CROSS], [
            c for c in eligible if c in CROSS
        ]


def controlled_states(h, n, k):
    return dict(
        attr="PASS" if kl_upper(h, n, k) <= 0.05 else "UNRESOLVED",
        lock="PASS",
        valid="PASS",
        coverage="PASS",
        dynamics="PASS",
        hybrid="PASS",
        collapse="FAIL",
    )


def run_route():
    rng = np.random.default_rng(20261120)
    p = np.r_[np.full(1500, 0.012), np.full(600, 0.28), np.full(1400, 0.01)]
    x = (rng.random(len(p)) < p).astype(int).tolist()
    db = db_engine()
    db.execute("CREATE TABLE line(idx INTEGER,fb INTEGER)")
    db.execute("INSERT INTO line SELECT unnest(?),unnest(?)", [list(range(len(x))), x])
    ivm = Incremental("C5")
    buf = deque()
    h = 0
    rows = []
    for k, end in enumerate(range(25, 3501, 25), 1):
        t = perf_counter_ns()
        for v in x[end - 25 : end]:
            if len(buf) == 1000:
                h -= buf.popleft()
            buf.append(v)
            h += v
        states = controlled_states(h, len(buf), k)
        inc = ivm.apply(states)
        dt = (perf_counter_ns() - t) / 1e6
        t = perf_counter_ns()
        n2, h2 = db.execute(
            "SELECT count(*),sum(fb) FROM line WHERE idx>=? AND idx<?",
            [max(0, end - 1000), end],
        ).fetchone()
        ref = sql_frontier(db, controlled_states(h2, n2, k), "C5")
        full = (perf_counter_ns() - t) / 1e6
        rows.append(
            dict(
                batch=k,
                t=end,
                window_n=n2,
                fb=h2,
                ucb=kl_upper(h2, n2, k),
                ivm_c=json.dumps(inc[0]),
                ivm_r=json.dumps(inc[1]),
                sql_c=json.dumps(ref[0]),
                sql_r=json.dumps(ref[1]),
                match=int(inc == ref and (len(buf), h) == (n2, h2)),
                ivm_ms=dt,
                full_ms=full,
            )
        )
    table("route_stream.csv", rows)
    switches = [
        (r["t"], r["ivm_c"])
        for i, r in enumerate(rows)
        if i == 0 or rows[i - 1]["ivm_c"] != r["ivm_c"]
    ]
    assert all(r["match"] for r in rows)
    return dict(
        checkpoints=len(rows),
        mismatches=sum(1 - r["match"] for r in rows),
        switches=switches,
        ivm_p50_ms=float(np.median([r["ivm_ms"] for r in rows])),
        full_p50_ms=float(np.median([r["full_ms"] for r in rows])),
    )


def monte_carlo():
    batch = 25
    w = 1000
    horizon = 3500
    rows = []
    z = norm.ppf(0.95)
    for adaptive in (False, True):
        for seed in range(500):
            rng = np.random.default_rng(180000 + seed)
            sums = deque()
            tot = 0
            old_rate = 0.065
            first = {}
            for k in range(1, horizon // batch + 1):
                p = (0.075 if old_rate < 0.045 else 0.058) if adaptive else 0.065
                n_new = int(rng.binomial(batch, p))
                old_rate = n_new / batch
                if len(sums) == w // batch:
                    tot -= sums.popleft()
                sums.append(n_new)
                tot += n_new
                n = len(sums) * batch
                q = tot / n
                wald = q + z * math.sqrt(q * (1 - q) / n) <= 0.05
                d = (
                    (q * math.log(q / 0.05) if q else 0)
                    + (1 - q) * math.log((1 - q) / 0.95)
                    if q < 0.05
                    else 0.0
                )
                kl = q < 0.05 and n * d >= math.log(math.pi**2 * k * k / (6 * 0.05))
                for start in ("first", "full"):
                    if start == "full" and n < w:
                        continue
                    for method, licensed in (("wald", wald), ("kl", kl)):
                        if licensed:
                            first.setdefault(start + "_" + method, k)
            rows.append(
                dict(
                    adaptive=adaptive,
                    seed=180000 + seed,
                    **{
                        key: first.get(key, 0)
                        for key in ["first_wald", "first_kl", "full_wald", "full_kl"]
                    },
                )
            )
    table("monte_carlo.csv", rows)
    summary = []
    for strategy in (False, True):
        sl = [r for r in rows if r["adaptive"] == strategy]
        for key in ["first_wald", "first_kl", "full_wald", "full_kl"]:
            h = sum(bool(r[key]) for r in sl)
            lo, hi = bounds(h, len(sl), 0.025)
            summary.append(
                dict(
                    adaptive=strategy,
                    variant=key,
                    events=h,
                    n=len(sl),
                    rate=h / len(sl),
                    ci95=[lo, hi],
                )
            )
    return summary


def counters(row):
    return (
        1,
        row["fb"],
        int(row["cap"] > row["base_ok"]),
        int(row["cap"] < row["base_ok"]),
        row["exec_valid"],
        row["retry"],
    )


def verdict(groups, k):
    n, fb, good, bad, valid, retries = map(int, np.sum(groups, axis=0))
    alpha = 6 * 0.025 / (math.pi**2 * k**2)
    upper = kl_upper(fb, n, k, 0.025)
    gl, gh = bounds(good, n, alpha / 2)
    bl, bh = bounds(bad, n, alpha / 2)
    status = (
        "PASS"
        if upper <= 0.05 and bh - gl < 0
        else "FAIL"
        if bl - gh >= 0
        else "UNRESOLVED"
    )
    states = dict(
        attr=status,
        lock="PASS",
        valid="PASS" if valid == n else "FAIL",
        coverage="PASS" if all(g[2] > g[3] for g in groups) else "FAIL",
        dynamics="MISSING",
        hybrid="PASS" if retries and valid == n else "FAIL",
        collapse="FAIL",
    )
    return states, dict(
        n=n,
        fb=fb,
        good=good,
        bad=bad,
        valid_count=valid,
        retries=retries,
        fb_joint_upper=upper,
        loss_joint_upper=bh - gl,
    )


def run_sql():
    cfg = {
        "seed": 1117,
        "window": 1000,
        "batch": 25,
        "lengths": [1500, 600, 1400],
        "migration_probability": 0.30,
    }
    w, batch = cfg["window"], cfg["batch"]
    source = pd.read_csv(DATA / "spider/migration.csv.gz")
    replay = {
        v: g.set_index("qid").to_dict("index") for v, g in source.groupby("version")
    }
    ids = np.array(sorted(replay["normal"]))
    hard = np.array([q for q in ids if replay["normal"][q]["join_bin"] == 2])
    rng = np.random.default_rng(cfg["seed"])
    events = []
    for phase, count in zip(("nominal", "shock", "recovery"), cfg["lengths"]):
        for _ in range(count):
            incident = (
                phase != "nominal" and rng.random() < cfg["migration_probability"]
            )
            qid = int(rng.choice(hard if incident else ids))
            version = (
                ("old_on_migrated" if phase == "shock" else "new_on_migrated")
                if incident
                else "normal"
            )
            r = replay[version][qid].copy()
            r.update(
                qid=qid,
                t=len(events) + 1,
                phase=phase,
                version=version,
                fb=int(r["retry"] or r["rep_ms"] > 15),
                cap=r["rep_ok"] if r["rep_ms"] <= 15 else 0,
            )
            events.append(r)
    table("sql_requests.csv", events)
    database = db_engine()
    logged = pd.DataFrame(events)
    database.register("request_log", logged)
    database.execute("CREATE TABLE events AS SELECT * FROM request_log")
    counts = np.zeros((3, 6), dtype=int)
    window = deque()
    ivm = Incremental("C4pa")
    checks = []
    for event in events:
        group = event["join_bin"]
        v = counters(event)
        counts[group] += v
        window.append((group, v))
        if len(window) > w:
            j, old = window.popleft()
            counts[j] -= old
        end = event["t"]
        if end % batch:
            continue
        totals = np.zeros_like(counts)
        independently = database.execute(
            """
            SELECT join_bin,count(*),sum(fb),sum((cap>base_ok)::int),
                   sum((cap<base_ok)::int),sum(exec_valid),sum(retry)
            FROM events WHERE t>? AND t<=? GROUP BY join_bin
        """,
            [end - w, end],
        ).fetchall()
        for j, *vals in independently:
            totals[j] = vals
        assert np.array_equal(totals, counts)
        k = end // batch
        gates, stats = verdict(counts, k)
        other, sqlstats = verdict(totals, k)
        c, r = ivm.apply(gates)
        assert (c, r) == sql_frontier(database, other, "C4pa") and stats == sqlstats
        checks.append(
            dict(
                t=end,
                phase=event["phase"],
                **stats,
                **gates,
                C=json.dumps(c),
                R=json.dumps(r),
                licensed="C4pa" in c,
                parity=True,
            )
        )
    assert len(checks) == 140
    table("sql_checkpoints.csv", checks)
    results = {}
    for phase in ["nominal", "shock", "recovery"]:
        obs = [e for e in events if e["phase"] == phase]
        phase_checks = [x for x in checks if x["phase"] == phase]
        results[phase] = dict(
            n=len(obs),
            unique_qids=len({x["qid"] for x in obs}),
            fallbacks=sum(x["fb"] for x in obs),
            checkpoints=len(phase_checks),
            licensed=sum(x["licensed"] for x in phase_checks),
            hybrid=sum("hyb_resil" in x["R"] for x in phase_checks),
            min_fb_upper=min(x["fb_joint_upper"] for x in phase_checks),
            last=phase_checks[-1],
        )
    return dict(
        phases=results,
        changes=[
            dict(t=x["t"], phase=x["phase"], licensed=x["licensed"], R=x["R"])
            for i, x in enumerate(checks)
            if i == 0
            or (checks[i - 1]["licensed"], checks[i - 1]["R"])
            != (x["licensed"], x["R"])
        ],
    )
