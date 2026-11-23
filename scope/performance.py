"""Report instrumented server timings or collect the same measurements on this host."""

import gzip
import hashlib
import json
import tempfile
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd

from .common import DATA, table, write_json
from .contract import db_engine, set_gates
from .queries import FULL_QUERY, STATS


def recorded():
    folder = DATA / "performance"
    records = pd.read_csv(folder / "profile.csv.gz")
    rows = []
    for (stage, size, unit), g in records.groupby(["stage", "size", "unit"]):
        rows.append(
            dict(
                stage=stage,
                size=int(size),
                unit=unit,
                n=len(g),
                p50=g.value.median(),
                p99=g.value.quantile(0.99),
            )
        )
    table("instrumentation.csv", rows)
    thread = pd.read_csv(folder / "threads.csv.gz")
    res = []
    for (size, workers, stage), g in thread.groupby(["size", "threads", "stage"]):
        res.append(
            dict(
                size=int(size),
                threads=int(workers),
                stage=stage,
                n=len(g),
                p50_ms=g.ms.median(),
                p99_ms=g.ms.quantile(0.99),
            )
        )
    table("duckdb_timings.csv", res)
    stream = pd.read_csv(folder / "route_updates.csv")
    ratio = round(stream.full_ms.median() / stream.ivm_ms.median())
    write_json(
        "stream_timing.json",
        dict(
            ivm_ms=stream.ivm_ms.median(),
            duckdb_ms=stream.full_ms.median(),
            speedup=ratio,
        ),
    )
    return len(thread), len(stream)


def measure():
    from .database import run, verify_databases

    verify_databases()
    folder = DATA / "performance"
    inputs = (
        pd.read_csv(folder / "scan_workload.csv")
        .sort_values("qid")
        .reset_index(drop=True)
    )
    inputs["j"] = np.arange(len(inputs))
    assert len(inputs) == 300
    db = db_engine()
    db.execute(
        "CREATE TABLE small AS SELECT j,fb,cap_loss AS cap,base_loss FROM inputs"
    )
    set_gates(
        db,
        dict(
            attr="UNRESOLVED",
            lock="PASS",
            valid="PASS",
            coverage="FAIL",
            hybrid="PASS",
            dynamics="MISSING",
            collapse="FAIL",
        ),
    )
    timings = []
    for size in (300, 22500, 1000000):
        db.execute(
            "CREATE OR REPLACE TABLE tele AS SELECT i qid,'P1' arm,fb,cap,1 feasible FROM range(?) t(i) JOIN small ON i%300=j",
            [size],
        )
        db.execute(
            "CREATE OR REPLACE TABLE refs AS SELECT i qid,base_loss FROM range(?) t(i) JOIN small ON i%300=j",
            [size],
        )
        for rep in range(103):
            checks = []
            for threads in [1, 8] if rep % 2 == 0 else [8, 1]:
                db.execute(f"SET threads={threads}")
                start = perf_counter_ns()
                bounds = db.execute(STATS).fetchone()
                mid = perf_counter_ns()
                front = db.execute(FULL_QUERY, ["C4pa"]).fetchall()
                end = perf_counter_ns()
                checks.append((bounds, front))
                if rep >= 3:
                    for stage, ns in [
                        ("scan_and_bounds", mid - start),
                        ("scan_to_frontier", end - mid),
                    ]:
                        timings.append(
                            dict(
                                size=size,
                                threads=threads,
                                iteration=rep - 3,
                                stage=stage,
                                ms=ns / 1e6,
                            )
                        )
            (a, f), (b, g) = checks
            assert (
                a[:4] == b[:4]
                and a[-1] == b[-1]
                and f == g
                and np.allclose(a[4:8], b[4:8], atol=1e-10, rtol=0)
            )
    db.close()
    table("local_duckdb_trials.csv", timings)
    qs = pd.read_csv(folder / "manifest_queries.csv")
    calls = list(zip(qs.db, qs.gold_sql))
    trace = []

    def wrapped(db_name, sql):
        start = perf_counter_ns()
        result = run(db_name, sql)
        trace.append((db_name, start, perf_counter_ns(), result[1]))
        return result

    samples = []
    for i in range(110):
        row = calls[i % len(calls)]
        elapsed = {}
        for name, fn in (
            [("raw", run), ("wrapped", wrapped)]
            if i % 2
            else [("wrapped", wrapped), ("raw", run)]
        ):
            start = perf_counter_ns()
            fn(*row)
            elapsed[name] = (perf_counter_ns() - start) / 1e3
        if i >= 10:
            samples.append(
                dict(
                    stage="sqlite_interceptor",
                    unit="us",
                    value=elapsed["wrapped"] - elapsed["raw"],
                )
            )
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for gz in sorted((folder / "hash_inputs").glob("*.gz")):
            target = Path(tmp) / gz.stem
            target.write_bytes(gzip.decompress(gz.read_bytes()))
            files.append(target)
        assert sum(p.stat().st_size for p in files) == 406320

        def rehash():
            from .common import digest

            manifest = {f.name: digest(f) for f in files}
            return hashlib.sha256(
                json.dumps(manifest, sort_keys=True).encode()
            ).digest()

        reference = rehash()
        for i in range(110):
            start = perf_counter_ns()
            current = rehash()
            ns = perf_counter_ns() - start
            assert current == reference
            if i >= 10:
                samples.append(dict(stage="manifest", unit="ms", value=ns / 1e6))
    table("local_instrumentation.csv", samples)
