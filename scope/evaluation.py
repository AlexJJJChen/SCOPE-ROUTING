"""Paired execution metrics, four-valued decisions, and gateway admission."""

import hashlib
import json
import random
from collections import Counter

import numpy as np
import pandas as pd

from .common import DATA, table, write_json
from .contract import classify, db_engine, frontier, sql_frontier

THRESHOLDS = (0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20, 0.30, 0.35)
BUDGETS = (5, 10, 15, 25, 50)
REFERENCES = ("Zero-shot", "Zero-shot + binder", "Graphix + binder", "Sol + binder")


def grid_record(pipeline, group, baseline, tau, ms):
    known = group.aux_calls.notna().all()
    admissible = (group.aux_calls == 0) & (group.repair_ms <= ms)
    capped = group.rep_ok.where(admissible, 0)
    ref = baseline.rep_ok.where(
        (baseline.aux_calls == 0) & (baseline.repair_ms <= ms), 0
    )
    delta = capped - ref
    falls = int(((~admissible) | (group.solver_called > 0)).sum())
    good = int((delta > 0).sum())
    bad = int((delta < 0).sum())
    attr, _, upper_fb, _, upper_loss = classify(
        falls, good, bad, len(group), tau, not known
    )
    if (
        baseline.pipeline.iloc[0] == "Zero-shot"
        and pipeline in {"P4", "P5"}
        and good == bad == 0
    ):
        attr = "FAIL"
    valid = group.valid.eq(1).all()
    coverage = all(delta[group.join_bin == i].mean() > 0 for i in range(3))
    gates = dict(
        attr=attr,
        lock="PASS",
        valid="PASS" if valid else "FAIL",
        coverage="PASS" if coverage else "FAIL",
        dynamics="MISSING",
        hybrid="PASS" if valid and group.solver_called.gt(0).any() else "FAIL",
        collapse="PASS" if pipeline == "P7" else "FAIL",
    )
    c, r = frontier(gates, "C4pa")
    return gates, dict(
        pipeline=pipeline,
        reference=baseline.pipeline.iloc[0],
        threshold=tau,
        budget_ms=ms,
        attribution=attr,
        upper_fallback=upper_fb,
        upper_loss=upper_loss,
        fallbacks=falls if known else None,
        primary=json.dumps(c),
        other=json.dumps(r),
        licensed="C4pa" in c,
    )


def clustered_interval(series, groups, rng_kind="python"):
    clusters = [series[groups == name].to_numpy() for name in sorted(groups.unique())]
    sums = np.array([x.sum() for x in clusters])
    sizes = np.array([len(x) for x in clusters])
    count = len(clusters)
    if rng_kind == "numpy":
        samples = np.random.default_rng(117).integers(count, size=(5000, count))
    else:
        rng = random.Random(117)
        samples = np.array(
            [[rng.choice(range(count)) for _ in range(count)] for _ in range(5000)]
        )
    rates = sums[samples].sum(axis=1) / sizes[samples].sum(axis=1)
    return (float(series.mean()), *np.quantile(rates, [0.025, 0.975]).tolist())


def run():
    questions = pd.read_csv(DATA / "spider/questions.csv").set_index("qid")
    assert (
        len(questions) == 459
        and questions.db.nunique() == 20
        and questions.index.is_unique
    )
    selected = sorted(
        questions.index,
        key=lambda i: hashlib.sha256(f"scope-v2-static/{i}".encode()).hexdigest(),
    )[:126]
    assert set(selected) == set(questions.index[questions.migrated == 1])
    lineage = pd.read_csv(DATA / "spider/lineage.csv.gz").join(
        questions[["join_bin", "db"]], on="qid", validate="many_to_one"
    )
    assert not lineage.duplicated(["qid", "pipeline"]).any()
    groups = {
        k: g.set_index("qid").sort_index() for k, g in lineage.groupby("pipeline")
    }
    assert len(groups) == 15 and all(
        g.index.equals(questions.sort_index().index) for g in groups.values()
    )
    summary = []
    for name, g in groups.items():
        known = g.aux_calls.notna().all()
        ok = (g.aux_calls == 0) & (g.repair_ms <= 15)

        def count(s):
            return int(s.sum()) if known else None

        summary.append(
            dict(
                pipeline=name,
                n=len(g),
                raw=count(g.raw_ok),
                repaired=count(g.rep_ok),
                capped=count(g.rep_ok.where(ok, 0)),
                final=int(g.final_ok.sum()),
                valid=int(g.valid.sum()),
                solver_switches=count(g.solver_called),
                fallbacks=count((~ok) | (g.solver_called > 0)),
            )
        )
    table("spider_accuracy.csv", summary)
    db = db_engine()
    records = []
    for ref in REFERENCES:
        for name, g in groups.items():
            if name in {
                "Zero-shot",
                "Zero-shot + binder",
                "Graphix",
                "Graphix + binder",
            }:
                continue
            for tau in THRESHOLDS:
                for ms in BUDGETS:
                    states, row = grid_record(name, g, groups[ref], tau, ms)
                    if tau == 0.05 and ms == 15:
                        c, r = sql_frontier(db, states, "C4pa")
                        assert (
                            json.loads(row["primary"]) == c
                            and json.loads(row["other"]) == r
                        )
                    records.append(row)
    db.close()
    table("spider_grid.csv", records)
    table(
        "default_gates.csv",
        [r for r in records if r["threshold"] == 0.05 and r["budget_ms"] == 15],
    )
    intervals = []
    for metric, pairs in [
        (
            "capped",
            [
                ("P1", "Zero-shot + binder"),
                ("P1", "Graphix + binder"),
                ("Sol binder + retry", "Zero-shot + binder"),
                ("Sol binder + retry", "P1"),
                ("Sol single", "Zero-shot"),
                ("Sol + binder", "Zero-shot + binder"),
                ("Sol + binder", "Graphix + binder"),
                ("Sol + binder", "P1"),
                ("Sol + binder", "Sol single"),
            ],
        ),
        (
            "final",
            [
                ("P3", "P1"),
                ("Sol deferral", "Sol binder + retry"),
                ("Sol binder + retry", "P1"),
                ("Sol deferral", "P3"),
                ("Sol binder + retry", "Sol + binder"),
                ("Sol deferral", "Sol single"),
            ],
        ),
    ]:

        def score(g):
            return (
                g.final_ok
                if metric == "final"
                else g.rep_ok.where((g.aux_calls == 0) & (g.repair_ms <= 15), 0)
            )

        for a, b in pairs:
            d, lo, hi = clustered_interval(
                score(groups[a]) - score(groups[b]), groups[a].db
            )
            intervals.append(
                dict(
                    metric=metric,
                    comparison=a + " minus " + b,
                    delta=d,
                    low=lo,
                    high=hi,
                )
            )
    interaction = (groups["P1"].rep_ok - groups["P2"].raw_ok) - (
        groups["Zero-shot + binder"].rep_ok - groups["Zero-shot"].raw_ok
    )
    d, lo, hi = clustered_interval(interaction, groups["P1"].db, "numpy")
    intervals.append(
        dict(
            metric="repair",
            comparison="difference in binding gains",
            delta=d,
            low=lo,
            high=hi,
        )
    )
    table("paired_intervals.csv", intervals)
    # Sensitivity to treating a served vote as a single, on-budget primary call.
    vote_delta = groups["P6"].final_ok - groups["Zero-shot"].raw_ok
    gains, losses = int((vote_delta > 0).sum()), int((vote_delta < 0).sum())
    state, _, upper_fb, _, upper_loss = classify(
        0, gains, losses, len(vote_delta), 0.05
    )
    write_json(
        "vote_sensitivity.json",
        dict(
            reference="Zero-shot",
            observed_attribution=next(
                row["attribution"]
                for row in records
                if row["pipeline"] == "P6"
                and row["reference"] == "Zero-shot"
                and row["threshold"] == 0.05
                and row["budget_ms"] == 15
            ),
            assumption="Charge the served vote to the primary path with zero auxiliary calls, zero repair time and zero switches.",
            paired_gains=gains,
            paired_losses=losses,
            hypothetical_attribution=state,
            hypothetical_upper_fallback=upper_fb,
            hypothetical_upper_loss=upper_loss,
        ),
    )
    votes = pd.read_csv(DATA / "spider/vote_control.csv")
    accesses = []
    for row in votes.itertuples():
        candidates = [row.dail, row.zero_shot, row.vote, row.template]
        accesses.append(
            dict(
                qid=row.qid,
                calls=len(candidates),
                selected=Counter(candidates).most_common(1)[0][0],
            )
        )
    table("instrumented_vote.csv", accesses)
    effective_fallbacks = sum(row["calls"] > 0 for row in accesses)
    failed = sum(
        classify(effective_fallbacks, 0, 0, len(accesses), tau)[0] == "FAIL"
        for tau in THRESHOLDS
        for _ in BUDGETS
    )
    p1, p3 = groups["P1"], groups["P3"]
    summary = dict(
        questions=len(questions),
        databases=questions.db.nunique(),
        migrated=len(selected),
        routing_to_interactive_switches=int(p1.solver_called.sum()),
        routing_to_retry_switches=int(p3.solver_called.sum()),
        avoided_switches=int(p3.solver_called.sum() - p1.solver_called.sum()),
        vote_control_questions=len(accesses),
        vote_control_failed_cells=failed,
    )
    write_json("gateway.json", summary)
    return summary
