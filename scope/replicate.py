"""Run the executable paper artifact from a source checkout."""

import argparse

import pandas as pd

from . import evaluation, performance, routing, streams
from .common import OUTPUT, ROOT, verify_inputs, write_json
from .contract import db_engine, frontier, sql_frontier


def verify_closure():
    db = db_engine()
    defaults = dict(
        attr="PASS",
        lock="PASS",
        valid="PASS",
        coverage="PASS",
        dynamics="PASS",
        hybrid="PASS",
        collapse="PASS",
    )
    examples = [
        ("C5", {}, (["C5"], ["ast_probe", "hyb_resil"])),
        ("C4pa", {}, (["C4pa"], ["ast_probe", "hyb_resil"])),
        ("C5", {"attr": "MISSING"}, ([], ["ast_probe", "hyb_resil"])),
        ("C5", {"attr": "FAIL", "collapse": "FAIL"}, ([], ["hyb_resil"])),
    ]
    for target, overrides, answer in examples:
        gates = defaults | overrides
        assert frontier(gates, target) == sql_frontier(db, gates, target) == answer
    # The two branches of the primary diamond are incomparable.
    from .contract import TC

    assert ("C3pref", "C3attr") not in TC and ("C3attr", "C3pref") not in TC
    db.close()


def compare_checkpoint_file(file, expected_file):
    expected = pd.read_csv(ROOT / "reference" / expected_file)
    actual = pd.read_csv(OUTPUT / file)
    pd.testing.assert_frame_equal(
        actual[expected.columns].reset_index(drop=True),
        expected,
        check_exact=False,
        rtol=1e-10,
        atol=1e-12,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sqlite",
        action="store_true",
        help="regenerate SQL and verify SQLite outcomes using the 20 source databases",
    )
    ap.add_argument(
        "--measure",
        action="store_true",
        help="run the host-specific timing experiments with the source databases",
    )
    opts = ap.parse_args()
    if not __debug__:
        raise RuntimeError(
            "Verification requires Python assertions. Remove -O or PYTHONOPTIMIZE."
        )
    count = verify_inputs()
    verify_closure()
    audit = evaluation.run()
    assert audit["avoided_switches"] == 158 and audit["vote_control_failed_cells"] == 45
    compare_checkpoint_file("spider_accuracy.csv", "spider_accuracy.csv")
    compare_checkpoint_file("default_gates.csv", "default_gates.csv")
    compare_checkpoint_file("paired_intervals.csv", "paired_intervals.csv")
    routes = routing.run()
    controlled = streams.run_route()
    sql = streams.run_sql()
    compare_checkpoint_file("route_stream.csv", "route_checkpoints.csv")
    compare_checkpoint_file("sql_checkpoints.csv", "sql_checkpoints.csv")
    assert (
        controlled["mismatches"] == 0
        and sum(x["licensed"] for x in sql["phases"].values()) == 0
    )
    simulation = streams.monte_carlo()
    assert [r["events"] for r in simulation] == [120, 0, 0, 0, 134, 0, 0, 0]
    write_json("repeated_testing.json", simulation)
    write_json("stream_summary.json", dict(route=controlled, sql=sql))
    assert performance.recorded() == (1200, 140)
    compare_checkpoint_file("duckdb_timings.csv", "duckdb_timings.csv")
    compare_checkpoint_file("instrumentation.csv", "instrumentation.csv")
    if opts.sqlite:
        from .sqlite_replay import run

        run()
    if opts.measure:
        performance.measure()
    print(
        f"Verified {count} input files, {audit['questions']} Spider questions, "
        f"{routes['model_observations']} RG64 model observations and 280 checkpoints."
    )
    print(f"Results: {OUTPUT}")


if __name__ == "__main__":
    main()
