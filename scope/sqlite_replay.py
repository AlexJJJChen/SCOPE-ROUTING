"""Regenerate pipeline SQL and check every observed answer on the locked databases."""

import pandas as pd

from . import database as db
from .common import DATA, table, write_json


def run():
    db.verify_databases()
    cases = pd.read_csv(DATA / "spider/questions.csv").set_index("qid")
    preds = pd.read_csv(DATA / "spider/predictions.csv.gz").set_index("qid")
    logs = pd.read_csv(DATA / "spider/lineage.csv.gz", keep_default_na=False).set_index(
        ["qid", "pipeline"]
    )
    migration = pd.read_csv(DATA / "spider/migration.csv.gz").set_index(
        ["qid", "version"]
    )
    clean = []
    checked = 0
    for q in cases.itertuples():
        qid = q.Index
        pred = preds.loc[qid]
        ordered = db.parse(q.gold_sql).args.get("order") is not None
        gold, err = db.execute(q.db, q.gold_sql, False)
        assert not err
        gold_migrated, _ = db.execute(q.db, db.compatibility(q.gold_sql, q.db), True)
        assert db.equivalent(gold_migrated, gold, ordered), (qid, "gold migration")
        cache = {}

        def query(sql, mig):
            if (sql, mig) not in cache:
                res, error = db.execute(q.db, sql, mig)
                cache[sql, mig] = (
                    int(db.equivalent(res, gold, ordered)),
                    error,
                    0 if res is None else len(res[1]),
                )
            return cache[sql, mig]

        template = "SELECT * FROM " + db.quote(
            (db.referenced(db.parse(pred.dail), q.db) or db.schema(q.db)[0])[0]
        )
        for active in sorted({False, bool(q.migrated)}):

            def pipeline(name, raw, binder=False, guard="", rescue=None):
                first, _, _ = query(raw, active)
                rep = db.repair(raw, q.db, active) if binder else raw
                hit, error, n = query(rep, active)
                trigger = bool(error) or (guard in ("error_empty", "broad") and n == 0)
                if guard == "broad":
                    trigger = trigger or len(db.referenced(db.parse(raw), q.db)) < 2
                called = bool(rescue and trigger)
                final = db.repair(rescue, q.db, active) if called else rep
                served, finalerr, _ = query(final, active)
                if not active:
                    clean.append(
                        dict(
                            qid=qid,
                            pipeline=name,
                            raw=None if name == "P6" else first,
                            repaired=None if name == "P6" else hit,
                            final=served,
                        )
                    )
                if active != bool(q.migrated):
                    return
                r = logs.loc[qid, name]
                assert (r.served_sql, int(r.final_ok), int(r.valid)) == (
                    final,
                    served,
                    int(not finalerr),
                ), (qid, name, "served")
                if name != "P6":
                    assert (
                        r.primary_sql,
                        r.repaired_sql,
                        int(r.raw_ok),
                        int(r.rep_ok),
                        int(r.solver_called),
                    ) == (raw, rep, first, hit, int(called)), (qid, name, "stages")

            pipeline("Zero-shot", pred.zero_shot)
            pipeline("Zero-shot + binder", pred.zero_shot, True)
            pipeline("Graphix", pred.graphix)
            pipeline("Graphix + binder", pred.graphix, True)
            pipeline("P1", pred.dail, True, "error", pred.zero_shot)
            pipeline("P2", pred.dail)
            pipeline("P3", pred.graphix, False, "error_empty", pred.vote)
            pipeline("P4", pred.zero_shot, False, "broad", pred.vote)
            pipeline("P5", pred.zero_shot, False, "error", pred.dail)
            pipeline("P6", pred.vote)
            pipeline("P7", template)
            pipeline("Sol single", pred.sol)
            pipeline("Sol + binder", pred.sol, True)
            pipeline("Sol binder + retry", pred.sol, True, "error", pred.zero_shot)
            pipeline("Sol deferral", pred.sol, False, "error_empty", pred.vote)
        for version, active, updated in [
            ("normal", False, False),
            ("old_on_migrated", True, False),
            ("new_on_migrated", True, True),
        ]:
            rep = db.repair(pred.dail, q.db, updated)
            hit, error, _ = query(rep, active)
            final = db.repair(pred.zero_shot, q.db, active) if error else rep
            served, finalerr, _ = query(final, active)
            base, _, _ = query(pred.zero_shot, active)
            r = migration.loc[qid, version]
            assert (r.rep_ok, r.retry, r.exec_ok, r.exec_valid, r.base_ok) == (
                hit,
                int(bool(error)),
                served,
                int(not finalerr),
                base,
            ), (qid, version)
        checked += 1
    table("original_schema.csv", clean)
    write_json(
        "sqlite_checks.json",
        dict(
            gold_equivalent=checked,
            pipeline_rows=len(logs),
            migration_rows=len(migration),
        ),
    )
