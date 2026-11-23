"""Read-only SQLite execution, result equivalence, and catalog binding."""

import json
import os
import re
import sqlite3
from collections import Counter
from functools import lru_cache
from pathlib import Path

import sqlglot
from sqlglot import exp

from .common import DATA, ROOT, digest

DATABASES = Path(os.environ.get("SCOPE_DB", ROOT / "datasets/databases")).resolve()


def verify_databases():
    for name, expected in json.loads(
        (DATA / "spider/databases.json").read_text()
    ).items():
        path = DATABASES / name / (name + ".sqlite")
        if not path.exists() or digest(path) != expected:
            raise ValueError(f"Missing or mismatched Spider database: {name}")


def parse(sql):
    try:
        return sqlglot.parse_one(sql, read="sqlite")
    except (sqlglot.errors.SqlglotError, RecursionError):
        return None


@lru_cache(None)
def conn(db):
    path = DATABASES / db / (db + ".sqlite")
    c = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    c.text_factory = lambda b: b.decode("utf-8", "replace")
    c.execute("PRAGMA query_only=ON")
    return c


@lru_cache(None)
def schema(db):
    c = conn(db)
    names = [
        r[0]
        for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    cols = {
        r[1].lower()
        for n in names
        for r in c.execute("SELECT * FROM pragma_table_info(?)", (n,))
    }
    return names, cols


def run(db, sql):
    c = conn(db)
    calls = 0

    def progress():
        nonlocal calls
        calls += 1
        return calls > 200000

    c.set_progress_handler(progress, 1000)
    try:
        cur = c.execute(sql)
        rows = cur.fetchmany(30001)
        if len(rows) > 30000:
            return None, "row_limit"
        return (len(cur.description or ()), rows), ""
    except (sqlite3.Error, ValueError) as error:
        return None, str(error)
    finally:
        c.set_progress_handler(None, 0)


def equivalent(res, gold, ordered):
    if res is None or res[0] != gold[0]:
        return False
    return res[1] == gold[1] if ordered else Counter(res[1]) == Counter(gold[1])


def normalize(name):
    return re.sub(r"[^a-z0-9]", "", name.lower()).rstrip("s")


def bind(sql, db, mapping=None):
    tree = parse(sql)
    if tree is None:
        return sql
    names, cols = schema(db)
    mappings = [(exp.Table, {n.lower() for n in names}), (exp.Column, cols)]
    changed = False
    for kind, vocabulary in mappings:
        for obj in tree.find_all(kind):
            ident = obj.this
            if not isinstance(ident, exp.Identifier):
                continue
            raw = ident.name.lower()
            if mapping and raw in mapping:
                ident.set("this", mapping[raw])
                changed = True
            elif raw not in vocabulary:
                matches = [x for x in vocabulary if normalize(x) == normalize(raw)]
                if len(matches) == 1:
                    ident.set("this", matches[0])
                    changed = True
    return tree.sql(dialect="sqlite") if changed else sql


def quote(name):
    return '"' + name.replace('"', '""') + '"'


@lru_cache(None)
def registry(db):
    connection = conn(db)
    return {
        n.lower(): (
            n,
            [
                r[1]
                for r in connection.execute("SELECT * FROM pragma_table_info(?)", (n,))
            ],
        )
        for n in schema(db)[0]
    }


@lru_cache(None)
def migrated_connection(db):
    path = DATABASES / db / (db + ".sqlite")
    c = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    c.text_factory = lambda b: b.decode("utf-8", "replace")
    for name, cols in registry(db).values():
        project = ", ".join(f"{quote(x)} AS {quote('scope_v2_' + x)}" for x in cols)
        c.execute(
            f"CREATE TEMP VIEW {quote(name)} AS SELECT {project} FROM main.{quote(name)}"
        )
    c.execute("PRAGMA query_only=ON")
    return c


def execute(db, sql, migrated):
    if not migrated:
        return run(db, sql)
    c = migrated_connection(db)
    ticks = 0

    def timeout():
        nonlocal ticks
        ticks += 1
        return ticks > 200000

    c.set_progress_handler(timeout, 1000)
    try:
        cur = c.execute(sql)
        vals = cur.fetchmany(30001)
        if len(vals) > 30000:
            return None, "row_limit"
        return (len(cur.description or ()), vals), ""
    except (sqlite3.Error, ValueError) as exc:
        return None, str(exc)
    finally:
        c.set_progress_handler(None, 0)


def referenced(tree, db):
    if tree is None:
        return []
    aliases = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    return sorted(
        ({t.name.lower() for t in tree.find_all(exp.Table)} - aliases)
        & registry(db).keys()
    )


def compatibility(sql, db):
    tree = parse(sql)
    tables = referenced(tree, db)
    if not tables:
        return sql
    # Supply a versioned projection only for referenced entities. No model/candidate lookups.
    views = []
    for t in tables:
        n, cols = registry(db)[t]
        projection = ", ".join(f"{quote('scope_v2_' + x)} AS {quote(x)}" for x in cols)
        views.append(f"{quote(n)} AS (SELECT {projection} FROM temp.{quote(n)})")
    import re

    head = re.match(r"^\s*WITH\s+(RECURSIVE\s+)?", sql, flags=re.I)
    return (
        (head.group() + ", ".join(views) + ", " + sql[head.end() :])
        if head
        else "WITH " + ", ".join(views) + " " + sql
    )


def repair(sql, db, active):
    rep = bind(sql, db)
    return compatibility(rep, db) if active else rep
