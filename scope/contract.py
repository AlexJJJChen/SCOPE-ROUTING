"""Auditable executable test contract. Missing evidence never licenses a claim."""

import math
from functools import lru_cache

import duckdb
from scipy.stats import beta

# Each obligation list is cumulative. The primary subject is a diamond.
OBL = {
    "C0": {"attr", "lock"},
    "C3attr": {"attr", "lock", "valid"},
    "C3pref": {"attr", "lock", "coverage"},
    "C4pa": {"attr", "lock", "valid", "coverage"},
    "C5": {"attr", "lock", "valid", "coverage", "dynamics"},
    "hyb_resil": {"hybrid"},
    "ast_probe": {"collapse"},
}
EDGES = [
    ("C0", "C3attr"),
    ("C0", "C3pref"),
    ("C3attr", "C4pa"),
    ("C3pref", "C4pa"),
    ("C4pa", "C5"),
]
CROSS = {"hyb_resil", "ast_probe"}
TC = {(k, k) for k in OBL} | set(EDGES)
while True:
    new = {(a, d) for a, b in TC for c, d in TC if b == c} - TC
    if not new:
        break
    TC |= new


def frontier(states, target):
    live = {c for c, os in OBL.items() if all(states.get(o) == "PASS" for o in os)}
    within = {c for c in live if (c, target) in TC}
    return sorted(
        c for c in within if not any(d != c and (c, d) in TC for d in within)
    ), sorted(live & CROSS)


QUERY = """WITH licensed AS (
 SELECT c FROM claims WHERE NOT EXISTS (SELECT 1 FROM obligations o LEFT JOIN gates g USING (o)
 WHERE o.c=claims.c AND NOT coalesce(g.pass,false))),
 candidates AS (SELECT c FROM licensed JOIN tc ON c=anc WHERE dst=?)
 SELECT 'C', a.c FROM candidates a WHERE NOT EXISTS
 (SELECT 1 FROM candidates b JOIN tc ON tc.dst=b.c WHERE tc.anc=a.c AND b.c<>a.c)
 UNION ALL SELECT 'R', c FROM licensed WHERE c IN ('hyb_resil','ast_probe') ORDER BY 1,2"""


def db_engine():
    db = duckdb.connect()
    db.execute("SET threads=1")
    db.execute(
        "CREATE TABLE claims(c VARCHAR); CREATE TABLE edges(anc VARCHAR,dst VARCHAR); CREATE TABLE obligations(c VARCHAR,o VARCHAR); CREATE TABLE gates(o VARCHAR,pass BOOLEAN)"
    )
    db.executemany("INSERT INTO claims VALUES (?)", [(c,) for c in OBL])
    db.executemany("INSERT INTO edges VALUES (?,?)", EDGES)
    db.executemany(
        "INSERT INTO obligations VALUES (?,?)",
        [(c, o) for c, obs in OBL.items() for o in obs],
    )
    db.execute("""CREATE TABLE tc AS WITH RECURSIVE reach(anc,dst) AS
    (SELECT c,c FROM claims UNION SELECT r.anc,e.dst FROM reach r JOIN edges e ON r.dst=e.anc)
    SELECT DISTINCT * FROM reach""")
    assert set(db.execute("SELECT * FROM tc").fetchall()) == TC
    return db


def set_gates(db, states):
    db.execute("DELETE FROM gates")
    # Single statement includes catalog loading when used in full-recompute benchmark.
    keys = list(states)
    db.execute(
        "INSERT INTO gates SELECT unnest(?), unnest(?)",
        [keys, [states[k] == "PASS" for k in keys]],
    )


def sql_frontier(db, states, target):
    set_gates(db, states)
    rows = db.execute(QUERY, [target]).fetchall()
    return [c for t, c in rows if t == "C"], [c for t, c in rows if t == "R"]


@lru_cache(65536)
def bounds(k, n, alpha):
    if n <= 0:
        return 0.0, 1.0
    return (
        float(beta.ppf(alpha, k, n - k + 1)) if k else 0.0,
        float(beta.ppf(1 - alpha, k + 1, n - k)) if k < n else 1.0,
    )


def classify(fb, good, bad, n, tau, unknown=False):
    # Bonferroni across the upper or lower directional decision. Two marginal
    # binomial bounds on discordant outcome types bound the paired mean difference.
    if unknown or not n:
        return "MISSING", None, None, None, None
    flo, fhi = bounds(fb, n, 0.025)
    gl, gh = bounds(good, n, 0.0125)
    bl, bh = bounds(bad, n, 0.0125)
    lo, hi = bl - gh, bh - gl
    if fhi <= tau and hi < 0:
        state = "PASS"
    elif flo > tau or lo >= 0:
        state = "FAIL"
    else:
        state = "UNRESOLVED"
    return state, flo, fhi, lo, hi


def kl_upper(h, n, k, alpha=0.05):
    if h == n or n == 0:
        return 1.0
    q = h / n
    cost = -math.log(6 * alpha / (math.pi**2 * k * k)) / n
    low = q
    high = 1.0
    for _ in range(38):
        mid = (low + high) / 2
        kl = (q * math.log(q / mid) if h else 0) + (1 - q) * (
            math.log1p(-q) - math.log1p(-mid)
        )
        if kl < cost:
            low = mid
        else:
            high = mid
    return high
