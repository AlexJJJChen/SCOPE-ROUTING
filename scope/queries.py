"""Paired empirical Bernstein bounds and the composed frontier query."""

from .contract import QUERY

STATS = """WITH g AS (
SELECT arm,count(*) n,count(fb) nf,count(cap-base_loss) nd,
avg(fb) fm, stddev_samp(fb) fs,avg(cap-base_loss) dm,stddev_samp(cap-base_loss) ds
FROM tele LEFT JOIN refs USING(qid) WHERE feasible=1 GROUP BY arm),
r AS (SELECT *,sqrt(2*ln(2/.025)/nullif(n,0)) s,7*ln(2/.025)/(3*nullif(n-1,0)) c FROM g)
 , bounded AS (SELECT arm,n,nf,nd,fm-fs*s-c fb_lo,fm+fs*s+c fb_hi,dm-ds*s-2*c d_lo,dm+ds*s+2*c d_hi FROM r)
 SELECT *, CASE WHEN nf<>n OR nd<>n THEN 'MISSING' WHEN n<2 THEN 'UNRESOLVED'
 WHEN fb_hi<=0.05 AND d_hi<0 THEN 'PASS' WHEN fb_lo>0.05 OR d_lo>=0 THEN 'FAIL'
 ELSE 'UNRESOLVED' END AS decision FROM bounded ORDER BY arm"""
FULL_QUERY = (
    "WITH evaluated AS (" + STATS + "), runtime_gates AS "
    "(SELECT o,pass FROM gates WHERE o<>'attr' UNION ALL SELECT 'attr',decision='PASS' FROM evaluated), "
    + QUERY.removeprefix("WITH ").replace(
        "LEFT JOIN gates g", "LEFT JOIN runtime_gates g"
    )
)
