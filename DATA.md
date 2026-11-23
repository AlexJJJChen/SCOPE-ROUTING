# Data and experiment protocols

`checksums.json` records the SHA-256 of each file under `data/`. The reference tables contain the recorded paper observations; the launcher computes its outputs from the underlying inputs before comparing them.

## Spider

- The question IDs are zero-based indices in the Spider development split. Reference SQL must mention at least two distinct table names. The fixed alias-migration subset was selected before scoring by hashing the original IDs. The split, questions and reference SQL are in `data/spider/questions.csv`. The database hashes apply to the unpacked SQLite files. [Spider dataset](https://yale-lily.github.io/spider) · [Pinned database distribution](https://huggingface.co/datasets/HAL-9001/spider-databases/tree/4a01bbac6520cd35b216db9e1724e5e1ada60aa4)
- `predictions.csv.gz` aligns the saved model outputs by question ID. The zero-shot release stores a continuation of the prompt-final `SELECT`. The normalized predictions in this artifact include the SQL prefix. Graphix-T5, the DAIL single prediction and the DAIL vote come from the same pinned DAIL release. [THU-BPM zero-shot predictions](https://github.com/THU-BPM/chatgpt-sql/blob/116c7fc54f3134a24b93f147d834b58b84265da3/spider/predict.json) · [DAIL-SQL prediction files](https://github.com/BeachWang/DAIL-SQL/tree/2061f68112222083134a0c9e2877961ff315ff44/results)
- `lineage.csv.gz` has one row per question and configuration. A successful execution is checked for column count, ordered columns, row values and duplicates. A top-level reference ORDER BY makes row order significant. For P6 the served prediction is available but the call history is absent. Empty primary fields preserve that distinction. The separate sensitivity check assigns P6 zero auxiliary calls, zero repair time and zero switches and uses its served vote as primary. Against the raw zero-shot reference, it gains 72 paired answers and loses eight. Under these assumptions the attribution gate passes. The observed gate is MISSING. `vote_control.csv` supplies the archived DAIL single, zero-shot, DAIL vote and single-table candidates read by a four-access control on 300 requests.
- `migration.csv.gz` records the three dictionary conditions per question: original schema, migrated schema with stale names, and migrated schema with the current binder. The column prefix `scope_v2_` is part of the measured SQLite schema. The input prompt always describes the original schema.
- The optional Sol comparison comprises 459 completed Responses API calls on 2026-11-21. The configured relay, `api.openai-next.com`, returned the model ID `gpt-6.2-sol` for every response. It reported 65,105 input and 71,297 output tokens. The request used medium reasoning, no tools and a 4,096-output-token cap. Prompts and per-question usage are in `sol_prompts.jsonl.gz`; the saved SQL is in `predictions.csv.gz`. To generate a fresh sample using your own account, set `OPENAI_API_KEY` in your environment and run `python scripts/generate_sol.py --output /path/to/responses`. The script accepts an explicit HTTPS `--base-url`. This optional command makes billable requests; the replication launcher uses saved predictions.

## Claims and streams

The executable catalog in `scope/contract.py` includes the obligations, order edges and subject boundaries exercised by these experiments. Its SQL anti-join and incremental counters independently compute the reported frontiers. The static attribution gate combines paired binary loss bounds and the observed fallback rate. Primary validity and coverage across the three JOIN strata are additional obligations. A high streaming fallback upper bound withholds passage; the streaming attribution classifier returns FAIL when the paired-loss lower bound is nonnegative.

Each controlled route and Spider stream has 1,500 nominal, 600 shock and 1,400 recovery requests. Microbatches contain 25 requests and the rolling window contains up to 1,000. The route generator uses seed 20261120 and the Spider replay uses seed 1117. Recorded event-level checks are in `reference/`. The controlled routing stream regains its primary license at request 3,100. The SQL stream has zero primary licenses in 140 checkpoints. Its hybrid resilience credit is present at 58/60 nominal, 3/24 shock and 18/56 recovery checkpoints, including the final checkpoint.

## Routing

RouteScope-Lite contributes 22,500 policy observations across 1,500 requests. Its two stored rankings cover 14 non-oracle policies. The release reconstructs the fallback gate and computes Kendall's tau-b on those orders. The data do not expose the original per-request capped primary losses required for full primary attribution.

The separate CAP-DR RG64 experiment contains episodes 20–23 (64 requests) and five checkpoints. The graph input includes true edge objectives for evaluation, observed proxy features for path costs, per-request constraints, oracle cost and model edge scores. The replay reconstructs up to four feasible candidates, selects each model's and proxy's path, applies the single-substitution feasibility guard and compares paths and regret to the recorded decisions. An injected infeasible path exercises the repair branch independently of the natural requests.

`data/routing/checkpoints.json` pins the source episode archive and model weights. With those inputs and the CAP-DR inference dependencies available, the scores can be exported again:

```sh
python scripts/export_rg64.py --capdr-code /path/to/CAP-DR/code \
  --episodes /path/to/episodes.jsonl.gz --checkpoints /path/to/checkpoint-directory \
  --output /path/to/graphs.json.gz
```

The exporter requires separate access to the CAP-DR checkout at commit `07cdcc116a893ccf810905acdbc1840cd518baca`, the episodes and the five weight files. Each checkpoint subdirectory is named `seed_0` through `seed_4` and contains `capdr_final.pt`. The offline replay includes the CAP-DR path enumeration and feasibility routines in `scope/paths.py` and the exported edge scores for all five checkpoints.

## Performance

Server measurements used a Xeon Platinum 8358P with DuckDB 1.4.3. The single/eight-thread series contains 100 post-warmup samples per size, thread count and stage. The 22,500-row and million-row performance relations repeat measured telemetry with distinct pairing keys. The SHA-256 microbenchmark reopens four files totaling 406,320 bytes. The routing microbatch profile measures insertion, eviction, the KL update and frontier maintenance. Wall-clock measurements depend on the CPU, available workers and process load.
