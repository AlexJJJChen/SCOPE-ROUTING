# SCOPE

**Claim-Aware Data Systems: Managing Capability Claims as Stateful Views for Composite AI Pipelines**

SCOPE evaluates a pipeline's primary model, bounded repair and fallback solver separately. A DuckDB contract computes the strongest supported primary claims and the independently supported hybrid capabilities. The artifact includes frozen per-query SQL lineage, scored routing graphs, both 3,500-request streams, the repeated-testing simulation and the raw timing observations.

## Reproduce the paper

Use Python 3.11 or 3.12. From the repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
./run_all.sh
```

The default run uses the bundled data offline. It verifies the input hashes, recomputes the statistics and compares them with the reference files. It also re-enumerates routing candidates from graph data, runs the sliding windows incrementally and in SQL, and simulates 500 trajectories per testing condition. Output goes to `results/`. Use `SCOPE_OUTPUT` to choose another directory or `PYTHON` to select the interpreter.

| Research question | Output in `results/` |
|---|---|
| Which path produced the correct SQL answer? | `spider_accuracy.csv`, `paired_intervals.csv` |
| Which claims clear the 5% fallback and 15 ms repair limits? | `default_gates.csv`, `gateway.json` |
| How do the baseline, fallback cap and repair budget affect the decision? | `spider_grid.csv` |
| How does imputing zero hidden calls affect the public vote? | `vote_sensitivity.json`, `instrumented_vote.csv` |
| How does model-based routing compare with proxy costs? | `routing_policies.csv`, `routing.json`, `rg64.csv` |
| When are claims licensed, withdrawn and restored? | `route_stream.csv`, `sql_checkpoints.csv`, `stream_summary.json` |
| How often do repeated tests license a noncompliant stream? | `monte_carlo.csv`, `repeated_testing.json` |
| What did the instrumented server runs cost? | `instrumentation.csv`, `duckdb_timings.csv`, `stream_timing.json` |

## Interpreting the SQL configurations

All 459 multi-table questions are scored on one fixed workload spanning 20 databases. In 126 cases the execution schema changes after prediction. The binder projects catalog column names through compatibility views. Primary credit requires an admissible call count and repair time. A missing observation stays null.

| Pipeline | Primary path and fallback |
|---|---|
| P1 | DAIL-SQL with the catalog binder. Retry residual execution errors with repaired zero-shot SQL. |
| P2 | DAIL-SQL single prediction. |
| P3 | Graphix-T5. Retry an execution error or empty result with the schema-aware DAIL vote. |
| P4 | Zero-shot ChatGPT. Retry an execution error, empty result or prediction referencing fewer than two tables with the same vote. |
| P5 | Zero-shot ChatGPT. Retry execution errors with schema-aware single DAIL predictions. |
| P6 | Published DAIL vote. Its final SQL is observed and its primary call counts are missing. |
| P7 | Single-table projection probe. |

P1 answers 330 questions correctly with eight solver switches. P3 answers 337 correctly with 166 switches. Against the equally repaired zero-shot reference, P1 receives the primary license at the default threshold; P3 receives hybrid resilience credit. The 5,000-draw confidence intervals resample whole databases.

The supplementary GPT-6.2 Sol experiment uses the same questions and original-schema zero-shot prompts. The result files identify its four configurations with the prefix `Sol`. Sol with binding and error retry serves 310 correct answers; Sol with error-or-empty deferral serves 320 and switches to the external solver on 150 questions. Choosing repaired Sol as the reference changes which primary claims clear the gate. The model identifier is the one reported by the inference service. Sol and DAIL used different prompting protocols.

## Re-execute SQL and measure local runtime

```sh
python scripts/prepare_spider.py
./run_all.sh --sqlite
./run_all.sh --measure
```

The preparation step fetches the pinned Spider archive (about 206 MB), verifies its SHA-256 and extracts only the 20 evaluation databases. It also accepts `--archive /path/to/spider_data.zip`. Set `SCOPE_DB` to use an existing verified directory with the layout `<db>/<db>.sqlite`.

`--sqlite` rebuilds submitted, repaired and served SQL, executes it against read-only SQLite connections, checks all 459 reference answers on both schemas, and verifies all 6,885 mixed-schema pipeline records and 1,377 drift inputs. It writes the original-schema control results as well. The gate sweep uses the recorded repair latencies so that machine speed does not change the published decisions.

`--measure` collects fresh on/off interception, input rehash and one/eight-thread DuckDB measurements. Both flags can be combined. Read the measured host's timings from the `local_*` outputs. The bundled Xeon measurements are summarized separately from the current host's data.

[Input provenance, coverage and optional model inference](DATA.md)
