# ChatCat RAG evaluation workflows

Use this reference only for the evaluation mode being run. Commands assume the repository root and the WSL development environment described by `AGENTS.md`.

## Shared inputs and prerequisites

- Dataset: `eval/rag_v1.jsonl` (inspect the current JSONL sample count before each run).
- Retrieval baseline: `eval/baselines/rag_v1-retrieval.json`.
- Answer baseline: `eval/baselines/rag_v1-answer.json`.
- Generated reports: `eval/results/*.json` (gitignored).
- Configuration: `.env`; never print secrets.
- Commands must use a WSL-native `uv`. If `uv --version` resolves to a `/mnt/c/` pyenv shim or fails with a `^M` interpreter error, diagnose with `type -a uv` and use the installed WSL binary; do not run the Windows shim against this workspace.
- Retrieval modes require the configured Milvus collection and embedding provider/model.
- Real retrieval additionally exercises reranking when enabled by the application configuration.
- Answer evaluation requires PostgreSQL/checkpointer initialization and the generation model. Judged runs require `LLM_API_KEY` plus `LLM_JUDGE_MODEL`, `LLM_FAST_MODEL`, or `LLM_MODEL`.

Before a live run, useful read-only checks are:

```bash
docker compose ps
git status --short
```

Do not assume an absent baseline is an error. The scripts emit measurements without a gate when no baseline exists.

## Fast smoke tests

Validate the real retrieval path without a gate:

```bash
uv run python eval/eval_retrieval.py --limit 3 --no-gate --warmup 0
```

Validate one end-to-end answer without paying for a judge call:

```bash
uv run python eval/eval_answer.py --limit 1 --no-judge --no-gate
```

Smoke runs answer “can this path execute?” They do not support baseline conclusions.

## Raw retrieval-mode comparison

Run dense, server-side sparse/BM25, and hybrid retrieval against leaf chunks, without rerank or auto-merging:

```bash
uv run python eval/eval_embeding.py --modes dense sparse hybrid --k 1 3 5 8 --top-k 8
```

For an initial bounded comparison, add `--limit 3`. The report is written as `eval/results/embedding-<timestamp>.json`.

Compare, per mode:

- document and evidence hit rates at each K;
- document MRR;
- mean latency;
- provider failures;
- individual returned chunk IDs for misses.

This mode isolates retrieval strategy. Do not attribute its result to rerank, auto-merging, or answer generation.

## Real retrieval regression

Trial without a gate:

```bash
uv run python eval/eval_retrieval.py --no-gate
```

Normal regression run using the committed baseline when present:

```bash
uv run python eval/eval_retrieval.py
```

The default gate allows at most a 0.03 absolute drop in document/evidence Hit@K, no increase in retrieval or rerank failure rate, and at most a 25% P95 latency increase. A failed gate exits with status 1.

Inspect these report fields:

- `summary.document_hit`, `evidence_hit`, `evidence_coverage`, and `document_mrr`;
- `summary.latency_ms`, `failure_rate`, and `empty_hits`;
- `summary.rerank`, `auto_merge`, `retrieval_mode`, and `by_tag`;
- `cases` for missed evidence, errors, latency outliers, and returned chunk IDs;
- `gates` for current, baseline, threshold, and pass/fail values.

Only after explicit approval, a healthy full-dataset run, and human review of the result may the baseline be written:

```bash
uv run python eval/eval_retrieval.py --update-baseline
```

Review the baseline diff before calling the update complete. Never use `--limit` for baseline creation.

## Complete answer evaluation

Full trial without a gate:

```bash
uv run python eval/eval_answer.py --no-gate
```

Normal regression run:

```bash
uv run python eval/eval_answer.py
```

Report groundedness, correctness, relevance, completeness, unsupported claims, generation failure rate, judge failure rate, and empty-answer rate. Separate generation/HITL failures from judge failures.

Only after explicit approval and review may a healthy, judged, full-dataset result become the answer baseline:

```bash
uv run python eval/eval_answer.py --update-baseline
```

The script rejects baseline updates made with `--limit`, `--no-judge`, generation failures, or judge failures.

## Comprehensive evaluation and detailed report

When the user requests a general RAG evaluation or a detailed quality report, run the complete `eval/rag_v1.jsonl` dataset through `eval/eval_retrieval.py`, then `eval/eval_answer.py` with the judge enabled. Use their normal baseline gates if baseline files exist; otherwise report that no baseline comparison was available. Do not create a baseline as part of an ordinary evaluation request. Save the detailed human-readable Markdown report under `eval/reports/`; include both JSON source paths and the dataset SHA, verify that the file exists, and link it in the final response.

For retrieval, include hit and coverage at each K, MRR, latency distribution, failure and empty-result counts, rerank/fallback/auto-merging activity, and the question IDs with missed evidence or unusual latency. For answers, include completed and judged counts, mean Judge metrics, unsupported claims, failure rates, and the weakest or failed question IDs with short evidence-backed explanations. Explain whether a poor answer is associated with missing retrieved evidence or with generation/judging. When a grounded answer conflicts with a reference answer, inspect the cited source before treating the Judge score as a model error; flag suspected dataset defects without editing gold labels during the evaluation. If one stage cannot run, report the successful stage and the exact blocker instead of inferring missing metrics.

The answer evaluator creates evaluation users, sessions, runs, and checkpoints in PostgreSQL and sends generation and Judge requests to configured model providers. Mention these side effects and model usage before starting a comprehensive run. Record the generation and Judge model names; they may resolve to the same model. Keep a detailed report concise enough to scan, with per-question findings focused on failures and outliers rather than dumping every returned chunk.

## Optional LangSmith experiment

Use only when explicitly requested and after confirming the LangSmith environment variables and the cloud dataset named `RAG`:

```bash
uv run python eval/eval_langsmith.py
```

Treat this as a remote experiment. Do not substitute it for the local retrieval or answer baselines.

## Failure classification

- Dataset/schema: malformed JSONL, missing fields, duplicate IDs, empty `gold_chunks`, or gold chunks absent from the current collection.
- Milvus/embedding: missing collection, connection/proxy issue, embedding error, collection schema or dense dimension mismatch.
- Retrieval/rerank: provider exceptions, empty hits, fallback activation, or rerank errors.
- PostgreSQL/checkpointer: database connection, schema initialization, checkpoint, user, session, or ChatRun persistence failure.
- Generation/HITL: model failure, empty answer, or `waiting_hitl` during a supposedly answerable evaluation sample.
- Judge: missing judge configuration, invalid JSON, timeout, or provider failure.
- Gate: the run completed, but one or more measured regressions crossed the configured threshold.

When comparing two reports, first confirm matching dataset SHA, sample count, `top_k`, K values, models/providers, and relevant feature flags. If these differ, label the comparison as non-equivalent.
