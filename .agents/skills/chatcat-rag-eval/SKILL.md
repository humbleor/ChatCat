---
name: chatcat-rag-eval
description: Evaluate ChatCat RAG retrieval and answer quality with the repository's versioned dataset and baseline gates. Use when asked to benchmark, regression-test, compare retrieval modes or embeddings, diagnose RAG quality, or update an explicitly approved RAG baseline; do not use for ordinary unit tests or unrelated RAG implementation work.
---

# ChatCat RAG Evaluation

Evaluate the current repository state with its existing `eval/` entry points. Run all commands from the repository root inside the configured WSL environment. Do not replace the evaluation scripts with ad hoc metrics.

## Choose the smallest useful evaluation

- For a quick environment or generation smoke test, run one to three samples with gates disabled.
- To compare dense, sparse/BM25, and hybrid retrieval before rerank and auto-merging, use `eval/eval_embeding.py`.
- To measure the real retrieval chain, including rerank, auto-merging, fallback, latency, and baseline gates, use `eval/eval_retrieval.py`.
- To evaluate complete graph answers with a separate LLM Judge invocation, use `eval/eval_answer.py`.
- For a comprehensive RAG evaluation or a detailed quality report, run full-dataset real retrieval followed by full-dataset judged answers. Keep the two result artifacts and interpret their failures separately.
- Use `eval/eval_langsmith.py` only when the user explicitly wants a LangSmith experiment; it is not the source of local regression baselines.

Read [references/workflows.md](references/workflows.md) before executing an evaluation or interpreting its output. Read only the sections for the selected mode.

## Example invocations

- `$chatcat-rag-eval 对当前 RAG 做 3 条样本的冒烟评测`
- `$chatcat-rag-eval 比较 dense、sparse 和 hybrid 的检索质量`
- `$chatcat-rag-eval 跑完整检索回归并说明失败样本`
- `$chatcat-rag-eval 评估完整回答质量和 Judge 分数`
- `$chatcat-rag-eval 对当前 RAG 做完整评估，给出逐项报告`
- `$chatcat-rag-eval 更新检索 baseline` (requires an explicit baseline update request)

## Operating rules

1. Inspect `git status --short` before running. Treat existing changes and files as user-owned; evaluation is read-only except for ignored `eval/results/`, evaluation database records, and an explicitly requested baseline update.
2. Check prerequisites for the selected mode. Do not start, stop, recreate, or delete infrastructure unless the user requested it. A missing service is a diagnostic result, not permission to mutate infrastructure.
3. Prefer a bounded smoke run before a full or costly run when the environment has not been validated in the current task.
4. Do not use `--update-baseline` unless the user explicitly asked to create or update a baseline. Never combine a baseline update with `--limit`; answer baselines also require the judge.
5. Reject degraded baselines by default. Use `--allow-degraded-baseline` only when the user explicitly accepts recording provider failures.
6. Preserve the committed dataset and baseline files unless their modification is the requested task. Do not rewrite gold chunks merely to make a regression pass.
7. On failure, identify the stage: dataset/schema, PostgreSQL/checkpointer, Milvus/embedding, retrieval/rerank, generation/HITL, judge, or baseline gate. Avoid presenting provider or infrastructure failures as quality regressions.

## Report the outcome

When the user requests a detailed report, save a human-readable Markdown artifact under `eval/reports/` and link it in the final response. The report must identify its source JSON artifacts and include findings, limitations, and follow-up actions. A chat-only summary or raw JSON does not fulfill a request for a generated report.

State the command and configuration used, dataset/sample count, result artifact path, relevant metrics, gate status, and any failed cases or environmental limitations. For comparisons, report absolute values and deltas rather than only saying “better” or “worse.” A successful process exit without a baseline means the run produced measurements; it does not establish regression safety.
