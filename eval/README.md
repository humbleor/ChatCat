# ChatCat RAG Evaluation

正式数据集是 `eval/rag_v1.jsonl`。每条样本包含稳定 ID、问题、参考答案、当前
Milvus 的 gold leaf chunks 和标签。

## 1. 裸检索对比

比较 dense、sparse/BM25 和 hybrid，不经过线上 rerank、auto-merging 或回答生成：

```bash
uv run python eval/eval_embeding.py
uv run python eval/eval_embeding.py --modes dense sparse hybrid --k 1 3 5 8
```

结果写入 `eval/results/embedding-*.json`。

## 2. 真实检索链与 baseline

运行线上 `retrieve_documents()`，覆盖 hybrid、rerank、auto-merging 和 fallback：

```bash
# 先试跑，不执行 gate
uv run python eval/eval_retrieval.py --no-gate

# 在所有 Provider 健康、结果经人工确认后建立 baseline
uv run python eval/eval_retrieval.py --update-baseline

# 后续回归评测：自动读取 baseline，失败时退出码为 1
uv run python eval/eval_retrieval.py
```

baseline 保存到 `eval/baselines/rag_v1-retrieval.json`，应提交 Git。
原始运行结果写入已忽略的 `eval/results/`。

默认 gate：

- Document Hit@K 和 Evidence Hit@K 相对 baseline 最多下降 0.03；
- retrieval/rerank failure rate 不得增加；
- P95 延迟最多增加 25%。

Provider 失败时默认拒绝更新 baseline。只有明确需要记录降级环境时才使用
`--allow-degraded-baseline`。

## 3. 完整回答与 baseline

每条样本创建独立 Session 和 ChatRun，运行完整图，并使用独立 Judge 模型评分：

```bash
# 一条无 Judge 冒烟
uv run python eval/eval_answer.py --limit 1 --no-judge --no-gate

# 完整试跑
uv run python eval/eval_answer.py --no-gate

# 人工检查结果后建立答案 baseline
uv run python eval/eval_answer.py --update-baseline

# 后续回归 gate
uv run python eval/eval_answer.py
```

可通过 `LLM_JUDGE_MODEL` 指定 Judge；未配置时依次回退到
`LLM_FAST_MODEL`、`LLM_MODEL`。指标包括 correctness、groundedness、
relevance、completeness、unsupported claims，以及 generation/judge/empty
失败率。

若任何 generation 或 Judge 请求失败，脚本拒绝更新答案 baseline。

## 4. LangSmith

`eval_langsmith.py` 是可选的 LangSmith 实验入口，仍依赖云端名为 `RAG`
的数据集，不是本地 baseline 的来源。正式回归以 `eval_retrieval.py` 和
`eval_answer.py` 为准。
