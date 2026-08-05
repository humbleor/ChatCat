import os

# 保证纯函数测试不触发外部连接；各测试如需真实服务自行设 env 标记
os.environ.setdefault("LLM_API_KEY", "")
os.environ.setdefault("MILVUS_HOST", "localhost")

# 以下为历史遗留的独立实验脚本（非 pytest 用例：顶层代码含模型下载 / LLM 调用 /
# 外部文件读取 / LangSmith evaluate()），会破坏 pytest 收集，故在此忽略。
# 待后续任务清理后移除。
collect_ignore = [
    "test_doc_loader.py",
    "test_embedding.py",
    "test_langsmith_eval.py",
    "test_milvus.py",
    "test_model.py",
]
