from typing import Optional
import os
import contextvars
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

AMAP_WEATHER_API = os.getenv("AMAP_WEATHER_API")
AMAP_API_KEY = os.getenv("AMAP_API_KEY")

# Per-request state — 优先用 contextvars（同线程零开销），
# 同时维护模块级全局变量作为线程池跨线程访问的 fallback。
# contextvars 从主线程复制到线程池后，子线程能读但不能写回主线程，
# 因此 _last_rag_context 必须以全局变量为主存储。
_last_rag_context: contextvars.ContextVar = contextvars.ContextVar("last_rag_context", default=None)
_knowledge_tool_calls: contextvars.ContextVar = contextvars.ContextVar("knowledge_tool_calls", default=0)
_rag_step_queue: contextvars.ContextVar = contextvars.ContextVar("rag_step_queue", default=None)
_rag_step_loop: contextvars.ContextVar = contextvars.ContextVar("rag_step_loop", default=None)

# 模块级 fallback，供线程池线程写入后主线程读取
_global_rag_context: Optional[dict] = None
_global_rag_queue: Optional[object] = None
_global_rag_loop: Optional[object] = None


def _set_last_rag_context(context: dict):
    global _global_rag_context
    _last_rag_context.set(context)
    _global_rag_context = context


def get_last_rag_context(clear: bool = True) -> Optional[dict]:
    """获取最近一次 RAG 检索上下文，默认读取后清空。"""
    global _global_rag_context
    ctx = _last_rag_context.get() or _global_rag_context
    if clear:
        _last_rag_context.set(None)
        _global_rag_context = None
    return ctx


def reset_tool_call_guards():
    """每轮对话开始时重置工具调用计数。"""
    _knowledge_tool_calls.set(0)


def set_rag_step_queue(queue):
    """设置 RAG 步骤队列及其事件循环。"""
    global _global_rag_queue, _global_rag_loop
    _rag_step_queue.set(queue)
    _global_rag_queue = queue
    if queue:
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        _rag_step_loop.set(loop)
        _global_rag_loop = loop
    else:
        _rag_step_loop.set(None)
        _global_rag_loop = None


def emit_rag_step(icon: str, label: str, detail: str = ""):
    """向当前请求的输出队列发送一个 RAG 检索步骤。

    优先从 contextvars 读取队列和事件循环（同线程），失败时回退到
    模块级全局变量（跨线程）。当两者均不可用时静默跳过。
    """
    queue = _rag_step_queue.get() or _global_rag_queue
    loop = _rag_step_loop.get() or _global_rag_loop
    if queue is not None and loop is not None:
        step = {"icon": icon, "label": label, "detail": detail}
        try:
            if not loop.is_closed():
                loop.call_soon_threadsafe(queue.put_nowait, step)
        except Exception:
            pass


@tool("get_current_weather")
def get_current_weather(location: str, extensions: Optional[str] = "base") -> str:
    """获取指定城市的天气信息。location 为城市名（如"武汉"或"420100"）。
    extensions: "base" 仅返回实时天气，"all" 返回未来多日天气预报。用户询问未来天气时必须使用 extensions="all"。"""
    if not location:
        return "location参数不能为空"
    if extensions not in ("base", "all"):
        return "extensions参数错误，请输入base或all"

    if not AMAP_WEATHER_API or not AMAP_API_KEY:
        return "天气服务未配置（缺少 AMAP_WEATHER_API 或 AMAP_API_KEY）"

    params = {
        "key": AMAP_API_KEY,
        "city": location,
        "extensions": extensions,
        "output": "json",
    }

    try:
        resp = requests.get(AMAP_WEATHER_API, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "1":
            return f"查询失败：{data.get('info', '未知错误')}"

        if extensions == "base":
            lives = data.get("lives", [])
            if not lives:
                return f"未查询到 {location} 的天气数据"
            w = lives[0]
            return (
                f"【{w.get('city', location)} 实时天气】\n"
                f"天气状况：{w.get('weather', '未知')}\n"
                f"温度：{w.get('temperature', '未知')}℃\n"
                f"湿度：{w.get('humidity', '未知')}%\n"
                f"风向：{w.get('winddirection', '未知')}\n"
                f"风力：{w.get('windpower', '未知')}级\n"
                f"更新时间：{w.get('reporttime', '未知')}"
            )

        forecasts = data.get("forecasts", [])
        if not forecasts:
            return f"未查询到 {location} 的天气预报数据"
        f0 = forecasts[0]
        out = [f"【{f0.get('city', location)} 天气预报】", f"更新时间：{f0.get('reporttime', '未知')}", ""]
        casts = f0.get("casts") or []
        if not casts:
            return f"未查询到 {location} 的天气预报数据"
        for i, day in enumerate(casts):
            label = "今日天气" if i == 0 else f"未来第{i}天 ({day.get('date','')})"
            out += [
                f"{label}：",
                f"  白天：{day.get('dayweather','未知')}",
                f"  夜间：{day.get('nightweather','未知')}",
                f"  气温：{day.get('nighttemp','未知')}~{day.get('daytemp','未知')}℃",
                f"  风向：{day.get('daywind','未知')} {day.get('daypower','未知')}级",
                f"  降水量：{day.get('daytemp_float','未知')}",
            ]
        return "\n".join(out)

    except requests.exceptions.Timeout:
        return "错误：请求天气服务超时"
    except requests.exceptions.RequestException as e:
        return f"错误：天气服务请求失败 - {e}"
    except Exception as e:
        return f"错误：解析天气数据失败 - {e}"


@tool("search_knowledge_base")
def search_knowledge_base(query: str) -> str:
    """Search for information in the knowledge base using hybrid retrieval (dense + sparse vectors)."""
    calls = _knowledge_tool_calls.get()
    if calls >= 1:
        return (
            "TOOL_CALL_LIMIT_REACHED: search_knowledge_base has already been called once in this turn. "
            "Use the existing retrieval result and provide the final answer directly."
        )
    _knowledge_tool_calls.set(calls + 1)

    from rag_pipeline import run_rag_graph

    rag_result = run_rag_graph(query)

    docs = rag_result.get("docs", []) if isinstance(rag_result, dict) else []
    rag_trace = rag_result.get("rag_trace", {}) if isinstance(rag_result, dict) else {}
    if rag_trace:
        _set_last_rag_context({"rag_trace": rag_trace})

    if not docs:
        return "No relevant documents found in the knowledge base."

    formatted = []
    for i, result in enumerate(docs, 1):
        source = result.get("filename", "Unknown")
        page = result.get("page_number", "N/A")
        text = result.get("text", "")
        formatted.append(f"[{i}] {source} (Page {page}):\n{text}")

    return "Retrieved Chunks:\n" + "\n\n---\n\n".join(formatted)
