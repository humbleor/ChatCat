import os
from typing import Optional

import requests
from langchain_core.tools import tool

AMAP_WEATHER_API = os.getenv("AMAP_WEATHER_API")
AMAP_API_KEY = os.getenv("AMAP_API_KEY")


def emit_rag_step(icon: str, label: str, detail: str = "") -> None:
    """向当前图的 custom 流发送一个 RAG 步骤（图节点内调用有效，否则静默跳过）。"""
    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()({"type": "rag_step", "step": {"icon": icon, "label": label, "detail": detail}})
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
            label = "今日天气" if i == 0 else f"未来第{i}天 ({day.get('date', '')})"
            out += [
                f"{label}：",
                f"  白天：{day.get('dayweather', '未知')}",
                f"  夜间：{day.get('nightweather', '未知')}",
                f"  气温：{day.get('nighttemp', '未知')}~{day.get('daytemp', '未知')}℃",
                f"  风向：{day.get('daywind', '未知')} {day.get('daypower', '未知')}级",
                f"  降水量：{day.get('daytemp_float', '未知')}",
            ]
        return "\n".join(out)

    except requests.exceptions.Timeout:
        return "错误：请求天气服务超时"
    except requests.exceptions.RequestException as e:
        return f"错误：天气服务请求失败 - {e}"
    except Exception as e:
        return f"错误：解析天气数据失败 - {e}"
