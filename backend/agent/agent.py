"""聊天编排层：驱动统一 StateGraph（HITL + 持久化笔记 + 标题）完成一轮对话。

同步版 `chat_with_agent` 与流式版 `chat_with_agent_stream` 共享
「跑图 + 收事件 + 收尾持久化」的同一套逻辑。
"""

import asyncio
import json
import logging
import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command

from backend.infra.cache import cache
from backend.infra.database import SessionLocal
from backend.models.models import ChatMessage, ChatSession, User
from backend.rag.hitl_detect import format_hitl_message, normalize_rag_trace

load_dotenv()

logger = logging.getLogger(__name__)

# ---------- 快模型（笔记维护等轻任务） ----------
_fast_model = None


def _get_fast_model():
    """笔记/标题等轻任务用的廉价快模型（LLM_FAST_MODEL，缺省 LLM_MODEL）。"""
    global _fast_model
    if _fast_model is None:
        from langchain.chat_models import init_chat_model

        _fast_model = init_chat_model(
            model=os.getenv("LLM_FAST_MODEL") or os.getenv("LLM_MODEL"),
            model_provider="openai",
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            temperature=0.0,
        )
    return _fast_model


def _get_context_model():
    """_manage_context_window 摘要用的模型。"""
    return _get_fast_model()


# Tokenizer setup
_tokenizer = None
TOKENIZER_ENCODING = os.getenv("TOKENIZER_ENCODING", "o200k_base")


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        import tiktoken

        try:
            _tokenizer = tiktoken.get_encoding(TOKENIZER_ENCODING)
        except Exception:
            _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def count_tokens(text: str) -> int:
    """返回文本的 token 数量，失败时返回字符数/4 的粗略估算。"""
    try:
        return len(_get_tokenizer().encode(text))
    except Exception:
        return len(text) // 4


class ConversationStorage:
    """对话存储(PostgreSQL + Redis), append-only + superseded_by 标记。"""

    _CACHE_VERSION = "v2"  # bump when cached data format changes

    @staticmethod
    def _messages_cache_key(user_id: str, session_id: str) -> str:
        return f"chat_messages:{ConversationStorage._CACHE_VERSION}:{user_id}:{session_id}"

    @staticmethod
    def _sessions_cache_key(user_id: str) -> str:
        return f"chat_sessions:{ConversationStorage._CACHE_VERSION}:{user_id}"

    @staticmethod
    def _to_langchain_messages(records: list[dict]) -> list:
        """将 DB 记录转为 LangChain 消息对象，同时在 additional_kwargs 中注入 _db_id 和 _token_count。"""
        messages = []
        for msg_data in records:
            msg_type = msg_data.get("type")
            content = msg_data.get("content", "")
            extra = {
                "_db_id": msg_data.get("id"),
                "_token_count": msg_data.get("token_count", 0),
            }
            if msg_type in ("system", "summary"):
                extra["_msg_type"] = msg_type
                extra["_is_summary"] = msg_type == "summary"

            if msg_type == "human":
                msg = HumanMessage(content=content)
            elif msg_type == "ai":
                msg = AIMessage(content=content)
            else:
                msg = SystemMessage(content=content)
            msg.additional_kwargs.update(extra)
            messages.append(msg)
        return messages

    def save(
        self, user_id: str, session_id: str, messages: list, metadata: dict = None, extra_message_data: list = None
    ):
        """增量保存对话：只 INSERT 新消息，UPDATE superseded_by 标记。"""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return

            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                session = ChatSession(user_id=user.id, session_id=session_id, metadata_json=metadata or {})
                db.add(session)
                db.flush()
            elif metadata:
                session.metadata_json = metadata

            # 获取当前最大 message_index
            max_idx_row = (
                db.query(ChatMessage.message_index)
                .filter(ChatMessage.session_ref_id == session.id)
                .order_by(ChatMessage.message_index.desc())
                .first()
            )
            next_index = (max_idx_row[0] + 1) if max_idx_row else 0

            serialized = []
            now = datetime.utcnow()
            for idx, msg in enumerate(messages):
                db_id = msg.additional_kwargs.get("_db_id")
                supersedes_ids = msg.additional_kwargs.get("_supersedes_ids", [])

                if db_id is not None:
                    # 已有消息：检查是否需要更新 superseded_by
                    if supersedes_ids:
                        self._mark_superseded(db, msg, db_id, supersedes_ids)
                    serialized.append(
                        {
                            "type": msg.type,
                            "content": str(msg.content),
                            "id": db_id,
                            "timestamp": msg.additional_kwargs.get("_timestamp", now.isoformat()),
                            "token_count": msg.additional_kwargs.get("_token_count", 0),
                            "rag_trace": normalize_rag_trace(msg.additional_kwargs.get("_rag_trace")),
                        }
                    )
                    continue

                # 新消息：INSERT
                rag_trace = normalize_rag_trace(None)
                if extra_message_data and idx < len(extra_message_data):
                    extra = extra_message_data[idx] or {}
                    rag_trace = normalize_rag_trace(extra.get("rag_trace"))

                tk = count_tokens(str(msg.content))
                msg_type = msg.additional_kwargs.get("_msg_type", msg.type)
                if msg_type == "summary":
                    msg_type = "summary"

                new_msg = ChatMessage(
                    session_ref_id=session.id,
                    message_type=msg_type,
                    content=str(msg.content),
                    timestamp=now,
                    rag_trace=rag_trace,
                    token_count=tk,
                    message_index=next_index,
                )
                db.add(new_msg)
                db.flush()  # 获取 id

                db_id = new_msg.id
                next_index += 1

                # 标记被此摘要覆盖的消息
                if supersedes_ids and db_id:
                    self._mark_superseded(db, msg, db_id, supersedes_ids)

                serialized.append(
                    {
                        "type": msg_type,
                        "content": str(msg.content),
                        "id": db_id,
                        "token_count": tk,
                        "timestamp": now.isoformat(),
                        "rag_trace": rag_trace,
                    }
                )

            session.updated_at = now
            db.commit()

            cache.set_json(self._messages_cache_key(user_id, session_id), serialized)
            cache.delete(self._sessions_cache_key(user_id))
        except Exception:
            db.rollback()
            logger.exception("Failed to save conversation for user=%s session=%s", user_id, session_id)
        finally:
            db.close()

    @staticmethod
    def _mark_superseded(db, msg, summary_db_id: int, supersedes_ids: list[int]):
        """将 supersedes_ids 中的消息标记为被 summary_db_id 替代。"""
        if not supersedes_ids:
            return
        db.query(ChatMessage).filter(
            ChatMessage.id.in_(supersedes_ids),
            ChatMessage.superseded_by.is_(None),
        ).update(
            {"superseded_by": summary_db_id},
            synchronize_session=False,
        )

    def load(self, user_id: str, session_id: str) -> list:
        """加载对话（过滤 superseded_by IS NULL）。"""
        cached = cache.get_json(self._messages_cache_key(user_id, session_id))
        if cached is not None:
            return self._to_langchain_messages(cached)

        records = self.get_session_messages(user_id, session_id)
        cache.set_json(self._messages_cache_key(user_id, session_id), records)
        return self._to_langchain_messages(records)

    def load_with_meta(self, user_id: str, session_id: str) -> tuple[list, dict]:
        """加载对话消息及会话元数据（标题、持久化笔记、HITL 状态等）。"""
        messages = self.load(user_id, session_id)
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return messages, {}
            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return messages, {}
            return messages, dict(session.metadata_json or {})
        finally:
            db.close()

    def list_sessions(self, user_id: str) -> list:
        """列出用户的所有会话"""
        return [item["session_id"] for item in self.list_session_infos(user_id)]

    def list_session_infos(self, user_id: str) -> list[dict]:
        cached = cache.get_json(self._sessions_cache_key(user_id))
        if cached is not None:
            return cached

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return []

            sessions = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id)
                .order_by(ChatSession.updated_at.desc())
                .all()
            )
            result = []
            for s in sessions:
                count = (
                    db.query(ChatMessage)
                    .filter(ChatMessage.session_ref_id == s.id, ChatMessage.superseded_by.is_(None))
                    .count()
                )
                result.append(
                    {
                        "session_id": s.session_id,
                        "updated_at": s.updated_at.isoformat(),
                        "message_count": count,
                        "title": (s.metadata_json or {}).get("title"),
                    }
                )
            cache.set_json(self._sessions_cache_key(user_id), result)
            return result
        finally:
            db.close()

    def get_session_messages(self, user_id: str, session_id: str) -> list[dict]:
        cached = cache.get_json(self._messages_cache_key(user_id, session_id))
        if cached is not None:
            return cached

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return []
            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return []

            rows = (
                db.query(ChatMessage)
                .filter(
                    ChatMessage.session_ref_id == session.id,
                    ChatMessage.superseded_by.is_(None),
                )
                .order_by(ChatMessage.message_index.asc())
                .all()
            )
            result = [
                {
                    "id": row.id,
                    "type": row.message_type,
                    "content": row.content,
                    "timestamp": row.timestamp.isoformat(),
                    "rag_trace": normalize_rag_trace(row.rag_trace),
                    "token_count": row.token_count,
                }
                for row in rows
            ]
            cache.set_json(self._messages_cache_key(user_id, session_id), result)
            return result
        finally:
            db.close()

    def delete_session(self, user_id: str, session_id: str) -> bool:
        """删除指定用户的会话（CASCADE 删除消息），返回是否删除成功。"""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return False
            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return False

            db.delete(session)
            db.commit()
            cache.delete(self._messages_cache_key(user_id, session_id))
            cache.delete(self._sessions_cache_key(user_id))
            return True
        finally:
            db.close()


storage = ConversationStorage()

# 上下文窗口预算
CONTEXT_WINDOW_TOKENS = int(os.getenv("CONTEXT_WINDOW_TOKENS", "128000"))
CONTEXT_BUDGET_RATIO = 0.8
RECENT_TURNS_KEEP = 5  # 始终保留最近 N 轮


def _available_budget() -> int:
    """可用于历史消息的总 token 预算。"""
    total = int(CONTEXT_WINDOW_TOKENS * CONTEXT_BUDGET_RATIO)
    return max(total, 4000)


def summarize_old_messages(model, messages: list) -> str:
    """将旧消息总结为摘要。"""
    old_conversation = "\n".join([f"{'用户' if msg.type == 'human' else 'AI'}: {msg.content}" for msg in messages])

    summary_prompt = f"""请总结以下对话的关键信息：

{old_conversation}
总结（包含用户信息、重要事实、待办事项）："""

    summary = model.invoke(summary_prompt).content
    return summary


def _manage_context_window(messages: list, model) -> list:
    """Token-aware 上下文窗口管理。

    从旧到新累积 token，超出预算时总结最旧的非摘要消息。
    已标记为 _is_summary 的消息不会被二次压缩。
    """
    budget = _available_budget()
    total_tokens = sum(msg.additional_kwargs.get("_token_count", 0) for msg in messages)

    if total_tokens <= budget:
        return messages

    # 从旧到新遍历，找出需要总结的消息范围
    accumulated = 0
    split_idx = 0
    for i, msg in enumerate(messages):
        tk = msg.additional_kwargs.get("_token_count", 0)
        accumulated += tk
        split_idx = i + 1
        # 保留的消息从后往前算，当保留部分 + 当前 <= 预算时停止
        remaining = total_tokens - accumulated
        if remaining <= budget:
            break

    if split_idx == 0:
        return messages

    old_messages = messages[:split_idx]
    # 已经被总结过的消息不再压缩
    candidates = [m for m in old_messages if not m.additional_kwargs.get("_is_summary")]

    if not candidates:
        return messages  # 全是摘要，无法进一步压缩

    summary = summarize_old_messages(model, candidates)
    supersedes_ids = [m.additional_kwargs["_db_id"] for m in candidates if m.additional_kwargs.get("_db_id")]

    summary_msg = SystemMessage(content=f"之前的对话摘要：\n{summary}")
    summary_msg.additional_kwargs["_msg_type"] = "summary"
    summary_msg.additional_kwargs["_is_summary"] = True
    summary_msg.additional_kwargs["_supersedes_ids"] = supersedes_ids
    summary_msg.additional_kwargs["_token_count"] = count_tokens(str(summary_msg.content))

    return [summary_msg] + messages[split_idx:]


# ---------- 持久化笔记 / 会话标题辅助 ----------
CONTEXT_WINDOW_MESSAGES = 6  # 笔记维护触发阈值（消息条数）


def _should_update_persistent_note(messages: list, current_note: str) -> bool:
    """只在短期上下文真正开始裁剪时才花钱维护笔记。"""
    return bool(current_note) or len(messages) > CONTEXT_WINDOW_MESSAGES


def generate_session_title(user_text: str) -> str:
    compact_title = " ".join(user_text.split()).strip(" \t\r\n。！？!?，,；;：:")
    return compact_title[:16] or "新会话"


def _title_for(metadata: dict, is_first_message: bool, user_text: str) -> str | None:
    """首条消息生成标题；否则沿用已有标题（避免覆盖 metadata_json）。"""
    if is_first_message:
        return generate_session_title(user_text)
    return metadata.get("title")


async def update_persistent_note(current_note, user_text, ai_response, history_messages=None):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: _update_persistent_note_sync(current_note, user_text, ai_response, history_messages=history_messages),
    )


def _update_persistent_note_sync(current_note, user_text, ai_response, *, history_messages=None):
    try:
        history_text = ""
        if history_messages:
            lines = [f"{'用户' if isinstance(m, HumanMessage) else 'AI'}：{str(m.content)}" for m in history_messages]
            history_text = "\n\n▼ 首次建立笔记时需要一并概括的此前对话：\n" + "\n".join(lines) + "\n\n"
        prompt = (
            "你是一个【Context Manager Agent】，负责维护多轮对话中的「持久化笔记」。\n"
            "笔记是模型在有限上下文窗口下的长效工作记忆，记录已解决的问题与关键事实。\n\n"
            "更新规则：\n"
            "1. 将新信息与现有笔记智能合并，不要简单拼接。\n"
            "2. 过滤噪音，控制在 500 字以内，用简明条目输出。\n"
            "3. 若信息冲突，保留最可靠或最新版本。\n\n"
            f"▼ 现有笔记：\n{current_note if current_note else '无'}\n\n"
            f"{history_text}"
            f"▼ 最新一轮对话：\n用户：{user_text}\nAI：{ai_response}\n\n"
            "请直接输出更新后的笔记（纯文本，不要解释或 Markdown 代码块）："
        )
        res = _get_fast_model().invoke([HumanMessage(content=prompt)])
        return (res.content or "").strip()
    except Exception as e:
        print(f"Context Manager Error: {e}")
        return current_note


# ---------- 图编排 ----------
def _session_thread_config(user_id: str, session_id: str) -> dict:
    return {"configurable": {"thread_id": f"{user_id}:{session_id}"}}


def _history_dicts(messages: list) -> list[dict]:
    out = []
    for m in messages:
        role = "system" if m.type in ("system", "summary") else ("user" if m.type == "human" else "assistant")
        out.append({"role": role, "content": str(m.content)})
    return out


def _run_graph_turn(input_data: dict, cfg: dict) -> tuple[list[dict], dict, dict]:
    """跑一轮图，返回 (custom_events, final_state, pending_hitl_value)。

    final_state 为 graph.get_state(cfg) 快照；pending_hitl_value 为中断值或 None。
    """
    from backend.infra.checkpointer import get_checkpointer
    from backend.rag.chat_graph import build_chat_graph

    graph = build_chat_graph(checkpointer=get_checkpointer())
    events = list(graph.stream(input_data, config=cfg, stream_mode=["custom"]))
    snap = graph.get_state(cfg)
    pending = None
    if snap.tasks and getattr(snap.tasks[0], "interrupts", None):
        pending = snap.tasks[0].interrupts[0].value
    return [d for _kind, d in events], snap, pending


def chat_with_agent(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    messages, metadata = storage.load_with_meta(user_id, session_id)
    persistent_note = metadata.get("persistent_note", "")
    is_first_message = len(messages) == 0

    user_msg = HumanMessage(content=user_text)
    user_msg.additional_kwargs["_token_count"] = count_tokens(user_text)
    messages.append(user_msg)
    messages = _manage_context_window(messages, _get_context_model())

    from backend.infra.checkpointer import get_checkpointer
    from backend.rag.chat_graph import build_chat_graph

    cfg = _session_thread_config(user_id, session_id)
    graph = build_chat_graph(checkpointer=get_checkpointer())
    snap = graph.get_state(cfg)
    pending = bool(snap.tasks and getattr(snap.tasks[0], "interrupts", None))

    if pending:
        input_data = Command(resume=user_text)
    else:
        input_data = {
            "user_text": user_text,
            "persistent_note": persistent_note,
            "history": _history_dicts(messages[:-1]),
        }

    events, final_snap, hitl_value = _run_graph_turn(input_data, cfg)
    full_response = ""
    for ev in events:
        if ev.get("type") == "content":
            full_response += ev.get("content", "")

    if hitl_value is not None:
        prompt = hitl_value.get("prompt", "")
        options = hitl_value.get("options") or []
        hitl_text = format_hitl_message(prompt, options)
        rag_trace = normalize_rag_trace(final_snap.values.get("rag_trace") if hasattr(final_snap, "values") else None)
        messages.append(AIMessage(content=hitl_text))
        extra = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
        save_meta = dict(metadata)
        title = _title_for(metadata, is_first_message, user_text)
        if title:
            save_meta["title"] = title
        storage.save(user_id, session_id, messages, metadata=save_meta, extra_message_data=extra)
        return {"response": hitl_text, "rag_trace": rag_trace, "hitl": hitl_value}

    rag_trace = normalize_rag_trace(final_snap.values.get("rag_trace") if hasattr(final_snap, "values") else None)
    full_response = full_response or (final_snap.values.get("response") if hasattr(final_snap, "values") else "")
    save_meta = dict(metadata)
    if is_first_message:
        save_meta["title"] = generate_session_title(user_text)
    if _should_update_persistent_note(messages, persistent_note):
        save_meta["persistent_note"] = _update_persistent_note_sync(
            persistent_note,
            user_text,
            full_response,
            history_messages=messages[:-1] if not persistent_note else None,
        )
    messages.append(AIMessage(content=full_response))
    extra = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    storage.save(user_id, session_id, messages, metadata=save_meta, extra_message_data=extra)
    return {"response": full_response, "rag_trace": rag_trace}


async def chat_with_agent_stream(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    """流式驱动图：把整轮跑进线程，将 custom 事件经 asyncio.Queue 转发为 SSE。"""
    yield f"data: {json.dumps({'type': 'rag_step', 'step': {'icon': '📨', 'label': '请求已接收，正在准备回答'}})}\n\n"

    messages, metadata = storage.load_with_meta(user_id, session_id)
    persistent_note = metadata.get("persistent_note", "")
    is_first_message = len(messages) == 0

    user_msg = HumanMessage(content=user_text)
    user_msg.additional_kwargs["_token_count"] = count_tokens(user_text)
    messages.append(user_msg)
    messages = _manage_context_window(messages, _get_context_model())

    from backend.infra.checkpointer import get_checkpointer
    from backend.rag.chat_graph import build_chat_graph

    cfg = _session_thread_config(user_id, session_id)
    graph = build_chat_graph(checkpointer=get_checkpointer())
    snap = await asyncio.to_thread(graph.get_state, cfg)
    pending = bool(snap.tasks and getattr(snap.tasks[0], "interrupts", None))

    input_data = (
        Command(resume=user_text)
        if pending
        else {
            "user_text": user_text,
            "persistent_note": persistent_note,
            "history": _history_dicts(messages[:-1]),
        }
    )

    output_queue: asyncio.Queue = asyncio.Queue()
    # asyncio.Queue 非线程安全：_worker 跑在 to_thread 的 executor 线程里，
    # 必须经由捕获的事件循环用 call_soon_threadsafe 投递，否则并发 put 会撕裂队列。
    loop = asyncio.get_running_loop()

    def _safe_put(data):
        loop.call_soon_threadsafe(output_queue.put_nowait, data)

    def _worker():
        try:
            for kind, data in graph.stream(input_data, config=cfg, stream_mode=["custom"]):
                if kind != "custom":
                    continue
                _safe_put(data)
            _safe_put(None)
        except Exception as e:
            _safe_put({"type": "error", "content": str(e)})
            _safe_put(None)

    task = asyncio.create_task(asyncio.to_thread(_worker))

    full_response = ""
    session_title = None
    if is_first_message:
        session_title = generate_session_title(user_text)
        yield f"data: {json.dumps({'type': 'session_title', 'title': session_title, 'session_id': session_id})}\n\n"

    try:
        while True:
            ev = await output_queue.get()
            if ev is None:
                break
            if ev.get("type") == "content":
                full_response += ev.get("content", "")
                yield f"data: {json.dumps(ev)}\n\n"
            elif ev.get("type") == "rag_step":
                yield f"data: {json.dumps(ev)}\n\n"
            elif ev.get("type") == "error":
                yield f"data: {json.dumps(ev)}\n\n"
    except GeneratorExit:
        task.cancel()
        raise
    finally:
        if not task.done():
            task.cancel()

    snap = await asyncio.to_thread(graph.get_state, cfg)
    hitl_value = None
    if snap.tasks and getattr(snap.tasks[0], "interrupts", None):
        hitl_value = snap.tasks[0].interrupts[0].value

    rag_trace = normalize_rag_trace(snap.values.get("rag_trace") if hasattr(snap, "values") else None)
    if rag_trace:
        yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace})}\n\n"

    if hitl_value is not None:
        hitl_event = {
            "route": hitl_value.get("route"),
            "prompt": hitl_value.get("prompt"),
            "options": hitl_value.get("options") or [],
        }
        hitl_text = format_hitl_message(hitl_event["prompt"], hitl_event["options"])
        messages.append(AIMessage(content=hitl_text))
        extra = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
        save_meta = dict(metadata)
        if session_title:
            save_meta["title"] = session_title
        storage.save(user_id, session_id, messages, metadata=save_meta, extra_message_data=extra)
        yield f"data: {json.dumps({'type': 'hitl_request', 'hitl': hitl_event})}\n\n"
        yield "data: [DONE]\n\n"
        return

    full_response = full_response or snap.values.get("response", "")

    save_meta = dict(metadata)
    if session_title:
        save_meta["title"] = session_title
    if _should_update_persistent_note(messages, persistent_note):
        try:
            save_meta["persistent_note"] = update_persistent_note(
                persistent_note,
                user_text,
                full_response,
                history_messages=messages[:-1] if not persistent_note else None,
            )
        except Exception as e:
            print(f"Update persistent note error: {e}")
    messages.append(AIMessage(content=full_response))
    extra = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    storage.save(user_id, session_id, messages, metadata=save_meta, extra_message_data=extra)

    yield "data: [DONE]\n\n"
