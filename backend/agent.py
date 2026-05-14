from dotenv import load_dotenv
import os
import json
import asyncio
import logging
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage
from tools import get_current_weather, search_knowledge_base, get_last_rag_context, reset_tool_call_guards, set_rag_step_queue
from datetime import datetime
from cache import cache
from database import SessionLocal
from models import User, ChatSession, ChatMessage

load_dotenv()

logger = logging.getLogger(__name__)

API_KEY = os.getenv("LLM_API_KEY")
MODEL = os.getenv("LLM_MODEL")
BASE_URL = os.getenv("LLM_BASE_URL")

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
                extra["_is_summary"] = (msg_type == "summary")

            if msg_type == "human":
                msg = HumanMessage(content=content)
            elif msg_type == "ai":
                msg = AIMessage(content=content)
            else:
                msg = SystemMessage(content=content)
            msg.additional_kwargs.update(extra)
            messages.append(msg)
        return messages

    def save(self, user_id: str, session_id: str, messages: list, metadata: dict = None, extra_message_data: list = None):
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
                    serialized.append({
                        "type": msg.type,
                        "content": str(msg.content),
                        "id": db_id,
                        "timestamp": msg.additional_kwargs.get("_timestamp", now.isoformat()),
                        "token_count": msg.additional_kwargs.get("_token_count", 0),
                        "rag_trace": msg.additional_kwargs.get("_rag_trace"),
                    })
                    continue

                # 新消息：INSERT
                rag_trace = None
                if extra_message_data and idx < len(extra_message_data):
                    extra = extra_message_data[idx] or {}
                    rag_trace = extra.get("rag_trace")

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

                serialized.append({
                    "type": msg_type,
                    "content": str(msg.content),
                    "id": db_id,
                    "token_count": tk,
                    "timestamp": now.isoformat(),
                    "rag_trace": rag_trace,
                })

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
                result.append({
                    "session_id": s.session_id,
                    "updated_at": s.updated_at.isoformat(),
                    "message_count": count,
                })
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
                    "rag_trace": row.rag_trace,
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



def create_agent_instance():
    model = init_chat_model(
        model=MODEL,
        model_provider="openai",
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.3,
        stream_usage=True,
    )

    agent = create_agent(
        model=model,
        tools=[get_current_weather, search_knowledge_base],
        system_prompt=(
            "You are a cute cat bot that loves to help users. "
            "When responding, you may use tools to assist. "
            "Use get_current_weather when users ask about weather. "
            "Use search_knowledge_base when users ask document/knowledge questions. "
            "Do not call the same tool repeatedly in one turn. At most one knowledge tool call per turn. "
            "Once you call search_knowledge_base and receive its result, you MUST immediately produce the Final Answer based on that result. "
            "After receiving search_knowledge_base result, you MUST NOT call any tool again (including get_current_weather or search_knowledge_base). "
            "If the retrieved context is insufficient, answer honestly that you don't know instead of making up facts. "
            "If tool results include a Step-back Question/Answer, use that general principle to reason and answer, "
            "but do not reveal chain-of-thought. "
            "If you don't know the answer, admit it honestly."
        ),
    )
    return agent, model


agent, model = create_agent_instance()

storage = ConversationStorage()

# 上下文窗口预算
CONTEXT_WINDOW_TOKENS = int(os.getenv("CONTEXT_WINDOW_TOKENS", "128000"))
CONTEXT_BUDGET_RATIO = 0.8
RECENT_TURNS_KEEP = 5  # 始终保留最近 N 轮

# 系统提示词 token 数（启动时结算一次）
_system_prompt_tokens = 0


def _get_system_prompt_tokens() -> int:
    global _system_prompt_tokens
    if _system_prompt_tokens == 0:
        sp = (
            "You are a cute cat bot that loves to help users. "
            "When responding, you may use tools to assist. "
            "Use get_current_weather when users ask about weather. "
            "Use search_knowledge_base when users ask document/knowledge questions. "
            "Do not call the same tool repeatedly in one turn. At most one knowledge tool call per turn. "
            "Once you call search_knowledge_base and receive its result, you MUST immediately produce the Final Answer based on that result. "
            "After receiving search_knowledge_base result, you MUST NOT call any tool again (including get_current_weather or search_knowledge_base). "
            "If the retrieved context is insufficient, answer honestly that you don't know instead of making up facts. "
            "If tool results include a Step-back Question/Answer, use that general principle to reason and answer, "
            "but do not reveal chain-of-thought. "
            "If you don't know the answer, admit it honestly."
        )
        _system_prompt_tokens = count_tokens(sp)
    return _system_prompt_tokens


def _available_budget() -> int:
    """可用于历史消息的总 token 预算。"""
    total = int(CONTEXT_WINDOW_TOKENS * CONTEXT_BUDGET_RATIO)
    return max(total - _get_system_prompt_tokens(), 4000)


def summarize_old_messages(model, messages: list) -> str:
    """将旧消息总结为摘要。"""
    old_conversation = "\n".join([
        f"{'用户' if msg.type == 'human' else 'AI'}: {msg.content}"
        for msg in messages
    ])

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
    total_tokens = sum(
        msg.additional_kwargs.get("_token_count", 0)
        for msg in messages
    )

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
    supersedes_ids = [
        m.additional_kwargs["_db_id"]
        for m in candidates
        if m.additional_kwargs.get("_db_id")
    ]

    summary_msg = SystemMessage(content=f"之前的对话摘要：\n{summary}")
    summary_msg.additional_kwargs["_msg_type"] = "summary"
    summary_msg.additional_kwargs["_is_summary"] = True
    summary_msg.additional_kwargs["_supersedes_ids"] = supersedes_ids
    summary_msg.additional_kwargs["_token_count"] = count_tokens(str(summary_msg.content))

    return [summary_msg] + messages[split_idx:]


def chat_with_agent(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    """使用 Agent 处理用户消息并返回响应。"""
    messages = storage.load(user_id, session_id)

    # 清理可能残留的 RAG 上下文，避免跨请求污染
    get_last_rag_context(clear=True)
    reset_tool_call_guards()

    user_msg = HumanMessage(content=user_text)
    user_msg.additional_kwargs["_token_count"] = count_tokens(user_text)
    messages.append(user_msg)

    messages = _manage_context_window(messages, model)

    result = agent.invoke(
        {"messages": messages},
        config={"recursion_limit": 8},
    )

    response_content = ""
    if isinstance(result, dict):
        if "output" in result:
            response_content = result["output"]
        elif "messages" in result and result["messages"]:
            msg = result["messages"][-1]
            response_content = getattr(msg, "content", str(msg))
        else:
            response_content = str(result)
    elif hasattr(result, "content"):
        response_content = result.content
    else:
        response_content = str(result)

    ai_msg = AIMessage(content=response_content)
    ai_msg.additional_kwargs["_token_count"] = count_tokens(response_content)
    messages.append(ai_msg)

    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None

    extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    storage.save(user_id, session_id, messages, extra_message_data=extra_message_data)

    return {
        "response": response_content,
        "rag_trace": rag_trace,
    }


async def chat_with_agent_stream(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    """使用 Agent 处理用户消息并流式返回响应。

    架构：使用统一输出队列 + 后台任务，确保 RAG 检索步骤在工具执行期间实时推送。
    """
    messages = storage.load(user_id, session_id)

    # 清理可能残留的 RAG 上下文
    get_last_rag_context(clear=True)
    reset_tool_call_guards()

    # 统一输出队列：所有事件（content / rag_step）都汇入这里
    output_queue = asyncio.Queue()

    class _RagStepProxy:
        """代理对象：将 emit_rag_step 的原始 step dict 包装后放入统一输出队列。"""
        def put_nowait(self, step):
            output_queue.put_nowait({"type": "rag_step", "step": step})

    set_rag_step_queue(_RagStepProxy())

    user_msg = HumanMessage(content=user_text)
    user_msg.additional_kwargs["_token_count"] = count_tokens(user_text)
    messages.append(user_msg)

    # Token-aware 上下文窗口管理
    messages = _manage_context_window(messages, model)

    full_response = ""

    async def _agent_worker():
        """后台任务：运行 agent 并将内容 chunk 推入输出队列。"""
        nonlocal full_response
        try:
            async for msg, metadata in agent.astream(
                {"messages": messages},
                stream_mode="messages",
                config={"recursion_limit": 8},
            ):
                if not isinstance(msg, AIMessageChunk):
                    continue
                if getattr(msg, "tool_call_chunks", None):
                    continue

                content = ""
                if isinstance(msg.content, str):
                    content = msg.content
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if isinstance(block, str):
                            content += block
                        elif isinstance(block, dict) and block.get("type") == "text":
                            content += block.get("text", "")

                if content:
                    full_response += content
                    await output_queue.put({"type": "content", "content": content})
        except Exception as e:
            await output_queue.put({"type": "error", "content": str(e)})
        finally:
            # 哨兵：通知主循环 agent 已完成
            await output_queue.put(None)

    # 启动后台任务
    agent_task = asyncio.create_task(_agent_worker())

    try:
        # 主循环：持续从统一队列取事件并 yield SSE
        while True:
            event = await output_queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"
    except GeneratorExit:
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass
        raise
    finally:
        set_rag_step_queue(None)
        if not agent_task.done():
            agent_task.cancel()

    # 获取 RAG trace
    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None

    # 发送 trace 信息
    if rag_trace:
        yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace})}\n\n"

    # 发送结束信号
    yield "data: [DONE]\n\n"

    # 保存对话
    ai_msg = AIMessage(content=full_response)
    ai_msg.additional_kwargs["_token_count"] = count_tokens(full_response)
    messages.append(ai_msg)
    extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    storage.save(user_id, session_id, messages, extra_message_data=extra_message_data)
