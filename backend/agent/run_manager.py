"""ChatRun 生命周期：预订、幂等、单会话并发、HITL 恢复与终态。"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime
from uuid import uuid4

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError

from backend.infra.cache import cache
from backend.infra.database import SessionLocal
from backend.models.models import ChatMessage, ChatRun, ChatRunResume, ChatSession, User

ACTIVE_STATUSES = {"running", "waiting_hitl", "cancelling"}
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


class ChatRunError(Exception):
    def __init__(self, code: str, message: str, *, status_code: int = 409, run_id: str | None = None):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.run_id = run_id


@dataclass(frozen=True)
class RunReservation:
    run_id: str
    session_id: str
    status: str
    created: bool
    checkpoint_thread_id: str
    output_text: str = ""
    rag_trace: dict | None = None
    hitl: dict | None = None
    error_code: str | None = None
    error_detail: str | None = None

    def public_dict(self) -> dict:
        return asdict(self)


def _hash_payload(value: str) -> str:
    return hashlib.sha256(value.strip().encode("utf-8")).hexdigest()


def _record(run: ChatRun, session_id: str, *, created: bool) -> RunReservation:
    return RunReservation(
        run_id=run.id,
        session_id=session_id,
        status=run.status,
        created=created,
        checkpoint_thread_id=run.checkpoint_thread_id,
        output_text=run.output_text or "",
        rag_trace=run.rag_trace,
        hitl=run.hitl_json,
        error_code=run.error_code,
        error_detail=run.error_detail,
    )


def _validate_request_id(request_id: str) -> str:
    value = (request_id or "").strip()
    if not value or len(value) > 128:
        raise ChatRunError("INVALID_REQUEST_ID", "request_id 必须为 1-128 个字符", status_code=400)
    return value


def _get_user(db, username: str) -> User:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise ChatRunError("USER_NOT_FOUND", "用户不存在或已失效", status_code=401)
    return user


def _get_session(db, user: User, session_id: str, *, create: bool) -> ChatSession:
    session = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
        .with_for_update()
        .first()
    )
    if session is not None:
        return session
    if not create:
        raise ChatRunError("SESSION_NOT_FOUND", "会话不存在", status_code=404)
    session = ChatSession(user_id=user.id, session_id=session_id, metadata_json={})
    db.add(session)
    db.flush()
    return session


def _append_user_message(db, session: ChatSession, run_id: str, content: str) -> None:
    max_index = db.query(func.max(ChatMessage.message_index)).filter(ChatMessage.session_ref_id == session.id).scalar()
    db.add(
        ChatMessage(
            session_ref_id=session.id,
            run_id=run_id,
            message_type="human",
            content=content,
            timestamp=datetime.utcnow(),
            token_count=max(1, len(content) // 4),
            message_index=(max_index + 1) if max_index is not None else 0,
        )
    )
    session.updated_at = datetime.utcnow()


def _invalidate(username: str, session_id: str) -> None:
    cache.delete(f"chat_messages:v2:{username}:{session_id}")
    cache.delete(f"chat_sessions:v2:{username}")


def reserve_run(*, username: str, session_id: str, message: str, request_id: str) -> RunReservation:
    """原子预订新 Run；相同 request_id+payload 返回原 Run，不重复执行。"""
    key = _validate_request_id(request_id)
    request_hash = _hash_payload(message)
    db = SessionLocal()
    try:
        with db.begin():
            user = _get_user(db, username)
            session = _get_session(db, user, session_id, create=True)
            existing = db.query(ChatRun).filter(ChatRun.session_ref_id == session.id, ChatRun.request_id == key).first()
            if existing:
                if existing.request_hash != request_hash:
                    raise ChatRunError(
                        "IDEMPOTENCY_CONFLICT",
                        "相同 request_id 对应了不同消息",
                        run_id=existing.id,
                    )
                return _record(existing, session_id, created=False)

            active = (
                db.query(ChatRun)
                .filter(ChatRun.session_ref_id == session.id, ChatRun.status.in_(ACTIVE_STATUSES))
                .first()
            )
            if active:
                raise ChatRunError(
                    "RUN_ACTIVE",
                    "该会话已有正在执行或等待补充的 Run",
                    run_id=active.id,
                )

            run_id = f"run_{uuid4().hex}"
            run = ChatRun(
                id=run_id,
                session_ref_id=session.id,
                request_id=key,
                request_hash=request_hash,
                status="running",
                checkpoint_thread_id=f"{username}:{session_id}:{run_id}",
            )
            db.add(run)
            db.flush()
            _append_user_message(db, session, run_id, message)
        _invalidate(username, session_id)
        return _record(run, session_id, created=True)
    except IntegrityError as exc:
        db.rollback()
        raise ChatRunError("RUN_CONFLICT", "会话状态已变化，请重试") from exc
    finally:
        db.close()


def reserve_resume(
    *,
    username: str,
    session_id: str,
    run_id: str,
    answer: str,
    request_id: str,
) -> RunReservation:
    """原子接受一次 HITL 回答；重复提交同一 request_id 不会二次 resume。"""
    key = _validate_request_id(request_id)
    request_hash = _hash_payload(answer)
    db = SessionLocal()
    try:
        with db.begin():
            user = _get_user(db, username)
            session = _get_session(db, user, session_id, create=False)
            run = (
                db.query(ChatRun)
                .filter(ChatRun.id == run_id, ChatRun.session_ref_id == session.id)
                .with_for_update()
                .first()
            )
            if not run:
                raise ChatRunError("RUN_NOT_FOUND", "Run 不存在", status_code=404, run_id=run_id)
            existing = (
                db.query(ChatRunResume).filter(ChatRunResume.run_id == run.id, ChatRunResume.request_id == key).first()
            )
            if existing:
                if existing.request_hash != request_hash:
                    raise ChatRunError(
                        "IDEMPOTENCY_CONFLICT",
                        "相同 request_id 对应了不同 HITL 回答",
                        run_id=run.id,
                    )
                return _record(run, session_id, created=False)
            if run.status != "waiting_hitl":
                raise ChatRunError(
                    "RUN_NOT_WAITING_HITL",
                    f"状态为 {run.status} 的 Run 不能恢复",
                    run_id=run.id,
                )
            db.add(ChatRunResume(run_id=run.id, request_id=key, request_hash=request_hash))
            _append_user_message(db, session, run.id, answer)
            run.status = "running"
            run.hitl_json = None
            run.updated_at = datetime.utcnow()
        _invalidate(username, session_id)
        return _record(run, session_id, created=True)
    finally:
        db.close()


def get_run(*, username: str, run_id: str) -> RunReservation:
    db = SessionLocal()
    try:
        row = (
            db.query(ChatRun, ChatSession.session_id)
            .join(ChatSession, ChatRun.session_ref_id == ChatSession.id)
            .join(User, ChatSession.user_id == User.id)
            .filter(ChatRun.id == run_id, User.username == username)
            .first()
        )
        if not row:
            raise ChatRunError("RUN_NOT_FOUND", "Run 不存在", status_code=404, run_id=run_id)
        run, session_id = row
        return _record(run, session_id, created=False)
    finally:
        db.close()


def mark_waiting(run_id: str, hitl: dict, rag_trace: dict | None = None) -> bool:
    return _transition(run_id, {"running"}, "waiting_hitl", hitl=hitl, rag_trace=rag_trace)


def mark_completed(run_id: str, output_text: str, rag_trace: dict | None = None) -> bool:
    return _transition(run_id, {"running"}, "completed", output_text=output_text, rag_trace=rag_trace)


def mark_failed(run_id: str, detail: str, *, code: str = "RUN_FAILED") -> bool:
    return _transition(run_id, ACTIVE_STATUSES, "failed", error_code=code, error_detail=detail)


def cancel_run(*, username: str, run_id: str) -> RunReservation:
    db = SessionLocal()
    try:
        with db.begin():
            row = (
                db.query(ChatRun, ChatSession.session_id)
                .join(ChatSession, ChatRun.session_ref_id == ChatSession.id)
                .join(User, ChatSession.user_id == User.id)
                .filter(ChatRun.id == run_id, User.username == username)
                .with_for_update()
                .first()
            )
            if not row:
                raise ChatRunError("RUN_NOT_FOUND", "Run 不存在", status_code=404, run_id=run_id)
            run, session_id = row
            if run.status not in TERMINAL_STATUSES:
                run.status = "cancelled"
                run.error_code = "RUN_CANCELLED"
                run.hitl_json = None
                run.error_detail = "运行已由用户取消"
                run.finished_at = datetime.utcnow()
                run.updated_at = datetime.utcnow()
        _invalidate(username, session_id)
        return _record(run, session_id, created=False)
    finally:
        db.close()


def _transition(
    run_id: str,
    allowed: set[str],
    target: str,
    *,
    output_text: str | None = None,
    rag_trace: dict | None = None,
    hitl: dict | None = None,
    error_code: str | None = None,
    error_detail: str | None = None,
) -> bool:
    db = SessionLocal()
    try:
        with db.begin():
            run = db.query(ChatRun).filter(ChatRun.id == run_id).with_for_update().first()
            if not run or run.status not in allowed:
                return False
            run.status = target
            if output_text is not None:
                run.output_text = output_text
            if rag_trace is not None:
                run.rag_trace = rag_trace
            run.hitl_json = hitl
            run.error_code = error_code
            run.error_detail = error_detail
            run.updated_at = datetime.utcnow()
            if target in TERMINAL_STATUSES:
                run.finished_at = datetime.utcnow()
        return True
    finally:
        db.close()
