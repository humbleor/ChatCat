import json
import os
import re
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.agent.agent import chat_with_agent, chat_with_agent_stream, storage
from backend.agent.run_manager import (
    ChatRunError,
    RunReservation,
    cancel_run,
    get_run,
    mark_failed,
    reserve_resume,
    reserve_run,
)
from backend.api.schemas import (
    AuthResponse,
    ChatRequest,
    ChatResponse,
    ChatRunResponse,
    CurrentUserResponse,
    DocumentDeleteJobResponse,
    DocumentDeleteResponse,
    DocumentDeleteStartResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadJobResponse,
    DocumentUploadResponse,
    DocumentUploadStartResponse,
    HitlCancelRequest,
    LoginRequest,
    MessageInfo,
    RegisterRequest,
    SessionDeleteResponse,
    SessionInfo,
    SessionListResponse,
    SessionMessagesResponse,
)
from backend.document.document_loader import DocumentLoader
from backend.document.document_registry import ACTIVE, FAILED, document_registry
from backend.document.document_service import soft_delete_document
from backend.document.parent_chunk_store import ParentChunkStore
from backend.infra.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    get_db,
    get_password_hash,
    require_admin,
    resolve_role,
)
from backend.jobs.upload_jobs import DELETE_STEPS, delete_job_manager, upload_job_manager
from backend.models.models import User
from backend.vector.embedding import embedding_service
from backend.vector.milvus_client import get_milvus_store
from backend.vector.milvus_writer import MilvusWriter

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR.parent / "data"
UPLOAD_DIR = DATA_DIR / "documents"
MILVUS_SOFT_DELETE_ENABLED = os.getenv("MILVUS_SOFT_DELETE_ENABLED", "false").lower() == "true"

loader = DocumentLoader()
parent_chunk_store = ParentChunkStore()
milvus_store = get_milvus_store()
milvus_writer = MilvusWriter(embedding_service=embedding_service, milvus_store=milvus_store)

router = APIRouter()


def _run_http_error(exc: ChatRunError) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail={"code": exc.code, "message": str(exc), "run_id": exc.run_id},
    )


def _reserve_chat(request: ChatRequest, username: str) -> RunReservation:
    session_id = request.session_id or "default_session"
    try:
        if request.resume_run_id:
            return reserve_resume(
                username=username,
                session_id=session_id,
                run_id=request.resume_run_id,
                answer=request.message,
                request_id=request.request_id,
            )
        return reserve_run(
            username=username,
            session_id=session_id,
            message=request.message,
            request_id=request.request_id,
        )
    except ChatRunError as exc:
        raise _run_http_error(exc) from exc


async def _replay_run_stream(run: RunReservation):
    yield f"data: {json.dumps({'type': 'run', 'run_id': run.run_id, 'status': run.status})}\n\n"
    if run.status == "completed":
        if run.output_text:
            yield f"data: {json.dumps({'type': 'content', 'content': run.output_text})}\n\n"
        if run.rag_trace:
            yield f"data: {json.dumps({'type': 'trace', 'rag_trace': run.rag_trace})}\n\n"
    elif run.status == "waiting_hitl" and run.hitl:
        yield f"data: {json.dumps({'type': 'hitl_request', 'run_id': run.run_id, 'hitl': run.hitl})}\n\n"
    elif run.status == "failed":
        yield f"data: {json.dumps({'type': 'error', 'content': run.error_detail or 'Run 执行失败'})}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/auth/register", response_model=AuthResponse)
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    username = (request.username or "").strip()
    password = (request.password or "").strip()
    if not username or not password:
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")

    exists = db.query(User).filter(User.username == username).first()
    if exists:
        raise HTTPException(status_code=409, detail="用户名已存在")

    role = resolve_role(request.role, request.admin_code)
    user = User(username=username, password_hash=get_password_hash(password), role=role)
    db.add(user)
    db.commit()

    token = create_access_token(username=username, role=role)
    return AuthResponse(access_token=token, username=username, role=role)


@router.post("/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = authenticate_user(db, request.username, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_access_token(username=user.username, role=user.role)
    return AuthResponse(access_token=token, username=user.username, role=user.role)


@router.get("/auth/me", response_model=CurrentUserResponse)
async def me(current_user: User = Depends(get_current_user)):
    return CurrentUserResponse(username=current_user.username, role=current_user.role)


@router.get("/sessions/{session_id}", response_model=SessionMessagesResponse)
async def get_session_messages(session_id: str, current_user: User = Depends(get_current_user)):
    """获取指定会话的所有消息"""
    try:
        messages = [
            MessageInfo(
                type=msg["type"],
                content=msg["content"],
                timestamp=msg.get("timestamp", ""),
                rag_trace=msg.get("rag_trace"),
                hitl=msg.get("hitl"),
                run_id=msg.get("run_id"),
            )
            for msg in storage.get_session_messages(current_user.username, session_id)
        ]
        return SessionMessagesResponse(messages=messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(current_user: User = Depends(get_current_user)):
    """获取当前用户的所有会话列表"""
    try:
        sessions = [SessionInfo(**item) for item in storage.list_session_infos(current_user.username)]
        sessions.sort(key=lambda x: x.updated_at, reverse=True)
        return SessionListResponse(sessions=sessions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(session_id: str, current_user: User = Depends(get_current_user)):
    """删除当前用户的指定会话"""
    try:
        deleted = storage.delete_session(current_user.username, session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="会话不存在")
        return SessionDeleteResponse(session_id=session_id, message="成功删除会话")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    run = _reserve_chat(request, current_user.username)
    if not run.created:
        return ChatResponse(
            response=run.output_text,
            rag_trace=run.rag_trace,
            hitl=run.hitl,
            run_id=run.run_id,
            status=run.status,
        )

    try:
        resp = chat_with_agent(
            request.message,
            current_user.username,
            run.session_id,
            run_id=run.run_id,
            resume=bool(request.resume_run_id),
        )
        if isinstance(resp, dict):
            return ChatResponse(**resp)
        return ChatResponse(response=resp)
    except Exception as e:
        message = str(e)
        mark_failed(run.run_id, message)
        match = re.search(r"Error code:\s*(\d{3})", message)
        if match:
            code = int(match.group(1))
            if code == 429:
                raise HTTPException(
                    status_code=429,
                    detail=(f"上游模型服务触发限流/额度限制（429）。请检查账号额度/模型状态。\n原始错误：{message}"),
                ) from e
            if code in (401, 403):
                raise HTTPException(status_code=code, detail=message) from e
            raise HTTPException(status_code=code, detail=message) from e
        raise HTTPException(status_code=500, detail=message) from e


@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    """跟 Agent 对话 (流式)"""
    run = _reserve_chat(request, current_user.username)
    if not run.created:
        stream = _replay_run_stream(run)
    else:

        async def event_generator():
            try:
                async for chunk in chat_with_agent_stream(
                    request.message,
                    current_user.username,
                    run.session_id,
                    run_id=run.run_id,
                    resume=bool(request.resume_run_id),
                ):
                    yield chunk
            except Exception as e:
                mark_failed(run.run_id, str(e))
                error_data = {"type": "error", "content": str(e), "run_id": run.run_id}
                yield f"data: {json.dumps(error_data)}\n\n"
                yield "data: [DONE]\n\n"

        stream = event_generator()

    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Chat-Run-Id": run.run_id,
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/chat/runs/{run_id}", response_model=ChatRunResponse)
async def get_chat_run(run_id: str, current_user: User = Depends(get_current_user)):
    try:
        return ChatRunResponse(**get_run(username=current_user.username, run_id=run_id).public_dict())
    except ChatRunError as exc:
        raise _run_http_error(exc) from exc


@router.post("/chat/runs/{run_id}/cancel", response_model=ChatRunResponse)
async def cancel_chat_run(run_id: str, current_user: User = Depends(get_current_user)):
    from backend.infra.checkpointer import get_checkpointer

    try:
        before = get_run(username=current_user.username, run_id=run_id)
        cancelled = cancel_run(username=current_user.username, run_id=run_id)
        if before.status == "waiting_hitl":
            get_checkpointer().delete_thread(before.checkpoint_thread_id)
        return ChatRunResponse(**cancelled.public_dict())
    except ChatRunError as exc:
        raise _run_http_error(exc) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/chat/hitl/cancel")
async def cancel_hitl(request: HitlCancelRequest, current_user: User = Depends(get_current_user)):
    """放弃当前 HITL 追问：删除该会话的 checkpoint 线程，下一条消息走新问题。"""
    from backend.agent.agent import _session_thread_config
    from backend.infra.checkpointer import get_checkpointer

    try:
        cfg = _session_thread_config(current_user.username, request.session_id or "default_session")
        # langgraph-checkpoint-postgres 的 delete_thread 签名是 (thread_id: str)；
        # 若传入 config dict，str(cfg) 不会匹配任何 thread_id，DELETE 命中 0 行而静默失效。
        get_checkpointer().delete_thread(cfg["configurable"]["thread_id"])
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _save_upload_file(file: UploadFile, file_path: Path) -> None:
    """按块写入上传文件，避免大文件一次性读入内存。"""
    with open(file_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _process_upload_job(
    job_id: str,
    file_path: str,
    filename: str,
    document_id: str | None = None,
) -> None:
    """Run parsing, chunking, and vector ingestion in the background."""
    failed_step = "cleanup"
    try:
        upload_job_manager.complete_step(job_id, "upload", "文件已保存到服务器")

        failed_step = "cleanup"
        upload_job_manager.update_step(job_id, "cleanup", 10, "running", "正在清理同名旧文档")
        milvus_store.init_collection()
        if MILVUS_SOFT_DELETE_ENABLED:
            previous = document_registry.find_active_by_filename(filename)
            if previous and previous["document_id"] != document_id:
                soft_delete_document(
                    previous["document_id"],
                    registry=document_registry,
                    milvus_store=milvus_store,
                    parent_store=parent_chunk_store,
                )
        else:
            delete_expr = f'filename == "{filename}"'
            try:
                milvus_store.delete(delete_expr)
            except Exception:
                pass
            try:
                parent_chunk_store.delete_by_filename(filename)
            except Exception:
                pass
        upload_job_manager.complete_step(job_id, "cleanup", "旧版本清理完成")

        failed_step = "parse"
        upload_job_manager.update_step(job_id, "parse", 5, "running", "正在解析文档并执行三级分块")
        new_docs = loader.load_document(
            file_path,
            filename,
            document_id=document_id or "",
            is_deleted=False,
        )
        if not new_docs:
            raise ValueError("文档处理失败，未能提取内容")

        parent_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
        leaf_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) == 3]
        if not leaf_docs:
            raise ValueError("文档处理失败，未生成可检索叶子分块")
        upload_job_manager.complete_step(
            job_id,
            "parse",
            f"解析完成：父级分块 {len(parent_docs)} 个，叶子分块 {len(leaf_docs)} 个",
        )

        failed_step = "parent_store"
        upload_job_manager.update_step(job_id, "parent_store", 20, "running", "正在写入父级分块")
        parent_chunk_store.upsert_documents(parent_docs)
        upload_job_manager.complete_step(job_id, "parent_store", f"父级分块已入库：{len(parent_docs)} 个")

        failed_step = "vector_store"
        total_leaf = len(leaf_docs)
        upload_job_manager.update_step(
            job_id,
            "vector_store",
            0,
            "running",
            f"正在向量化入库：0 / {total_leaf}",
            total_chunks=total_leaf,
            processed_chunks=0,
        )

        def _on_vector_progress(processed: int, total: int) -> None:
            percent = round(processed * 100 / total) if total else 100
            upload_job_manager.update_step(
                job_id,
                "vector_store",
                percent,
                "running",
                f"正在向量化入库：{processed} / {total}",
                total_chunks=total,
                processed_chunks=processed,
            )

        milvus_writer.write_documents(leaf_docs, progress_callback=_on_vector_progress)
        if MILVUS_SOFT_DELETE_ENABLED and document_id:
            document_registry.upsert(
                document_id=document_id,
                filename=filename,
                file_path=file_path,
                file_type=str(leaf_docs[0].get("file_type", "")),
                status=ACTIVE,
                leaf_chunk_count=len(leaf_docs),
                parent_chunk_count=len(parent_docs),
            )
        upload_job_manager.complete_step(job_id, "vector_store", f"向量化入库完成：{total_leaf} 个叶子分块")
        upload_job_manager.complete_job(job_id, f"成功上传并处理 {filename}")
    except Exception as e:
        if MILVUS_SOFT_DELETE_ENABLED and document_id:
            try:
                document_registry.set_status(document_id, FAILED, error_message=str(e))
            except Exception:
                pass
        upload_job_manager.fail_job(job_id, failed_step, str(e))


def _process_delete_job(job_id: str, filename: str, document_id: str | None = None) -> None:
    """Soft-delete v2 documents; keep the legacy physical-delete path for v1."""
    failed_step = "prepare"
    try:
        delete_job_manager.update_step(job_id, "prepare", 20, "running", "正在初始化删除任务")
        milvus_store.init_collection()

        if MILVUS_SOFT_DELETE_ENABLED:
            document = document_registry.get(document_id or "")
            if not document:
                document = document_registry.find_active_by_filename(filename)
            if not document:
                raise KeyError(f"未找到活动文档: {filename}")
            document_id = document["document_id"]
            delete_job_manager.complete_step(job_id, "prepare", "软删除任务已创建")

            failed_step = "milvus"
            delete_job_manager.update_step(job_id, "milvus", 30, "running", "正在标记向量为已删除")
            result = soft_delete_document(
                document_id,
                registry=document_registry,
                milvus_store=milvus_store,
                parent_store=parent_chunk_store,
            )
            vector_count = int(result.get("vector_count", 0))
            parent_count = int(result.get("parent_count", 0))
            delete_job_manager.complete_step(job_id, "milvus", f"向量已软删除：{vector_count} 条")

            failed_step = "parent_store"
            delete_job_manager.update_step(job_id, "parent_store", 100, "running", "正在确认父级分块状态")
            delete_job_manager.complete_step(job_id, "parent_store", f"父级分块已软删除：{parent_count} 条")
            delete_job_manager.complete_job(job_id, f"已软删除 {filename}，向量 {vector_count} 条")
            return

        delete_expr = f'filename == "{filename}"'
        delete_job_manager.complete_step(job_id, "prepare", "删除任务已创建")
        failed_step = "milvus"
        delete_job_manager.update_step(job_id, "milvus", 30, "running", "正在删除 Milvus 向量数据")
        result = milvus_store.delete(delete_expr)
        deleted_count = result.get("delete_count", 0) if isinstance(result, dict) else 0
        delete_job_manager.complete_step(job_id, "milvus", f"向量数据已删除：{deleted_count} 条")

        failed_step = "parent_store"
        delete_job_manager.update_step(job_id, "parent_store", 30, "running", "正在删除 PostgreSQL 父级分块")
        parent_chunk_store.delete_by_filename(filename)
        delete_job_manager.complete_step(job_id, "parent_store", "父级分块已删除")
        delete_job_manager.complete_job(job_id, f"已删除 {filename}，向量数据 {deleted_count} 条")
    except Exception as e:
        delete_job_manager.fail_job(job_id, failed_step, str(e))


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(_: User = Depends(require_admin)):
    """List active documents from PostgreSQL in v2, or all Milvus rows in legacy mode."""
    try:
        if MILVUS_SOFT_DELETE_ENABLED:
            documents = [
                DocumentInfo(
                    document_id=item["document_id"],
                    filename=item["filename"],
                    file_type=item["file_type"],
                    chunk_count=item["leaf_chunk_count"],
                    status=item["status"],
                    uploaded_at=item["created_at"].isoformat(),
                )
                for item in document_registry.list_active()
            ]
            return DocumentListResponse(documents=documents)

        milvus_store.init_collection()
        results = milvus_store.query_all(output_fields=["filename", "file_type"])
        file_stats = {}
        for item in results:
            filename = item.get("filename", "")
            file_type = item.get("file_type", "")
            if filename not in file_stats:
                file_stats[filename] = {
                    "filename": filename,
                    "file_type": file_type,
                    "chunk_count": 0,
                }
            file_stats[filename]["chunk_count"] += 1
        return DocumentListResponse(documents=[DocumentInfo(**stats) for stats in file_stats.values()])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


@router.post("/documents/upload/async", response_model=DocumentUploadStartResponse)
async def upload_document_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    _: User = Depends(require_admin),
):
    """轻量版异步上传：文件落盘后立即返回 job_id，后台继续解析和向量化。"""
    filename = file.filename or ""
    if not filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    if not loader._is_supported_document(filename):
        raise HTTPException(status_code=400, detail="仅支持 PDF、Word 和 Excel 文档")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    job = upload_job_manager.create_job(filename)
    file_path = UPLOAD_DIR / filename
    document_id = None
    if MILVUS_SOFT_DELETE_ENABLED:
        document_id = document_registry.create(
            filename=filename,
            file_path=str(file_path),
        )

    try:
        upload_job_manager.update_step(job["job_id"], "upload", 1, "running", "正在保存文件到服务器")
        await _save_upload_file(file, file_path)
        upload_job_manager.complete_step(job["job_id"], "upload", "文件已上传，等待后台处理")
    except Exception as e:
        if document_id:
            document_registry.set_status(document_id, FAILED, error_message=str(e))
        upload_job_manager.fail_job(job["job_id"], "upload", f"文件保存失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    background_tasks.add_task(
        _process_upload_job,
        job["job_id"],
        str(file_path),
        filename,
        document_id,
    )
    return DocumentUploadStartResponse(
        job_id=job["job_id"],
        filename=filename,
        message="文件已上传，正在后台解析和向量化入库",
        document_id=document_id,
    )


@router.get("/documents/upload/jobs/{job_id}", response_model=DocumentUploadJobResponse)
async def get_upload_job(job_id: str, _: User = Depends(require_admin)):
    job = upload_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="上传任务不存在或已过期")
    return DocumentUploadJobResponse(**job)


@router.get("/documents/upload/jobs", response_model=list[DocumentUploadJobResponse])
async def list_upload_jobs(_: User = Depends(require_admin)):
    jobs = upload_job_manager.list_jobs()
    jobs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return [DocumentUploadJobResponse(**job) for job in jobs]


@router.delete("/documents/delete/async/{filename}", response_model=DocumentDeleteStartResponse)
async def delete_document_async(
    filename: str,
    background_tasks: BackgroundTasks,
    _: User = Depends(require_admin),
):
    """轻量版异步删除：立即返回 job_id，实际删除在后台执行。"""
    document_id = None
    if MILVUS_SOFT_DELETE_ENABLED:
        document = document_registry.find_active_by_filename(filename)
        if not document:
            raise HTTPException(status_code=404, detail=f"未找到活动文档: {filename}")
        document_id = document["document_id"]

    job = delete_job_manager.create_job(
        filename,
        steps=DELETE_STEPS,
        current_step="prepare",
        message="等待删除",
        completion_step="parent_store",
    )
    delete_job_manager.update_step(job["job_id"], "prepare", 1, "running", "删除任务已提交")
    background_tasks.add_task(_process_delete_job, job["job_id"], filename, document_id)
    return DocumentDeleteStartResponse(
        job_id=job["job_id"],
        filename=filename,
        message=f"正在删除 {filename}",
    )


@router.get("/documents/delete/jobs/{job_id}", response_model=DocumentDeleteJobResponse)
async def get_delete_job(job_id: str, _: User = Depends(require_admin)):
    job = delete_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="删除任务不存在或已过期")
    return DocumentDeleteJobResponse(**job)


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...), _: User = Depends(require_admin)):
    """Legacy synchronous upload endpoint."""
    if MILVUS_SOFT_DELETE_ENABLED:
        raise HTTPException(status_code=409, detail="软删除模式请使用 /documents/upload/async")
    try:
        filename = file.filename or ""
        if not filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        if not loader._is_supported_document(filename):
            raise HTTPException(status_code=400, detail="仅支持 PDF、Word 和 Excel 文档")

        os.makedirs(UPLOAD_DIR, exist_ok=True)
        milvus_store.init_collection()

        delete_expr = f'filename == "{filename}"'
        try:
            milvus_store.delete(delete_expr)
        except Exception:
            pass
        try:
            parent_chunk_store.delete_by_filename(filename)
        except Exception:
            pass

        file_path = UPLOAD_DIR / filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        try:
            new_docs = loader.load_document(str(file_path), filename)
        except Exception as doc_err:
            raise HTTPException(status_code=500, detail=f"文档处理失败: {doc_err}")

        if not new_docs:
            raise HTTPException(status_code=500, detail="文档处理失败，未能提取内容")

        parent_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
        leaf_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) == 3]
        if not leaf_docs:
            raise HTTPException(status_code=500, detail="文档处理失败，未生成可检索叶子分块")

        parent_chunk_store.upsert_documents(parent_docs)
        milvus_writer.write_documents(leaf_docs)

        return DocumentUploadResponse(
            filename=filename,
            chunks_processed=len(leaf_docs),
            message=(
                f"成功上传并处理 {filename}，叶子分块 {len(leaf_docs)} 个，"
                f"父级分块 {len(parent_docs)} 个（存入 PostgreSQL）"
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档上传失败: {str(e)}")


@router.delete("/documents/{filename}", response_model=DocumentDeleteResponse)
async def delete_document(filename: str, _: User = Depends(require_admin)):
    """删除文档在 Milvus 中的向量（保留本地文件，管理员）"""
    try:
        milvus_store.init_collection()

        if MILVUS_SOFT_DELETE_ENABLED:
            document = document_registry.find_active_by_filename(filename)
            if not document:
                raise HTTPException(status_code=404, detail=f"未找到活动文档: {filename}")
            result = soft_delete_document(
                document["document_id"],
                registry=document_registry,
                milvus_store=milvus_store,
                parent_store=parent_chunk_store,
            )
            return DocumentDeleteResponse(
                filename=filename,
                chunks_deleted=int(result.get("vector_count", 0)),
                message=f"成功软删除文档 {filename}",
            )

        delete_expr = f'filename == "{filename}"'
        result = milvus_store.delete(delete_expr)
        parent_chunk_store.delete_by_filename(filename)

        return DocumentDeleteResponse(
            filename=filename,
            chunks_deleted=result.get("delete_count", 0) if isinstance(result, dict) else 0,
            message=f"成功删除文档 {filename} 的向量数据（本地文件已保留）",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")
