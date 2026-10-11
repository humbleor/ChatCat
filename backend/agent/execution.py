"""Cooperative graph cancellation with cancellable async model I/O.

The synchronous graph/checkpointer stay in worker threads. Model HTTP requests run
on the owning event loop, so stopping a Run can cancel a blocked network read.
ContextVars propagate the same cancellation scope into LangGraph fan-out nodes.
"""

import asyncio
import concurrent.futures
import logging
import threading
from contextvars import ContextVar
from functools import partial, wraps

logger = logging.getLogger(__name__)


class RunCancelled(BaseException):
    """Must bypass node fallback handlers that catch Exception."""


_scope: ContextVar = ContextVar("chat_run_execution", default=None)
_executions: dict[str, "Execution"] = {}
# Polling and cancellation must not queue behind long-running graph workers.
_control_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="chat-control")


async def control_call(fn, *args, **kwargs):
    return await asyncio.get_running_loop().run_in_executor(_control_pool, partial(fn, *args, **kwargs))


def check_cancelled():
    scope = _scope.get()
    if scope is not None and scope.stopped.is_set():
        raise RunCancelled()


def cancellable_node(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        check_cancelled()
        result = fn(*args, **kwargs)
        check_cancelled()
        return result

    return wrapped


def invoke_model(model, messages):
    scope = _scope.get()
    if scope is None:
        return model.invoke(messages)
    return scope.call(lambda: model.ainvoke(messages))


def stream_model(model, messages):
    scope = _scope.get()
    if scope is None:
        yield from model.stream(messages)
        return
    stream = model.astream(messages)
    try:
        while True:
            try:
                chunk = scope.call(stream.__anext__)
            except StopAsyncIteration:
                break
            check_cancelled()
            yield chunk
    finally:
        # Close this response stream, never the shared model/client. Closing is
        # allowed after cancellation and awaited before the worker exits.
        scope.call(stream.aclose, cleanup=True)


def emit_event(event):
    scope = _scope.get()
    if scope is not None and scope.emit is not None:
        check_cancelled()
        scope.loop.call_soon_threadsafe(scope.emit, event)


def cancel_local(run_id: str):
    execution = _executions.get(run_id)
    if execution is not None:
        execution.stop()


class Execution:
    def __init__(self, run_id, username, emit=None):
        self.run_id = run_id
        self.username = username
        self.emit = emit
        self.loop = asyncio.get_running_loop()
        self.stopped = threading.Event()
        self.requests: set[asyncio.Task] = set()
        self.done = asyncio.Event()

    def stop(self):
        self.stopped.set()
        self.loop.call_soon_threadsafe(self._cancel_requests)

    def _cancel_requests(self):
        for task in tuple(self.requests):
            if not task.cancelling():
                task.cancel()

    def call(self, factory, *, cleanup=False):
        async def request():
            if not cleanup:
                check_cancelled()
                task = asyncio.current_task()
                self.requests.add(task)
            try:
                return await factory()
            finally:
                if not cleanup:
                    self.requests.discard(task)

        if not cleanup:
            check_cancelled()
        future = asyncio.run_coroutine_threadsafe(request(), self.loop)
        try:
            return future.result()
        except concurrent.futures.CancelledError:
            raise RunCancelled() from None

    async def watch(self):
        from backend.agent.run_manager import get_run

        while True:
            await asyncio.sleep(0.25)
            try:
                run = await control_call(get_run, username=self.username, run_id=self.run_id)
                if run.status in {"cancelling", "cancelled"}:
                    self.stop()
                    return
            except Exception:
                logger.exception("Unable to check cancellation for %s", self.run_id)

    async def run(self, fn, *args, **kwargs):
        from backend.agent.run_manager import cancel_run, get_run, mark_cancelled, mark_failed

        _executions[self.run_id] = self
        token = _scope.set(self)
        watcher = None
        worker = None
        try:
            # Covers cancel arriving before the worker was registered.
            run = await control_call(get_run, username=self.username, run_id=self.run_id)
            if run.status in {"cancelling", "cancelled"}:
                self.stop()
            check_cancelled()
            watcher = asyncio.create_task(self.watch())
            worker = asyncio.create_task(asyncio.to_thread(fn, *args, **kwargs))
            result = await asyncio.shield(worker)
            check_cancelled()
            return result
        except (RunCancelled, asyncio.CancelledError):
            self.stop()
            if worker is not None:
                try:
                    await asyncio.shield(worker)
                except RunCancelled:
                    pass
                except Exception:
                    logger.debug("Worker failed during cancellation of %s", self.run_id, exc_info=True)
            return {"response": "", "run_id": self.run_id, "status": "cancelled"}
        except Exception as exc:
            await control_call(mark_failed, self.run_id, str(exc))
            raise
        finally:
            try:
                if watcher is not None:
                    watcher.cancel()
                    await asyncio.gather(watcher, return_exceptions=True)
                if self.stopped.is_set():
                    await control_call(cancel_run, username=self.username, run_id=self.run_id)
                    await control_call(mark_cancelled, self.run_id)
            finally:
                _scope.reset(token)
                _executions.pop(self.run_id, None)
                self.done.set()


async def shutdown_executions():
    executions = tuple(_executions.values())
    for execution in executions:
        execution.stop()
    if executions:
        try:
            await asyncio.wait_for(asyncio.gather(*(item.done.wait() for item in executions)), timeout=5)
        except TimeoutError:
            logger.warning("Chat workers are still finishing synchronous operations during shutdown")
