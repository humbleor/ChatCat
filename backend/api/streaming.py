"""SSE response that closes its producer even when no tokens are arriving."""

import anyio
from starlette.responses import StreamingResponse


class ChatStreamingResponse(StreamingResponse):
    def __init__(self, *args, on_disconnect=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_disconnect = on_disconnect

    async def __call__(self, scope, receive, send):
        completed = False
        try:
            async with anyio.create_task_group() as group:

                async def stream():
                    nonlocal completed
                    await self.stream_response(send)
                    completed = True
                    group.cancel_scope.cancel()

                async def disconnect():
                    await self.listen_for_disconnect(receive)
                    group.cancel_scope.cancel()

                group.start_soon(stream)
                await disconnect()
        finally:
            # Explicitly close the generator if disconnection occurs at a send
            # boundary, rather than relying on async-generator garbage collection.
            with anyio.CancelScope(shield=True):
                try:
                    await self.body_iterator.aclose()
                finally:
                    if not completed and self.on_disconnect is not None:
                        await self.on_disconnect()
        if self.background is not None:
            await self.background()
