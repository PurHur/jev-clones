"""Bound incoming JSON and base64 upload requests before parsing them."""
from starlette.responses import JSONResponse


class RequestSizeLimit:
    def __init__(self, app, max_bytes):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        chunks = []
        total = 0
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            data = event.get("body", b"")
            total += len(data)
            if total > self.max_bytes:
                return await JSONResponse({"detail": "Request exceeds the upload limit."}, status_code=413)(scope, receive, send)
            chunks.append(data)
            if not event.get("more_body", False):
                break
        body = b"".join(chunks)
        delivered = False

        async def replay():
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay, send)
