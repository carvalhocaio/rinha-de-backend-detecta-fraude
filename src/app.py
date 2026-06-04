from __future__ import annotations

import os
from pathlib import Path

import msgspec

from src.search import VectorIndex
from src.vectorize import FraudRequest, vectorize

_ROOT = Path(__file__).resolve().parent.parent
INDEX = VectorIndex(
    Path(os.environ.get("INDEX_PATH", _ROOT / "data" / "index.faiss")),
    Path(os.environ.get("LABELS_PATH", _ROOT / "data" / "labels.u8.npy")),
)

_RESPONSES: tuple[bytes, ...] = tuple(
    msgspec.json.encode({"approved": (f / 5) < 0.6, "fraud_score": f / 5})
    for f in range(6)
)

_HEADERS_TEXT = [(b"content-type", b"text/plain")]
_HEADERS_JSON = [(b"content-type", b"application/json")]


async def app(scope, receive, send):
    if scope["type"] != "http":
        return

    path = scope["path"]

    if path == "/ready":
        await send({"type": "http.response.start", "status": 200, "headers": _HEADERS_TEXT})
        await send({"type": "http.response.body", "body": b"ok"})
        return

    if path == "/fraud-score" and scope["method"] == "POST":
        body = b""
        while True:
            msg = await receive()
            body += msg.get("body", b"")
            if not msg.get("more_body", False):
                break
        try:
            data = msgspec.json.decode(body, type=FraudRequest)
            query = vectorize(data)
            frauds = int(INDEX.labels[INDEX.knn(query, k=5)].sum())
            resp = _RESPONSES[frauds]
        except Exception:
            resp = _RESPONSES[0]
        await send({"type": "http.response.start", "status": 200, "headers": _HEADERS_JSON})
        await send({"type": "http.response.body", "body": resp})
        return

    await send({"type": "http.response.start", "status": 404, "headers": []})
    await send({"type": "http.response.body", "body": b""})
