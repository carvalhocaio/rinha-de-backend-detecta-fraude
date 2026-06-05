from __future__ import annotations

import os
import threading
from pathlib import Path

import faiss
import msgspec

from src.search import VectorIndex
from src.vectorize import FraudRequest, vectorize

faiss.omp_set_num_threads(1)

_ROOT = Path(__file__).resolve().parent.parent
_INDEX_PATH = Path(os.environ.get("INDEX_PATH", _ROOT / "data" / "index.faiss"))
_LABELS_PATH = Path(os.environ.get("LABELS_PATH", _ROOT / "data" / "labels.u8.npy"))

_index: VectorIndex | None = None
_ready = threading.Event()


def _load_index() -> None:
    global _index
    _index = VectorIndex(_INDEX_PATH, _LABELS_PATH)
    _ready.set()


threading.Thread(target=_load_index, daemon=True).start()

_RESPONSES: tuple[bytes, ...] = tuple(
    msgspec.json.encode({"approved": (f / 5) < 0.6, "fraud_score": f / 5})
    for f in range(6)
)

_HEADERS_TEXT = [(b"content-type", b"text/plain")]
_HEADERS_JSON = [(b"content-type", b"application/json")]
_NOT_READY = msgspec.json.encode({"error": "loading"})


async def app(scope, receive, send):
    if scope["type"] != "http":
        return

    path = scope["path"]

    if path == "/ready":
        if _ready.is_set():
            await send({"type": "http.response.start", "status": 200, "headers": _HEADERS_TEXT})
            await send({"type": "http.response.body", "body": b"ok"})
        else:
            await send({"type": "http.response.start", "status": 503, "headers": _HEADERS_JSON})
            await send({"type": "http.response.body", "body": _NOT_READY})
        return

    if path == "/fraud-score" and scope["method"] == "POST":
        body = b""
        while True:
            msg = await receive()
            body += msg.get("body", b"")
            if not msg.get("more_body", False):
                break
        if not _ready.is_set():
            await send({"type": "http.response.start", "status": 503, "headers": _HEADERS_JSON})
            await send({"type": "http.response.body", "body": _NOT_READY})
            return
        try:
            data = msgspec.json.decode(body, type=FraudRequest)
            query = vectorize(data)
            frauds = int(_index.labels[_index.knn(query, k=5)].sum())
            resp = _RESPONSES[frauds]
        except Exception:
            resp = _RESPONSES[0]
        await send({"type": "http.response.start", "status": 200, "headers": _HEADERS_JSON})
        await send({"type": "http.response.body", "body": resp})
        return

    await send({"type": "http.response.start", "status": 404, "headers": []})
    await send({"type": "http.response.body", "body": b""})
