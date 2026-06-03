from __future__ import annotations

import os
from pathlib import Path

import msgspec
from litestar import Litestar, Response, get, post
from litestar.enums import MediaType

from src.search import VectorIndex
from src.vectorize import FraudRequest, vectorize

_ROOT = Path(__file__).resolve().parent.parent
_VECTORS = Path(os.environ.get("VECTORS_PATH", _ROOT / "data" / "vectors.f16.npy"))
_LABELS = Path(os.environ.get("LABELS_PATH", _ROOT / "data" / "labels.u8.npy"))

INDEX = VectorIndex(_VECTORS, _LABELS)

_RESPONSES: tuple[bytes, ...] = tuple(
    msgspec.json.encode({"approved": (frauds / 5) < 0.6, "fraud_score": frauds / 5})
    for frauds in range(6)
)


@get("/ready", media_type=MediaType.TEXT, sync_to_thread=False)
def ready() -> str:
    return "ok"


@post("/fraud-score", media_type=MediaType.JSON, sync_to_thread=True)
def fraud_score(data: FraudRequest) -> Response[bytes]:
    try:
        query = vectorize(data)
        frauds = int(INDEX.labels[INDEX.knn(query, k=5)].sum())
        body = _RESPONSES[frauds]
    except Exception:
        body = _RESPONSES[0]

    return Response(body, media_type=MediaType.JSON, status_code=200)


app = Litestar(route_handlers=[ready, fraud_score])
