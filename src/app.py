from __future__ import annotations

import os
from pathlib import Path

import msgspec
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from src.search import VectorIndex
from src.vectorize import FraudRequest, vectorize

_ROOT = Path(__file__).resolve().parent.parent
_INDEX = Path(os.environ.get("INDEX_PATH", _ROOT / "data" / "index.faiss"))
_LABELS = Path(os.environ.get("LABELS_PATH", _ROOT / "data" / "labels.u8.npy"))

INDEX = VectorIndex(_INDEX, _LABELS)

_RESPONSES: tuple[bytes, ...] = tuple(
    msgspec.json.encode({"approved": (frauds / 5) < 0.6, "fraud_score": frauds / 5})
    for frauds in range(6)
)


async def ready(request: Request) -> Response:
    return Response("ok", media_type="text/plain")


async def fraud_score(request: Request) -> Response:
    try:
        body = await request.body()
        data = msgspec.json.decode(body, type=FraudRequest)
        query = vectorize(data)
        frauds = int(INDEX.labels[INDEX.knn(query, k=5)].sum())
        resp_body = _RESPONSES[frauds]
    except Exception:
        resp_body = _RESPONSES[0]
    return Response(resp_body, media_type="application/json")


app = Starlette(routes=[
    Route("/ready", ready),
    Route("/fraud-score", fraud_score, methods=["POST"]),
])
