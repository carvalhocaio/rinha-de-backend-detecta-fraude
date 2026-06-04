from __future__ import annotations

import os
from pathlib import Path

import faiss
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
DATA = _ROOT / "data"


class VectorIndex:
    """Busca aproximada dos k vizinhos mais próximos via índice FAISS IVF."""

    def __init__(
        self,
        index_path: Path = DATA / "index.faiss",
        labels_path: Path = DATA / "labels.u8.npy",
        nprobe: int | None = None,
    ) -> None:
        self.index = faiss.read_index(str(index_path), faiss.IO_FLAG_MMAP)
        self.nprobe = int(
            nprobe if nprobe is not None else os.environ.get("FAISS_NPROBE", 16)
        )

        self.index.nprobe = self.nprobe  # type: ignore[assignment]
        self.labels = np.load(labels_path, mmap_mode="r")
        self.n = self.index.ntotal

    def knn(self, query: np.ndarray, k: int = 5) -> np.ndarray:
        """Índices dos k vizinhos mais próximos (aproximados)."""
        q = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        _, idx = self.index.search(q, k)
        idx = idx[0]
        return idx[idx >= 0]

    def score(self, query: np.ndarray, k: int = 5) -> tuple[float, bool]:
        """Retorna (fraud_score, approved) seguindo a regra da Rinha."""
        idx = self.knn(query, k)
        frauds = int(self.labels[idx].sum())
        fraud_score = frauds / k
        return fraud_score, fraud_score < 0.6
