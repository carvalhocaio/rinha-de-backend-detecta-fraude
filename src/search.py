from __future__ import annotations

from pathlib import Path

import numpy as np

DATA = Path("data")


class VectorIndex:
    """Busca por força bruta dos k vizinhos mais próximos (k-NN exato, distância L2)."""

    def __init__(
        self,
        vectors_path: Path = DATA / "vectors.f16.npy",
        labels_path: Path = DATA / "labels.u8.npy",
    ) -> None:
        self.vectors = np.load(vectors_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        self.n = self.vectors.shape[0]

    def knn(self, query: np.ndarray, k: int = 5, chunk: int = 261_144) -> np.ndarray:
        """índices dos k vetores de referência mais próximos (sem ordem garantida)."""
        dist_sq = np.empty(self.n, dtype=np.float32)

        for start in range(0, self.n, chunk):
            end = min(start + chunk, self.n)
            block = self.vectors[start:end].astype(np.float32)
            diff = block - query
            dist_sq[start:end] = np.einsum("ij,ij->i", diff, diff)

        return np.argpartition(dist_sq, k)[:k]

    def score(self, query: np.ndarray, k: int = 5) -> tuple[float, bool]:
        """Retorna (fraud_score, approved) seguindo a regra da Rinha."""
        idx = self.knn(query, k)
        frauds = int(self.labels[idx].sum())
        fraud_score = frauds / k
        return fraud_score, fraud_score < 0.6
