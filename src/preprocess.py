from __future__ import annotations

import gzip
import sys
import time
import urllib.request
from pathlib import Path

import msgspec
import numpy as np

RAW_BASE = "https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources"

FILES = {
    "references.json.gz": f"{RAW_BASE}/references.json.gz",
    "mcc_risk.json": f"{RAW_BASE}/mcc_risk.json",
    "normalization.json": f"{RAW_BASE}/normalization.json",
}

RESOURCES = Path("resources")
DATA = Path("data")


class Ref(msgspec.Struct):
    vector: list[float]
    label: str


def _progress(blocks: int, block_size: int, total: int) -> None:
    """Callback de progresso pro download de ~48 MB não parecer travado."""
    done = blocks * block_size
    if total > 0:
        pct = min(done / total * 100, 100)
        sys.stdout.write(f"\r {pct:5.1f}% ({done / 1e6:5.1f} MB)")
        sys.stdout.flush()


def download_all() -> None:
    RESOURCES.mkdir(exist_ok=True)
    for name, url in FILES.items():
        dest = RESOURCES / name
        if dest.exists():
            print(f"[skip] {name} já existe localmente")
            continue
        print(f"[download] {name}")
        try:
            urllib.request.urlretrieve(url, dest, _progress)
            print()
        except Exception as e:
            sys.exit(
                f"\nFalha ao baixar {name}: {e}\n"
                f"Verifique a URL abrindo o arquivo no GitHub e clicando em 'Raw'.\n"
                f"Se o caminho mudou, ajuste RAW_BASE no topo deste arquivo."
            )


def build_arrays() -> None:
    DATA.mkdir(exist_ok=True)
    gz = RESOURCES / "references.json.gz"

    print(f"[parse] descomprimindo e parseando {gz.name} ...")
    t0 = time.perf_counter()
    with gzip.open(gz, "rb") as f:
        raw = f.read()
    refs = msgspec.json.decode(raw, type=list[Ref])
    del raw
    n = len(refs)
    print(f" {n:,} vetores parseados em {time.perf_counter() - t0:.1f}s")

    print("[transform] montando arrays NumPy ...")
    vectors = np.array([r.vector for r in refs], dtype=np.float16)

    labels = np.fromiter(
        (1 if r.label == "fraud" else 0 for r in refs),
        dtype=np.uint8,
        count=n,
    )
    del refs

    assert vectors.shape == (n, 14), f"shape inesperado: {vectors.shape}"

    print(f" vetores: {vectors.shape} {vectors.dtype} ({vectors.nbytes / 1e6:.1f} MB)")

    np.save(DATA / "vectors.f16.npy", vectors)
    np.save(DATA / "labels.u8.npy", labels)
    print(f"[ok] artefatos salvos em {DATA}/")


if __name__ == "__main__":
    download_all()
    build_arrays()
