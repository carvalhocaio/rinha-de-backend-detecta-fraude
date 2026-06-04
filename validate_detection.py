from __future__ import annotations

import math
import time
from pathlib import Path

import msgspec

from src.search import VectorIndex
from src.vectorize import FraudRequest, vectorize


class Entry(msgspec.Struct):
    request: FraudRequest
    expected_approved: bool
    expected_fraud_score: float


class TestData(msgspec.Struct):
    entries: list[Entry]


TEST_DATA = Path("test/test-data.json")

print(f"[load] lendo {TEST_DATA} ...")
t0 = time.perf_counter()
data = msgspec.json.decode(TEST_DATA.read_bytes(), type=TestData)
n = len(data.entries)
print(f" {n:,} entradas carregadas em {time.perf_counter() - t0:.2f}s")

print("[load] carregando índice FAISS ...")
t0 = time.perf_counter()
INDEX = VectorIndex()
print(f" índice pronto em {time.perf_counter() - t0:.2f}s (nprobe={INDEX.nprobe})")

tp = tn = fp = fn = 0
exceptions = 0

print(f"[run] processando {n:,} entradas ...")
t0 = time.perf_counter()

for entry in data.entries:
    try:
        vec = vectorize(entry.request)
        idx = INDEX.knn(vec, k=5)
        approved = (int(INDEX.labels[idx].sum()) / 5) < 0.6
    except Exception:
        approved = True
        exceptions += 1

    expected = entry.expected_approved
    if expected and approved:
        tn += 1
    elif expected and not approved:
        fp += 1
    elif not expected and not approved:
        tp += 1
    else:
        fn += 1

elapsed = time.perf_counter() - t0
print(f" concluído em {elapsed:.2f}s ({n / elapsed:,.0f} req/s)")
if exceptions:
    print(f" [aviso] {exceptions} exceções no pipelone - investigar")


err = 0
E = 1 * fp + 3 * fn + 5 * err
epsilon = E / n
failure_rate = (fp + fn + err) / n

K, BETA, EPS_MIN = 1000, 300, 0.001

if failure_rate > 0.15:
    rate_component = absolute_penalty = None
    score_det = -3000.0
    cut = True
else:
    rate_component = K * math.log10(1 / max(epsilon, EPS_MIN))
    absolute_penalty = -BETA * math.log10(1 + E)
    score_det = rate_component + absolute_penalty
    cut = False

print()
print("=" * 60)
print(f"Resultado da validação de detecção  (nprobe={INDEX.nprobe})")
print("=" * 60)
print(f"  total                  : {n:,}")
print(f"  TP (fraude negada)     : {tp:,}")
print(f"  TN (legítima aprovada) : {tn:,}")
print(f"  FP (legítima negada)   : {fp:,}")
print(f"  FN (fraude aprovada)   : {fn:,}")
print(f"  failure_rate           : {failure_rate * 100:.2f}%")
print(f"  weighted_errors_E      : {E}")
print(f"  error_rate_epsilon     : {epsilon:.6f}")
print("-" * 60)
if cut:
    print("  CORTE DISPARADO (failure_rate > 15%) → score_det fixado em −3000")
else:
    print(f"  rate_component         : {rate_component:+9.2f}")
    print(f"  absolute_penalty       : {absolute_penalty:+9.2f}")
print(f"  -> detection_score      : {score_det:+9.2f}")
print("=" * 60)
