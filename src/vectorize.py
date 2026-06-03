from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import msgspec
import numpy as np


class Transaction(msgspec.Struct):
    amount: float
    installments: int
    requested_at: str


class Customer(msgspec.Struct):
    avg_amount: float
    tx_count_24h: int
    known_merchants: list[str]


class Merchant(msgspec.Struct):
    id: str
    mcc: str
    avg_amount: float


class Terminal(msgspec.Struct):
    is_online: bool
    card_present: bool
    km_from_home: float


class LastTransaction(msgspec.Struct):
    timestamp: str
    km_from_current: float


class FraudRequest(msgspec.Struct):
    id: str
    transaction: Transaction
    customer: Customer
    merchant: Merchant
    terminal: Terminal
    last_transaction: LastTransaction | None = None


_NORM = json.loads((Path("resources") / "normalization.json").read_text())
_MCC_RISK = json.loads((Path("resources") / "mcc_risk.json").read_text())

MAX_AMOUNT = _NORM["max_amount"]
MAX_INSTALLMENTS = _NORM["max_installments"]
AMOUNT_VS_AVG_RATIO = _NORM["amount_vs_avg_ratio"]
MAX_MINUTES = _NORM["max_minutes"]
MAX_KM = _NORM["max_km"]
MAX_TX_COUNT_24H = _NORM["max_tx_count_24h"]
MAX_MERCHANT_AVG = _NORM["max_merchant_avg_amount"]

HOUR_DIVISOR = 23
DOW_DIVISOR = 6
MCC_DEFAULT_RISK = 0.5


def _clamp(x: float) -> float:
    """limitar(x): restringe ao intervalo [0.0, 1.0]."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def vectorize(req: FraudRequest) -> np.ndarray:
    """Monta o vetor de 14 dimensões na ORDEM da especificação."""
    t, c, m, term = req.transaction, req.customer, req.merchant, req.terminal
    requested = _parse_ts(t.requested_at)

    if req.last_transaction is not None:
        minutes = (
            requested - _parse_ts(req.last_transaction.timestamp)
        ).total_seconds() / 60.0
        minutes_since_last = _clamp(minutes / MAX_MINUTES)
        km_from_last = _clamp(req.last_transaction.km_from_current / MAX_KM)
    else:
        minutes_since_last = -1.0
        km_from_last = -1.0

    vec = [
        _clamp(t.amount / MAX_AMOUNT),  # 0  amount
        _clamp(t.installments / MAX_INSTALLMENTS),  # 1  installments
        _clamp((t.amount / c.avg_amount) / AMOUNT_VS_AVG_RATIO),  # 2  amount_vs_avg
        requested.hour / HOUR_DIVISOR,  # 3  hour_of_day (0-23 já cabe em [0,1])
        requested.weekday() / DOW_DIVISOR,  # 4  day_of_week (0-6 já cabe em [0,1])
        minutes_since_last,  # 5  minutes_since_last_tx (-1 se null)
        km_from_last,  # 6  km_from_last_tx (-1 se null)
        _clamp(term.km_from_home / MAX_KM),  # 7  km_from_home
        _clamp(c.tx_count_24h / MAX_TX_COUNT_24H),  # 8  tx_count_24h
        1.0 if term.is_online else 0.0,  # 9  is_online
        1.0 if term.card_present else 0.0,  # 10 card_present
        0.0 if m.id in c.known_merchants else 1.0,  # 11 unknown_merchant (invertido!)
        _MCC_RISK.get(m.mcc, MCC_DEFAULT_RISK),  # 12 mcc_risk (default se ausente)
        _clamp(m.avg_amount / MAX_MERCHANT_AVG),  # 13 merchant_avg_amount
    ]

    return np.asarray(vec, dtype=np.float32)
