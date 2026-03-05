from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math


@dataclass
class CurvePoint:
    dx: float
    dy: float

def finite_diff_marginal(dxs: list[float], dys: list[float]) -> tuple[list[float], list[float]]:
    x_mid, m = [], []
    for a_dx, b_dx, a_dy, b_dy in zip(dxs[:-1], dxs[1:], dys[:-1], dys[1:]):
        ddx = b_dx - a_dx
        if ddx <= 0:
            continue
        x_mid.append(0.5 * (a_dx + b_dx))
        m.append((b_dy - a_dy) / ddx)  # marginal output
    return x_mid, m

def marginal_price_impact(dxs: list[float], dys: list[float]) -> tuple[list[float], list[float]]:
    x_mid, m = finite_diff_marginal(dxs, dys)
    pi = [max(0.0, 1.0 - v) for v in m]
    return x_mid, pi


def marginal_slippage(points) -> List[Tuple[float, float]]:
    # returns (dx_mid, d(dy)/d(dx)) using correct variable spacing
    out = []
    for a, b in zip(points[:-1], points[1:]):
        ddx = b.dx - a.dx
        if ddx <= 0:
            continue
        dx_mid = 0.5 * (a.dx + b.dx)
        dydx = (b.dy - a.dy) / ddx
        out.append((dx_mid, dydx))
    return out


def hhi_from_shares(shares: Dict[str, float]) -> float:
    s = sum(shares.values())
    if s <= 0:
        return 0.0
    return sum((v / s) ** 2 for v in shares.values())


def pool_use_count(shares: Dict[str, float], eps: float = 0.01) -> int:
    s = sum(shares.values())
    if s <= 0:
        return 0
    return sum(1 for v in shares.values() if (v / s) >= eps)
