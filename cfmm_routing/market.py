from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from cfmm_routing.config import MarketConfig, PoolSpec

_CURVE_CATEGORY_SCALE = {
    "stable_stable": 30.0,
    "major_major": 140.0,
    "mixed": 90.0,
    "other": 180.0,
}

def _is_cvx(x) -> bool:
    try:
        import cvxpy as cp
        return isinstance(x, cp.Expression)
    except Exception:
        return False


@dataclass
class Market:
    n_assets: int
    pools: List[PoolSpec]
    # adjacency: (i,j) -> list of pool indices
    adj: Dict[Tuple[int, int], List[int]]


def assert_unique_uids(pools: List[PoolSpec]) -> None:
    uids = [p.uid for p in pools]
    if len(set(uids)) != len(uids):
        dupes = sorted({u for u in uids if uids.count(u) > 1})
        raise ValueError(f"Non-unique pool uids: {dupes}")


def build_market(cfg: MarketConfig) -> Market:
    pools = list(cfg.pools)
    assert_unique_uids(pools)

    adj: Dict[Tuple[int, int], List[int]] = {}
    for k, p in enumerate(pools):
        for a, b in [(p.i, p.j), (p.j, p.i)]:
            adj.setdefault((a, b), []).append(k)

    return Market(n_assets=cfg.n_assets, pools=pools, adj=adj)


# ---------- Pool models (minimal, convex-friendly) ----------

def _fee(p: PoolSpec) -> float:
    return float(p.params.get("fee", 0.0))


def curve_proxy_k(p: PoolSpec) -> float:
    """Resolve the Curve proxy depth parameter, deriving it from A when needed."""
    if p.params.get("k") is not None:
        k = float(p.params["k"])
    else:
        category = str(p.params.get("curve_category", "other"))
        A = float(p.params.get("A", 120.0))
        if A <= 0:
            raise ValueError("curve proxy requires A > 0 when k is not provided")
        scale = float(_CURVE_CATEGORY_SCALE.get(category, _CURVE_CATEGORY_SCALE["other"]))
        k = scale / (scale + A)

    if k <= 0:
        raise ValueError("curve proxy requires k > 0")
    return k


def curve_proxy_virtual_reserve(p: PoolSpec) -> float:
    """Virtual reserve boost used by the calibrated Curve proxy."""
    L = float(p.liquidity)
    k = curve_proxy_k(p)
    return L * (1.0 / k - 1.0)


# def univ2_out_given_in(p: PoolSpec, dx):
#     fee = _fee(p)
#     dx_eff = dx * (1.0 - fee)
#     x = float(p.liquidity)
#     y = float(p.liquidity)

#     if _is_cvx(dx_eff):
#         import cvxpy as cp
#         # dy = y * (1 - x / (x + dx_eff)) = y * (1 - x * inv_pos(x + dx_eff))
#         return y * (1.0 - x * cp.inv_pos(x + dx_eff))
#     else:
#         dx_eff = max(float(dx_eff), 0.0)
#         return y * dx_eff / (x + dx_eff)

def univ2_out_given_in(p: PoolSpec, dx):
    fee = float(p.params.get("fee", 0.0))
    dx_eff = dx * (1.0 - fee)
    x = float(p.liquidity)
    y = float(p.liquidity)

    # CVXPY-safe concave form:
    # dy = y * (1 - x/(x+dx_eff)) = y * (1 - x * inv_pos(x + dx_eff))
    try:
        import cvxpy as cp
        if isinstance(dx_eff, cp.Expression):
            return y * (1.0 - x * cp.inv_pos(x + dx_eff))
    except Exception:
        pass

    # numeric fallback
    dx_eff = max(float(dx_eff), 0.0)
    return y * dx_eff / (x + dx_eff)





def bal_wgm_out_given_in(p: PoolSpec, dx: float) -> float:
    """
    Balancer weighted geometric mean pool (toy):
      reserves x=y=L, weights w_i, w_j.
      dy = y * (1 - (x/(x+dx'))^(w_i/w_j))
    Concave in dx for w_i,w_j>0.
    """
    fee = _fee(p)
    dx_eff = dx * (1.0 - fee)
    x = p.liquidity
    y = p.liquidity
    wi = float(p.params.get("w_i", 0.5))
    wj = float(p.params.get("w_j", 0.5))
    ratio = x / (x + dx_eff)
    power = wi / wj
    return y * (1.0 - ratio ** power)


# def curve_stableswap_out_given_in(p: PoolSpec, dx: float) -> float:
#     """
#     Minimal concave proxy for stableswap: treat as higher depth near 1.
#     This is NOT the exact Curve invariant; it’s a research-friendly concave
#     approximation you can replace later with exact math.
#     dy = dx_eff * (1 - k*dx_eff/(L + dx_eff))
#     Equivalent to dy = dx_eff * L/(L + dx_eff*k) with k in (0,1] approx.
#     """
#     fee = _fee(p)
#     dx_eff = dx * (1.0 - fee)
#     L = p.liquidity
#     k = float(p.params.get("k", 0.2))  # smaller k => deeper (less slippage)
#     return dx_eff * L / (L + k * dx_eff)

def curve_stableswap_out_given_in(p: PoolSpec, dx) -> float:
    """
    Calibrated StableSwap proxy used consistently in pricing and routing.

    This is not the exact Curve invariant. The research input may specify Curve
    depth via amplification-style `A`; the implementation resolves that into the
    proxy depth parameter `k` (or consumes an explicit `k` if already present).
    We then model the pool as a symmetric constant-product pool with equal
    virtual reserve offsets on both assets. For liquidity L and proxy parameter
    k in (0, 1], the virtual reserve offset is v = L * (1 / k - 1). With equal
    starting reserves, this yields

        dy = dx_eff * L / (L + k * dx_eff),

    while remaining compatible with the convex routing formulation. Smaller k
    means deeper near-peg liquidity.
    """
    fee = _fee(p)
    dx_eff = dx * (1.0 - fee)

    L = float(p.liquidity)
    v = curve_proxy_virtual_reserve(p)
    base = L + v

    try:
        import cvxpy as cp
        if isinstance(dx_eff, cp.Expression):
            return base * (1.0 - base * cp.inv_pos(base + dx_eff))
    except Exception:
        pass

    dx_eff = max(float(dx_eff), 0.0)
    return base * dx_eff / (base + dx_eff)



def out_given_in(p: PoolSpec, i: int, j: int, dx: float) -> float:
    """
    Dispatch per pool type. Assumes pool connects i<->j.
    """
    if not ((i == p.i and j == p.j) or (i == p.j and j == p.i)):
        raise ValueError(f"Pool {p.uid} does not connect {i}->{j}")

    if p.ptype == "univ2":
        return univ2_out_given_in(p, dx)
    if p.ptype == "bal_wgm":
        return bal_wgm_out_given_in(p, dx)
    if p.ptype == "curve":
        return curve_stableswap_out_given_in(p, dx)
    raise ValueError(f"Unknown pool type: {p.ptype}")

