#!/usr/bin/env python3
from __future__ import annotations

import math

import networkx as nx

from cfmm_routing.config import PoolSpec, RoutingConfig
from cfmm_routing.market import build_market, curve_proxy_k, curve_stableswap_out_given_in
from cfmm_routing.routing import solve_max_out
from cfmm_routing.sbm import build_market_config_from_graph, classify_curve_pair, curve_proxy_k_from_A


def _assert_close(a: float, b: float, tol: float = 1e-5) -> None:
    if not math.isclose(a, b, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"Expected {a} ~= {b}")


def main() -> int:
    categories = {
        ("stable", "stable"): "stable_stable",
        ("major", "major"): "major_major",
        ("stable", "major"): "mixed",
    }
    expected_order = []
    for pair, expected in categories.items():
        got = classify_curve_pair(*pair)
        if got != expected:
            raise AssertionError(f"category mismatch for {pair}: {got} != {expected}")
        expected_order.append(curve_proxy_k_from_A(400.0, got))

    if not (expected_order[0] < expected_order[2] < expected_order[1]):
        raise AssertionError(f"unexpected k ordering: {expected_order}")

    G = nx.Graph()
    G.add_node(0, token_type="stable")
    G.add_node(1, token_type="stable")
    G.add_edge(0, 1, amm="curve", liquidity=1_000_000.0, fee=0.001, A=1400)
    cfg = build_market_config_from_graph(G)
    pool = cfg.pools[0]

    if pool.params.get("curve_model") != "calibrated_stableswap_proxy_from_A":
        raise AssertionError(f"unexpected curve model tag: {pool.params}")
    if pool.params.get("curve_category") != "stable_stable":
        raise AssertionError(f"unexpected curve category: {pool.params}")
    _assert_close(pool.params["k"], curve_proxy_k_from_A(pool.params["A"], "stable_stable"), tol=1e-9)
    _assert_close(curve_proxy_k(pool), curve_proxy_k_from_A(pool.params["A"], "stable_stable"), tol=1e-9)

    a_only_pool = PoolSpec(
        uid="curve-a-only",
        ptype="curve",
        i=0,
        j=1,
        liquidity=1_000_000.0,
        params={"fee": 0.001, "A": 1400.0, "curve_category": "stable_stable"},
    )
    _assert_close(curve_proxy_k(a_only_pool), curve_proxy_k_from_A(1400.0, "stable_stable"), tol=1e-9)

    mkt = build_market(cfg)
    rcfg = RoutingConfig(solver="SCS", solver_opts={"max_iters": 50_000, "eps": 1e-7, "verbose": False})

    dx = 25_000.0
    priced = curve_stableswap_out_given_in(pool, dx)
    priced_from_a_only = curve_stableswap_out_given_in(a_only_pool, dx)
    solved = solve_max_out(mkt, in_asset=0, out_asset=1, dx_total=dx, rcfg=rcfg)
    if solved.status not in ("optimal", "optimal_inaccurate"):
        raise AssertionError(f"unexpected solve status: {solved.status} / {solved.solver_info}")
    _assert_close(priced, priced_from_a_only, tol=5e-3)
    _assert_close(priced, solved.dy_total, tol=5e-3)

    print({
        "stable_stable_k": expected_order[0],
        "major_major_k": expected_order[1],
        "mixed_k": expected_order[2],
        "priced": priced,
        "priced_from_a_only": priced_from_a_only,
        "solved": solved.dy_total,
        "status": solved.status,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
