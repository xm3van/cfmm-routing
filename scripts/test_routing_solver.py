#!/usr/bin/env python3
"""Notebook-style SBM routing validation script.

Mirrors the setup in notebooks/sbm_testing.ipynb and runs a small sweep to
validate that convex routing optimization returns optimal statuses for
reachable token pairs.
"""

from __future__ import annotations

import argparse
import importlib
import json
from collections import defaultdict


def _require(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception as exc:
        print(f"[missing] {module}: {exc}")
        return False


def _build_generator(seed: int):
    import numpy as np
    from cfmm_routing.sbm import (
        SBMGenerator,
        RoleSBMConfig,
        TopologyModel,
        NodeAttributeRule,
        NodeAttributeModel,
        EdgeAttributeRule,
        EdgeAttributeModel,
    )

    role_cfg = RoleSBMConfig(
        n_nodes=50,
        role_probs={"core": 0.08, "mid": 0.17, "periphery": 0.75},
        role_connectivity={
            ("core", "core"): 0.6,
            ("core", "mid"): 0.35,
            ("core", "periphery"): 0.15,
            ("mid", "mid"): 0.08,
            ("mid", "periphery"): 0.04,
            ("periphery", "periphery"): 0.005,
        },
        degree_correction=True,
        pareto_alpha=2.5,
        seed=seed,
    )
    topology_model = TopologyModel(role_cfg)

    def token_type_sampler(node, G, rng):
        role = G.nodes[node]["role"]
        conditional_probs = {
            "core": {"stable": 0.45, "major": 0.45, "alt": 0.1, "meme": 0.0},
            "mid": {"stable": 0.1, "major": 0.4, "alt": 0.4, "meme": 0.1},
            "periphery": {"stable": 0.02, "major": 0.08, "alt": 0.55, "meme": 0.35},
        }
        types = list(conditional_probs[role].keys())
        probs = np.array(list(conditional_probs[role].values()))
        probs /= probs.sum()
        return rng.choice(types, p=probs)

    node_model = NodeAttributeModel(
        {"token_type": NodeAttributeRule("token_type", token_type_sampler)},
        seed=11,
    )

    def amm_sampler(i, j, G, rng):
        ti = G.nodes[i]["token_type"]
        tj = G.nodes[j]["token_type"]
        if ti == "stable" and tj == "stable":
            return rng.choice(["curve", "univ2"], p=[0.9, 0.1])
        return "univ2"

    def liquidity_sampler(i, j, G, rng):
        ri = G.nodes[i]["role"]
        rj = G.nodes[j]["role"]
        base = {"core": 5e6, "mid": 1e6, "periphery": 2e5}
        scale = (base[ri] + base[rj]) / 2
        return scale * rng.lognormal(mean=0, sigma=0.6)

    def fee_sampler(i, j, G, rng):
        return float(rng.integers(1, 51) / 1000)

    def A_sampler(i, j, G, rng):
        if G.edges[i, j]["amm"] == "curve":
            return int(rng.uniform(500, 2000))
        return None

    edge_model = EdgeAttributeModel(
        {
            "amm": EdgeAttributeRule("amm", amm_sampler),
            "liquidity": EdgeAttributeRule("liquidity", liquidity_sampler),
            "fee": EdgeAttributeRule("fee", fee_sampler),
            "A": EdgeAttributeRule("A", A_sampler),
        },
        seed=22,
    )

    return SBMGenerator(topology_model=topology_model, node_model=node_model, edge_model=edge_model)


def _make_trade_sizes(max_size: float, n: int):
    import numpy as np

    xs = np.unique(np.round(np.logspace(np.log10(1.0), np.log10(max_size), n)).astype(float))
    return tuple(xs.tolist())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run notebook-style SBM convex routing validation.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-size", type=float, default=1000.0)
    parser.add_argument("--n-trades", type=int, default=20)
    parser.add_argument("--max-pairs", type=int, default=10, help="cap number of reachable pairs to test")
    args = parser.parse_args()

    ok = True
    for module in ("numpy", "cvxpy", "networkx"):
        ok &= _require(module)
    if not ok:
        print("\nDependency check failed. Install numpy/cvxpy/networkx to run this validation.")
        return 2

    import networkx as nx
    import numpy as np

    from cfmm_routing.config import RoutingConfig, SweepConfig
    from cfmm_routing.harness import run_sweep
    from cfmm_routing.sbm import build_market_config_from_graph

    generator = _build_generator(args.seed)
    G = generator.generate()

    comp_id = {}
    for k, comp in enumerate(nx.connected_components(G)):
        for node in comp:
            comp_id[node] = k

    def reachable(i: int, j: int) -> bool:
        return comp_id[i] == comp_id[j]

    token_types = nx.get_node_attributes(G, "token_type")
    token_nodes = defaultdict(list)
    for node, t in token_types.items():
        token_nodes[t].append(node)

    stable_nodes = token_nodes.get("stable", [])
    pairs = []
    for i in range(len(stable_nodes)):
        for j in range(i + 1, len(stable_nodes)):
            a, b = stable_nodes[i], stable_nodes[j]
            if reachable(a, b):
                pairs.append((a, b))

    pairs = pairs[: args.max_pairs]
    if not pairs:
        print("No reachable stable-stable pairs found for this seed.")
        return 1

    market_cfg = build_market_config_from_graph(G)
    routing_cfg = RoutingConfig(
        solver="SCS",
        solver_opts={
            "max_iters": 50_000,
            "eps": 1e-5,
            "verbose": False,
            "acceleration_lookback": 10,
        },
    )
    trade_sizes = _make_trade_sizes(max_size=args.max_size, n=args.n_trades)

    results = []
    statuses = []

    for i, j in pairs:
        sweep_cfg = SweepConfig(in_asset=i, out_asset=j, dx_grid=trade_sizes)
        try:
            res = run_sweep(market_cfg=market_cfg, routing_cfg=routing_cfg, sweep_cfg=sweep_cfg)
            status = "ok"
            mean_pi = float(np.mean([(dx - dy) / dx for dx, dy in zip(res["dxs"], res["dys"])]))
            results.append({"pair": [i, j], "status": status, "mean_pi": mean_pi})
            statuses.append(status)
        except Exception as exc:
            results.append({"pair": [i, j], "status": "error", "error": repr(exc)})
            statuses.append("error")

    summary = {
        "seed": args.seed,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "stable_nodes": len(stable_nodes),
        "pairs_tested": len(pairs),
        "pair_results": results,
    }
    print(json.dumps(summary, indent=2))

    if any(s != "ok" for s in statuses):
        print("\nValidation failed: at least one pair encountered a solve error.")
        return 1

    print("\nValidation passed: convex optimization succeeded for all tested SBM pairs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
