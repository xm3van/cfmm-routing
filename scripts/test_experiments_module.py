#!/usr/bin/env python3
from __future__ import annotations

import json
import numpy as np


from experiments.experiment_01_network_typology import (
    DEFAULT_DEGREE_CORRECTION,
    DEFAULT_PAIR_SAMPLING_POLICY,
    DEFAULT_PARETO_ALPHA,
    DEFAULT_ROLE_PROBS,
    TOPOLOGY_PRESETS,
)
from cfmm_routing.config import RoutingConfig
from cfmm_routing.experiments import (
    CategoryDefinition,
    ExperimentConfig,
    PairSamplingPolicy,
    VariedParameter,
    run_experiment,
)
from cfmm_routing.sbm import (
    EdgeAttributeModel,
    EdgeAttributeRule,
    NodeAttributeModel,
    NodeAttributeRule,
    RoleSBMConfig,
    SBMGenerator,
    TopologyModel,
)


def build_generator(config: ExperimentConfig, seed: int) -> SBMGenerator:
    role_cfg = RoleSBMConfig(
        n_nodes=int(config.fixed_parameters["n_nodes"]),
        role_probs={"core": 0.08, "mid": 0.17, "periphery": 0.75},
        role_connectivity={
            ("core", "core"): float(config.fixed_parameters["p_core_core"]),
            ("core", "mid"): 0.35,
            ("core", "periphery"): 0.15,
            ("mid", "mid"): 0.08,
            ("mid", "periphery"): 0.04,
            ("periphery", "periphery"): float(config.varied_parameter.value),
        },
        degree_correction=True,
        pareto_alpha=2.5,
        seed=seed,
    )
    topology_model = TopologyModel(role_cfg)

    def token_type_sampler(node, graph, rng):
        role = graph.nodes[node]["role"]
        conditional_probs = {
            "core": {"stable": 0.45, "major": 0.45, "alt": 0.1, "meme": 0.0},
            "mid": {"stable": 0.1, "major": 0.4, "alt": 0.4, "meme": 0.1},
            "periphery": {"stable": 0.02, "major": 0.08, "alt": 0.55, "meme": 0.35},
        }
        types = list(conditional_probs[role].keys())
        probs = np.array(list(conditional_probs[role].values()), dtype=float)
        probs /= probs.sum()
        return str(rng.choice(types, p=probs))

    node_model = NodeAttributeModel(
        {"token_type": NodeAttributeRule("token_type", token_type_sampler)},
        seed=seed + 1,
    )

    def amm_sampler(i, j, graph, rng):
        ti = graph.nodes[i]["token_type"]
        tj = graph.nodes[j]["token_type"]
        if ti == "stable" and tj == "stable":
            return str(rng.choice(["curve", "univ2"], p=[0.9, 0.1]))
        return "univ2"

    def liquidity_sampler(i, j, graph, rng):
        base = {"core": 5e6, "mid": 1e6, "periphery": 2e5}
        ri = graph.nodes[i]["role"]
        rj = graph.nodes[j]["role"]
        scale = (base[ri] + base[rj]) / 2
        return float(scale * rng.lognormal(mean=0, sigma=0.4))

    def fee_sampler(i, j, graph, rng):
        return float(rng.integers(1, 10) / 1000)

    def a_sampler(i, j, graph, rng):
        if graph.edges[i, j]["amm"] == "curve":
            return int(rng.integers(500, 1200))
        return None

    edge_model = EdgeAttributeModel(
        {
            "amm": EdgeAttributeRule("amm", amm_sampler),
            "liquidity": EdgeAttributeRule("liquidity", liquidity_sampler),
            "fee": EdgeAttributeRule("fee", fee_sampler),
            "A": EdgeAttributeRule("A", a_sampler),
        },
        seed=seed + 2,
    )
    return SBMGenerator(topology_model=topology_model, node_model=node_model, edge_model=edge_model)


def main() -> int:
    config = ExperimentConfig(
        varied_parameter=VariedParameter(name="p_periphery_periphery", value=0.01),
        fixed_parameters={"n_nodes": 28, "p_core_core": 0.55},
        seeds=(3, 4),
        pair_sampling_policy=PairSamplingPolicy(mode="sample", max_pairs_per_category=3, seed_offset=91),
        trade_size_grid=(1.0, 10.0, 100.0),
        category_definitions=(
            CategoryDefinition(name="stable->stable", source_token_types=("stable",), target_token_types=("stable",)),
            CategoryDefinition(name="stable->major", source_token_types=("stable",), target_token_types=("major",)),
            CategoryDefinition(name="major->major", source_token_types=("major",), target_token_types=("major",)),
        ),
        routing_config=RoutingConfig(
            solver="SCS",
            solver_opts={"max_iters": 20000, "eps": 1e-5, "verbose": False},
        ),
    )

    result = run_experiment(config, build_generator)

    assert len(result.graph_rows) == len(config.seeds)
    assert result.graph_curve_rows, "expected per-graph aggregated curves"
    assert result.aggregate_curve_rows, "expected aggregated curves across seeds"
    assert result.eligible_pair_rows, "expected eligible reachable pairs"
    assert result.pair_curve_rows, "expected pair-level sweep outputs"
    assert all("varied_parameter_name" in row for row in result.graph_rows)
    assert all(row["pair_count"] >= 1 for row in result.graph_curve_rows)
    assert all(row["n_nodes"] == 28 for row in result.aggregate_curve_rows)
    assert all(row["n_nodes"] == 28 for row in result.node_rows)
    assert all(row["n_nodes"] == 28 for row in result.edge_rows)
    assert all(preset.role_probs == DEFAULT_ROLE_PROBS for preset in TOPOLOGY_PRESETS.values())
    assert all(preset.degree_correction == DEFAULT_DEGREE_CORRECTION for preset in TOPOLOGY_PRESETS.values())
    assert all(preset.pareto_alpha == DEFAULT_PARETO_ALPHA for preset in TOPOLOGY_PRESETS.values())
    assert DEFAULT_PAIR_SAMPLING_POLICY.mode == "all"
    assert DEFAULT_PAIR_SAMPLING_POLICY.max_pairs_per_category is None


    print(json.dumps({
        "graphs": len(result.graph_rows),
        "eligible_pairs": len(result.eligible_pair_rows),
        "pair_curve_rows": len(result.pair_curve_rows),
        "graph_curve_rows": len(result.graph_curve_rows),
        "aggregate_curve_rows": len(result.aggregate_curve_rows),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
