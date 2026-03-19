from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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


@dataclass(frozen=True)
class TopologyPreset:
    role_probs: dict[str, float]
    role_connectivity: dict[tuple[str, str], float]
    degree_correction: bool
    pareto_alpha: float


TOPOLOGY_PRESETS: dict[str, TopologyPreset] = {
    "core_periphery_strong": TopologyPreset(
        role_probs={"core": 0.08, "mid": 0.17, "periphery": 0.75},
        role_connectivity={
            ("core", "core"): 0.65,
            ("core", "mid"): 0.34,
            ("core", "periphery"): 0.14,
            ("mid", "mid"): 0.09,
            ("mid", "periphery"): 0.03,
            ("periphery", "periphery"): 0.008,
        },
        degree_correction=True,
        pareto_alpha=2.1,
    ),
    "balanced": TopologyPreset(
        role_probs={"core": 0.22, "mid": 0.38, "periphery": 0.40},
        role_connectivity={
            ("core", "core"): 0.34,
            ("core", "mid"): 0.24,
            ("core", "periphery"): 0.18,
            ("mid", "mid"): 0.18,
            ("mid", "periphery"): 0.12,
            ("periphery", "periphery"): 0.08,
        },
        degree_correction=True,
        pareto_alpha=2.8,
    ),
    "fragmented_periphery": TopologyPreset(
        role_probs={"core": 0.10, "mid": 0.20, "periphery": 0.70},
        role_connectivity={
            ("core", "core"): 0.48,
            ("core", "mid"): 0.22,
            ("core", "periphery"): 0.08,
            ("mid", "mid"): 0.07,
            ("mid", "periphery"): 0.025,
            ("periphery", "periphery"): 0.002,
        },
        degree_correction=False,
        pareto_alpha=3.5,
    ),
    "hub_dominant": TopologyPreset(
        role_probs={"core": 0.05, "mid": 0.15, "periphery": 0.80},
        role_connectivity={
            ("core", "core"): 0.72,
            ("core", "mid"): 0.44,
            ("core", "periphery"): 0.22,
            ("mid", "mid"): 0.06,
            ("mid", "periphery"): 0.035,
            ("periphery", "periphery"): 0.004,
        },
        degree_correction=True,
        pareto_alpha=1.8,
    ),
}


def build_generator(config: ExperimentConfig, seed: int) -> SBMGenerator:
    preset_name = str(config.fixed_parameters["topology_preset"])
    preset = TOPOLOGY_PRESETS[preset_name]

    role_cfg = RoleSBMConfig(
        n_nodes=int(config.fixed_parameters["n_nodes"]),
        role_probs=preset.role_probs,
        role_connectivity=preset.role_connectivity,
        degree_correction=preset.degree_correction,
        pareto_alpha=preset.pareto_alpha,
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


def build_experiment_config(topology_preset: str) -> ExperimentConfig:
    if topology_preset not in TOPOLOGY_PRESETS:
        raise KeyError(f"Unknown topology preset: {topology_preset}")

    preset = TOPOLOGY_PRESETS[topology_preset]
    return ExperimentConfig(
        varied_parameter=VariedParameter(name="topology_preset", value=topology_preset),
        fixed_parameters={
            "n_nodes": 28,
            "topology_preset": topology_preset,
            "degree_correction": preset.degree_correction,
            "pareto_alpha": preset.pareto_alpha,
        },
        seeds=(3, 4, 5),
        pair_sampling_policy=PairSamplingPolicy(mode="sample", max_pairs_per_category=5, seed_offset=91),
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


def main() -> int:
    config = build_experiment_config("balanced")
    run_experiment(config, build_generator)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
