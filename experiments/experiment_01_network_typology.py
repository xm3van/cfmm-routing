from __future__ import annotations

import gzip
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

from cfmm_routing.config import RoutingConfig, SweepConfig
from cfmm_routing.experiments import (
    CategoryDefinition,
    ExperimentConfig,
    PairSamplingPolicy,
    VariedParameter,
    enumerate_pairs_by_category,
    generate_graph_artifacts,
    run_experiment,
    sample_pairs,
)
from cfmm_routing.harness import run_sweep
from cfmm_routing.results import aggregate_pool_flow_by_edge_category, write_csv
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


# Research hypothesis:
# Denser token-exchange connectivity should reduce routing cost across token
# categories by increasing the number and quality of feasible paths available to
# the solver. We evaluate that hypothesis by holding the non-topological layers
# fixed and comparing topology presets using category-level lowest-cost curves
# and routing-behavior summaries.
DEFAULT_ROLE_PROBS: dict[str, float] = {"core": 0.08, "mid": 0.17, "periphery": 0.75}
DEFAULT_DEGREE_CORRECTION = True
DEFAULT_PARETO_ALPHA = 2.5
DEFAULT_PAIR_SAMPLING_POLICY = PairSamplingPolicy(mode="all")
TOKEN_TYPES: tuple[str, ...] = ("stable", "major", "alt", "meme")
DEFAULT_OUTPUT_DIR = Path("outputs/experiment_01_network_typology")
RAW_OUTPUT_FILENAME = "raw_simulation_output.json.gz"
MANIFEST_FILENAME = "run_manifest.json"


TOPOLOGY_PRESETS: dict[str, TopologyPreset] = {
    "core_periphery_strong": TopologyPreset(
        role_probs=DEFAULT_ROLE_PROBS,
        role_connectivity={
            ("core", "core"): 0.65,
            ("core", "mid"): 0.34,
            ("core", "periphery"): 0.14,
            ("mid", "mid"): 0.09,
            ("mid", "periphery"): 0.03,
            ("periphery", "periphery"): 0.008,
        },
        degree_correction=DEFAULT_DEGREE_CORRECTION,
        pareto_alpha=DEFAULT_PARETO_ALPHA,
    ),
    "balanced": TopologyPreset(
        role_probs=DEFAULT_ROLE_PROBS,
        role_connectivity={
            ("core", "core"): 0.34,
            ("core", "mid"): 0.24,
            ("core", "periphery"): 0.18,
            ("mid", "mid"): 0.18,
            ("mid", "periphery"): 0.12,
            ("periphery", "periphery"): 0.08,
        },
        degree_correction=DEFAULT_DEGREE_CORRECTION,
        pareto_alpha=DEFAULT_PARETO_ALPHA,
    ),
    "fragmented_periphery": TopologyPreset(
        role_probs=DEFAULT_ROLE_PROBS,
        role_connectivity={
            ("core", "core"): 0.48,
            ("core", "mid"): 0.22,
            ("core", "periphery"): 0.08,
            ("mid", "mid"): 0.07,
            ("mid", "periphery"): 0.025,
            ("periphery", "periphery"): 0.002,
        },
        degree_correction=DEFAULT_DEGREE_CORRECTION,
        pareto_alpha=DEFAULT_PARETO_ALPHA,
    ),
    "hub_dominant": TopologyPreset(
        role_probs=DEFAULT_ROLE_PROBS,
        role_connectivity={
            ("core", "core"): 0.72,
            ("core", "mid"): 0.44,
            ("core", "periphery"): 0.22,
            ("mid", "mid"): 0.06,
            ("mid", "periphery"): 0.035,
            ("periphery", "periphery"): 0.004,
        },
        degree_correction=DEFAULT_DEGREE_CORRECTION,
        pareto_alpha=DEFAULT_PARETO_ALPHA,
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
            "role_probs": dict(preset.role_probs),
        },
        seeds=(3, 4, 5),
        pair_sampling_policy=DEFAULT_PAIR_SAMPLING_POLICY,
        trade_size_grid=(1.0, 10.0, 100.0),
        category_definitions=build_exchange_route_categories(),
        routing_config=RoutingConfig(
            solver="SCS",
            solver_opts={"max_iters": 20000, "eps": 1e-5, "verbose": False},
        ),
    )


def build_exchange_route_categories(token_types: tuple[str, ...] = TOKEN_TYPES) -> tuple[CategoryDefinition, ...]:
    return tuple(
        CategoryDefinition(
            name=f"{source_token_type}<->{target_token_type}",
            source_token_types=(source_token_type,),
            target_token_types=(target_token_type,),
        )
        for source_token_type, target_token_type in combinations_with_replacement(token_types, 2)
    )


def _routing_cost(dx: float, dy: float) -> float:
    if dy <= 0:
        return float("inf")
    return float(dx) / float(dy)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        ).strip()
    except Exception:
        return "unknown"


def _environment_manifest() -> dict[str, object]:
    import cvxpy as cp
    import matplotlib

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "cvxpy_version": cp.__version__,
        "installed_solvers": sorted(cp.installed_solvers()),
        "git_commit": _git_commit(),
    }


def _write_json_gz(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True)


def collect_typology_outputs(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    lowest_cost_rows: list[dict[str, float | int | str]] = []
    routing_behavior_rows: list[dict[str, float | int | str]] = []
    raw_results_by_preset: dict[str, dict[str, object]] = {}

    for topology_preset in TOPOLOGY_PRESETS:
        config = build_experiment_config(topology_preset)
        experiment_result = run_experiment(config, build_generator)
        raw_results_by_preset[topology_preset] = {
            "config": {
                "varied_parameter": {
                    "name": config.varied_parameter.name,
                    "value": config.varied_parameter.value,
                },
                "fixed_parameters": dict(config.fixed_parameters),
                "seeds": list(config.seeds),
                "pair_sampling_policy": {
                    "mode": config.pair_sampling_policy.mode,
                    "max_pairs_per_category": config.pair_sampling_policy.max_pairs_per_category,
                    "seed_offset": config.pair_sampling_policy.seed_offset,
                    "sort_pairs": config.pair_sampling_policy.sort_pairs,
                },
                "trade_size_grid": list(config.trade_size_grid),
                "category_definitions": [
                    {
                        "name": category.name,
                        "source_token_types": list(category.source_token_types),
                        "target_token_types": list(category.target_token_types),
                        "allow_same_node": category.allow_same_node,
                    }
                    for category in config.category_definitions
                ],
                "routing_config": {
                    "objective": config.routing_config.objective,
                    "solver": config.routing_config.solver,
                    "solver_opts": dict(config.routing_config.solver_opts),
                },
            },
            "raw_rows": {
                "graph_rows": experiment_result.graph_rows,
                "eligible_pair_rows": experiment_result.eligible_pair_rows,
                "pair_curve_rows": experiment_result.pair_curve_rows,
                "graph_curve_rows": experiment_result.graph_curve_rows,
                "aggregate_curve_rows": experiment_result.aggregate_curve_rows,
                "node_rows": experiment_result.node_rows,
                "edge_rows": experiment_result.edge_rows,
            },
        }

        lowest_cost_by_key: dict[tuple[str, int, str, float], float] = {}
        for row in experiment_result.pair_curve_rows:
            key = (
                str(row["topology_preset"]),
                int(row["seed"]),
                str(row["category"]),
                float(row["dx"]),
            )
            lowest_cost_by_key[key] = min(
                lowest_cost_by_key.get(key, float("inf")),
                _routing_cost(float(row["dx"]), float(row["dy"])),
            )

        for (preset_name, seed, category, dx), routing_cost in sorted(lowest_cost_by_key.items()):
            lowest_cost_rows.append(
                {
                    "topology_preset": preset_name,
                    "seed": seed,
                    "category": category,
                    "dx": dx,
                    "lowest_routing_cost": routing_cost,
                }
            )

        for seed in config.seeds:
            generator = build_generator(config, seed)
            artifacts = generate_graph_artifacts(generator, seed)
            category_pairs = enumerate_pairs_by_category(artifacts.graph, config.category_definitions)

            for category_name, eligible_pairs in sorted(category_pairs.items()):
                for source_asset, target_asset in sample_pairs(
                    eligible_pairs,
                    config.pair_sampling_policy,
                    seed=seed,
                ):
                    sweep_result = run_sweep(
                        market_cfg=artifacts.market_config,
                        routing_cfg=config.routing_config,
                        sweep_cfg=SweepConfig(
                            in_asset=int(source_asset),
                            out_asset=int(target_asset),
                            dx_grid=tuple(float(dx) for dx in config.trade_size_grid),
                        ),
                    )
                    for dx, pool_out in zip(sweep_result["dxs"], sweep_result["composition"]):
                        flow_by_edge_category = aggregate_pool_flow_by_edge_category(
                            pool_out,
                            artifacts.market_config,
                            mode="sum",
                        )
                        total_flow = sum(flow_by_edge_category.values()) or 1.0
                        for route_edge_category, flow in sorted(flow_by_edge_category.items()):
                            routing_behavior_rows.append(
                                {
                                    "topology_preset": topology_preset,
                                    "seed": seed,
                                    "category": category_name,
                                    "dx": float(dx),
                                    "source_asset": int(source_asset),
                                    "target_asset": int(target_asset),
                                    "route_edge_category": route_edge_category,
                                    "flow": float(flow),
                                    "share": float(flow) / float(total_flow),
                                }
                            )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(str(output_dir / "lowest_cost_rows.csv"), lowest_cost_rows)
    write_csv(str(output_dir / "routing_behavior_rows.csv"), routing_behavior_rows)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": "experiment_01_network_typology",
        "raw_results_by_preset": raw_results_by_preset,
        "lowest_cost_rows": lowest_cost_rows,
        "routing_behavior_rows": routing_behavior_rows,
    }


def plot_lowest_cost_curves(lowest_cost_rows: list[dict[str, float | int | str]], output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    by_category: dict[str, dict[str, dict[float, list[float]]]] = {}
    for row in lowest_cost_rows:
        category = str(row["category"])
        preset = str(row["topology_preset"])
        dx = float(row["dx"])
        by_category.setdefault(category, {}).setdefault(preset, {}).setdefault(dx, []).append(
            float(row["lowest_routing_cost"])
        )

    categories = sorted(by_category)
    ncols = 2
    nrows = int(np.ceil(len(categories) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, max(4, 3.5 * nrows)), sharex=False)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax, category in zip(axes.flatten(), categories):
        preset_map = by_category[category]
        for preset, dx_map in sorted(preset_map.items()):
            xs = sorted(dx_map)
            ys = [float(np.mean(dx_map[dx])) for dx in xs]
            ax.plot(xs, ys, marker="o", linewidth=2, label=preset)
        ax.set_title(f"Lowest routing cost curves: {category}")
        ax.set_xlabel("Trade size (dx)")
        ax.set_ylabel("Lowest routing cost (dx / dy)")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    for ax in axes.flatten()[len(categories):]:
        ax.axis("off")
    fig.tight_layout()
    out_path = output_dir / "lowest_cost_curves_summary.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_routing_behavior(routing_behavior_rows: list[dict[str, float | int | str]], output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    by_category: dict[str, dict[str, dict[str, dict[float, float]]]] = {}
    for row in routing_behavior_rows:
        category = str(row["category"])
        preset = str(row["topology_preset"])
        route_edge_category = str(row["route_edge_category"])
        dx = float(row["dx"])
        by_category.setdefault(category, {}).setdefault(preset, {}).setdefault(route_edge_category, {}).setdefault(dx, 0.0)
        by_category[category][preset][route_edge_category][dx] += float(row["flow"])

    categories = sorted(by_category)
    ncols = 2
    nrows = int(np.ceil(len(categories) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, max(4, 3.8 * nrows)), sharex=False)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax, category in zip(axes.flatten(), categories):
        preset_map = by_category[category]
        for preset, edge_category_map in sorted(preset_map.items()):
            dxs = sorted({dx for dx_map in edge_category_map.values() for dx in dx_map})
            for route_edge_category, dx_map in sorted(edge_category_map.items()):
                flows = np.array([dx_map.get(dx, 0.0) for dx in dxs], dtype=float)
                totals = np.array(
                    [
                        sum(category_map.get(dx, 0.0) for category_map in edge_category_map.values())
                        for dx in dxs
                    ],
                    dtype=float,
                )
                shares = np.divide(flows, np.where(totals > 0, totals, 1.0))
                ax.plot(
                    dxs,
                    shares,
                    linewidth=2,
                    marker="o",
                    label=f"{preset} | {route_edge_category}",
                )
        ax.set_title(f"Routing behaviour shares: {category}")
        ax.set_xlabel("Trade size (dx)")
        ax.set_ylabel("Flow share")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.2)
    for ax in axes.flatten()[len(categories):]:
        ax.axis("off")
    axes[0, 0].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    out_path = output_dir / "routing_behaviour_summary.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_run_manifest(
    output_dir: Path,
    analysis_payload: dict[str, object],
    image_paths: list[Path],
) -> Path:
    manifest = {
        "generated_at_utc": analysis_payload["generated_at_utc"],
        "experiment_name": analysis_payload["experiment_name"],
        "token_types": list(TOKEN_TYPES),
        "topology_presets": sorted(TOPOLOGY_PRESETS),
        "environment": _environment_manifest(),
        "files": {
            "raw_simulation_output": str(output_dir / RAW_OUTPUT_FILENAME),
            "lowest_cost_rows_csv": str(output_dir / "lowest_cost_rows.csv"),
            "routing_behavior_rows_csv": str(output_dir / "routing_behavior_rows.csv"),
            "images": [str(path) for path in image_paths],
        },
    }
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest_path


def run_typology_analysis(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, object]:
    analysis_payload = collect_typology_outputs(output_dir)
    raw_output_path = output_dir / RAW_OUTPUT_FILENAME
    _write_json_gz(
        raw_output_path,
        {
            "generated_at_utc": analysis_payload["generated_at_utc"],
            "experiment_name": analysis_payload["experiment_name"],
            "raw_results_by_preset": analysis_payload["raw_results_by_preset"],
            "lowest_cost_rows": analysis_payload["lowest_cost_rows"],
            "routing_behavior_rows": analysis_payload["routing_behavior_rows"],
        },
    )
    lowest_cost_path = plot_lowest_cost_curves(analysis_payload["lowest_cost_rows"], output_dir)
    routing_behavior_path = plot_routing_behavior(analysis_payload["routing_behavior_rows"], output_dir)
    manifest_path = write_run_manifest(output_dir, analysis_payload, [lowest_cost_path, routing_behavior_path])
    return {
        **analysis_payload,
        "raw_output_path": raw_output_path,
        "manifest_path": manifest_path,
        "image_paths": [lowest_cost_path, routing_behavior_path],
    }


def main() -> int:
    run_typology_analysis()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
