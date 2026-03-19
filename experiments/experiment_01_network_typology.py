from __future__ import annotations

import gzip
import json
import math
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
DEFAULT_TRADE_SIZE_GRID: tuple[float, ...] = tuple(float(dx) for dx in np.geomspace(1.0, 10_000.0, num=16))
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
        trade_size_grid=DEFAULT_TRADE_SIZE_GRID,
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


def _safe_avg_price(dx: float, dy: float) -> float:
    if dx <= 0:
        return float("nan")
    return float(dy) / float(dx)


def _routing_cost(dx: float, dy: float) -> float:
    if dy <= 0:
        return float("inf")
    return float(dx) / float(dy)


def _log_cost(avg_price: float) -> float:
    if avg_price <= 0 or not math.isfinite(avg_price):
        return float("inf")
    return -math.log(avg_price)


def _relative_price_deterioration(avg_price: float, baseline_avg_price: float) -> float:
    if baseline_avg_price <= 0 or not math.isfinite(baseline_avg_price):
        return float("nan")
    return max(0.0, (baseline_avg_price - avg_price) / baseline_avg_price)


def _normalized_avg_price(avg_price: float, baseline_avg_price: float) -> float:
    if baseline_avg_price <= 0 or not math.isfinite(baseline_avg_price):
        return float("nan")
    return avg_price / baseline_avg_price


def _append_metric_rows(
    rows: list[dict[str, float | int | str]],
    grouped_rows: dict[tuple[object, ...], list[dict[str, object]]],
    *,
    value_field: str,
    metadata_fields: tuple[str, ...],
) -> None:
    for _, group in sorted(grouped_rows.items()):
        group_sorted = sorted(group, key=lambda row: float(row["dx"]))
        if not group_sorted:
            continue
        baseline_dx = float(group_sorted[0]["dx"])
        baseline_avg_price = _safe_avg_price(float(group_sorted[0]["dx"]), float(group_sorted[0][value_field]))
        metadata = {field: group_sorted[0][field] for field in metadata_fields}

        for row in group_sorted:
            dx = float(row["dx"])
            dy = float(row[value_field])
            avg_price = _safe_avg_price(dx, dy)
            normalized_avg_price = _normalized_avg_price(avg_price, baseline_avg_price)
            rows.append(
                {
                    **metadata,
                    "metric_kind": "level",
                    "dx": dx,
                    "dy": dy,
                    "avg_price": avg_price,
                    "baseline_dx": baseline_dx,
                    "baseline_avg_price": baseline_avg_price,
                    "normalized_avg_price": normalized_avg_price,
                    "liquidity_contraction": 1.0 - normalized_avg_price if math.isfinite(normalized_avg_price) else float("nan"),
                    "price_deterioration": _relative_price_deterioration(avg_price, baseline_avg_price),
                    "routing_cost_diagnostic": _routing_cost(dx, dy),
                    "log_cost": _log_cost(avg_price),
                    "marginal_avg_price": float("nan"),
                    "marginal_price_impact": float("nan"),
                    "marginal_log_cost": float("nan"),
                    "marginal_cost": float("nan"),
                }
            )

        for previous_row, current_row in zip(group_sorted[:-1], group_sorted[1:]):
            dx_prev = float(previous_row["dx"])
            dx_curr = float(current_row["dx"])
            dy_prev = float(previous_row[value_field])
            dy_curr = float(current_row[value_field])
            ddx = dx_curr - dx_prev
            marginal_avg_price = float("nan")
            marginal_price_impact = float("nan")
            if ddx > 0:
                marginal_avg_price = (dy_curr - dy_prev) / ddx
                if baseline_avg_price > 0 and math.isfinite(baseline_avg_price):
                    marginal_price_impact = max(0.0, 1.0 - (marginal_avg_price / baseline_avg_price))
            rows.append(
                {
                    **metadata,
                    "metric_kind": "marginal",
                    "dx": 0.5 * (dx_prev + dx_curr),
                    "dy": float("nan"),
                    "avg_price": float("nan"),
                    "baseline_dx": baseline_dx,
                    "baseline_avg_price": baseline_avg_price,
                    "normalized_avg_price": float("nan"),
                    "liquidity_contraction": float("nan"),
                    "price_deterioration": float("nan"),
                    "routing_cost_diagnostic": float("nan"),
                    "log_cost": float("nan"),
                    "dx_left": dx_prev,
                    "dx_right": dx_curr,
                    "marginal_avg_price": marginal_avg_price,
                    "marginal_price_impact": marginal_price_impact,
                    "marginal_log_cost": _log_cost(marginal_avg_price),
                    "marginal_cost": _routing_cost(1.0, marginal_avg_price),
                }
            )


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
    pair_metrics_rows: list[dict[str, float | int | str]] = []
    category_metrics_rows: list[dict[str, float | int | str]] = []
    lowest_marginal_cost_rows: list[dict[str, float | int | str]] = []
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

        pair_groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
        for row in experiment_result.pair_curve_rows:
            key = (
                str(row["topology_preset"]),
                int(row["seed"]),
                str(row["category"]),
                int(row["source_asset"]),
                int(row["target_asset"]),
            )
            pair_groups.setdefault(key, []).append(row)
        _append_metric_rows(
            pair_metrics_rows,
            pair_groups,
            value_field="dy",
            metadata_fields=("topology_preset", "seed", "category", "source_asset", "target_asset"),
        )

        category_groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
        for row in experiment_result.aggregate_curve_rows:
            key = (str(row["topology_preset"]), str(row["category"]))
            category_groups.setdefault(key, []).append(
                {
                    "topology_preset": row["topology_preset"],
                    "category": row["category"],
                    "dx": row["dx"],
                    "dy_mean": row["dy_mean_across_graphs"],
                }
            )
        _append_metric_rows(
            category_metrics_rows,
            category_groups,
            value_field="dy_mean",
            metadata_fields=("topology_preset", "category"),
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

    lowest_marginal_cost_by_key: dict[tuple[str, int, str, float], float] = {}
    for row in pair_metrics_rows:
        if row.get("metric_kind") != "marginal":
            continue
        key = (
            str(row["topology_preset"]),
            int(row["seed"]),
            str(row["category"]),
            float(row["dx"]),
        )
        lowest_marginal_cost_by_key[key] = min(
            lowest_marginal_cost_by_key.get(key, float("inf")),
            float(row["marginal_cost"]),
        )

    for (topology_preset, seed, category, dx), marginal_cost in sorted(lowest_marginal_cost_by_key.items()):
        lowest_marginal_cost_rows.append(
            {
                "topology_preset": topology_preset,
                "seed": seed,
                "category": category,
                "dx": dx,
                "lowest_marginal_cost": marginal_cost,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(str(output_dir / "pair_metrics_rows.csv"), pair_metrics_rows)
    write_csv(str(output_dir / "category_metrics_rows.csv"), category_metrics_rows)
    write_csv(str(output_dir / "lowest_marginal_cost_rows.csv"), lowest_marginal_cost_rows)
    write_csv(str(output_dir / "routing_behavior_rows.csv"), routing_behavior_rows)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": "experiment_01_network_typology",
        "raw_results_by_preset": raw_results_by_preset,
        "pair_metrics_rows": pair_metrics_rows,
        "category_metrics_rows": category_metrics_rows,
        "lowest_marginal_cost_rows": lowest_marginal_cost_rows,
        "routing_behavior_rows": routing_behavior_rows,
    }


def plot_category_metric_summary(category_metrics_rows: list[dict[str, float | int | str]], output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    level_rows = [row for row in category_metrics_rows if row.get("metric_kind", "level") == "level"]
    marginal_rows = [row for row in category_metrics_rows if row.get("metric_kind") == "marginal"]

    level_metrics = {
        "avg_price": ("Average price (dy / dx)", None),
        "normalized_avg_price": ("Normalized avg price vs smallest trade", None),
        "routing_cost_diagnostic": ("Routing cost diagnostic (dx / dy)", None),
    }

    categories = sorted({str(row["category"]) for row in category_metrics_rows})
    fig, axes = plt.subplots(
        nrows=len(categories),
        ncols=4,
        figsize=(20, max(4, 3.8 * len(categories))),
        sharex=False,
    )
    axes = np.atleast_2d(axes)

    for row_axes, category in zip(axes, categories):
        category_level_rows = [row for row in level_rows if str(row["category"]) == category]

        for ax, (metric_name, (ylabel, ylim)) in zip(row_axes, level_metrics.items()):
            preset_map: dict[str, dict[float, list[float]]] = {}
            for row in category_level_rows:
                preset = str(row["topology_preset"])
                dx = float(row["dx"])
                value = float(row[metric_name])
                preset_map.setdefault(preset, {}).setdefault(dx, []).append(value)
            for preset, dx_map in sorted(preset_map.items()):
                xs = sorted(dx_map)
                ys = [float(np.mean(dx_map[dx])) for dx in xs]
                ax.plot(xs, ys, marker="o", linewidth=2, label=preset)
            ax.set_title(f"{category}\n{ylabel}")
            ax.set_xlabel("Trade size (dx)")
            ax.set_ylabel(ylabel)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)

        marginal_ax = row_axes[-1]
        marginal_preset_map: dict[str, dict[float, list[float]]] = {}
        for row in marginal_rows:
            if str(row["category"]) != category:
                continue
            preset = str(row["topology_preset"])
            dx = float(row["dx"])
            marginal_preset_map.setdefault(preset, {}).setdefault(dx, []).append(float(row["marginal_avg_price"]))
        for preset, dx_map in sorted(marginal_preset_map.items()):
            xs = sorted(dx_map)
            ys = [float(np.mean(dx_map[dx])) for dx in xs]
            marginal_ax.plot(xs, ys, marker="o", linewidth=2, label=preset)
        marginal_ax.set_title(f"{category}\nMarginal price (Δdy / Δdx)")
        marginal_ax.set_xlabel("Trade size midpoint")
        marginal_ax.set_ylabel("Marginal price")
        marginal_ax.set_xscale("log")
        marginal_ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=False, fontsize=8)
            break
    fig.tight_layout()
    out_path = output_dir / "category_metric_summary.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_marginal_price_impact_summary(
    category_metrics_rows: list[dict[str, float | int | str]],
    output_dir: Path,
) -> Path:
    import matplotlib.pyplot as plt

    marginal_rows = [row for row in category_metrics_rows if row.get("metric_kind") == "marginal"]
    categories = sorted({str(row["category"]) for row in marginal_rows})
    ncols = 2
    nrows = max(1, int(np.ceil(len(categories) / ncols)))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, max(4, 3.8 * nrows)), sharex=False)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax, category in zip(axes.flatten(), categories):
        preset_map: dict[str, dict[float, list[float]]] = {}
        for row in marginal_rows:
            if str(row["category"]) != category:
                continue
            preset = str(row["topology_preset"])
            dx = float(row["dx"])
            preset_map.setdefault(preset, {}).setdefault(dx, []).append(float(row["marginal_avg_price"]))
        for preset, dx_map in sorted(preset_map.items()):
            xs = sorted(dx_map)
            ys = [float(np.mean(dx_map[dx])) for dx in xs]
            ax.plot(xs, ys, marker="o", linewidth=2, label=preset)
        ax.set_title(f"Mean marginal price: {category}")
        ax.set_xlabel("Trade size midpoint")
        ax.set_ylabel("Marginal price (Δdy / Δdx)")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    for ax in axes.flatten()[len(categories):]:
        ax.axis("off")
    if categories:
        axes[0, 0].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    out_path = output_dir / "marginal_price_summary.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_lowest_marginal_cost_grid(
    lowest_marginal_cost_rows: list[dict[str, float | int | str]],
    output_dir: Path,
) -> Path:
    import matplotlib.pyplot as plt

    by_category: dict[str, dict[str, dict[float, list[float]]]] = {}
    for row in lowest_marginal_cost_rows:
        category = str(row["category"])
        preset = str(row["topology_preset"])
        dx = float(row["dx"])
        value = float(row["lowest_marginal_cost"])
        by_category.setdefault(category, {}).setdefault(preset, {}).setdefault(dx, []).append(value)

    categories = sorted(by_category)
    ncols = 2
    nrows = max(1, int(np.ceil(len(categories) / ncols)))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, max(4, 3.8 * nrows)), sharex=False, squeeze=False)

    for ax, category in zip(axes.flatten(), categories):
        preset_map = by_category[category]
        for preset in sorted(TOPOLOGY_PRESETS):
            dx_map = preset_map.get(preset, {})
            if not dx_map:
                continue
            xs = sorted(dx_map)
            mean_curve = np.array([float(np.mean(dx_map[dx])) for dx in xs], dtype=float)
            std_curve = np.array([float(np.std(dx_map[dx], ddof=0)) for dx in xs], dtype=float)
            ax.plot(xs, mean_curve, marker="o", linewidth=2, label=preset)
            ax.fill_between(xs, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15)
        ax.set_title(f"Lowest marginal cost: {category}")
        ax.set_xlabel("Trade size midpoint")
        ax.set_ylabel("Lowest marginal cost (1 / (Δdy / Δdx))")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.25)

    for ax in axes.flatten()[len(categories):]:
        ax.axis("off")
    if categories:
        axes[0, 0].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    out_path = output_dir / "lowest_marginal_cost_summary.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pair_liquidity_contraction_distribution(
    pair_metrics_rows: list[dict[str, float | int | str]],
    output_dir: Path,
) -> Path:
    import matplotlib.pyplot as plt

    level_rows = [row for row in pair_metrics_rows if row.get("metric_kind", "level") == "level"]
    categories = sorted({str(row["category"]) for row in level_rows})
    ncols = 2
    nrows = max(1, int(np.ceil(len(categories) / ncols)))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, max(4, 3.8 * nrows)), sharex=False)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax, category in zip(axes.flatten(), categories):
        preset_dx_map: dict[str, dict[float, list[float]]] = {}
        for row in level_rows:
            if str(row["category"]) != category:
                continue
            preset = str(row["topology_preset"])
            dx = float(row["dx"])
            preset_dx_map.setdefault(preset, {}).setdefault(dx, []).append(float(row["price_deterioration"]))

        for preset, dx_map in sorted(preset_dx_map.items()):
            xs = sorted(dx_map)
            medians = [float(np.median(dx_map[dx])) for dx in xs]
            p10 = [float(np.percentile(dx_map[dx], 10)) for dx in xs]
            p90 = [float(np.percentile(dx_map[dx], 90)) for dx in xs]
            ax.plot(xs, medians, marker="o", linewidth=2, label=f"{preset} median")
            ax.fill_between(xs, p10, p90, alpha=0.18)

        ax.set_title(f"Liquidity contraction distribution: {category}")
        ax.set_xlabel("Trade size (dx)")
        ax.set_ylabel("Price deterioration vs smallest trade")
        ax.set_ylim(0.0, 1.0)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    for ax in axes.flatten()[len(categories):]:
        ax.axis("off")
    if categories:
        axes[0, 0].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    out_path = output_dir / "pair_liquidity_contraction_distribution.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_role_block_average_network_heatmap(output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    roles = ("core", "mid", "periphery")
    presets = sorted(TOPOLOGY_PRESETS)
    ncols = 2
    nrows = max(1, int(np.ceil(len(presets) / ncols)))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, max(4, 4.2 * nrows)), squeeze=False)

    for ax, preset_name in zip(axes.flatten(), presets):
        preset = TOPOLOGY_PRESETS[preset_name]
        matrix = np.array(
            [
                [
                    float(
                        preset.role_connectivity.get((role_i, role_j), preset.role_connectivity.get((role_j, role_i), 0.0))
                    )
                    for role_j in roles
                ]
                for role_i in roles
            ],
            dtype=float,
        )
        im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues", aspect="equal")
        ax.set_title(
            f"{preset_name}\nrole probs="
            + ", ".join(f"{role}={preset.role_probs[role]:.2f}" for role in roles)
        )
        ax.set_xticks(range(len(roles)), roles, rotation=30, ha="right")
        ax.set_yticks(range(len(roles)), roles)
        for row_idx, role_i in enumerate(roles):
            for col_idx, role_j in enumerate(roles):
                ax.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]:.3f}", ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Expected edge probability")

    for ax in axes.flatten()[len(presets):]:
        ax.axis("off")

    fig.suptitle("Role-block average network heatmap", fontsize=14)
    fig.tight_layout()
    out_path = output_dir / "role_block_average_network_heatmap.png"
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
    presets = sorted(TOPOLOGY_PRESETS)
    nrows = max(1, len(categories))
    ncols = max(1, len(presets))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols, max(3.5, 2.8 * nrows)),
        sharex=False,
        sharey=True,
        squeeze=False,
    )

    legend_handles = None
    legend_labels = None
    for row_idx, category in enumerate(categories):
        preset_map = by_category[category]
        route_edge_categories = sorted(
            {
                route_edge_category
                for edge_category_map in preset_map.values()
                for route_edge_category in edge_category_map
            }
        )
        for col_idx, preset in enumerate(presets):
            ax = axes[row_idx, col_idx]
            edge_category_map = preset_map.get(preset, {})
            dxs = sorted({dx for dx_map in edge_category_map.values() for dx in dx_map})

            stacked_shares: list[np.ndarray] = []
            for route_edge_category in route_edge_categories:
                dx_map = edge_category_map.get(route_edge_category, {})
                flows = np.array([dx_map.get(dx, 0.0) for dx in dxs], dtype=float)
                totals = np.array(
                    [
                        sum(category_map.get(dx, 0.0) for category_map in edge_category_map.values())
                        for dx in dxs
                    ],
                    dtype=float,
                )
                shares = np.divide(flows, np.where(totals > 0, totals, 1.0), where=np.ones_like(flows, dtype=bool))
                stacked_shares.append(shares)

            if dxs and stacked_shares:
                ax.stackplot(dxs, stacked_shares, labels=route_edge_categories, alpha=0.85)
                if legend_handles is None or legend_labels is None:
                    legend_handles, legend_labels = ax.get_legend_handles_labels()
            else:
                ax.text(0.5, 0.5, "No routed flow", ha="center", va="center", transform=ax.transAxes, fontsize=9)

            ax.set_title(f"{category}\n{preset}")
            ax.set_xlabel("Trade size (dx)")
            if col_idx == 0:
                ax.set_ylabel("Share of output flow")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.2)

    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels, frameon=False, loc="upper center", ncol=min(4, len(legend_labels)))
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
            "pair_metrics_rows_csv": str(output_dir / "pair_metrics_rows.csv"),
            "category_metrics_rows_csv": str(output_dir / "category_metrics_rows.csv"),
            "lowest_marginal_cost_rows_csv": str(output_dir / "lowest_marginal_cost_rows.csv"),
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
            "pair_metrics_rows": analysis_payload["pair_metrics_rows"],
            "category_metrics_rows": analysis_payload["category_metrics_rows"],
            "lowest_marginal_cost_rows": analysis_payload["lowest_marginal_cost_rows"],
            "routing_behavior_rows": analysis_payload["routing_behavior_rows"],
        },
    )
    lowest_marginal_cost_path = plot_lowest_marginal_cost_grid(
        analysis_payload["lowest_marginal_cost_rows"],
        output_dir,
    )
    category_metric_path = plot_category_metric_summary(analysis_payload["category_metrics_rows"], output_dir)
    marginal_price_path = plot_marginal_price_impact_summary(
        analysis_payload["category_metrics_rows"],
        output_dir,
    )
    liquidity_contraction_path = plot_pair_liquidity_contraction_distribution(
        analysis_payload["pair_metrics_rows"],
        output_dir,
    )
    role_block_heatmap_path = plot_role_block_average_network_heatmap(output_dir)
    routing_behavior_path = plot_routing_behavior(analysis_payload["routing_behavior_rows"], output_dir)
    manifest_path = write_run_manifest(
        output_dir,
        analysis_payload,
        [
            lowest_marginal_cost_path,
            category_metric_path,
            marginal_price_path,
            liquidity_contraction_path,
            role_block_heatmap_path,
            routing_behavior_path,
        ],
    )
    return {
        **analysis_payload,
        "raw_output_path": raw_output_path,
        "manifest_path": manifest_path,
        "image_paths": [
            lowest_marginal_cost_path,
            category_metric_path,
            marginal_price_path,
            liquidity_contraction_path,
            role_block_heatmap_path,
            routing_behavior_path,
        ],
    }


def main() -> int:
    run_typology_analysis()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
