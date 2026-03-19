from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np

from cfmm_routing.config import MarketConfig, RoutingConfig, SweepConfig
from cfmm_routing.harness import run_sweep
from cfmm_routing.sbm import SBMGenerator, build_market_config_from_graph


@dataclass(frozen=True)
class VariedParameter:
    name: str
    value: Any


@dataclass(frozen=True)
class PairSamplingPolicy:
    mode: str = "all"
    max_pairs_per_category: int | None = None
    seed_offset: int = 0
    sort_pairs: bool = True


@dataclass(frozen=True)
class CategoryDefinition:
    name: str
    source_token_types: tuple[str, ...]
    target_token_types: tuple[str, ...]
    allow_same_node: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    varied_parameter: VariedParameter
    fixed_parameters: Mapping[str, Any]
    seeds: tuple[int, ...]
    pair_sampling_policy: PairSamplingPolicy
    trade_size_grid: tuple[float, ...]
    category_definitions: tuple[CategoryDefinition, ...]
    routing_config: RoutingConfig


@dataclass(frozen=True)
class GraphExperimentArtifacts:
    graph: nx.Graph
    market_config: MarketConfig
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class PairRun:
    category_name: str
    source_asset: int
    target_asset: int
    dxs: tuple[float, ...]
    dys: tuple[float, ...]


def _set_seed_on_object(obj: Any, seed: int) -> None:
    if obj is None:
        return
    if hasattr(obj, "cfg") and getattr(obj, "cfg") is not None and hasattr(obj.cfg, "seed"):
        obj.cfg = replace(obj.cfg, seed=seed)
        return
    if hasattr(obj, "seed"):
        setattr(obj, "seed", seed)


def seed_generator(generator: SBMGenerator, seed: int) -> SBMGenerator:
    """Return the same generator instance with deterministic per-component seeds applied."""
    _set_seed_on_object(generator.topology_model, seed)
    _set_seed_on_object(generator.node_model, seed + 1)
    _set_seed_on_object(generator.edge_model, seed + 2)
    return generator


def generate_graph_artifacts(generator: SBMGenerator, seed: int) -> GraphExperimentArtifacts:
    seeded_generator = seed_generator(generator, seed)
    graph = seeded_generator.generate()
    market_config = build_market_config_from_graph(graph)
    metadata = graph_metadata_rows(graph, seed)
    return GraphExperimentArtifacts(graph=graph, market_config=market_config, metadata=metadata)


def graph_metadata_rows(graph: nx.Graph, seed: int) -> Dict[str, Any]:
    token_type_counts: dict[str, int] = defaultdict(int)
    role_counts: dict[str, int] = defaultdict(int)
    for _, attrs in graph.nodes(data=True):
        token_type_counts[str(attrs.get("token_type", "unknown"))] += 1
        role_counts[str(attrs.get("role", "unknown"))] += 1

    component_sizes = sorted((len(component) for component in nx.connected_components(graph)), reverse=True)
    metadata: dict[str, Any] = {
        "seed": seed,
        "n_nodes": int(graph.number_of_nodes()),
        "n_edges": int(graph.number_of_edges()),
        "n_components": len(component_sizes),
        "largest_component_size": int(component_sizes[0]) if component_sizes else 0,
    }
    for token_type, count in sorted(token_type_counts.items()):
        metadata[f"token_type_count__{token_type}"] = int(count)
    for role, count in sorted(role_counts.items()):
        metadata[f"role_count__{role}"] = int(count)
    return metadata


def node_rows(
    graph: nx.Graph,
    seed: int,
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    row_metadata = dict(metadata or {})
    for node, attrs in graph.nodes(data=True):
        rows.append({"seed": seed, "node": int(node), **row_metadata, **dict(attrs)})
    return rows


def edge_rows(
    graph: nx.Graph,
    seed: int,
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    row_metadata = dict(metadata or {})
    for source, target, attrs in graph.edges(data=True):
        row = {"seed": seed, "source": int(source), "target": int(target), **row_metadata}
        row.update(dict(attrs))
        rows.append(row)
    return rows


def build_token_type_index(graph: nx.Graph) -> dict[str, list[int]]:
    token_nodes: dict[str, list[int]] = defaultdict(list)
    for node, token_type in nx.get_node_attributes(graph, "token_type").items():
        token_nodes[str(token_type)].append(int(node))
    for nodes in token_nodes.values():
        nodes.sort()
    return dict(token_nodes)


def get_pairs(
    token_nodes: Mapping[str, Sequence[int]],
    source_token_type: str,
    target_token_type: str,
    *,
    allow_same_node: bool = False,
) -> list[tuple[int, int]]:
    nodes1 = list(token_nodes.get(source_token_type, ()))
    nodes2 = list(token_nodes.get(target_token_type, ()))

    pairs: set[tuple[int, int]] = set()
    if source_token_type == target_token_type:
        start = 0 if allow_same_node else 1
        for i, source in enumerate(nodes1):
            for target in nodes1[i + start:]:
                if source == target and not allow_same_node:
                    continue
                pairs.add((int(source), int(target)))
    else:
        for source in nodes1:
            for target in nodes2:
                if source == target and not allow_same_node:
                    continue
                pairs.add((int(source), int(target)))

    return sorted(pairs)


def component_map(graph: nx.Graph) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for component_id, component in enumerate(nx.connected_components(graph)):
        for node in component:
            mapping[int(node)] = component_id
    return mapping


def filter_reachable_pairs(pairs: Iterable[tuple[int, int]], comp_map: Mapping[int, int]) -> list[tuple[int, int]]:
    return [
        (source, target)
        for source, target in pairs
        if source in comp_map and target in comp_map and comp_map[source] == comp_map[target]
    ]


def enumerate_pairs_by_category(
    graph: nx.Graph,
    category_definitions: Sequence[CategoryDefinition],
) -> dict[str, list[tuple[int, int]]]:
    token_nodes = build_token_type_index(graph)
    comp_map = component_map(graph)
    category_pairs: dict[str, list[tuple[int, int]]] = {}

    for category in category_definitions:
        pairs: set[tuple[int, int]] = set()
        for source_token_type in category.source_token_types:
            for target_token_type in category.target_token_types:
                pairs.update(
                    get_pairs(
                        token_nodes,
                        source_token_type,
                        target_token_type,
                        allow_same_node=category.allow_same_node,
                    )
                )
        category_pairs[category.name] = filter_reachable_pairs(sorted(pairs), comp_map)

    return category_pairs


def sample_pairs(
    pairs: Sequence[tuple[int, int]],
    policy: PairSamplingPolicy,
    *,
    seed: int,
) -> list[tuple[int, int]]:
    ordered_pairs = sorted(pairs) if policy.sort_pairs else list(pairs)
    if policy.mode == "all" or policy.max_pairs_per_category is None:
        return ordered_pairs
    if policy.mode != "sample":
        raise ValueError(f"Unsupported pair sampling mode: {policy.mode}")
    if len(ordered_pairs) <= policy.max_pairs_per_category:
        return ordered_pairs

    rng = np.random.default_rng(seed + policy.seed_offset)
    indices = sorted(rng.choice(len(ordered_pairs), size=policy.max_pairs_per_category, replace=False).tolist())
    return [ordered_pairs[index] for index in indices]


def run_category_sweeps(
    *,
    market_config: MarketConfig,
    routing_config: RoutingConfig,
    trade_size_grid: Sequence[float],
    category_name: str,
    pairs: Sequence[tuple[int, int]],
) -> list[PairRun]:
    pair_runs: list[PairRun] = []
    for source_asset, target_asset in pairs:
        sweep = SweepConfig(
            in_asset=int(source_asset),
            out_asset=int(target_asset),
            dx_grid=tuple(float(dx) for dx in trade_size_grid),
        )
        result = run_sweep(
            market_cfg=market_config,
            routing_cfg=routing_config,
            sweep_cfg=sweep,
        )
        pair_runs.append(
            PairRun(
                category_name=category_name,
                source_asset=int(source_asset),
                target_asset=int(target_asset),
                dxs=tuple(float(dx) for dx in result["dxs"]),
                dys=tuple(float(dy) for dy in result["dys"]),
            )
        )
    return pair_runs


def pair_curve_rows(pair_runs: Sequence[PairRun], *, seed: int, metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_run in pair_runs:
        for dx, dy in zip(pair_run.dxs, pair_run.dys):
            rows.append(
                {
                    "seed": seed,
                    "category": pair_run.category_name,
                    "source_asset": pair_run.source_asset,
                    "target_asset": pair_run.target_asset,
                    "dx": float(dx),
                    "dy": float(dy),
                    "avg_price": float(dy) / float(dx) if dx else float("nan"),
                    **metadata,
                }
            )
    return rows


def eligible_pair_rows(
    category_pairs: Mapping[str, Sequence[tuple[int, int]]],
    *,
    seed: int,
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category_name, pairs in sorted(category_pairs.items()):
        for source_asset, target_asset in pairs:
            rows.append(
                {
                    "seed": seed,
                    "category": category_name,
                    "source_asset": int(source_asset),
                    "target_asset": int(target_asset),
                    **metadata,
                }
            )
    return rows


def aggregate_curves_within_graph(pair_runs: Sequence[PairRun], *, seed: int, metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, float], list[float]] = defaultdict(list)
    pair_counts: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for pair_run in pair_runs:
        pair_counts[pair_run.category_name].add((pair_run.source_asset, pair_run.target_asset))
        for dx, dy in zip(pair_run.dxs, pair_run.dys):
            buckets[(pair_run.category_name, float(dx))].append(float(dy))

    rows: list[dict[str, Any]] = []
    for (category_name, dx), dys in sorted(buckets.items()):
        dys_array = np.asarray(dys, dtype=float)
        rows.append(
            {
                "seed": seed,
                "category": category_name,
                "dx": dx,
                "pair_count": len(pair_counts[category_name]),
                "dy_mean": float(dys_array.mean()),
                "dy_std": float(dys_array.std(ddof=0)),
                "dy_min": float(dys_array.min()),
                "dy_max": float(dys_array.max()),
                "avg_price_mean": float((dys_array / dx).mean()) if dx else float("nan"),
                **metadata,
            }
        )
    return rows


def aggregate_curves_across_graphs(graph_curve_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, float], list[float]] = defaultdict(list)
    pair_counts: dict[tuple[str, float], list[int]] = defaultdict(list)
    seeds: dict[tuple[str, float], set[int]] = defaultdict(set)
    metadata_by_key: dict[tuple[str, float], dict[str, Any]] = {}
    excluded_fields = {
        "seed",
        "category",
        "dx",
        "pair_count",
        "dy_mean",
        "dy_std",
        "dy_min",
        "dy_max",
        "avg_price_mean",
    }

    for row in graph_curve_rows:
        key = (str(row["category"]), float(row["dx"]))
        buckets[key].append(float(row["dy_mean"]))
        pair_counts[key].append(int(row.get("pair_count", 0)))
        seeds[key].add(int(row["seed"]))
        metadata_by_key.setdefault(key, {k: v for k, v in row.items() if k not in excluded_fields})

    aggregated_rows: list[dict[str, Any]] = []
    for (category_name, dx), values in sorted(buckets.items()):
        values_array = np.asarray(values, dtype=float)
        pairs_array = np.asarray(pair_counts[(category_name, dx)], dtype=float)
        aggregated_rows.append(
            {
                "category": category_name,
                "dx": dx,
                "graph_count": len(seeds[(category_name, dx)]),
                "dy_mean_across_graphs": float(values_array.mean()),
                "dy_std_across_graphs": float(values_array.std(ddof=0)),
                "pair_count_mean": float(pairs_array.mean()) if len(pairs_array) else 0.0,
                **metadata_by_key[(category_name, dx)],
            }
        )
    return aggregated_rows


@dataclass(frozen=True)
class ExperimentResult:
    graph_rows: list[dict[str, Any]]
    eligible_pair_rows: list[dict[str, Any]]
    pair_curve_rows: list[dict[str, Any]]
    graph_curve_rows: list[dict[str, Any]]
    aggregate_curve_rows: list[dict[str, Any]]
    node_rows: list[dict[str, Any]] = field(default_factory=list)
    edge_rows: list[dict[str, Any]] = field(default_factory=list)


GeneratorFactory = Callable[[ExperimentConfig, int], SBMGenerator]


def run_experiment(config: ExperimentConfig, generator_factory: GeneratorFactory) -> ExperimentResult:
    graph_rows: list[dict[str, Any]] = []
    pair_selection_rows: list[dict[str, Any]] = []
    pair_curve_output_rows: list[dict[str, Any]] = []
    graph_curve_rows: list[dict[str, Any]] = []
    node_metadata_rows: list[dict[str, Any]] = []
    edge_metadata_rows: list[dict[str, Any]] = []

    for seed in config.seeds:
        generator = generator_factory(config, seed)
        artifacts = generate_graph_artifacts(generator, seed)
        graph = artifacts.graph

        experiment_metadata = {
            "varied_parameter_name": config.varied_parameter.name,
            "varied_parameter_value": config.varied_parameter.value,
            **dict(config.fixed_parameters),
        }
        graph_metadata = {
            **experiment_metadata,
            **artifacts.metadata,
        }
        graph_rows.append(graph_metadata)
        node_metadata_rows.extend(node_rows(graph, seed, metadata=experiment_metadata))
        edge_metadata_rows.extend(edge_rows(graph, seed, metadata=experiment_metadata))

        category_pairs = enumerate_pairs_by_category(graph, config.category_definitions)
        pair_metadata = dict(experiment_metadata)
        pair_selection_rows.extend(
            eligible_pair_rows(category_pairs, seed=seed, metadata=pair_metadata)
        )
        pair_runs: list[PairRun] = []
        for category_name, eligible_pairs in sorted(category_pairs.items()):
            sampled_pairs = sample_pairs(
                eligible_pairs,
                config.pair_sampling_policy,
                seed=seed,
            )
            pair_runs.extend(
                run_category_sweeps(
                    market_config=artifacts.market_config,
                    routing_config=config.routing_config,
                    trade_size_grid=config.trade_size_grid,
                    category_name=category_name,
                    pairs=sampled_pairs,
                )
            )

        graph_curve_rows.extend(aggregate_curves_within_graph(pair_runs, seed=seed, metadata=pair_metadata))
        pair_curve_output_rows.extend(pair_curve_rows(pair_runs, seed=seed, metadata=pair_metadata))

    aggregate_curve_rows = aggregate_curves_across_graphs(graph_curve_rows)
    return ExperimentResult(
        graph_rows=graph_rows,
        eligible_pair_rows=pair_selection_rows,
        pair_curve_rows=pair_curve_output_rows,
        graph_curve_rows=graph_curve_rows,
        aggregate_curve_rows=aggregate_curve_rows,
        node_rows=node_metadata_rows,
        edge_rows=edge_metadata_rows,
    )
