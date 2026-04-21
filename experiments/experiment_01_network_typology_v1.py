from __future__ import annotations

import gzip
import json
import logging
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import networkx as nx
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


# ============================================================================
# Calibration-backed synthetic typology experiment
# - topology remains synthetic (SBM)
# - node / edge attributes are sampled from calibration.json
# - node regimes: usd_like / volatile
# - pool types match engine dispatch: univ2 / bal_wgm / curve / univ3_proxy
# ============================================================================

CALIBRATION_PATH = Path("outputs/geckoterminal_calibration_experiment/calibration.json")
CALIBRATION_WEIGHTING = "liquidity_weighted"  # or "count_weighted"
LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def load_calibration(path_str: str) -> dict[str, object]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {path}. "
            "Generate it first with the calibration script."
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_prob_dict(d: dict[str, float]) -> dict[str, float]:
    total = float(sum(float(v) for v in d.values()))
    if total <= 0:
        n = len(d)
        return {str(k): 1.0 / n for k in d}
    return {str(k): float(v) / total for k, v in d.items()}


def _configure_logging_from_env() -> None:
    enabled = os.environ.get("CFMM_EXPERIMENT_LOG", "").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return

    level_name = os.environ.get("CFMM_EXPERIMENT_LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    LOGGER.info(
        "Diagnostic logging enabled (level=%s). Set CFMM_EXPERIMENT_LOG=0 to disable.",
        logging.getLevelName(level),
    )


def _sample_from_prob_dict(rng, probs: dict[str, float]) -> str:
    probs = _normalize_prob_dict(probs)
    keys = list(probs.keys())
    vals = np.array([probs[k] for k in keys], dtype=float)
    vals /= vals.sum()
    return str(rng.choice(keys, p=vals))


def _lookup_prob_table(
    calibration: dict,
    table_name: str,
    key: str,
    weighting: str,
) -> dict[str, float] | None:
    table = calibration.get(table_name, {})
    weighted_table = table.get(weighting, {})
    return weighted_table.get(key)


def _nearest_numeric_key(d: dict, x: float) -> str | None:
    if not d:
        return None
    numeric_keys: list[tuple[float, str]] = []
    for k in d.keys():
        try:
            numeric_keys.append((float(k), str(k)))
        except Exception:
            continue
    if not numeric_keys:
        return None
    return min(numeric_keys, key=lambda kv: abs(kv[0] - x))[1]


def _find_liquidity_bucket(
    calibration: dict,
    role_pair: str,
    ptype: str,
    regime_pair: str,
) -> dict[str, float] | None:
    """
    Fallback order:
      1. exact role_pair | ptype | regime_pair
      2. any role_pair for same ptype + regime_pair
      3. any regime_pair for same ptype
    """
    buckets = calibration.get("liquidity_distributions", {})
    exact = buckets.get(f"{role_pair} | {ptype} | {regime_pair}")
    if exact is not None:
        return exact

    matches = []
    for k, v in buckets.items():
        parts = [x.strip() for x in k.split("|")]
        if len(parts) != 3:
            continue
        rp, pt, gp = parts
        if pt == ptype and gp == regime_pair:
            matches.append(v)
    if matches:
        return max(matches, key=lambda x: float(x.get("count", 0)))

    matches = []
    for k, v in buckets.items():
        parts = [x.strip() for x in k.split("|")]
        if len(parts) != 3:
            continue
        _, pt, _ = parts
        if pt == ptype:
            matches.append(v)
    if matches:
        return max(matches, key=lambda x: float(x.get("count", 0)))

    return None


def _sample_lognormal_from_bucket(
    rng,
    bucket: dict[str, float] | None,
    default_mean: float = 1e6,
    default_sigma: float = 0.7,
) -> float:
    if bucket is None:
        return float(rng.lognormal(mean=np.log(default_mean), sigma=default_sigma))
    mu = float(bucket.get("lognormal_mu", np.log(default_mean)))
    sigma = float(bucket.get("lognormal_sigma", default_sigma))
    sigma = max(1e-8, sigma)
    return float(rng.lognormal(mean=mu, sigma=sigma))


def _sample_fee_from_calibration(
    calibration: dict,
    regime_pair: str,
    ptype: str,
    rng,
    weighting: str,
) -> float:
    key = f"{regime_pair} | {ptype}"
    probs = _lookup_prob_table(calibration, "fee_given_ptype_and_regime_pair", key, weighting)
    if probs:
        sampled = _sample_from_prob_dict(rng, probs)
        return float(sampled)

    # robust fallbacks
    if ptype == "univ2":
        return 0.003
    if ptype == "curve":
        return 0.0004
    if ptype == "univ3_proxy":
        return 0.003
    if ptype == "bal_wgm":
        return 0.0025
    return 0.003


def _get_univ3_proxy_params(calibration: dict, regime_pair: str, fee: float) -> tuple[float, float]:
    priors = calibration.get("metadata", {}).get("univ3_proxy_priors", {})
    regime_priors = priors.get(regime_pair, {})
    if not regime_priors:
        return 0.25, 0.60

    nearest_key = _nearest_numeric_key(regime_priors, fee)
    if nearest_key is None:
        return 0.25, 0.60

    params = regime_priors[nearest_key]
    alpha = float(params.get("alpha", 0.25))
    beta = float(params.get("beta", 0.60))
    return alpha, beta


def _get_curve_k(calibration: dict, regime_pair: str) -> float:
    return float(
        calibration.get("metadata", {})
        .get("curve_k_by_regime_pair", {})
        .get(regime_pair, 0.2)
    )


def _get_bal_weights(calibration: dict) -> tuple[float, float]:
    defaults = calibration.get("metadata", {}).get("bal_wgm_defaults", {})
    return float(defaults.get("w_i", 0.5)), float(defaults.get("w_j", 0.5))


def _initial_role_probs() -> dict[str, float]:
    try:
        calibration = load_calibration(str(CALIBRATION_PATH))
        return _normalize_prob_dict(calibration["role_probs"])
    except Exception:
        # fallback only for import-time convenience
        return {"core": 0.08, "mid": 0.17, "periphery": 0.75}


@dataclass(frozen=True)
class TopologyPreset:
    role_probs: dict[str, float]
    role_connectivity: dict[tuple[str, str], float]
    degree_correction: bool
    pareto_alpha: float


DEFAULT_ROLE_PROBS: dict[str, float] = _initial_role_probs()
DEFAULT_DEGREE_CORRECTION = True
DEFAULT_PARETO_ALPHA = 2.5
DEFAULT_PAIR_SAMPLING_POLICY = PairSamplingPolicy(mode="all")

# keep framework-compatible field name token_type, but values are now value regimes
TOKEN_TYPES: tuple[str, ...] = ("usd_like", "volatile")

DEFAULT_TRADE_SIZE_GRID: tuple[float, ...] = tuple(
    float(dx) for dx in np.geomspace(1000, 140000000.0, num=25)
)
DEFAULT_SEEDS: tuple[int, ...] = tuple(range(3, 9))
MARGINAL_AVG_PRICE_FLOOR = 1e-9
FRONTIER_PAIR_COST_QUANTILE = 0.50
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
    calibration = load_calibration(str(config.fixed_parameters["calibration_path"]))
    weighting = str(config.fixed_parameters.get("calibration_weighting", CALIBRATION_WEIGHTING))

    preset_name = str(config.fixed_parameters["topology_preset"])
    preset = TOPOLOGY_PRESETS[preset_name]

    role_probs = _normalize_prob_dict(config.fixed_parameters["role_probs"])

    role_cfg = RoleSBMConfig(
        n_nodes=30,
        role_probs=role_probs,
        role_connectivity=preset.role_connectivity,
        degree_correction=preset.degree_correction,
        pareto_alpha=preset.pareto_alpha,
        seed=seed,
    )
    topology_model = TopologyModel(role_cfg)

    def token_type_sampler(node, graph, rng):
        role = graph.nodes[node]["role"]
        probs = _lookup_prob_table(calibration, "value_regime_given_role", role, weighting)
        if not probs:
            probs = {"usd_like": 0.5, "volatile": 0.5}
        return _sample_from_prob_dict(rng, probs)

    node_model = NodeAttributeModel(
        {"token_type": NodeAttributeRule("token_type", token_type_sampler)},
        seed=seed + 1,
    )

    def amm_sampler(i, j, graph, rng):
        ti = graph.nodes[i]["token_type"]
        tj = graph.nodes[j]["token_type"]
        regime_pair = "__".join(sorted([ti, tj]))

        probs = _lookup_prob_table(calibration, "ptype_given_regime_pair", regime_pair, weighting)
        if not probs:
            if regime_pair == "usd_like__usd_like":
                probs = {"curve": 0.5, "univ2": 0.25, "univ3_proxy": 0.25}
            else:
                probs = {"univ2": 0.6, "univ3_proxy": 0.4}

        return _sample_from_prob_dict(rng, probs)

    def liquidity_sampler(i, j, graph, rng):
        ti = graph.nodes[i]["token_type"]
        tj = graph.nodes[j]["token_type"]
        ri = graph.nodes[i]["role"]
        rj = graph.nodes[j]["role"]
        ptype = graph.edges[i, j]["amm"]

        regime_pair = "__".join(sorted([ti, tj]))
        role_pair = "__".join(sorted([ri, rj]))

        bucket = _find_liquidity_bucket(calibration, role_pair, ptype, regime_pair)
        return _sample_lognormal_from_bucket(rng, bucket, default_mean=1e6, default_sigma=0.7)

    def fee_sampler(i, j, graph, rng):
        ti = graph.nodes[i]["token_type"]
        tj = graph.nodes[j]["token_type"]
        ptype = graph.edges[i, j]["amm"]
        regime_pair = "__".join(sorted([ti, tj]))
        return _sample_fee_from_calibration(calibration, regime_pair, ptype, rng, weighting)

    def alpha_sampler(i, j, graph, rng):
        if graph.edges[i, j]["amm"] != "univ3_proxy":
            return 0.25
        ti = graph.nodes[i]["token_type"]
        tj = graph.nodes[j]["token_type"]
        regime_pair = "__".join(sorted([ti, tj]))
        fee = float(graph.edges[i, j]["fee"])
        alpha, _ = _get_univ3_proxy_params(calibration, regime_pair, fee)
        return float(alpha)


    def beta_sampler(i, j, graph, rng):
        if graph.edges[i, j]["amm"] != "univ3_proxy":
            return 0.60
        ti = graph.nodes[i]["token_type"]
        tj = graph.nodes[j]["token_type"]
        regime_pair = "__".join(sorted([ti, tj]))
        fee = float(graph.edges[i, j]["fee"])
        _, beta = _get_univ3_proxy_params(calibration, regime_pair, fee)
        return float(beta)


    def k_sampler(i, j, graph, rng):
        if graph.edges[i, j]["amm"] != "curve":
            return 0.20
        ti = graph.nodes[i]["token_type"]
        tj = graph.nodes[j]["token_type"]
        regime_pair = "__".join(sorted([ti, tj]))
        return float(_get_curve_k(calibration, regime_pair))


    def wi_sampler(i, j, graph, rng):
        if graph.edges[i, j]["amm"] != "bal_wgm":
            return 0.5
        w_i, _ = _get_bal_weights(calibration)
        return float(w_i)


    def wj_sampler(i, j, graph, rng):
        if graph.edges[i, j]["amm"] != "bal_wgm":
            return 0.5
        _, w_j = _get_bal_weights(calibration)
        return float(w_j)

    edge_model = EdgeAttributeModel(
        {
            "amm": EdgeAttributeRule("amm", amm_sampler),
            "liquidity": EdgeAttributeRule("liquidity", liquidity_sampler),
            "fee": EdgeAttributeRule("fee", fee_sampler),
            "alpha": EdgeAttributeRule("alpha", alpha_sampler),
            "beta": EdgeAttributeRule("beta", beta_sampler),
            "k": EdgeAttributeRule("k", k_sampler),
            "w_i": EdgeAttributeRule("w_i", wi_sampler),
            "w_j": EdgeAttributeRule("w_j", wj_sampler),
        },
        seed=seed + 2,
    )
    return SBMGenerator(topology_model=topology_model, node_model=node_model, edge_model=edge_model)


def build_experiment_config(
    topology_preset: str,
    *,
    n_nodes: int = 28,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    pair_sampling_policy: PairSamplingPolicy = DEFAULT_PAIR_SAMPLING_POLICY,
    trade_size_grid: tuple[float, ...] = DEFAULT_TRADE_SIZE_GRID,
    calibration_path: Path = CALIBRATION_PATH,
    calibration_weighting: str = CALIBRATION_WEIGHTING,
) -> ExperimentConfig:
    if topology_preset not in TOPOLOGY_PRESETS:
        raise KeyError(f"Unknown topology preset: {topology_preset}")

    preset = TOPOLOGY_PRESETS[topology_preset]
    calibration = load_calibration(str(calibration_path))
    empirical_role_probs = _normalize_prob_dict(calibration["role_probs"])

    return ExperimentConfig(
        varied_parameter=VariedParameter(name="topology_preset", value=topology_preset),
        fixed_parameters={
            "n_nodes": int(n_nodes),
            "topology_preset": topology_preset,
            "degree_correction": preset.degree_correction,
            "pareto_alpha": preset.pareto_alpha,
            "role_probs": empirical_role_probs,
            "calibration_path": str(calibration_path),
            "calibration_weighting": calibration_weighting,
        },
        seeds=tuple(seeds),
        pair_sampling_policy=pair_sampling_policy,
        trade_size_grid=tuple(float(dx) for dx in trade_size_grid),
        category_definitions=build_exchange_route_categories(),
        routing_config=RoutingConfig(
            solver="SCS",
            solver_opts={"max_iters": 5000, "eps": 1e-5, "verbose": False},
            diagnostic_logging=os.environ.get("CFMM_EXPERIMENT_LOG", "").strip().lower() in {"1", "true", "yes", "on"},
        ),
    )


def build_exchange_route_categories(
    token_types: tuple[str, ...] = TOKEN_TYPES,
) -> tuple[CategoryDefinition, ...]:
    return tuple(
        CategoryDefinition(
            name=f"{source_token_type}->{target_token_type}",
            source_token_types=(source_token_type,),
            target_token_types=(target_token_type,),
        )
        for source_token_type in token_types
        for target_token_type in token_types
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


def _shannon_entropy(values: list[float]) -> float:
    total = float(sum(v for v in values if v > 0))
    if total <= 0:
        return float("nan")
    probs = [v / total for v in values if v > 0]
    return float(-sum(p * math.log(p) for p in probs if p > 0))


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
            marginal_avg_price_floored = (
                max(MARGINAL_AVG_PRICE_FLOOR, marginal_avg_price)
                if math.isfinite(marginal_avg_price)
                else float("nan")
            )
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
                    "marginal_avg_price_floored": marginal_avg_price_floored,
                    "marginal_price_impact": marginal_price_impact,
                    "marginal_log_cost": _log_cost(marginal_avg_price),
                    "marginal_cost": _routing_cost(1.0, marginal_avg_price_floored),
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
    *,
    topology_presets: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    trade_size_grid: tuple[float, ...] = DEFAULT_TRADE_SIZE_GRID,
    n_nodes: int = 28,
    pair_sampling_policy: PairSamplingPolicy = DEFAULT_PAIR_SAMPLING_POLICY,
    calibration_path: Path = CALIBRATION_PATH,
    calibration_weighting: str = CALIBRATION_WEIGHTING,
) -> dict[str, object]:
    pair_metrics_rows: list[dict[str, float | int | str]] = []
    category_metrics_rows: list[dict[str, float | int | str]] = []
    lowest_marginal_cost_rows: list[dict[str, float | int | str]] = []
    routing_behavior_rows: list[dict[str, float | int | str]] = []
    routing_path_metrics_rows: list[dict[str, float | int | str]] = []
    raw_results_by_preset: dict[str, dict[str, object]] = {}

    presets_to_run = topology_presets if topology_presets is not None else tuple(TOPOLOGY_PRESETS)
    for topology_preset in presets_to_run:
        config = build_experiment_config(
            topology_preset,
            n_nodes=n_nodes,
            seeds=seeds,
            pair_sampling_policy=pair_sampling_policy,
            trade_size_grid=trade_size_grid,
            calibration_path=calibration_path,
            calibration_weighting=calibration_weighting,
        )
        experiment_result = run_experiment(config, build_generator)
        raw_results_by_preset[topology_preset] = {
            "config": {
                "varied_parameter": {
                    "name": config.varied_parameter.name,
                    "value": config.varied_parameter.value,
                },
                "fixed_parameters": dict(config.fixed_parameters),
                "calibration_path": str(config.fixed_parameters["calibration_path"]),
                "calibration_weighting": str(config.fixed_parameters["calibration_weighting"]),
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
                    baseline_shortest_hops = float("nan")
                    if source_asset in artifacts.graph and target_asset in artifacts.graph:
                        try:
                            baseline_shortest_hops = float(
                                nx.shortest_path_length(
                                    artifacts.graph,
                                    source=int(source_asset),
                                    target=int(target_asset),
                                )
                            )
                        except nx.NetworkXNoPath:
                            baseline_shortest_hops = float("nan")

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
                        active_pool_uids = {
                            pool_uid
                            for pool_uid, value in pool_out.items()
                            if float(value) > 0
                        }
                        routed_graph = nx.Graph()
                        routed_graph.add_nodes_from((int(source_asset), int(target_asset)))
                        for pool in artifacts.market_config.pools:
                            if pool.uid not in active_pool_uids:
                                continue
                            routed_graph.add_edge(int(pool.i), int(pool.j))
                        routed_shortest_hops = float("nan")
                        if routed_graph.has_node(int(source_asset)) and routed_graph.has_node(int(target_asset)):
                            try:
                                routed_shortest_hops = float(
                                    nx.shortest_path_length(
                                        routed_graph,
                                        source=int(source_asset),
                                        target=int(target_asset),
                                    )
                                )
                            except nx.NetworkXNoPath:
                                routed_shortest_hops = float("nan")
                        edge_flow_values = [float(flow) for flow in flow_by_edge_category.values() if float(flow) > 0]
                        route_entropy = _shannon_entropy(edge_flow_values)
                        max_edge_share = (
                            float(max(edge_flow_values) / total_flow)
                            if edge_flow_values and total_flow > 0
                            else float("nan")
                        )
                        excess_hops = (
                            routed_shortest_hops - baseline_shortest_hops
                            if math.isfinite(routed_shortest_hops) and math.isfinite(baseline_shortest_hops)
                            else float("nan")
                        )
                        routing_path_metrics_rows.append(
                            {
                                "topology_preset": topology_preset,
                                "seed": seed,
                                "category": category_name,
                                "dx": float(dx),
                                "source_asset": int(source_asset),
                                "target_asset": int(target_asset),
                                "baseline_shortest_hops": baseline_shortest_hops,
                                "routed_shortest_hops": routed_shortest_hops,
                                "excess_hops": excess_hops,
                                "active_pool_count": len(active_pool_uids),
                                "route_entropy": route_entropy,
                                "max_edge_share": max_edge_share,
                            }
                        )
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

    marginal_cost_values_by_key: dict[tuple[str, int, str, float], list[float]] = {}
    for row in pair_metrics_rows:
        if row.get("metric_kind") != "marginal":
            continue
        marginal_cost = float(row["marginal_cost"])
        if not math.isfinite(marginal_cost) or marginal_cost <= 0:
            continue
        key = (
            str(row["topology_preset"]),
            int(row["seed"]),
            str(row["category"]),
            float(row["dx"]),
        )
        marginal_cost_values_by_key.setdefault(key, []).append(marginal_cost)

    for (topology_preset, seed, category, dx), marginal_costs in sorted(marginal_cost_values_by_key.items()):
        costs = np.array(marginal_costs, dtype=float)
        frontier_cost = float(np.quantile(costs, FRONTIER_PAIR_COST_QUANTILE))
        pair_median_cost = float(np.median(costs))
        lowest_marginal_cost_rows.append(
            {
                "topology_preset": topology_preset,
                "seed": seed,
                "category": category,
                "dx": dx,
                "lowest_marginal_cost": frontier_cost,
                "pair_median_marginal_cost": pair_median_cost,
                "pair_count": int(costs.size),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(str(output_dir / "pair_metrics_rows.csv"), pair_metrics_rows)
    write_csv(str(output_dir / "category_metrics_rows.csv"), category_metrics_rows)
    write_csv(str(output_dir / "lowest_marginal_cost_rows.csv"), lowest_marginal_cost_rows)
    write_csv(str(output_dir / "routing_behavior_rows.csv"), routing_behavior_rows)
    write_csv(str(output_dir / "routing_path_metrics_rows.csv"), routing_path_metrics_rows)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": "experiment_01_network_typology",
        "raw_results_by_preset": raw_results_by_preset,
        "pair_metrics_rows": pair_metrics_rows,
        "category_metrics_rows": category_metrics_rows,
        "lowest_marginal_cost_rows": lowest_marginal_cost_rows,
        "routing_behavior_rows": routing_behavior_rows,
        "routing_path_metrics_rows": routing_path_metrics_rows,
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
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, max(4, 3.8 * nrows)), sharex=False, squeeze=False)

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
            center_curve: list[float] = []
            low_curve: list[float] = []
            high_curve: list[float] = []
            filtered_xs: list[float] = []
            for dx in xs:
                vals = np.array(
                    [float(v) for v in dx_map[dx] if math.isfinite(float(v)) and float(v) > 0],
                    dtype=float,
                )
                if vals.size == 0:
                    continue
                filtered_xs.append(float(dx))
                center_curve.append(float(np.median(vals)))
                low_curve.append(float(np.percentile(vals, 10)))
                high_curve.append(float(np.percentile(vals, 90)))
            if not filtered_xs:
                continue
            ax.plot(filtered_xs, center_curve, marker="o", linewidth=2, label=preset)
            ax.fill_between(filtered_xs, low_curve, high_curve, alpha=0.15)
        ax.set_title(f"Robust frontier marginal cost: {category}")
        ax.set_xlabel("Trade size midpoint")
        ax.set_ylabel("Frontier marginal cost (lower is better)")
        ax.set_xscale("log")
        ax.set_yscale("log")
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

        has_positive_x = False
        for preset, dx_map in sorted(preset_dx_map.items()):
            xs = sorted(dx_map)
            medians = [float(np.median(dx_map[dx])) for dx in xs]
            p10 = [float(np.percentile(dx_map[dx], 10)) for dx in xs]
            p90 = [float(np.percentile(dx_map[dx], 90)) for dx in xs]

            filtered = [
                (x, med, low, high)
                for x, med, low, high in zip(xs, medians, p10, p90)
                if x > 0 and all(math.isfinite(v) for v in (x, med, low, high))
            ]
            if not filtered:
                continue
            has_positive_x = True
            xvals = [row[0] for row in filtered]
            medvals = [row[1] for row in filtered]
            p10vals = [row[2] for row in filtered]
            p90vals = [row[3] for row in filtered]
            ax.plot(xvals, medvals, marker="o", linewidth=2, label=f"{preset} median")
            ax.fill_between(xvals, p10vals, p90vals, alpha=0.18)

        ax.set_title(f"Liquidity contraction distribution: {category}")
        ax.set_xlabel("Trade size (dx)")
        ax.set_ylabel("Price deterioration vs smallest trade")
        ax.set_ylim(0.0, 1.0)
        if has_positive_x:
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


def _format_edge_category_label(route_edge_category: str) -> str:
    if route_edge_category == "unknown":
        return route_edge_category
    if "__" not in route_edge_category:
        return route_edge_category
    left, right = route_edge_category.split("__", maxsplit=1)
    return f"{left}<->{right}"


def _all_edge_categories() -> tuple[str, ...]:
    categories: list[str] = []
    for i, left in enumerate(TOKEN_TYPES):
        for right in TOKEN_TYPES[i:]:
            categories.append("__".join(sorted((left, right))))
    return tuple(categories)


def plot_routing_behavior(routing_behavior_rows: list[dict[str, float | int | str]], output_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    by_category: dict[str, dict[str, dict[str, dict[float, float]]]] = {}
    observed_edge_categories: set[str] = set()
    for row in routing_behavior_rows:
        category = str(row["category"])
        preset = str(row["topology_preset"])
        route_edge_category = str(row["route_edge_category"])
        dx = float(row["dx"])
        observed_edge_categories.add(route_edge_category)
        by_category.setdefault(category, {}).setdefault(preset, {}).setdefault(route_edge_category, {}).setdefault(dx, 0.0)
        by_category[category][preset][route_edge_category][dx] += float(row["flow"])

    default_edge_categories = _all_edge_categories()
    route_edge_categories = sorted(set(default_edge_categories) | observed_edge_categories)
    color_lookup = {
        edge_category: color
        for edge_category, color in zip(
            route_edge_categories,
            plt.cm.tab20(np.linspace(0.0, 1.0, len(route_edge_categories), endpoint=False)),
        )
    }

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

    for row_idx, category in enumerate(categories):
        preset_map = by_category[category]
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
                ax.stackplot(
                    dxs,
                    stacked_shares,
                    labels=[_format_edge_category_label(edge_category) for edge_category in route_edge_categories],
                    colors=[color_lookup[edge_category] for edge_category in route_edge_categories],
                    alpha=0.85,
                )
            else:
                ax.text(0.5, 0.5, "No routed flow", ha="center", va="center", transform=ax.transAxes, fontsize=9)

            ax.set_title(f"{category}\n{preset}")
            ax.set_xlabel("Trade size (dx)")
            if col_idx == 0:
                ax.set_ylabel("Share of output flow")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.2)

    handles = [
        Patch(
            color=color_lookup[edge_category],
            label=_format_edge_category_label(edge_category),
        )
        for edge_category in route_edge_categories
    ]
    if handles:
        fig.legend(handles=handles, frameon=False, loc="upper center", ncol=min(5, len(handles)))
    fig.tight_layout()
    out_path = output_dir / "routing_behaviour_summary.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_routing_path_metrics_summary(
    routing_path_metrics_rows: list[dict[str, float | int | str]],
    output_dir: Path,
) -> Path:
    import matplotlib.pyplot as plt

    metrics = {
        "excess_hops": "Excess hops (routed shortest - baseline shortest)",
        "route_entropy": "Route entropy over active edge-category flow",
    }
    categories = sorted({str(row["category"]) for row in routing_path_metrics_rows})
    fig, axes = plt.subplots(
        nrows=max(1, len(categories)),
        ncols=2,
        figsize=(14, max(4, 3.5 * max(1, len(categories)))),
        squeeze=False,
        sharex=False,
    )

    for row_idx, category in enumerate(categories):
        category_rows = [row for row in routing_path_metrics_rows if str(row["category"]) == category]
        for col_idx, (metric_name, ylabel) in enumerate(metrics.items()):
            ax = axes[row_idx, col_idx]
            preset_map: dict[str, dict[float, list[float]]] = {}
            for row in category_rows:
                value = float(row[metric_name])
                if not math.isfinite(value):
                    continue
                preset = str(row["topology_preset"])
                dx = float(row["dx"])
                preset_map.setdefault(preset, {}).setdefault(dx, []).append(value)

            for preset, dx_map in sorted(preset_map.items()):
                xs = sorted(dx_map)
                ys = np.array([float(np.mean(dx_map[dx])) for dx in xs], dtype=float)
                std = np.array([float(np.std(dx_map[dx], ddof=0)) for dx in xs], dtype=float)
                ax.plot(xs, ys, marker="o", linewidth=2, label=preset)
                ax.fill_between(xs, ys - std, ys + std, alpha=0.15)

            ax.set_title(f"{category}\n{ylabel}")
            ax.set_xlabel("Trade size (dx)")
            ax.set_ylabel(ylabel)
            ax.set_xscale("log")
            ax.grid(True, alpha=0.25)

    for ax in axes.flatten()[len(categories) * 2:]:
        ax.axis("off")
    if categories:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, frameon=False, loc="upper center", ncol=min(4, len(labels)))
    fig.tight_layout()
    out_path = output_dir / "routing_path_metrics_summary.png"
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
            "routing_path_metrics_rows_csv": str(output_dir / "routing_path_metrics_rows.csv"),
            "images": [str(path) for path in image_paths],
        },
    }
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest_path


def run_typology_analysis(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    calibration_path: Path = CALIBRATION_PATH,
    calibration_weighting: str = CALIBRATION_WEIGHTING,
) -> dict[str, object]:
    analysis_payload = collect_typology_outputs(
        output_dir,
        calibration_path=calibration_path,
        calibration_weighting=calibration_weighting,
    )
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
            "routing_path_metrics_rows": analysis_payload["routing_path_metrics_rows"],
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
    role_block_heatmap_path = plot_role_block_average_network_heatmap(output_dir)
    routing_behavior_path = plot_routing_behavior(analysis_payload["routing_behavior_rows"], output_dir)
    routing_path_metrics_path = plot_routing_path_metrics_summary(
        analysis_payload["routing_path_metrics_rows"],
        output_dir,
    )
    manifest_path = write_run_manifest(
        output_dir,
        analysis_payload,
        [
            lowest_marginal_cost_path,
            category_metric_path,
            marginal_price_path,
            role_block_heatmap_path,
            routing_behavior_path,
            routing_path_metrics_path,
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
            role_block_heatmap_path,
            routing_behavior_path,
            routing_path_metrics_path,
        ],
    }


def main() -> int:
    _configure_logging_from_env()
    run_typology_analysis()

    # Smoke-test example:
    # small_output_dir = Path("outputs/experiment_01_network_typology_smoke")
    # analysis_payload = collect_typology_outputs(
    #     small_output_dir,
    #     topology_presets=("balanced",),
    #     seeds=(3,),
    #     trade_size_grid=(1.0, 10.0, 100.0),
    #     n_nodes=14,
    #     pair_sampling_policy=PairSamplingPolicy(mode="sample", max_pairs_per_category=24),
    #     calibration_path=CALIBRATION_PATH,
    #     calibration_weighting=CALIBRATION_WEIGHTING,
    # )
    # plot_lowest_marginal_cost_grid(analysis_payload["lowest_marginal_cost_rows"], small_output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
