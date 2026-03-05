# sbm.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, Mapping, Optional, Tuple, Union

import numpy as np
import networkx as nx

Role = str
TokenType = str
AMMType = str

# A key describing an (ordered) (role_i, role_j, token_i, token_j) condition.
AttrKey = Tuple[Role, Role, TokenType, TokenType]


# -----------------------------
# Utilities
# -----------------------------

def _normalize_prob_dict(d: Mapping[Hashable, float]) -> Tuple[list, np.ndarray]:
    """Return (keys, probs) with probs normalized; raises if sum<=0."""
    keys = list(d.keys())
    probs = np.asarray([float(d[k]) for k in keys], dtype=float)
    probs = np.clip(probs, 0.0, np.inf)
    s = float(probs.sum())
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError("Probability dict must have positive finite total mass.")
    return keys, probs / s


def _sym_key(key: AttrKey) -> AttrKey:
    """Swap (i,j) side of the key."""
    ri, rj, ti, tj = key
    return (rj, ri, tj, ti)


def _get_amm_dist(
    amm_probs: Mapping[AttrKey, Mapping[AMMType, float]],
    key: AttrKey,
    default_amm: AMMType = "univ2",
) -> Mapping[AMMType, float]:
    """Fetch AMM distribution for key with symmetric fallback, else default."""
    dist = amm_probs.get(key)
    if dist is None:
        dist = amm_probs.get(_sym_key(key))
    if dist is None:
        return {default_amm: 1.0}
    return dist


def _call_sampler(
    sampler: Callable[..., float],
    *,
    rng: np.random.Generator,
    ri: Role,
    rj: Role,
    ti: TokenType,
    tj: TokenType,
    amm: AMMType,
) -> float:
    """
    Calls sampler with best-effort signature compatibility:
      - preferred: sampler(rng, ri, rj, ti, tj, amm)
      - fallback: sampler(ri, rj, ti, tj, amm)
    """
    try:
        return float(sampler(rng, ri, rj, ti, tj, amm))
    except TypeError:
        return float(sampler(ri, rj, ti, tj, amm))


# -----------------------------
# Orchestrator
# -----------------------------

class SBMGenerator:
    """
    Pipeline orchestrator:
      topology_model.generate() -> G
      node_model.assign(G)      -> G
      edge_model.assign(G)      -> G
    """

    def __init__(self, topology_model: Any, node_model: Any = None, edge_model: Any = None):
        self.topology_model = topology_model
        self.node_model = node_model
        self.edge_model = edge_model

    def generate(self) -> nx.Graph:
        G = self.topology_model.generate()
        if self.node_model is not None:
            G = self.node_model.assign(G)
        if self.edge_model is not None:
            G = self.edge_model.assign(G)
        return G


# -----------------------------
# 1) Topology (Role-SBM, optional degree correction)
# -----------------------------

@dataclass(frozen=True)
class RoleSBMConfig:
    n_nodes: int
    role_probs: Dict[Role, float]
    role_connectivity: Dict[Tuple[Role, Role], float]
    degree_correction: bool = True
    # Pareto tail index (larger => less heavy-tailed). Used only if degree_correction=True.
    pareto_alpha: float = 2.5
    seed: int = 42


class TopologyModel:
    """
    Generates an undirected simple graph with node attribute 'role' and
    a core-periphery-style SBM on roles.

    Edge probability: p_ij = clip( P[role_i, role_j] * omega_i * omega_j, 0, 1 )
    where omega_i are heavy-tailed node activity weights if degree_correction=True.
    """

    def __init__(self, cfg: RoleSBMConfig):
        if cfg.n_nodes <= 0:
            raise ValueError("n_nodes must be positive.")
        if cfg.pareto_alpha <= 0:
            raise ValueError("pareto_alpha must be positive.")
        self.cfg = cfg

        # Validate role probs now
        _normalize_prob_dict(cfg.role_probs)

    def generate(self) -> nx.Graph:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)

        roles, role_pvals = _normalize_prob_dict(cfg.role_probs)

        G = nx.Graph()
        G.add_nodes_from(range(cfg.n_nodes))

        # Assign roles
        role_assignments = rng.choice(roles, size=cfg.n_nodes, p=role_pvals)
        nx.set_node_attributes(G, {i: str(role_assignments[i]) for i in range(cfg.n_nodes)}, "role")

        # Degree correction weights
        if cfg.degree_correction:
            # Pareto(a) + 1 => support [1, inf). Then normalize to mean 1.
            omega = rng.pareto(cfg.pareto_alpha, cfg.n_nodes) + 1.0
            omega = omega / omega.mean()
        else:
            omega = np.ones(cfg.n_nodes, dtype=float)

        # Add edges
        for i in range(cfg.n_nodes):
            ri = G.nodes[i]["role"]
            for j in range(i + 1, cfg.n_nodes):
                rj = G.nodes[j]["role"]

                base_p = cfg.role_connectivity.get((ri, rj))
                if base_p is None:
                    base_p = cfg.role_connectivity.get((rj, ri), 0.0)

                p = float(base_p) * float(omega[i]) * float(omega[j])
                if p <= 0.0:
                    continue
                if p >= 1.0 or rng.random() < p:
                    G.add_edge(i, j)

        return G


# -----------------------------
# 2) Node attributes (token_type conditional on role)
# -----------------------------
NodeSampler = Callable[[int, nx.Graph, np.random.Generator], Any]


@dataclass
class NodeAttributeRule:
    name: str
    sampler: NodeSampler


class NodeAttributeModel:

    def __init__(self, rules: Dict[str, NodeAttributeRule], seed: int = 0):
        self.rules = rules
        self.seed = seed

    def assign(self, G: nx.Graph) -> nx.Graph:
        rng = np.random.default_rng(self.seed)

        for node in G.nodes():
            for rule in self.rules.values():
                value = rule.sampler(node, G, rng)
                G.nodes[node][rule.name] = value

        return G


# -----------------------------
# 3) Edge attributes (AMM + liquidity)
# -----------------------------
EdgeSampler = Callable[[int, int, nx.Graph, np.random.Generator], Any]


@dataclass
class EdgeAttributeRule:
    name: str
    sampler: EdgeSampler


class EdgeAttributeModel:

    def __init__(self, rules: Dict[str, EdgeAttributeRule], seed: int = 0):
        self.rules = rules
        self.seed = seed

    def assign(self, G: nx.Graph) -> nx.Graph:
        rng = np.random.default_rng(self.seed)

        for i, j in G.edges():
            for rule in self.rules.values():
                value = rule.sampler(i, j, G, rng)
                G.edges[i, j][rule.name] = value
                # if value is not None:
                #     G.edges[i, j][rule.name] = value

        return G
    

from cfmm_routing.config import MarketConfig, PoolSpec

# def build_market_config_from_sbm(
#     sbm_generator: SBMGenerator,
# ) -> MarketConfig:
#     """
#     Generate SBM graph and convert to MarketConfig.
#     Assumes edges have attributes:
#         - 'amm'
#         - 'liquidity'
#         - optionally 'fee'
#         - optionally AMM-specific params
#     """

#     G = sbm_generator.generate()

#     pools = []

#     for idx, (i, j, data) in enumerate(G.edges(data=True)):

#         # Required
#         ptype = data.get("amm")
#         liquidity = float(data.get("liquidity"))

#         # Optional params
#         params = {}

#         # Fee
#         if "fee" in data:
#             params["fee"] = float(data["fee"])

#         # Curve-style param
#         # if "A" in data:
#         #     params["A"] = float(data["A"])
#         A = data.get("A")
#         if A is not None:
#             params["A"] = int(A)

#         # Balancer-style params
#         if "w_i" in data:
#             params["w_i"] = float(data["w_i"])
#         if "w_j" in data:
#             params["w_j"] = float(data["w_j"])

#         uid = f"{ptype}-{idx}:{i}-{j}"

#         pools.append(
#             PoolSpec(
#                 uid=uid,
#                 ptype=ptype,
#                 i=int(i),
#                 j=int(j),
#                 liquidity=liquidity,
#                 params=params,
#             )
#         )

#     return MarketConfig(
#         n_assets=G.number_of_nodes(),
#         pools=tuple(pools),
#     )

def build_market_config_from_graph(
    G: nx.Graph,
) -> MarketConfig:
    """
    Generate SBM graph and convert to MarketConfig.
    Assumes edges have attributes:
        - 'amm'
        - 'liquidity'
        - optionally 'fee'
        - optionally AMM-specific params
    """

    pools = []

    for idx, (i, j, data) in enumerate(G.edges(data=True)):

        # Required
        ptype = data.get("amm")
        liquidity = float(data.get("liquidity"))

        # Optional params
        params = {}

        # Fee
        if "fee" in data:
            params["fee"] = float(data["fee"])

        # Curve-style param
        # if "A" in data:
        #     params["A"] = float(data["A"])
        A = data.get("A")
        if A is not None:
            params["A"] = int(A)

        # Balancer-style params
        if "w_i" in data:
            params["w_i"] = float(data["w_i"])
        if "w_j" in data:
            params["w_j"] = float(data["w_j"])

        uid = f"{ptype}-{idx}:{i}-{j}"

        pools.append(
            PoolSpec(
                uid=uid,
                ptype=ptype,
                i=int(i),
                j=int(j),
                liquidity=liquidity,
                params=params,
            )
        )

    return MarketConfig(
        n_assets=G.number_of_nodes(),
        pools=tuple(pools),
    )