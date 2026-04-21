from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Literal, Optional, Tuple
import json
import hashlib

PoolType = Literal["univ2", "bal_wgm", "curve"]  # extend later
EdgeCategory = Tuple[str, str]


def normalize_edge_category(token_type_i: str, token_type_j: str) -> EdgeCategory:
    """Return a canonical symmetric edge category."""
    pair = (str(token_type_i), str(token_type_j))
    return tuple(sorted(pair))  # type: ignore[return-value]


@dataclass(frozen=True)
class PoolSpec:
    uid: str
    ptype: PoolType
    i: int
    j: int
    liquidity: float
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PoolMetadata:
    token_type_i: Optional[str] = None
    token_type_j: Optional[str] = None
    role_i: Optional[str] = None
    role_j: Optional[str] = None
    edge_category: Optional[EdgeCategory] = None


@dataclass(frozen=True)
class MarketMetadata:
    asset_token_types: Dict[int, str] = field(default_factory=dict)
    pool_metadata: Dict[str, PoolMetadata] = field(default_factory=dict)

    def edge_category_by_pool_uid(self) -> Dict[str, EdgeCategory]:
        return {
            pool_uid: meta.edge_category
            for pool_uid, meta in self.pool_metadata.items()
            if meta.edge_category is not None
        }

    def token_type_by_asset_id(self) -> Dict[int, str]:
        return dict(self.asset_token_types)


@dataclass(frozen=True)
class MarketConfig:
    n_assets: int
    pools: Tuple[PoolSpec, ...]
    metadata: MarketMetadata = field(default_factory=MarketMetadata)

    def edge_category_by_pool_uid(self) -> Dict[str, EdgeCategory]:
        return self.metadata.edge_category_by_pool_uid()

    def token_type_by_asset_id(self) -> Dict[int, str]:
        return self.metadata.token_type_by_asset_id()


@dataclass(frozen=True)
class RoutingConfig:
    objective: Literal["max_out"] = "max_out"
    solver: Literal["ECOS", "OSQP", "SCS", "CLARABEL"] = "ECOS"
    solver_opts: Dict[str, Any] = field(default_factory=dict)
    diagnostic_logging: bool = False


@dataclass(frozen=True)
class SweepConfig:
    in_asset: int
    out_asset: int
    dx_grid: Tuple[float, ...]


@dataclass(frozen=True)
class HarnessConfig:
    seed: int = 0
    output_dir: str = "outputs"
    atol: float = 1e-8


def as_dict(x: Any) -> Dict[str, Any]:
    if hasattr(x, "__dataclass_fields__"):
        return asdict(x)
    raise TypeError(f"Not a dataclass: {type(x)}")


def stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]
