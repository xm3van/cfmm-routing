from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Literal, Tuple
import json
import hashlib

PoolType = Literal["univ2", "bal_wgm", "curve"]  # extend later


@dataclass(frozen=True)
class PoolSpec:
    uid: str
    ptype: PoolType
    i: int
    j: int
    liquidity: float
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketConfig:
    n_assets: int
    pools: Tuple[PoolSpec, ...]


@dataclass(frozen=True)
class RoutingConfig:
    objective: Literal["max_out"] = "max_out"
    solver: Literal["ECOS", "OSQP", "SCS"] = "ECOS"
    solver_opts: Dict[str, Any] = field(default_factory=dict)


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
