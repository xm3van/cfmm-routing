from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def write_csv(path: str, rows: list[dict]) -> None:
    _ensure_dir(path)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def safe_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def safe_div(a: float, b: float) -> float:
    a = float(a)
    b = float(b)
    return a / b if b != 0.0 else float("nan")


@dataclass(frozen=True)
class RunMeta:
    """
    Generic metadata you want attached to EVERY row, across experiments.

    Keep it small. Add fields only if they truly generalize.
    """
    experiment: str              # e.g. "rq1_2_path_length"
    run: str                     # e.g. "hop_len=3" or "zipf_s=1.2"
    seed: int = 0
    solver: str = ""
    fee: float = float("nan")
    L_total: float = float("nan")
    n_routes: int = 0
    hop_len: int = 0
    symmetry_eps: float = float("nan")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "experiment": self.experiment,
            "run": self.run,
            "seed": self.seed,
            "solver": self.solver,
            "fee": self.fee,
            "L_total": self.L_total,
            "n_routes": self.n_routes,
            "hop_len": self.hop_len,
            "symmetry_eps": self.symmetry_eps,
        }


def build_curve_rows(
    meta: RunMeta,
    dxs: list[float],
    dys: list[float],
    extra: Optional[Dict[str, Any]] = None,
) -> list[dict]:
    """
    Core simulation output: one row per dx in the sweep.
    Stores only what is universal: dx, dy, avg_price.

    Everything else (marginal PI, slippage variants) can be derived later.
    """
    extra = extra or {}
    rows: list[dict] = []
    for dx, dy in zip(dxs, dys):
        dx = safe_float(dx)
        dy = safe_float(dy)
        rows.append(
            {
                **meta.as_dict(),
                **extra,
                "dx": dx,
                "dy": dy,
                "avg_price": safe_div(dy, dx),
            }
        )
    return rows


def build_flow_rows(
    meta: RunMeta,
    dxs: list[float],
    comp: list[dict[str, float]],
    *,
    aggregate_fn,
    aggregate_mode: str = "bottleneck",
    extra: Optional[Dict[str, Any]] = None,
) -> list[dict]:
    """
    Optional routing detail: one row per (dx, key), where key is typically route id.

    aggregate_fn must be: aggregate_fn(pool_out_dict, mode=str) -> dict[key -> flow]
    Example: route_aggregate(pool_out, mode="bottleneck") -> {"DIRECT": ..., "R0": ...}

    Stored columns:
      - key (route/pool/etc)
      - flow
      - share (flow / total)
    """
    extra = extra or {}
    rows: list[dict] = []

    for dx, pool_out in zip(dxs, comp):
        dx = safe_float(dx)
        agg = aggregate_fn(pool_out, mode=aggregate_mode)

        total = float(sum(max(0.0, safe_float(v)) for v in agg.values()))
        total = total if total > 0 else 1.0

        for key, v in agg.items():
            v = safe_float(v)
            if not (v > 0):
                continue
            rows.append(
                {
                    **meta.as_dict(),
                    **extra,
                    "dx": dx,
                    "key": str(key),
                    "flow": v,
                    "share": v / total,
                    "agg_mode": aggregate_mode,
                }
            )

    return rows
