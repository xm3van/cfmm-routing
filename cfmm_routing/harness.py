from __future__ import annotations

from cfmm_routing.config import MarketConfig, RoutingConfig, SweepConfig
from .market import build_market
from .routing import solve_max_out_sweep

def run_sweep(
    *,
    market_cfg: MarketConfig,
    routing_cfg: RoutingConfig,
    sweep_cfg: SweepConfig,
) -> dict:
    mkt = build_market(market_cfg)

    dys = []
    comp = []
    flow_results = solve_max_out_sweep(
        mkt,
        sweep_cfg.in_asset,
        sweep_cfg.out_asset,
        [float(dx) for dx in sweep_cfg.dx_grid],
        routing_cfg,
    )
    for dx, fr in zip(sweep_cfg.dx_grid, flow_results):
        if fr.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Solve failed at dx={dx}: {fr.status} {fr.solver_info}")
        dys.append(fr.dy_total)
        comp.append(fr.pool_out_to_sink)

    return {
        "dxs": list(sweep_cfg.dx_grid),
        "dys": dys,
        "composition": comp,  # list of dict uid->out_to_sink
    }
