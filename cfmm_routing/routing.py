from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import cvxpy as cp

from cfmm_routing.config import RoutingConfig, PoolSpec
from cfmm_routing.market import Market


_CONIC_FALLBACK_SOLVERS: Tuple[str, ...] = ("CLARABEL", "SCS", "ECOS")


def _ordered_solvers(primary: str) -> List[str]:
    ordered = [primary]
    for s in _CONIC_FALLBACK_SOLVERS:
        if s not in ordered:
            ordered.append(s)
    return ordered


def _solver_opts_for(solver: str, solver_opts: Dict[str, object]) -> Dict[str, object]:
    # Keep options portable across fallback solvers.
    opts = dict(solver_opts)

    if solver == "SCS":
        return opts

    # SCS-only options should not be forwarded to other solvers.
    for key in ("eps", "acceleration_lookback", "scale", "normalize", "rho_x"):
        opts.pop(key, None)

    # Clarabel expects max_iter (not max_iters).
    if solver == "CLARABEL":
        if "max_iters" in opts and "max_iter" not in opts:
            opts["max_iter"] = opts.pop("max_iters")
        opts.setdefault("max_iter", 1_000)

    return opts


@dataclass
class FlowResult:
    status: str
    dy_total: float
    # uid -> total amount_in to pool (sum of deltas across local assets)
    pool_in: Dict[str, float]
    # uid -> net amount of out_asset contributed (lambda_out - delta_out, floored at 0)
    pool_out_to_sink: Dict[str, float]
    solver_info: Dict[str, object]


def _pool_local_assets(p: PoolSpec) -> List[int]:
    # current codebase: all pools are 2-asset (i, j)
    return [p.i, p.j]


def _pool_reserves(p: PoolSpec, local_assets: List[int]) -> np.ndarray:
    # current modeling assumption: "liquidity" is symmetric reserves per asset
    # (matches your E0 config: liquidity=1e6 -> reserves [1e6, 1e6])
    return np.array([float(p.liquidity) for _ in local_assets], dtype=float)


def _pool_gamma(p: PoolSpec) -> float:
    # repo uses multiplicative gamma (e.g., 0.997). Here fee is fraction (e.g., 0.003).
    fee = float(p.params.get("fee", 0.0))
    return 1.0 - fee


def _pool_invariant_constraint(p: PoolSpec, R: np.ndarray, new_R: cp.Expression):
    """
    Convex-friendly pool feasibility constraints, close to the original repo.
    - univ2: constant product via geo_mean(new_R) >= geo_mean(R)
    - bal_wgm: weighted geometric mean via geo_mean(new_R, p=w) >= geo_mean(R, p=w)
    - curve: use constant-sum proxy (transparent; replace later with exact stable-swap if desired)
    """
    cons = [new_R >= 0]

    if p.ptype == "univ2":
        cons.append(cp.geo_mean(new_R) >= cp.geo_mean(R))

    elif p.ptype == "bal_wgm":
        wi = float(p.params.get("w_i", 0.5))
        wj = float(p.params.get("w_j", 0.5))
        w = np.array([wi, wj], dtype=float)
        cons.append(cp.geo_mean(new_R, p=w) >= cp.geo_mean(R, p=w))

    elif p.ptype == "curve":
        # transparent proxy consistent with convex modeling:
        # constant-sum feasibility (no free removal of total reserves)
        cons.append(cp.sum(new_R) >= float(np.sum(R)))

    else:
        raise ValueError(f"Unknown pool type: {p.ptype}")

    return cons


def solve_max_out(mkt: Market, in_asset: int, out_asset: int, dx_total: float, rcfg: RoutingConfig) -> FlowResult:
    n = int(mkt.n_assets)
    pools = mkt.pools

    if len(pools) == 0:
        return FlowResult(
            status="infeasible",
            dy_total=0.0,
            pool_in={},
            pool_out_to_sink={},
            solver_info={"reason": "no pools"},
        )

    local_assets_list: List[List[int]] = []
    A_list: List[np.ndarray] = []
    reserves_list: List[np.ndarray] = []
    gamma_list: List[float] = []
    deltas: List[cp.Variable] = []
    lambdas: List[cp.Variable] = []

    for p in pools:
        local_assets = _pool_local_assets(p)
        local_assets_list.append(local_assets)

        A_k = np.zeros((n, len(local_assets)), dtype=float)
        for t, idx in enumerate(local_assets):
            A_k[idx, t] = 1.0
        A_list.append(A_k)

        reserves_list.append(_pool_reserves(p, local_assets))
        gamma_list.append(_pool_gamma(p))

        deltas.append(cp.Variable(len(local_assets), nonneg=True))
        lambdas.append(cp.Variable(len(local_assets), nonneg=True))

    psi = 0
    for A_k, D, L in zip(A_list, deltas, lambdas):
        psi = psi + A_k @ (L - D)

    obj = cp.Maximize(psi[out_asset])
    cons: List[cp.Constraint] = []

    # paper endowment constraint
    w = np.zeros(n, dtype=float)
    w[in_asset] = float(dx_total)
    cons.append(psi + w >= 0)

    # pool constraints
    for p, R, gamma_k, D, L in zip(pools, reserves_list, gamma_list, deltas, lambdas):
        new_R = R + gamma_k * D - L
        cons.extend(_pool_invariant_constraint(p, R, new_R))

    prob = cp.Problem(obj, cons)

    solve_errors: List[Dict[str, str]] = []
    selected_solver = rcfg.solver
    solvers_to_try: List[str] = _ordered_solvers(selected_solver)

    status = "error"
    for solver in solvers_to_try:
        selected_solver = solver
        solver_opts = _solver_opts_for(solver, rcfg.solver_opts)
        try:
            prob.solve(solver=solver, **solver_opts)
        except Exception as e:
            solve_errors.append({"solver": solver, "exception": repr(e)})
            continue

        status = str(prob.status)
        if status in ("optimal", "optimal_inaccurate"):
            break

    if status == "error":
        return FlowResult(
            status="error",
            dy_total=0.0,
            pool_in={},
            pool_out_to_sink={},
            solver_info={
                "cvxpy_status": "error",
                "solver": selected_solver,
                "attempted_solvers": solvers_to_try,
                "exceptions": solve_errors,
            },
        )

    if prob.value is None or status not in ("optimal", "optimal_inaccurate"):
        spent = float(-psi.value[in_asset]) if psi.value is not None else 0.0
        return FlowResult(
            status=status,
            dy_total=0.0,
            pool_in={},
            pool_out_to_sink={},
            solver_info={
                "cvxpy_status": status,
                "solver": selected_solver,
                "attempted_solvers": solvers_to_try,
                "spent": spent,
                "exceptions": solve_errors,
            },
        )

    # compute spent/received AFTER solve
    spent = float(-psi.value[in_asset]) if psi.value is not None else 0.0
    received = float(psi.value[out_asset]) if psi.value is not None else 0.0

    pool_in: Dict[str, float] = {}
    pool_out_to_sink: Dict[str, float] = {}

    for p, local_assets, D, L in zip(pools, local_assets_list, deltas, lambdas):
        uid = p.uid
        Dv = np.array(D.value).astype(float) if D.value is not None else np.zeros(len(local_assets))
        Lv = np.array(L.value).astype(float) if L.value is not None else np.zeros(len(local_assets))

        pool_in[uid] = float(np.sum(Dv))

        contrib = 0.0
        if out_asset in local_assets:
            t = local_assets.index(out_asset)
            contrib = float(Lv[t] - Dv[t])
            if contrib < 0:
                contrib = 0.0
        pool_out_to_sink[uid] = contrib

    return FlowResult(
        status=status,
        dy_total=received,  # this is psi[out_asset]
        pool_in=pool_in,
        pool_out_to_sink=pool_out_to_sink,
        solver_info={
            "cvxpy_status": status,
            "solver": selected_solver,
            "attempted_solvers": solvers_to_try,
            "spent": spent,
            "received": received,
            "exceptions": solve_errors,
        },
    )
