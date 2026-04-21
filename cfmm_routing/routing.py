from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Dict, List, Tuple

import numpy as np
import cvxpy as cp

from cfmm_routing.config import RoutingConfig, PoolSpec

_LOGGER = logging.getLogger(__name__)

@dataclass
class Market:
    n_assets: int
    pools: List[PoolSpec]
    # adjacency: (i,j) -> list of pool indices
    adj: Dict[Tuple[int, int], List[int]]


_CONIC_FALLBACK_SOLVERS: Tuple[str, ...] = ("CLARABEL", "SCS", "ECOS")


@dataclass
class FlowResult:
    status: str
    dy_total: float
    # uid -> total amount_in to pool (sum of deltas across local assets)
    pool_in: Dict[str, float]
    # uid -> net amount of out_asset contributed (lambda_out - delta_out, floored at 0)
    pool_out_to_sink: Dict[str, float]
    solver_info: Dict[str, object]


@dataclass
class _CompiledMaxOutProblem:
    mkt: Market
    in_asset: int
    out_asset: int
    flow_scale: float
    dx_param: cp.Parameter
    psi: cp.Expression
    deltas: List[cp.Variable]
    lambdas: List[cp.Variable]
    local_assets_list: List[List[int]]
    prob: cp.Problem


def _solver_opts_for(solver: str, solver_opts: Dict[str, object]) -> Dict[str, object]:
    """Normalize generic solver options to backend-specific CVXPY kwargs."""
    opts: Dict[str, object] = dict(solver_opts)

    if solver == "CLARABEL":
        # Clarabel expects singular `max_iter` and different tolerance names.
        if "max_iters" in opts and "max_iter" not in opts:
            opts["max_iter"] = opts.pop("max_iters")
        if "eps" in opts:
            eps = opts.pop("eps")
            if "tol_gap_abs" not in opts:
                opts["tol_gap_abs"] = eps
            if "tol_gap_rel" not in opts:
                opts["tol_gap_rel"] = eps
            if "tol_feas" not in opts:
                opts["tol_feas"] = eps
        return opts

    if solver == "ECOS":
        # ECOS does not accept a generic `eps` kwarg; map to its tolerances.
        if "eps" in opts:
            eps = opts.pop("eps")
            if "abstol" not in opts:
                opts["abstol"] = eps
            if "reltol" not in opts:
                opts["reltol"] = eps
            if "feastol" not in opts:
                opts["feastol"] = eps
        return opts

    return opts


def _pool_local_assets(p: PoolSpec) -> List[int]:
    # current codebase: all pools are 2-asset (i, j)
    return [p.i, p.j]


def _pool_reserve_param(p: PoolSpec, names: Tuple[str, ...]) -> float | None:
    for name in names:
        value = p.params.get(name)
        if value is not None:
            return float(value)
    return None


def _pool_reserves(p: PoolSpec, local_assets: List[int]) -> np.ndarray:
    reserve_i = _pool_reserve_param(p, ("reserve_i", "x_reserve", f"reserve_{p.i}"))
    reserve_j = _pool_reserve_param(p, ("reserve_j", "y_reserve", f"reserve_{p.j}"))
    if reserve_i is not None and reserve_j is not None:
        reserve_by_asset = {p.i: reserve_i, p.j: reserve_j}
        return np.array([reserve_by_asset[idx] for idx in local_assets], dtype=float)

    if p.ptype == "bal_wgm":
        wi = max(float(p.params.get("w_i", 0.5)), 1e-6)
        wj = max(float(p.params.get("w_j", 0.5)), 1e-6)
        total_w = wi + wj
        reserve_by_asset = {
            p.i: float(p.liquidity) * wi / total_w,
            p.j: float(p.liquidity) * wj / total_w,
        }
        return np.array([reserve_by_asset[idx] for idx in local_assets], dtype=float)
    
    if p.ptype == "univ3_proxy":
        alpha = min(max(float(p.params.get("alpha", 0.25)), 1e-6), 1.0 - 1e-6)
        reserve_by_asset = {
            p.i: float(p.liquidity) * alpha,
            p.j: float(p.liquidity) * (1.0 - alpha),
        }
        return np.array([reserve_by_asset[idx] for idx in local_assets], dtype=float)
    
    reserve_ratio = _pool_reserve_param(p, ("reserve_ratio", "price_scale", "peg_offset"))
    if reserve_ratio is not None and reserve_ratio > 0:
        ratio = float(reserve_ratio)
        base = float(p.liquidity)
        reserve_by_asset = {
            p.i: base * ratio / (1.0 + ratio),
            p.j: base / (1.0 + ratio),
        }
        return np.array([reserve_by_asset[idx] for idx in local_assets], dtype=float)

    # Fallback: use symmetric reserves only when the upstream data contains no
    # asset-specific reserve or weighting information.
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
    - curve: calibrated stable-swap proxy using a constant-sum envelope plus a
      geometric-mean floor tied to amplification A to penalize reserve imbalance.
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
        # Stable-swap is not represented exactly here because the exact invariant
        # is not DCP-friendly. Instead we use a calibrated convex proxy:
        #   1) preserve total amplified depth via a constant-sum envelope
        #   2) preserve a floor on the geometric mean to prevent unrealistic
        #      one-sided depletion that the pure constant-sum proxy allowed
        # The geometric-mean floor scales with A and asymptotes toward 1.0 as A
        # increases, which keeps high-A pools flatter near the peg while still
        # introducing meaningful curvature away from balance.
        A = max(float(p.params.get("A", 100.0)), 1.0)
        gm_floor = min(0.995, A / (A + 50.0))
        reference_gm = float(np.sqrt(np.prod(R)))
        cons.append(cp.sum(new_R) >= float(np.sum(R)))
        cons.append(cp.geo_mean(new_R) >= gm_floor * reference_gm)

    elif p.ptype == "univ3_proxy":
        alpha = float(p.params.get("alpha", 0.25))
        beta = float(p.params.get("beta", 0.60))

        # Clamp to safe range
        alpha = min(max(alpha, 1e-6), 1.0 - 1e-6)
        beta = min(max(beta, 1e-6), 1.0 - 1e-6)

        # Concentration intensity:
        # beta > alpha => more of the local slope comes from a smaller active band.
        conc = max(0.0, beta - alpha)

        # Reduced-form DCP-safe proxy:
        # - constant-sum envelope gives flatter near-spot behavior
        # - gm floor prevents unrealistic one-sided depletion
        # - stronger concentration => more relaxed gm floor
        #
        # beta == alpha collapses toward univ2-like behavior.
        if conc <= 1e-8:
            cons.append(cp.geo_mean(new_R) >= cp.geo_mean(R))
        else:
            reference_gm = float(np.sqrt(np.prod(R)))

            # Typical range roughly 0.90 .. 1.00
            gm_floor = max(0.90, min(0.995, 1.0 - 0.18 * conc))

            cons.append(cp.sum(new_R) >= float(np.sum(R)))
            cons.append(cp.geo_mean(new_R) >= gm_floor * reference_gm)

    else:
        raise ValueError(f"Unknown pool type: {p.ptype}")

    return cons


def _build_compiled_max_out_problem(mkt: Market, in_asset: int, out_asset: int) -> _CompiledMaxOutProblem:
    n = int(mkt.n_assets)
    pools = mkt.pools

    if len(pools) == 0:
        raise ValueError("Cannot compile max-out problem with empty pool set")

    # Homogeneous scaling improves conditioning for conic solvers when
    # reserves/endowment are very large (common in on-chain units). Since dx is
    # a runtime parameter, scaling here depends only on market reserves.
    reserve_scale_candidates = [1.0]
    for p in pools:
        reserve_scale_candidates.extend(abs(float(r)) for r in _pool_reserves(p, _pool_local_assets(p)))
    flow_scale = max(1.0, max(reserve_scale_candidates, default=1.0))

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

        reserves_list.append(_pool_reserves(p, local_assets) / flow_scale)
        gamma_list.append(_pool_gamma(p))

        deltas.append(cp.Variable(len(local_assets), nonneg=True))
        lambdas.append(cp.Variable(len(local_assets), nonneg=True))

    psi = 0
    for A_k, D, L in zip(A_list, deltas, lambdas):
        psi = psi + A_k @ (L - D)

    obj = cp.Maximize(psi[out_asset])
    cons: List[cp.Constraint] = []

    # paper endowment constraint
    dx_param = cp.Parameter(nonneg=True, value=0.0)
    w = np.zeros(n, dtype=float)
    w[in_asset] = 1.0
    cons.append(psi + dx_param * w >= 0)

    # pool constraints
    for p, R, gamma_k, D, L in zip(pools, reserves_list, gamma_list, deltas, lambdas):
        new_R = R + gamma_k * D - L
        cons.extend(_pool_invariant_constraint(p, R, new_R))

    prob = cp.Problem(obj, cons)
    return _CompiledMaxOutProblem(
        mkt=mkt,
        in_asset=int(in_asset),
        out_asset=int(out_asset),
        flow_scale=float(flow_scale),
        dx_param=dx_param,
        psi=psi,
        deltas=deltas,
        lambdas=lambdas,
        local_assets_list=local_assets_list,
        prob=prob,
    )


def _solve_compiled_max_out_problem(
    compiled: _CompiledMaxOutProblem,
    dx_total: float,
    rcfg: RoutingConfig,
) -> FlowResult:
    pools = compiled.mkt.pools
    if len(pools) == 0:
        return FlowResult(
            status="infeasible",
            dy_total=0.0,
            pool_in={},
            pool_out_to_sink={},
            solver_info={"reason": "no pools"},
        )

    if dx_total < 0:
        return FlowResult(
            status="error",
            dy_total=0.0,
            pool_in={},
            pool_out_to_sink={},
            solver_info={"reason": "dx_total must be non-negative"},
        )

    compiled.dx_param.value = float(dx_total) / compiled.flow_scale

    solve_errors: List[Dict[str, str]] = []
    selected_solver = rcfg.solver

    installed = set(cp.installed_solvers())
    candidate_solvers: List[str] = [selected_solver]

    # This model uses conic atoms (geo_mean), so OSQP is incompatible.
    # Also, user environments often miss one or more optional solver packages.
    # We therefore try other conic solvers deterministically as a fallback.
    for s in _CONIC_FALLBACK_SOLVERS:
        if s not in candidate_solvers:
            candidate_solvers.append(s)

    # Keep ordering stable but skip clearly unavailable backends.
    solvers_to_try = [s for s in candidate_solvers if s in installed]
    if not solvers_to_try:
        solvers_to_try = candidate_solvers

    status = "error"
    for solver in solvers_to_try:
        selected_solver = solver
        solver_opts = _solver_opts_for(solver, rcfg.solver_opts)
        started_at = time.perf_counter()
        if rcfg.diagnostic_logging:
            _LOGGER.info(
                "Routing solve attempt: dx=%s solver=%s opts=%s",
                dx_total,
                solver,
                solver_opts,
            )
        try:
            compiled.prob.solve(solver=solver, warm_start=True, **solver_opts)
        except Exception as e:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            solve_errors.append({"solver": solver, "exception": repr(e)})
            if rcfg.diagnostic_logging:
                _LOGGER.exception(
                    "Routing solver failed: dx=%s solver=%s elapsed_ms=%.2f",
                    dx_total,
                    solver,
                    elapsed_ms,
                )
            continue

        status = str(compiled.prob.status)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        if rcfg.diagnostic_logging:
            _LOGGER.info(
                "Routing solver completed: dx=%s solver=%s status=%s elapsed_ms=%.2f objective=%s",
                dx_total,
                solver,
                status,
                elapsed_ms,
                compiled.prob.value,
            )
        if compiled.prob.value is not None and status in ("optimal", "optimal_inaccurate"):
            break

        solve_errors.append({
            "solver": solver,
            "status": status,
            "reason": "non_optimal_status",
        })
    else:
        return FlowResult(
            status=status,
            dy_total=0.0,
            pool_in={},
            pool_out_to_sink={},
            solver_info={
                "cvxpy_status": status,
                "solver": selected_solver,
                "attempted_solvers": solvers_to_try,
                "exceptions": solve_errors,
            },
        )

    if compiled.prob.value is None or status not in ("optimal", "optimal_inaccurate"):
        spent = (
            float(-compiled.psi.value[compiled.in_asset] * compiled.flow_scale)
            if compiled.psi.value is not None
            else 0.0
        )
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
    spent = (
        float(-compiled.psi.value[compiled.in_asset] * compiled.flow_scale)
        if compiled.psi.value is not None
        else 0.0
    )
    received = (
        float(compiled.psi.value[compiled.out_asset] * compiled.flow_scale)
        if compiled.psi.value is not None
        else 0.0
    )

    pool_in: Dict[str, float] = {}
    pool_out_to_sink: Dict[str, float] = {}

    for p, local_assets, D, L in zip(
        pools,
        compiled.local_assets_list,
        compiled.deltas,
        compiled.lambdas,
    ):
        uid = p.uid
        Dv = np.array(D.value).astype(float) if D.value is not None else np.zeros(len(local_assets))
        Lv = np.array(L.value).astype(float) if L.value is not None else np.zeros(len(local_assets))

        pool_in[uid] = float(np.sum(Dv) * compiled.flow_scale)

        contrib = 0.0
        if compiled.out_asset in local_assets:
            t = local_assets.index(compiled.out_asset)
            contrib = float((Lv[t] - Dv[t]) * compiled.flow_scale)
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


def solve_max_out(mkt: Market, in_asset: int, out_asset: int, dx_total: float, rcfg: RoutingConfig) -> FlowResult:
    if len(mkt.pools) == 0:
        return FlowResult(
            status="infeasible",
            dy_total=0.0,
            pool_in={},
            pool_out_to_sink={},
            solver_info={"reason": "no pools"},
        )
    compiled = _build_compiled_max_out_problem(mkt, in_asset, out_asset)
    return _solve_compiled_max_out_problem(compiled, dx_total, rcfg)


def solve_max_out_sweep(
    mkt: Market,
    in_asset: int,
    out_asset: int,
    dx_values: List[float],
    rcfg: RoutingConfig,
) -> List[FlowResult]:
    if len(mkt.pools) == 0:
        return [
            FlowResult(
                status="infeasible",
                dy_total=0.0,
                pool_in={},
                pool_out_to_sink={},
                solver_info={"reason": "no pools"},
            )
            for _ in dx_values
        ]
    compiled = _build_compiled_max_out_problem(mkt, in_asset, out_asset)
    return [_solve_compiled_max_out_problem(compiled, float(dx), rcfg) for dx in dx_values]
