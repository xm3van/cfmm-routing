from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple
import os

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from cfmm_routing.metrics import marginal_price_impact


# ============================================================
# Helpers
# ============================================================

def _parse_route(uid: str) -> str:
    """
    Expected uid formats:
      - univ2-DIRECT:0-1
      - univ2-R{t}-H{e}:{a}-{b}
    Returns route key: "DIRECT" or "R{t}".
    """
    if "DIRECT" in uid:
        return "DIRECT"

    try:
        left = uid.split(":")[0]     # "univ2-R3-H1"
        parts = left.split("-")      # ["univ2", "R3", "H1"]
        r = next(p for p in parts if p.startswith("R"))
        return r
    except Exception:
        # fallback: treat each uid as its own route
        return uid


def route_aggregate(pool_out: Dict[str, float], mode: str = "bottleneck") -> Dict[str, float]:
    """
    Aggregate pool-level contributions into route-level contributions.

    For serial k-hop routes, edges should carry equal flow; we use:
      - bottleneck: min(edge_flows)  [recommended]
      - mean: mean(edge_flows)
      - sum: sum(edge_flows)         [debug only; not serial-safe]
    """
    mode = mode.lower().strip()
    by_route: Dict[str, List[float]] = defaultdict(list)

    for uid, v in pool_out.items():
        v = float(v)
        if v <= 0:
            continue
        by_route[_parse_route(uid)].append(v)

    out: Dict[str, float] = {}
    for r, flows in by_route.items():
        if not flows:
            out[r] = 0.0
        elif mode == "bottleneck":
            out[r] = float(min(flows))
        elif mode == "mean":
            out[r] = float(sum(flows) / len(flows))
        elif mode == "sum":
            out[r] = float(sum(flows))
        else:
            raise ValueError(f"Unknown mode={mode!r}")
    return out


def _route_sort_key(r: str):
    if r == "DIRECT":
        return (-1, -1, r)
    if r.startswith("R"):
        try:
            return (0, int(r[1:]), r)
        except Exception:
            return (0, 10**9, r)
    return (1, 10**9, r)


# ============================================================
# Plot: Marginal price impact overlay (full + zoom)
# ============================================================
def plot_marginal_price_impact_overlay(
    runs: dict[str, tuple[list[float], list[float]]],
    title: str,
    out_path: str,
    L_ref: float | None = None,
    zoom_x: tuple[float, float] | None = (0.0, 0.05),
    zoom_y: tuple[float, float] | None = (0.0, 0.1),
):
    """
    Produces a 1- or 2-row grid:
      - top: full x-range
      - bottom: zoomed x-range (if zoom_x is not None)

    If L_ref is provided, x-axis uses x = dx_mid / L_ref, so zoom_x is a
    fraction of liquidity (e.g. (0.05, 0.10) = 5–10% of liquidity).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from cfmm_routing.metrics import marginal_price_impact

    has_zoom = zoom_x is not None
    nrows = 2 if has_zoom else 1

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(8, 7.0 if has_zoom else 4.5),
        sharex=False,
        sharey=False
    )

    # normalize axes to flat list
    if isinstance(axes, np.ndarray):
        axes = axes.ravel().tolist()
    else:
        axes = [axes]

    ax_full = axes[0]
    ax_zoom = axes[1] if has_zoom else None

    for label, (dxs, dys) in runs.items():
        x_mid, pi = marginal_price_impact(dxs, dys)

        if L_ref:
            x_mid = [float(x) / float(L_ref) for x in x_mid]
        else:
            x_mid = [float(x) for x in x_mid]

        pi = [float(p) for p in pi]

        ax_full.plot(x_mid, pi, label=label, lw=2)
        if has_zoom and ax_zoom is not None:
            ax_zoom.plot(x_mid, pi, label=label, lw=2)

    xlabel = "trade fraction (dx / liquidity)" if L_ref else "trade size"

    ax_full.set_xlabel(xlabel)
    ax_full.set_ylabel("marginal price impact (1 − dY/dX)")
    ax_full.set_ylim(bottom=0)
    ax_full.grid(True, alpha=0.3)
    ax_full.legend(frameon=False)

    if has_zoom and ax_zoom is not None:
        x0, x1 = zoom_x
        ax_zoom.set_xlim(float(x0), float(x1))
        ax_zoom.set_xlabel(xlabel)
        ax_zoom.set_ylabel("marginal PI (zoom)")
        ax_zoom.set_ylim(bottom=0)
        if zoom_y is not None:
            ax_zoom.set_ylim(float(zoom_y[0]), float(zoom_y[1]))
        ax_zoom.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)



# ============================================================
# Plot: Route share heatmap grid (faceted by run label)
# ============================================================

def plot_route_share_heatmap_grid(
    runs: Dict[str, Tuple[List[float], List[Dict[str, float]]]],
    title: str,
    out_path: str,
    agg_mode: str = "bottleneck",
    top_n: int = 12,
    include_other: bool = True,
    vmax: float | None = None,
    cmap: str = "viridis",
):
    if not runs:
        return

    # Aggregate to routes + compute global totals to choose top_n consistently
    runs_route: Dict[str, Tuple[List[float], List[Dict[str, float]]]] = {}
    global_totals: Dict[str, float] = {}

    for lbl, (dxs, comp) in runs.items():
        comp_route: List[Dict[str, float]] = []
        for d in comp:
            rd = route_aggregate(d, mode=agg_mode)
            comp_route.append(rd)
            for r, v in rd.items():
                global_totals[r] = global_totals.get(r, 0.0) + float(v)
        runs_route[lbl] = (dxs, comp_route)

    if not global_totals:
        return

    keep = [r for r, _ in sorted(global_totals.items(), key=lambda kv: kv[1], reverse=True)[:top_n]]
    keep_sorted = sorted(keep, key=_route_sort_key)
    keep_set = set(keep_sorted)

    labels = list(runs_route.keys())
    n_panels = len(labels)
    n_rows = len(keep_sorted) + (1 if include_other else 0)

    # Build matrices + determine a consistent vmax
    mats: List[np.ndarray] = []
    panel_dxs: List[List[float]] = []
    global_max = 0.0

    for lbl in labels:
        dxs, comp_route = runs_route[lbl]
        m = np.zeros((n_rows, len(dxs)), dtype=float)

        for t, rd in enumerate(comp_route):
            tot = float(sum(rd.values())) or 1.0
            # kept routes
            for i, r in enumerate(keep_sorted):
                m[i, t] = float(rd.get(r, 0.0)) / tot
            # other
            if include_other:
                other = 0.0
                for r, v in rd.items():
                    if r not in keep_set:
                        other += float(v) / tot
                m[len(keep_sorted), t] = other

        mats.append(m)
        panel_dxs.append(dxs)
        global_max = max(global_max, float(m.max()))

    vmax_use = float(vmax) if vmax is not None else (global_max if global_max > 0 else 1.0)

    # Plot
    fig_h = max(2.2 * n_panels, 3.2)
    fig, axes = plt.subplots(nrows=n_panels, ncols=1, figsize=(10.0, fig_h), sharex=False)
    if n_panels == 1:
        axes = [axes]

    im0 = None
    for ax, lbl, m, dxs in zip(axes, labels, mats, panel_dxs):
        im0 = ax.imshow(m, aspect="auto", interpolation="nearest", vmin=0.0, vmax=vmax_use, cmap=cmap)

        yticklabels = keep_sorted + (["OTHER"] if include_other else [])
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(yticklabels)

        # sparse x ticks
        if len(dxs) <= 10:
            xt = list(range(len(dxs)))
        else:
            xt = np.unique(np.round(np.linspace(0, len(dxs) - 1, 8)).astype(int)).tolist()
        ax.set_xticks(xt)
        ax.set_xticklabels([f"{dxs[i]:g}" for i in xt])

        ax.set_xlabel("trade size")
        ax.set_ylabel(lbl)

    fig.suptitle(f"{title} (mode={agg_mode})")

    if im0 is not None:
        # keep layout simple: reserve space on the right for the colorbar
        cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
        cbar = fig.colorbar(im0, cax=cax)
        cbar.set_label("route share")

    fig.subplots_adjust(left=0.12, right=0.88, top=0.90, bottom=0.10, hspace=0.25)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ============================================================
# Plot: Sankey snapshot (expects already-aggregated route flows)
# ============================================================

def plot_route_choice_sankey(
    route_flow: Dict[str, float],
    title: str,
    out_html: str,
    min_share: float = 0.01,
) -> None:
    os.makedirs(os.path.dirname(out_html), exist_ok=True)

    total = float(sum(max(0.0, v) for v in route_flow.values()))
    if total <= 0:
        raise ValueError("No flow to plot (total <= 0).")

    items = [(r, float(v)) for r, v in route_flow.items() if float(v) > 0]
    items.sort(key=lambda x: x[1], reverse=True)

    items = [(r, v) for r, v in items if (v / total) >= float(min_share)]
    if not items:
        r0, v0 = max(route_flow.items(), key=lambda kv: kv[1])
        items = [(r0, float(v0))]

    node_labels: List[str] = ["SOURCE"] + [r for r, _ in items] + ["SINK"]

    n_routes = len(items)
    sources, targets, values, colors = [], [], [], []
    palette = [
        "rgba(31,119,180,0.65)", "rgba(255,127,14,0.65)", "rgba(44,160,44,0.65)",
        "rgba(214,39,40,0.65)", "rgba(148,103,189,0.65)", "rgba(140,86,75,0.65)",
        "rgba(227,119,194,0.65)", "rgba(127,127,127,0.65)", "rgba(188,189,34,0.65)",
        "rgba(23,190,207,0.65)"
    ]

    for idx, (r, v) in enumerate(items):
        route_node = 1 + idx
        sources += [0, route_node]
        targets += [route_node, 1 + n_routes]
        values += [v, v]
        colors += [palette[idx % len(palette)], palette[idx % len(palette)]]

    fig = go.Figure(
        data=[go.Sankey(
            arrangement="snap",
            node=dict(label=node_labels, pad=18, thickness=18, line=dict(width=0.5)),
            link=dict(source=sources, target=targets, value=values, color=colors),
        )]
    )
    fig.update_layout(title=title, font=dict(size=12))
    fig.write_html(out_html)
