import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from graspologic.embed import LaplacianSpectralEmbed


# ============================================================
# CONFIGURATION
# ============================================================

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR.parent / "data/geckoterminal/eth/20260416/pools_detailed.jsonl"
OUTPUT_DIR = BASE_DIR.parent / "outputs/geckoterminal_network"

# ----------------------------
# Data parsing / cleaning
# ----------------------------
DROP_DUPLICATE_POOLS = True
DUPLICATE_POOL_SUBSET = ["pool_address"]

LOWERCASE_ADDRESSES = True
SKIP_ROWS_WITH_MISSING_TOKEN_OR_POOL_ADDRESS = True

# ----------------------------
# Graph filtering
# ----------------------------
REQUIRE_POSITIVE_EDGE_RESERVE = True 
REMOVE_ISOLATES_AFTER_FILTERING = True # removes nodes with zero edges after filtering.

USE_LARGEST_CONNECTED_COMPONENT = True
REMOVE_SELF_LOOPS_BEFORE_MODEL = True # removes identical token-token edges

# ----------------------------
# Complexity inference
# ----------------------------
CLUSTER_RANGE = range(1, 6)   # candidate k values to evaluate

# graspologic embedding
EMBED_FORM = "R-DAD" # Controls how degree heterogeneity is handled.
EMBED_N_COMPONENTS = None #Let the algorithm choose embedding dimension automatically.
EMBED_N_ELBOWS = 1 # number of elbows to consider in the spectral embedding for automatic dimension selection. We explicitly set this to 1 to get the smallest embedding dimension that captures the main structure of the graph for simple model complexity. 

# Gaussian mixture settings
GMM_COVARIANCE_TYPE = "diag" # "full": each cluster has its own general covariance matrix
GMM_N_INIT = 100 # number of random initializations to try
GMM_REG_COVAR = 1e-6
RANDOM_STATE = 42

# anti-fragmentation rule
USE_MIN_CLUSTER_SIZE_RULE = True
MIN_CLUSTER_SIZE_ABS = 0 # if using the min cluster size rule, this is the minimum number of nodes that a cluster must have to be considered valid
MIN_CLUSTER_SIZE_SHARE = 0.03 # if using the min cluster size rule, this is the minimum share of total nodes that a cluster must have to be considered valid
# 1387 nodes -> min cluster size 42 for 3% rule

# ----------------------------
# Role assignment
# ----------------------------
ROLE_ORDER_FEATURES = [
    "core_number",
    "degree",
    "weighted_degree_reserve_usd",
    "total_liquidity_usd",
]
ROLE_ORDER_ASCENDING = True

USE_STANDARD_ROLE_NAMES = True
GENERIC_ROLE_PREFIX = "layer"

# ----------------------------
# Static network figure
# ----------------------------
STATIC_FIGSIZE = (16, 12)
STATIC_LAYOUT_SEED = 42
STATIC_LAYOUT_ITERATIONS = 100
LABEL_TOP_N_STATIC = 40

NODE_SIZE_MIN_STATIC = 40
NODE_SIZE_SCALE_STATIC = 2200
NODE_ALPHA_STATIC = 0.9

EDGE_WIDTH_MIN_STATIC = 0.4
EDGE_WIDTH_SCALE_STATIC = 5.0
EDGE_ALPHA_STATIC = 0.18

# ----------------------------
# Clustered network figure
# ----------------------------
CLUSTERED_FIGSIZE = (18, 14)
CLUSTERED_LAYOUT_SEED = 42
CLUSTERED_LAYOUT_ITERATIONS = 150
LABEL_TOP_N_CLUSTERED = 50

NODE_SIZE_MIN_CLUSTERED = 35
NODE_SIZE_SCALE_CLUSTERED = 1800
NODE_ALPHA_CLUSTERED = 0.92

EDGE_WIDTH_MIN_CLUSTERED = 0.3
EDGE_WIDTH_SCALE_CLUSTERED = 3.0
EDGE_ALPHA_CLUSTERED = 0.12

CLUSTER_COLORMAP = "tab10"

# ----------------------------
# Model selection figure
# ----------------------------
MODEL_SELECTION_FIGSIZE = (10, 6)
MODEL_SELECTION_SHOW_BIC = True
MODEL_SELECTION_SHOW_AIC = True
MODEL_SELECTION_TITLE = "Model selection by Gaussian mixture criteria"

# ----------------------------
# Figure output settings
# ----------------------------
SAVEFIG_DPI = 220

# ----------------------------
# Output file names
# ----------------------------
NETWORK_PNG_NAME = "ethereum_pool_network.png"
CLUSTERED_NETWORK_PNG_NAME = "ethereum_pool_network_clustered.png"
MODEL_SELECTION_PNG_NAME = "graspologic_model_selection.png"

NODE_CSV_NAME = "network_nodes.csv"
EDGE_CSV_NAME = "network_edges.csv"
FIT_CSV_NAME = "graspologic_model_selection.csv"
ROLE_CSV_NAME = "graspologic_node_roles.csv"
BLOCK_CSV_NAME = "graspologic_block_summary.csv"


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_float(x, default=float("nan")):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def scale_log1p(x: float, x_max: float, min_value: float, scale_value: float) -> float:
    denom = max(1.0, math.log1p(x_max if x_max > 0 else 1.0))
    return min_value + scale_value * (math.log1p(max(x, 0.0)) / denom)


# ============================================================
# DATA PARSING
# ============================================================

def parse_pools(raw_rows: list[dict]) -> pd.DataFrame:
    rows = []

    for r in raw_rows:
        base = r.get("base_token", {}) or {}
        quote = r.get("quote_token", {}) or {}
        dex = r.get("dex", {}) or {}

        base_address = base.get("address") or ""
        quote_address = quote.get("address") or ""
        pool_address = r.get("pool_address") or ""

        if LOWERCASE_ADDRESSES:
            base_address = base_address.lower()
            quote_address = quote_address.lower()
            pool_address = pool_address.lower()

        if SKIP_ROWS_WITH_MISSING_TOKEN_OR_POOL_ADDRESS:
            if not base_address or not quote_address or not pool_address:
                continue

        tx_h24 = ((r.get("transactions") or {}).get("h24") or {})
        vol_h24 = (r.get("volume_usd") or {}).get("h24")

        rows.append(
            {
                "pool_id": r.get("pool_id"),
                "pool_address": pool_address,
                "pool_name": r.get("pool_name") or r.get("name"),
                "dex_id": dex.get("id"),
                "dex_name": dex.get("name"),
                "base_address": base_address,
                "base_symbol": base.get("symbol") or base_address[:6],
                "base_name": base.get("name"),
                "quote_address": quote_address,
                "quote_symbol": quote.get("symbol") or quote_address[:6],
                "quote_name": quote.get("name"),
                "reserve_usd": safe_float(r.get("reserve_in_usd")),
                "base_token_liquidity_usd": safe_float(r.get("base_token_liquidity_usd")),
                "quote_token_liquidity_usd": safe_float(r.get("quote_token_liquidity_usd")),
                "volume_usd_h24": safe_float(vol_h24),
                "transactions_h24": safe_float(tx_h24.get("buys"), 0.0) + safe_float(tx_h24.get("sells"), 0.0),
                "pool_fee_percentage": safe_float(r.get("pool_fee_percentage")),
                "pool_created_at": r.get("pool_created_at"),
            }
        )

    df = pd.DataFrame(rows)

    if DROP_DUPLICATE_POOLS:
        df = df.drop_duplicates(subset=DUPLICATE_POOL_SUBSET)

    return df


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def build_token_graph(pools: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()

    node_liquidity = defaultdict(float)
    node_volume = defaultdict(float)
    node_symbol = {}

    for _, row in pools.iterrows():
        a = row["base_address"]
        b = row["quote_address"]

        liq = 0.0 if pd.isna(row["reserve_usd"]) else float(row["reserve_usd"])
        vol = 0.0 if pd.isna(row["volume_usd_h24"]) else float(row["volume_usd_h24"])

        node_liquidity[a] += 0.5 * liq
        node_liquidity[b] += 0.5 * liq
        node_volume[a] += 0.5 * vol
        node_volume[b] += 0.5 * vol
        node_symbol[a] = row["base_symbol"]
        node_symbol[b] = row["quote_symbol"]

    for addr in node_symbol.keys():
        G.add_node(
            addr,
            symbol=node_symbol.get(addr, addr[:6]),
            total_liquidity_usd=node_liquidity.get(addr, 0.0),
            total_volume_usd_h24=node_volume.get(addr, 0.0),
        )

    edge_bucket = defaultdict(
        lambda: {
            "pool_count": 0,
            "reserve_usd_sum": 0.0,
            "volume_usd_h24_sum": 0.0,
            "dexes": set(),
            "pool_names": [],
        }
    )

    for _, row in pools.iterrows():
        a = row["base_address"]
        b = row["quote_address"]

        if a == b:
            continue

        key = tuple(sorted([a, b]))
        edge_bucket[key]["pool_count"] += 1
        edge_bucket[key]["reserve_usd_sum"] += 0.0 if pd.isna(row["reserve_usd"]) else float(row["reserve_usd"])
        edge_bucket[key]["volume_usd_h24_sum"] += 0.0 if pd.isna(row["volume_usd_h24"]) else float(row["volume_usd_h24"])

        if row["dex_id"]:
            edge_bucket[key]["dexes"].add(row["dex_id"])
        if row["pool_name"]:
            edge_bucket[key]["pool_names"].append(row["pool_name"])

    for (a, b), x in edge_bucket.items():
        G.add_edge(
            a,
            b,
            pool_count=x["pool_count"],
            reserve_usd=x["reserve_usd_sum"],
            volume_usd_h24=x["volume_usd_h24_sum"],
            dexes=sorted(x["dexes"]),
            label=" | ".join(sorted(x["dexes"])) if x["dexes"] else "",
        )

    return G


def filter_graph(G: nx.Graph) -> nx.Graph:
    H = G.copy()
    print(f"[filter] start: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

    if REQUIRE_POSITIVE_EDGE_RESERVE:
        to_remove = []
        for u, v, data in H.edges(data=True):
            liq = safe_float(data.get("reserve_usd"), default=float("nan"))
            if pd.isna(liq) or liq <= 0:
                to_remove.append((u, v))
        H.remove_edges_from(to_remove)
        print(f"[filter] after reserve filter: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

    if REMOVE_ISOLATES_AFTER_FILTERING:
        H.remove_nodes_from(list(nx.isolates(H)))
        print(f"[filter] after isolate removal: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

    if USE_LARGEST_CONNECTED_COMPONENT and H.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(H), key=len)
        H = H.subgraph(largest_cc).copy()
        print(f"[filter] largest CC: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

    return H


# ============================================================
# PLOTTING
# ============================================================

def draw_static(G: nx.Graph, out_png: Path):
    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty after filtering.")

    pos = nx.spring_layout(
        G,
        seed=STATIC_LAYOUT_SEED,
        iterations=STATIC_LAYOUT_ITERATIONS,
    )

    node_liq = [float(G.nodes[n].get("total_liquidity_usd", 0.0)) for n in G.nodes()]
    edge_liq = [float(G.edges[e].get("reserve_usd", 0.0) or 0.0) for e in G.edges()]

    max_node_liq = max(node_liq) if node_liq else 1.0
    max_edge_liq = max(edge_liq) if edge_liq else 1.0

    node_sizes = [
        scale_log1p(x, max_node_liq, NODE_SIZE_MIN_STATIC, NODE_SIZE_SCALE_STATIC)
        for x in node_liq
    ]
    edge_widths = [
        scale_log1p(x, max_edge_liq, EDGE_WIDTH_MIN_STATIC, EDGE_WIDTH_SCALE_STATIC)
        for x in edge_liq
    ]

    plt.figure(figsize=STATIC_FIGSIZE)
    nx.draw_networkx_edges(G, pos, alpha=EDGE_ALPHA_STATIC, width=edge_widths)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=NODE_ALPHA_STATIC)

    top_nodes = sorted(
        G.nodes(),
        key=lambda n: float(G.nodes[n].get("total_liquidity_usd", 0.0)),
        reverse=True,
    )[:LABEL_TOP_N_STATIC]

    labels = {n: G.nodes[n].get("symbol", n[:6]) for n in top_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    plt.title(f"Ethereum token-pool network\nnodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=SAVEFIG_DPI, bbox_inches="tight")
    plt.close()


def draw_clustered_network(G: nx.Graph, node_roles: pd.DataFrame, out_png: Path):
    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty after filtering.")

    pos = nx.spring_layout(
        G,
        seed=CLUSTERED_LAYOUT_SEED,
        iterations=CLUSTERED_LAYOUT_ITERATIONS,
    )

    role_map = dict(zip(node_roles["address"], node_roles["role"]))
    unique_roles = list(dict.fromkeys(node_roles["role"].tolist()))

    cmap = plt.get_cmap(CLUSTER_COLORMAP)
    role_to_color = {role: cmap(i % 10) for i, role in enumerate(unique_roles)}

    node_colors = [
        role_to_color.get(role_map.get(n, "unknown"), (0.7, 0.7, 0.7, 1.0))
        for n in G.nodes()
    ]

    node_liq = [float(G.nodes[n].get("total_liquidity_usd", 0.0)) for n in G.nodes()]
    edge_liq = [float(G.edges[e].get("reserve_usd", 0.0) or 0.0) for e in G.edges()]

    max_node_liq = max(node_liq) if node_liq else 1.0
    max_edge_liq = max(edge_liq) if edge_liq else 1.0

    node_sizes = [
        scale_log1p(x, max_node_liq, NODE_SIZE_MIN_CLUSTERED, NODE_SIZE_SCALE_CLUSTERED)
        for x in node_liq
    ]
    edge_widths = [
        scale_log1p(x, max_edge_liq, EDGE_WIDTH_MIN_CLUSTERED, EDGE_WIDTH_SCALE_CLUSTERED)
        for x in edge_liq
    ]

    plt.figure(figsize=CLUSTERED_FIGSIZE)
    nx.draw_networkx_edges(G, pos, alpha=EDGE_ALPHA_CLUSTERED, width=edge_widths)
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=NODE_ALPHA_CLUSTERED,
    )

    top_nodes = sorted(
        G.nodes(),
        key=lambda n: float(G.nodes[n].get("total_liquidity_usd", 0.0)),
        reverse=True,
    )[:LABEL_TOP_N_CLUSTERED]

    labels = {n: G.nodes[n].get("symbol", n[:8]) for n in top_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            color="w",
            label=role,
            markerfacecolor=role_to_color[role],
            markersize=10,
        )
        for role in unique_roles
    ]
    plt.legend(handles=handles, title="Selected roles", loc="best")

    n_clusters = node_roles["cluster"].nunique()
    plt.title(f"Best clustering on filtered network (k={n_clusters})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=SAVEFIG_DPI, bbox_inches="tight")
    plt.close()


def draw_model_selection(fit_df: pd.DataFrame, best_k: int, out_png: Path):
    plt.figure(figsize=MODEL_SELECTION_FIGSIZE)

    ordered = fit_df.sort_values("k")

    if MODEL_SELECTION_SHOW_BIC:
        plt.plot(ordered["k"], ordered["bic"], marker="o", label="BIC")

    if MODEL_SELECTION_SHOW_AIC:
        plt.plot(ordered["k"], ordered["aic"], marker="o", label="AIC")

    plt.axvline(best_k, linestyle="--", label=f"Selected k = {best_k}")
    plt.xticks(list(ordered["k"]))
    plt.xlabel("Number of clusters")
    plt.ylabel("Criterion value")
    plt.title(MODEL_SELECTION_TITLE)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=SAVEFIG_DPI, bbox_inches="tight")
    plt.close()


# ============================================================
# COMPLEXITY INFERENCE
# ============================================================

def infer_complexity_graspologic(G: nx.Graph):
    H = G.copy()

    if REMOVE_SELF_LOOPS_BEFORE_MODEL:
        H.remove_edges_from(nx.selfloop_edges(H))

    if H.number_of_nodes() < 4:
        raise ValueError("Graph too small for meaningful complexity selection.")
    if H.number_of_edges() == 0:
        raise ValueError("Graph has no edges.")

    nodelist = list(H.nodes())
    A = nx.to_numpy_array(H, nodelist=nodelist, weight=None, dtype=float)

    lse = LaplacianSpectralEmbed(
        form=EMBED_FORM,
        n_components=EMBED_N_COMPONENTS,
        n_elbows=EMBED_N_ELBOWS,
    )
    X = lse.fit_transform(A)

    if isinstance(X, tuple):
        X = np.concatenate([np.asarray(part) for part in X], axis=1)
    else:
        X = np.asarray(X)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    X_scaled = StandardScaler().fit_transform(X)

    fit_rows = []
    label_store = {}

    for k in CLUSTER_RANGE:
        gm = GaussianMixture(
            n_components=k,
            covariance_type=GMM_COVARIANCE_TYPE,
            n_init=GMM_N_INIT,
            reg_covar=GMM_REG_COVAR,
            random_state=RANDOM_STATE,
        )
        gm.fit(X_scaled)
        labels = gm.predict(X_scaled)
        counts = pd.Series(labels).value_counts().sort_index()

        fit_rows.append(
            {
                "k": k,
                "bic": float(gm.bic(X_scaled)),
                "aic": float(gm.aic(X_scaled)),
                "embedding_dim": int(X.shape[1]),
                "min_cluster_size": int(counts.min()),
                "max_cluster_size": int(counts.max()),
            }
        )
        label_store[k] = labels

    fit_df = pd.DataFrame(fit_rows).sort_values("bic").reset_index(drop=True)

    if USE_MIN_CLUSTER_SIZE_RULE:
        min_size_required = max(
            MIN_CLUSTER_SIZE_ABS,
            int(np.ceil(MIN_CLUSTER_SIZE_SHARE * len(nodelist)))
        )
        valid_df = fit_df[fit_df["min_cluster_size"] >= min_size_required].copy()
    else:
        valid_df = fit_df.copy()

    best_k = int(valid_df.sort_values(["bic", "k"]).iloc[0]["k"])
    best_labels = label_store[best_k]

    core_num = nx.core_number(H)

    node_rows = []
    for idx, n in enumerate(nodelist):
        weighted_degree = sum(
            float(d.get("reserve_usd", 0.0) or 0.0)
            for _, _, d in H.edges(n, data=True)
        )

        node_rows.append(
            {
                "address": n,
                "symbol": H.nodes[n].get("symbol", n[:6]),
                "degree": H.degree(n),
                "core_number": core_num[n],
                "total_liquidity_usd": float(H.nodes[n].get("total_liquidity_usd", 0.0) or 0.0),
                "total_volume_usd_h24": float(H.nodes[n].get("total_volume_usd_h24", 0.0) or 0.0),
                "weighted_degree_reserve_usd": weighted_degree,
                "cluster": int(best_labels[idx]),
            }
        )

    node_df = pd.DataFrame(node_rows)

    cluster_order = (
        node_df.groupby("cluster")[ROLE_ORDER_FEATURES]
        .mean()
        .sort_values(ROLE_ORDER_FEATURES, ascending=ROLE_ORDER_ASCENDING)
        .index
        .tolist()
    )

    if USE_STANDARD_ROLE_NAMES:
        if best_k == 2:
            role_names = ["periphery", "core"]
        elif best_k == 3:
            role_names = ["periphery", "mid", "core"]
        else:
            role_names = [f"{GENERIC_ROLE_PREFIX}_{i+1}" for i in range(best_k)]
    else:
        role_names = [f"{GENERIC_ROLE_PREFIX}_{i+1}" for i in range(best_k)]

    role_map = {cluster_order[i]: role_names[i] for i in range(best_k)}
    node_df["role"] = node_df["cluster"].map(role_map)

    block_summary = (
        node_df.groupby(["cluster", "role"])
        .agg(
            n_nodes=("address", "count"),
            mean_core_number=("core_number", "mean"),
            median_core_number=("core_number", "median"),
            mean_degree=("degree", "mean"),
            total_liquidity_usd=("total_liquidity_usd", "sum"),
            mean_liquidity_usd=("total_liquidity_usd", "mean"),
            total_weighted_degree_reserve_usd=("weighted_degree_reserve_usd", "sum"),
        )
        .reset_index()
        .sort_values(["mean_core_number", "mean_degree", "total_liquidity_usd"], ascending=False)
    )

    return {
        "fit_df": fit_df,
        "valid_df": valid_df.sort_values("bic").reset_index(drop=True),
        "node_df": node_df.sort_values(
            ["cluster", "core_number", "weighted_degree_reserve_usd"],
            ascending=[True, False, False],
        ).reset_index(drop=True),
        "block_summary": block_summary.reset_index(drop=True),
        "best_k": best_k,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    outdir = ensure_dir(OUTPUT_DIR)

    raw_rows = load_jsonl(INPUT_PATH)
    print(f"Loaded raw rows: {len(raw_rows)}")

    pools = parse_pools(raw_rows)
    print(f"Parsed pools: {len(pools)}")
    print(pools[["base_symbol", "quote_symbol", "reserve_usd"]].head())

    G = build_token_graph(pools)
    print(f"Raw graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    H = filter_graph(G)

    out_png = outdir / NETWORK_PNG_NAME
    out_cluster_png = outdir / CLUSTERED_NETWORK_PNG_NAME
    out_model_png = outdir / MODEL_SELECTION_PNG_NAME
    out_nodes = outdir / NODE_CSV_NAME
    out_edges = outdir / EDGE_CSV_NAME
    out_fit = outdir / FIT_CSV_NAME
    out_roles = outdir / ROLE_CSV_NAME
    out_blocks = outdir / BLOCK_CSV_NAME

    draw_static(H, out_png)

    node_rows = []
    for n, d in H.nodes(data=True):
        node_rows.append(
            {
                "address": n,
                "symbol": d.get("symbol"),
                "total_liquidity_usd": d.get("total_liquidity_usd"),
                "total_volume_usd_h24": d.get("total_volume_usd_h24"),
                "degree": H.degree(n),
            }
        )
    pd.DataFrame(node_rows).sort_values("total_liquidity_usd", ascending=False).to_csv(out_nodes, index=False)

    edge_rows = []
    for u, v, d in H.edges(data=True):
        edge_rows.append(
            {
                "token0": u,
                "token1": v,
                "symbol0": H.nodes[u].get("symbol"),
                "symbol1": H.nodes[v].get("symbol"),
                "reserve_usd": d.get("reserve_usd"),
                "volume_usd_h24": d.get("volume_usd_h24"),
                "pool_count": d.get("pool_count"),
                "dexes": ",".join(d.get("dexes", [])),
            }
        )
    pd.DataFrame(edge_rows).sort_values("reserve_usd", ascending=False).to_csv(out_edges, index=False)

    results = infer_complexity_graspologic(H)

    fit_df = results["fit_df"]
    valid_df = results["valid_df"]
    node_roles = results["node_df"]
    block_summary = results["block_summary"]
    best_k = results["best_k"]

    fit_df.to_csv(out_fit, index=False)
    node_roles.to_csv(out_roles, index=False)
    block_summary.to_csv(out_blocks, index=False)

    draw_clustered_network(H, node_roles, out_cluster_png)
    draw_model_selection(fit_df, best_k, out_model_png)

    print("\n=== RAW MODEL SELECTION ===")
    print(fit_df.to_string(index=False))

    print("\n=== VALID MODEL SELECTION (tiny clusters removed) ===")
    print(valid_df.to_string(index=False))

    print(f"\nSuggested complexity (best k): {best_k}")
    if best_k == 2:
        print("Interpretation: core / periphery")
    elif best_k == 3:
        print("Interpretation: core / mid / periphery")
    else:
        print(f"Interpretation: {best_k} structural layers")

    print("\n=== BLOCK SUMMARY ===")
    print(block_summary.to_string(index=False))

    print(f"\nWrote: {out_png}")
    print(f"Wrote: {out_cluster_png}")
    print(f"Wrote: {out_model_png}")
    print(f"Wrote: {out_nodes}")
    print(f"Wrote: {out_edges}")
    print(f"Wrote: {out_fit}")
    print(f"Wrote: {out_roles}")
    print(f"Wrote: {out_blocks}")
    print(f"Final graph: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")


if __name__ == "__main__":
    main()