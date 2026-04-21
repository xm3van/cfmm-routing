import json
import math
from pathlib import Path
from collections import defaultdict

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

try:
    from pyvis.network import Network
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR.parent / "data/geckoterminal/eth/20260416/pools_detailed.jsonl"
OUTPUT_DIR = BASE_DIR.parent / "outputs/geckoterminal_network"

MIN_POOL_LIQUIDITY_USD = 0
TOP_N_EDGES_BY_LIQUIDITY = 1000
USE_LARGEST_CONNECTED_COMPONENT = True
STATIC_FIGSIZE = (16, 12)


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


def parse_pools(raw_rows: list[dict]) -> pd.DataFrame:
    rows = []

    for r in raw_rows:
        base = r.get("base_token", {}) or {}
        quote = r.get("quote_token", {}) or {}
        dex = r.get("dex", {}) or {}

        base_address = (base.get("address") or "").lower()
        quote_address = (quote.get("address") or "").lower()
        pool_address = (r.get("pool_address") or "").lower()

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
                "base_decimals": safe_float(base.get("decimals")),
                "base_coingecko_id": base.get("coingecko_coin_id"),
                "quote_address": quote_address,
                "quote_symbol": quote.get("symbol") or quote_address[:6],
                "quote_name": quote.get("name"),
                "quote_decimals": safe_float(quote.get("decimals")),
                "quote_coingecko_id": quote.get("coingecko_coin_id"),
                "reserve_usd": safe_float(r.get("reserve_in_usd")),
                "base_token_liquidity_usd": safe_float(r.get("base_token_liquidity_usd")),
                "quote_token_liquidity_usd": safe_float(r.get("quote_token_liquidity_usd")),
                "volume_usd_h24": safe_float(vol_h24),
                "transactions_h24": safe_float(tx_h24.get("buys"), 0.0) + safe_float(tx_h24.get("sells"), 0.0),
                "pool_fee_percentage": safe_float(r.get("pool_fee_percentage")),
                "pool_created_at": r.get("pool_created_at"),
            }
        )

    df = pd.DataFrame(rows).drop_duplicates(subset=["pool_address"])
    return df


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

    to_remove = []
    for u, v, data in H.edges(data=True):
        liq = data.get("reserve_usd", 0.0)
        try:
            liq = float(liq)
        except Exception:
            liq = float("nan")

        if pd.isna(liq) or liq < MIN_POOL_LIQUIDITY_USD:
            to_remove.append((u, v))

    H.remove_edges_from(to_remove)
    print(f"[filter] after liquidity filter: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

    isolates = list(nx.isolates(H))
    H.remove_nodes_from(isolates)
    print(f"[filter] after isolate removal: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

    if TOP_N_EDGES_BY_LIQUIDITY is not None and H.number_of_edges() > TOP_N_EDGES_BY_LIQUIDITY:
        ranked = sorted(
            H.edges(data=True),
            key=lambda x: float(x[2].get("reserve_usd", 0.0) or 0.0),
            reverse=True,
        )
        keep = set((u, v) for u, v, _ in ranked[:TOP_N_EDGES_BY_LIQUIDITY])

        H2 = nx.Graph()
        for n, d in H.nodes(data=True):
            H2.add_node(n, **d)
        for u, v, d in H.edges(data=True):
            if (u, v) in keep or (v, u) in keep:
                H2.add_edge(u, v, **d)
        H = H2

        isolates = list(nx.isolates(H))
        H.remove_nodes_from(isolates)
        print(f"[filter] after top-N filter: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

    if USE_LARGEST_CONNECTED_COMPONENT and H.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(H), key=len)
        H = H.subgraph(largest_cc).copy()
        print(f"[filter] largest CC: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

    return H


def draw_static(G: nx.Graph, out_png: Path):
    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty after filtering.")

    pos = nx.spring_layout(G, seed=42, iterations=100)

    node_liq = [float(G.nodes[n].get("total_liquidity_usd", 0.0)) for n in G.nodes()]
    edge_liq = [float(G.edges[e].get("reserve_usd", 0.0)) for e in G.edges()]

    def node_size(x):
        return 40 + 2200 * (math.log1p(x) / max(1.0, math.log1p(max(node_liq) if node_liq else 1.0)))

    def edge_width(x):
        return 0.4 + 5.0 * (math.log1p(x) / max(1.0, math.log1p(max(edge_liq) if edge_liq else 1.0)))

    node_sizes = [node_size(x) for x in node_liq]
    edge_widths = [edge_width(x) for x in edge_liq]

    plt.figure(figsize=STATIC_FIGSIZE)
    nx.draw_networkx_edges(G, pos, alpha=0.18, width=edge_widths)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.9)

    top_nodes = sorted(
        G.nodes(),
        key=lambda n: float(G.nodes[n].get("total_liquidity_usd", 0.0)),
        reverse=True,
    )[:40]
    labels = {n: G.nodes[n].get("symbol", n[:6]) for n in top_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    plt.title(
        f"Ethereum token-pool network\n"
        f"nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, "
        f"min_edge_liquidity=${MIN_POOL_LIQUIDITY_USD:,.0f}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def draw_interactive(G: nx.Graph, out_html: Path):
    if not HAS_PYVIS:
        print("pyvis not installed; skipping HTML export.")
        return

    nt = Network(height="900px", width="100%", bgcolor="white", font_color="black")
    nt.barnes_hut()

    node_liq = [float(G.nodes[n].get("total_liquidity_usd", 0.0)) for n in G.nodes()]
    max_node_liq = max(node_liq) if node_liq else 1.0
    edge_liq = [float(G.edges[e].get("reserve_usd", 0.0)) for e in G.edges()]
    max_edge_liq = max(edge_liq) if edge_liq else 1.0

    for n, d in G.nodes(data=True):
        liq = float(d.get("total_liquidity_usd", 0.0))
        vol = float(d.get("total_volume_usd_h24", 0.0))
        symbol = d.get("symbol", n[:6])
        size = 8 + 30 * (math.log1p(liq) / max(1.0, math.log1p(max_node_liq)))
        title = (
            f"{symbol}<br>{n}<br>"
            f"total_liquidity_usd: ${liq:,.0f}<br>"
            f"total_volume_usd_h24: ${vol:,.0f}"
        )
        nt.add_node(n, label=symbol, title=title, size=size)

    for u, v, d in G.edges(data=True):
        liq = float(d.get("reserve_usd", 0.0))
        vol = float(d.get("volume_usd_h24", 0.0))
        pool_count = int(d.get("pool_count", 0))
        width = 1 + 8 * (math.log1p(liq) / max(1.0, math.log1p(max_edge_liq)))
        title = (
            f"{G.nodes[u].get('symbol', u[:6])} ↔ {G.nodes[v].get('symbol', v[:6])}<br>"
            f"aggregate_reserve_usd: ${liq:,.0f}<br>"
            f"aggregate_volume_usd_h24: ${vol:,.0f}<br>"
            f"pool_count: {pool_count}<br>"
            f"dexes: {', '.join(d.get('dexes', []))}"
        )
        nt.add_edge(u, v, value=width, width=width, title=title)

    nt.show(str(out_html))


def main():
    outdir = ensure_dir(OUTPUT_DIR)

    raw_rows = load_jsonl(INPUT_PATH)
    print(f"Loaded raw rows: {len(raw_rows)}")

    pools = parse_pools(raw_rows)
    print(f"Parsed pools: {len(pools)}")
    print(pools[['base_symbol', 'quote_symbol', 'reserve_usd']].head())

    G = build_token_graph(pools)
    print(f"Raw graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    H = filter_graph(G)

    out_png = outdir / "ethereum_pool_network.png"
    out_html = outdir / "ethereum_pool_network.html"
    out_nodes = outdir / "network_nodes.csv"
    out_edges = outdir / "network_edges.csv"

    draw_static(H, out_png)
    draw_interactive(H, out_html)

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

    print(f"Wrote: {out_png}")
    if HAS_PYVIS:
        print(f"Wrote: {out_html}")
    print(f"Wrote: {out_nodes}")
    print(f"Wrote: {out_edges}")
    print(f"Final graph: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")


if __name__ == "__main__":
    main()