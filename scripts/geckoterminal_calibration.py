import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


# ============================================================
# Config
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR.parent / "data/geckoterminal/eth/20260416/pools_detailed.jsonl"
OUTPUT_DIR = BASE_DIR.parent / "outputs/geckoterminal_calibration_experiment"

# Optional manual overrides:
# dex_id,ptype,default_fee_decimal,w_i,w_j
# balancer,bal_wgm,0.0025,0.8,0.2
# my-custom-v3-fork,univ3_proxy,0.003,,
OVERRIDES_PATH = BASE_DIR.parent / "data/geckoterminal/ptype_overrides.csv"

USD_LIKE_PRICE_LOW = 0.98
USD_LIKE_PRICE_HIGH = 1.02
USD_LIKE_MAX_WEIGHTED_MAD = 0.02
USD_LIKE_MIN_TOTAL_LIQUIDITY_USD = 100_000.0
USD_LIKE_MIN_INCIDENT_POOLS = 2

DIRICHLET_ALPHA = 1.0
ROLE_LEVELS = ["periphery", "mid", "core"]
VALUE_REGIME_LEVELS = ["usd_like", "volatile"]
SUPPORTED_PTYPES = ["univ2", "bal_wgm", "curve", "univ3_proxy"]

UNIV3_STANDARD_TIERS = np.array([0.0001, 0.0005, 0.001, 0.003, 0.01], dtype=float)

# Heuristic reduced-form priors for the univ3 proxy.
# These are not identified from GeckoTerminal snapshot data; they are model priors.
UNIV3_PROXY_PRIORS = {
    "usd_like__usd_like": {
        0.0001: {"alpha": 0.10, "beta": 0.80},
        0.0005: {"alpha": 0.14, "beta": 0.75},
        0.0010: {"alpha": 0.18, "beta": 0.70},
        0.0030: {"alpha": 0.25, "beta": 0.60},
        0.0100: {"alpha": 0.35, "beta": 0.48},
    },
    "usd_like__volatile": {
        0.0005: {"alpha": 0.18, "beta": 0.68},
        0.0010: {"alpha": 0.22, "beta": 0.64},
        0.0030: {"alpha": 0.28, "beta": 0.56},
        0.0100: {"alpha": 0.38, "beta": 0.46},
    },
    "volatile__volatile": {
        0.0005: {"alpha": 0.22, "beta": 0.62},
        0.0010: {"alpha": 0.26, "beta": 0.58},
        0.0030: {"alpha": 0.32, "beta": 0.52},
        0.0100: {"alpha": 0.42, "beta": 0.45},
    },
}

CURVE_K_BY_REGIME_PAIR = {
    "usd_like__usd_like": 0.20,
    "usd_like__volatile": 0.30,
    "volatile__volatile": 0.40,
}

BAL_WGM_DEFAULTS = {
    "w_i": 0.5,
    "w_j": 0.5,
}


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_float(x, default=np.nan):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def normalize_probs(d: dict[str, float]) -> dict[str, float]:
    s = float(sum(d.values()))
    if s <= 0:
        n = len(d)
        return {k: 1.0 / n for k in d}
    return {k: float(v) / s for k, v in d.items()}


def pair_class(a: str, b: str) -> str:
    return "__".join(sorted([a, b]))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return np.nan
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cdf = np.cumsum(w) / np.sum(w)
    idx = np.searchsorted(cdf, 0.5)
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def weighted_mad(values: np.ndarray, weights: np.ndarray, center: float) -> float:
    dev = np.abs(values - center)
    return weighted_median(dev, weights)


def load_ptype_overrides(path: Path):
    if not path.exists():
        return {}, {}, {}
    df = pd.read_csv(path)
    df["dex_id"] = df["dex_id"].astype(str).str.lower()

    ptype_map = df.set_index("dex_id")["ptype"].to_dict()

    fee_map = {}
    if "default_fee_decimal" in df.columns:
        fee_map = (
            df.dropna(subset=["default_fee_decimal"])
            .set_index("dex_id")["default_fee_decimal"]
            .astype(float)
            .to_dict()
        )

    weights_map = {}
    if {"w_i", "w_j"}.issubset(df.columns):
        tmp = df.dropna(subset=["w_i", "w_j"]).copy()
        for _, row in tmp.iterrows():
            weights_map[row["dex_id"]] = {
                "w_i": float(row["w_i"]),
                "w_j": float(row["w_j"]),
            }

    return ptype_map, fee_map, weights_map


PTYPE_OVERRIDE_MAP, OVERRIDE_FEE_MAP, OVERRIDE_BAL_WEIGHTS = load_ptype_overrides(OVERRIDES_PATH)


# ============================================================
# Strict DEX -> ptype mapping
# ============================================================

DEX_TO_PTYPE_EXACT = {
    # strict v2-like
    "uniswap_v2": "univ2",
    "univ2": "univ2",
    "sushiswap": "univ2",
    "shibaswap": "univ2",
    "pancakeswap_ethereum": "univ2",
    "fraxswap_ethereum": "univ2",
    "swapr_ethereum": "univ2",
    "apeswap_ethereum": "univ2",
    "radioshack_ethereum": "univ2",
    "verse": "univ2",
    "whiteswap": "univ2",
    "9inch-ethereum": "univ2",

    # strict curve stable bucket
    "curve": "curve",

    # strict concentrated-liquidity bucket
    "uniswap_v3": "univ3_proxy",
    "univ3": "univ3_proxy",
    "pancakeswap-v3-ethereum": "univ3_proxy",
}

DEX_TO_PTYPE_SUBSTR = [
    ("uniswap_v2", "univ2"),
    ("univ2", "univ2"),
    ("sushi", "univ2"),
    ("shibaswap", "univ2"),
    ("fraxswap", "univ2"),
    ("pancakeswap_ethereum", "univ2"),
    ("9inch", "univ2"),

    ("curve", "curve"),

    ("uniswap_v3", "univ3_proxy"),
    ("pancakeswap-v3", "univ3_proxy"),
]


def map_ptype(dex_id: str, regime_pair: str) -> str:
    x = (dex_id or "").lower()

    if x in PTYPE_OVERRIDE_MAP:
        ptype = PTYPE_OVERRIDE_MAP[x]
        if ptype == "curve" and regime_pair != "usd_like__usd_like":
            return "exclude"
        return ptype

    if x in DEX_TO_PTYPE_EXACT:
        ptype = DEX_TO_PTYPE_EXACT[x]
        if ptype == "curve":
            return "curve" if regime_pair == "usd_like__usd_like" else "exclude"
        return ptype

    for token, ptype in DEX_TO_PTYPE_SUBSTR:
        if token in x:
            if ptype == "curve":
                return "curve" if regime_pair == "usd_like__usd_like" else "exclude"
            return ptype

    return "exclude"


def normalize_fee_decimal(pool_fee_percentage):
    # GeckoTerminal percentage units:
    # 0.05 -> 0.05% -> 0.0005
    # 0.3  -> 0.3%  -> 0.003
    # 1    -> 1%    -> 0.01
    if pd.isna(pool_fee_percentage):
        return np.nan
    return float(pool_fee_percentage) / 100.0


def snap_univ3_fee_tier(fee_decimal: float) -> float:
    if pd.isna(fee_decimal):
        return np.nan
    idx = np.argmin(np.abs(UNIV3_STANDARD_TIERS - fee_decimal))
    return float(UNIV3_STANDARD_TIERS[idx])


def infer_fee_for_edge(ptype: str, dex_id: str, pool_fee_percentage):
    fee_decimal = normalize_fee_decimal(pool_fee_percentage)
    x = (dex_id or "").lower()

    if pd.isna(fee_decimal) and x in OVERRIDE_FEE_MAP:
        fee_decimal = float(OVERRIDE_FEE_MAP[x])

    if ptype == "univ2":
        if pd.isna(fee_decimal):
            fee_decimal = 0.003
        return round(float(fee_decimal), 6)

    if ptype == "univ3_proxy":
        if pd.isna(fee_decimal):
            return np.nan
        return round(snap_univ3_fee_tier(fee_decimal), 6)

    if ptype == "curve":
        if pd.isna(fee_decimal):
            return np.nan
        return round(float(fee_decimal), 6)

    if ptype == "bal_wgm":
        if pd.isna(fee_decimal):
            return np.nan
        return round(float(fee_decimal), 6)

    return np.nan


def get_univ3_proxy_params(regime_pair: str, fee: float) -> dict[str, float]:
    if pd.isna(fee):
        return {"alpha": np.nan, "beta": np.nan}

    rp = UNIV3_PROXY_PRIORS.get(regime_pair)
    if rp is None:
        return {"alpha": np.nan, "beta": np.nan}

    keys = np.array(sorted(rp.keys()), dtype=float)
    idx = np.argmin(np.abs(keys - float(fee)))
    chosen = float(keys[idx])
    return {
        "alpha": float(rp[chosen]["alpha"]),
        "beta": float(rp[chosen]["beta"]),
    }


def get_bal_wgm_params(dex_id: str) -> dict[str, float]:
    x = (dex_id or "").lower()
    if x in OVERRIDE_BAL_WEIGHTS:
        return {
            "w_i": float(OVERRIDE_BAL_WEIGHTS[x]["w_i"]),
            "w_j": float(OVERRIDE_BAL_WEIGHTS[x]["w_j"]),
        }
    return {
        "w_i": np.nan,
        "w_j": np.nan,
    }


# ============================================================
# Parse flattened pools_detailed.jsonl
# ============================================================

def parse_pools(raw_rows: list[dict]) -> pd.DataFrame:
    rows = []

    for r in raw_rows:
        base = r.get("base_token", {}) or {}
        quote = r.get("quote_token", {}) or {}
        dex = r.get("dex", {}) or {}
        tx_h24 = ((r.get("transactions") or {}).get("h24") or {})
        vol_h24 = (r.get("volume_usd") or {}).get("h24")

        base_address = (base.get("address") or "").lower()
        quote_address = (quote.get("address") or "").lower()
        pool_address = (r.get("pool_address") or "").lower()

        if not base_address or not quote_address or not pool_address:
            continue

        rows.append(
            {
                "pool_address": pool_address,
                "pool_name": r.get("pool_name") or r.get("name"),
                "dex_id": dex.get("id"),
                "dex_name": dex.get("name"),

                "base_address": base_address,
                "base_symbol": base.get("symbol"),
                "base_name": base.get("name"),
                "base_coingecko_id": base.get("coingecko_coin_id"),
                "base_price_usd": safe_float(r.get("base_token_price_usd")),
                "base_token_liquidity_usd": safe_float(r.get("base_token_liquidity_usd")),

                "quote_address": quote_address,
                "quote_symbol": quote.get("symbol"),
                "quote_name": quote.get("name"),
                "quote_coingecko_id": quote.get("coingecko_coin_id"),
                "quote_price_usd": safe_float(r.get("quote_token_price_usd")),
                "quote_token_liquidity_usd": safe_float(r.get("quote_token_liquidity_usd")),

                "reserve_usd": safe_float(r.get("reserve_in_usd")),
                "volume_usd_h24": safe_float(vol_h24, 0.0),
                "transactions_h24": safe_float(tx_h24.get("buys"), 0.0) + safe_float(tx_h24.get("sells"), 0.0),
                "pool_fee_percentage": safe_float(r.get("pool_fee_percentage")),
                "pool_created_at": r.get("pool_created_at"),
            }
        )

    return pd.DataFrame(rows).drop_duplicates(subset=["pool_address"])


# ============================================================
# Token table + price observations
# ============================================================

def build_token_table(pools: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    token_stats = defaultdict(lambda: {
        "address": None,
        "symbol": None,
        "name": None,
        "coingecko_id": None,
        "incident_pools": 0,
        "counterparties": set(),
        "total_liquidity_usd": 0.0,
        "total_volume_usd_h24": 0.0,
        "total_transactions_h24": 0.0,
    })

    token_obs = []

    for _, row in pools.iterrows():
        a = row["base_address"]
        b = row["quote_address"]
        liq_pool = 0.0 if pd.isna(row["reserve_usd"]) else float(row["reserve_usd"])
        vol = 0.0 if pd.isna(row["volume_usd_h24"]) else float(row["volume_usd_h24"])
        tx = 0.0 if pd.isna(row["transactions_h24"]) else float(row["transactions_h24"])

        if not pd.isna(row["base_price_usd"]) and not pd.isna(row["base_token_liquidity_usd"]):
            token_obs.append(
                {
                    "address": a,
                    "symbol": row["base_symbol"],
                    "price_usd": float(row["base_price_usd"]),
                    "obs_liquidity_usd": float(row["base_token_liquidity_usd"]),
                    "pool_address": row["pool_address"],
                }
            )

        if not pd.isna(row["quote_price_usd"]) and not pd.isna(row["quote_token_liquidity_usd"]):
            token_obs.append(
                {
                    "address": b,
                    "symbol": row["quote_symbol"],
                    "price_usd": float(row["quote_price_usd"]),
                    "obs_liquidity_usd": float(row["quote_token_liquidity_usd"]),
                    "pool_address": row["pool_address"],
                }
            )

        for addr, sym, name, cg, cp in [
            (a, row["base_symbol"], row["base_name"], row["base_coingecko_id"], b),
            (b, row["quote_symbol"], row["quote_name"], row["quote_coingecko_id"], a),
        ]:
            token_stats[addr]["address"] = addr
            token_stats[addr]["symbol"] = sym
            token_stats[addr]["name"] = name
            token_stats[addr]["coingecko_id"] = cg
            token_stats[addr]["incident_pools"] += 1
            token_stats[addr]["counterparties"].add(cp)
            token_stats[addr]["total_liquidity_usd"] += 0.5 * liq_pool
            token_stats[addr]["total_volume_usd_h24"] += 0.5 * vol
            token_stats[addr]["total_transactions_h24"] += 0.5 * tx

    token_rows = []
    for addr, x in token_stats.items():
        token_rows.append(
            {
                "address": addr,
                "symbol": x["symbol"],
                "name": x["name"],
                "coingecko_id": x["coingecko_id"],
                "incident_pools": x["incident_pools"],
                "degree": len(x["counterparties"]),
                "total_liquidity_usd": x["total_liquidity_usd"],
                "total_volume_usd_h24": x["total_volume_usd_h24"],
                "total_transactions_h24": x["total_transactions_h24"],
            }
        )

    tokens = pd.DataFrame(token_rows)
    observations = pd.DataFrame(token_obs)
    return tokens, observations


# ============================================================
# Classifier
# ============================================================

def classify_value_regime(tokens: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    summaries = []

    for addr, g in observations.groupby("address"):
        prices = g["price_usd"].to_numpy(dtype=float)
        weights = g["obs_liquidity_usd"].fillna(0.0).to_numpy(dtype=float)

        mask = np.isfinite(prices) & np.isfinite(weights) & (weights > 0) & (prices > 0)
        prices = prices[mask]
        weights = weights[mask]

        if len(prices) == 0:
            summaries.append(
                {
                    "address": addr,
                    "weighted_median_price_usd": np.nan,
                    "weighted_mad_price_usd": np.nan,
                    "obs_count": 0,
                    "obs_liquidity_usd_sum": 0.0,
                }
            )
            continue

        med = weighted_median(prices, weights)
        mad = weighted_mad(prices, weights, med)

        summaries.append(
            {
                "address": addr,
                "weighted_median_price_usd": med,
                "weighted_mad_price_usd": mad,
                "obs_count": int(len(prices)),
                "obs_liquidity_usd_sum": float(np.sum(weights)),
            }
        )

    price_summary = pd.DataFrame(summaries)
    tokens = tokens.merge(price_summary, on="address", how="left")

    def regime_rule(row):
        p = row["weighted_median_price_usd"]
        mad = row["weighted_mad_price_usd"]
        liq = row["total_liquidity_usd"]
        pools = row["incident_pools"]

        if pd.isna(p) or pd.isna(mad):
            return "volatile"

        is_usd_like = (
            USD_LIKE_PRICE_LOW <= p <= USD_LIKE_PRICE_HIGH
            and mad <= USD_LIKE_MAX_WEIGHTED_MAD
            and liq >= USD_LIKE_MIN_TOTAL_LIQUIDITY_USD
            and pools >= USD_LIKE_MIN_INCIDENT_POOLS
        )
        return "usd_like" if is_usd_like else "volatile"

    tokens["value_regime"] = tokens.apply(regime_rule, axis=1)
    return tokens


def assign_roles(tokens: pd.DataFrame) -> pd.DataFrame:
    tokens = tokens.copy()

    tokens["liq_score"] = tokens["total_liquidity_usd"].rank(pct=True)
    tokens["vol_score"] = tokens["total_volume_usd_h24"].rank(pct=True)
    tokens["deg_score"] = tokens["degree"].rank(pct=True)

    tokens["role_score"] = (
        0.45 * tokens["liq_score"]
        + 0.35 * tokens["vol_score"]
        + 0.20 * tokens["deg_score"]
    )

    q1 = tokens["role_score"].quantile(1 / 3)
    q2 = tokens["role_score"].quantile(2 / 3)

    def role_fn(x):
        if x <= q1:
            return "periphery"
        elif x <= q2:
            return "mid"
        return "core"

    tokens["role"] = tokens["role_score"].apply(role_fn)
    return tokens


# ============================================================
# Edge table for experiment file
# ============================================================

def build_edge_table(pools: pd.DataFrame, tokens: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = tokens.set_index("address")[["value_regime", "role"]].to_dict(orient="index")

    edges = pools.copy()
    edges["base_value_regime"] = edges["base_address"].map(lambda x: lookup.get(x, {}).get("value_regime"))
    edges["quote_value_regime"] = edges["quote_address"].map(lambda x: lookup.get(x, {}).get("value_regime"))
    edges["base_role"] = edges["base_address"].map(lambda x: lookup.get(x, {}).get("role"))
    edges["quote_role"] = edges["quote_address"].map(lambda x: lookup.get(x, {}).get("role"))

    edges = edges.dropna(subset=["base_value_regime", "quote_value_regime", "base_role", "quote_role"]).copy()

    edges["regime_pair"] = edges.apply(
        lambda r: pair_class(r["base_value_regime"], r["quote_value_regime"]),
        axis=1,
    )
    edges["role_pair"] = edges.apply(
        lambda r: pair_class(r["base_role"], r["quote_role"]),
        axis=1,
    )

    edges["ptype"] = edges.apply(
        lambda r: map_ptype(r["dex_id"], r["regime_pair"]),
        axis=1,
    )

    edges["fee"] = edges.apply(
        lambda r: infer_fee_for_edge(r["ptype"], r["dex_id"], r["pool_fee_percentage"]),
        axis=1,
    )

    # model params
    edges["alpha"] = np.nan
    edges["beta"] = np.nan
    edges["k"] = np.nan
    edges["w_i"] = np.nan
    edges["w_j"] = np.nan

    # univ3 proxy params
    mask_u3 = edges["ptype"] == "univ3_proxy"
    for idx, row in edges.loc[mask_u3].iterrows():
        params = get_univ3_proxy_params(row["regime_pair"], row["fee"])
        edges.at[idx, "alpha"] = params["alpha"]
        edges.at[idx, "beta"] = params["beta"]

    # curve params
    mask_curve = edges["ptype"] == "curve"
    for idx, row in edges.loc[mask_curve].iterrows():
        edges.at[idx, "k"] = CURVE_K_BY_REGIME_PAIR.get(row["regime_pair"], np.nan)

    # balancer weighted params (override-only)
    mask_bal = edges["ptype"] == "bal_wgm"
    for idx, row in edges.loc[mask_bal].iterrows():
        params = get_bal_wgm_params(row["dex_id"])
        edges.at[idx, "w_i"] = params["w_i"]
        edges.at[idx, "w_j"] = params["w_j"]

    excluded = edges[edges["ptype"] == "exclude"].copy()
    supported = edges[edges["ptype"] != "exclude"].copy()

    return supported, excluded


# ============================================================
# Estimators
# ============================================================

def estimate_role_probs(tokens: pd.DataFrame) -> dict:
    vc = tokens["role"].value_counts()
    total = vc.sum()
    return {k: float(v / total) for k, v in vc.sort_index().items()}


def estimate_value_regime_given_role(tokens: pd.DataFrame, weight_col: str | None = None) -> dict:
    out = {}

    for role, g in tokens.groupby("role"):
        counts = {vr: DIRICHLET_ALPHA for vr in VALUE_REGIME_LEVELS}

        if weight_col is None:
            vc = g["value_regime"].value_counts()
            for vr in VALUE_REGIME_LEVELS:
                counts[vr] += float(vc.get(vr, 0.0))
        else:
            agg = g.groupby("value_regime")[weight_col].sum()
            for vr in VALUE_REGIME_LEVELS:
                counts[vr] += float(agg.get(vr, 0.0))

        out[role] = normalize_probs(counts)

    return out


def estimate_ptype_given_regime_pair(edges: pd.DataFrame, weight_col: str | None = None) -> dict:
    out = {}

    for regime_pair, g in edges.groupby("regime_pair"):
        if weight_col is None:
            agg = g["ptype"].value_counts().to_dict()
        else:
            agg = g.groupby("ptype")[weight_col].sum().to_dict()

        probs = {ptype: float(agg.get(ptype, 0.0)) for ptype in SUPPORTED_PTYPES}
        out[regime_pair] = normalize_probs(probs)

    return out


def estimate_fee_given_ptype_and_regime_pair(edges: pd.DataFrame, weight_col: str | None = None) -> dict:
    out = {}
    usable = edges.dropna(subset=["fee"]).copy()
    usable["fee"] = usable["fee"].astype(float).round(6)

    for (regime_pair, ptype), g in usable.groupby(["regime_pair", "ptype"]):
        key = f"{regime_pair} | {ptype}"
        if weight_col is None:
            agg = g["fee"].astype(str).value_counts().to_dict()
        else:
            agg = g.groupby(g["fee"].astype(str))[weight_col].sum().to_dict()

        out[key] = normalize_probs({str(k): float(v) for k, v in agg.items()})

    return out


def estimate_liquidity_buckets(edges: pd.DataFrame) -> dict:
    out = {}

    for key, g in edges.groupby(["role_pair", "ptype", "regime_pair"]):
        role_pair, ptype, regime_pair = key
        x = g["reserve_usd"].dropna()
        x = x[x > 0]
        if len(x) == 0:
            continue

        lx = np.log(x)

        out[f"{role_pair} | {ptype} | {regime_pair}"] = {
            "count": int(len(x)),
            "mean_usd": float(np.mean(x)),
            "median_usd": float(np.median(x)),
            "p10_usd": float(np.percentile(x, 10)),
            "p90_usd": float(np.percentile(x, 90)),
            "lognormal_mu": float(np.mean(lx)),
            "lognormal_sigma": float(np.std(lx, ddof=0)),
        }

    return out


# ============================================================
# Main
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    raw_rows = load_jsonl(INPUT_PATH)
    pools = parse_pools(raw_rows)

    tokens, observations = build_token_table(pools)
    tokens = classify_value_regime(tokens, observations)
    tokens = assign_roles(tokens)

    edges_supported, edges_excluded = build_edge_table(pools, tokens)

    calibration = {
        "metadata": {
            "n_pools": int(len(pools)),
            "n_tokens": int(len(tokens)),
            "n_edges_supported": int(len(edges_supported)),
            "n_edges_excluded": int(len(edges_excluded)),
            "dirichlet_alpha": DIRICHLET_ALPHA,
            "supported_ptypes": SUPPORTED_PTYPES,
            "usd_like_classifier": {
                "price_low": USD_LIKE_PRICE_LOW,
                "price_high": USD_LIKE_PRICE_HIGH,
                "max_weighted_mad": USD_LIKE_MAX_WEIGHTED_MAD,
                "min_total_liquidity_usd": USD_LIKE_MIN_TOTAL_LIQUIDITY_USD,
                "min_incident_pools": USD_LIKE_MIN_INCIDENT_POOLS,
            },
            "curve_k_by_regime_pair": CURVE_K_BY_REGIME_PAIR,
            "univ3_proxy_priors": UNIV3_PROXY_PRIORS,
            "bal_wgm_defaults": BAL_WGM_DEFAULTS,
        },
        "role_probs": estimate_role_probs(tokens),
        "value_regime_given_role": {
            "count_weighted": estimate_value_regime_given_role(tokens, weight_col=None),
            "liquidity_weighted": estimate_value_regime_given_role(tokens, weight_col="total_liquidity_usd"),
        },
        "ptype_given_regime_pair": {
            "count_weighted": estimate_ptype_given_regime_pair(edges_supported, weight_col=None),
            "liquidity_weighted": estimate_ptype_given_regime_pair(edges_supported, weight_col="reserve_usd"),
        },
        "fee_given_ptype_and_regime_pair": {
            "count_weighted": estimate_fee_given_ptype_and_regime_pair(edges_supported, weight_col=None),
            "liquidity_weighted": estimate_fee_given_ptype_and_regime_pair(edges_supported, weight_col="reserve_usd"),
        },
        "liquidity_distributions": estimate_liquidity_buckets(edges_supported),
    }

    tokens.to_csv(OUTPUT_DIR / "tokens_with_classifier.csv", index=False)
    observations.to_csv(OUTPUT_DIR / "token_price_observations.csv", index=False)
    edges_supported.to_csv(OUTPUT_DIR / "edges_supported_for_experiment.csv", index=False)
    edges_excluded.to_csv(OUTPUT_DIR / "edges_excluded.csv", index=False)

    excluded_summary = (
        edges_excluded.groupby("dex_id")
        .agg(
            edge_count=("pool_address", "count"),
            reserve_usd_sum=("reserve_usd", "sum"),
        )
        .sort_values("reserve_usd_sum", ascending=False)
        .reset_index()
    )
    excluded_summary.to_csv(OUTPUT_DIR / "excluded_dex_summary.csv", index=False)

    supported_summary = (
        edges_supported.groupby(["regime_pair", "ptype"])
        .agg(
            edge_count=("pool_address", "count"),
            reserve_usd_sum=("reserve_usd", "sum"),
        )
        .reset_index()
    )
    supported_summary.to_csv(OUTPUT_DIR / "supported_ptype_summary.csv", index=False)

    with open(OUTPUT_DIR / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)

    print("Wrote:")
    print(OUTPUT_DIR / "tokens_with_classifier.csv")
    print(OUTPUT_DIR / "token_price_observations.csv")
    print(OUTPUT_DIR / "edges_supported_for_experiment.csv")
    print(OUTPUT_DIR / "edges_excluded.csv")
    print(OUTPUT_DIR / "excluded_dex_summary.csv")
    print(OUTPUT_DIR / "supported_ptype_summary.csv")
    print(OUTPUT_DIR / "calibration.json")

    print("\nP(value_regime | role), count-weighted:")
    print(json.dumps(calibration["value_regime_given_role"]["count_weighted"], indent=2))

    print("\nP(ptype | regime_pair), count-weighted:")
    print(json.dumps(calibration["ptype_given_regime_pair"]["count_weighted"], indent=2))


if __name__ == "__main__":
    main()