#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import urlencode

import requests

BASE_URL = "https://api.geckoterminal.com/api/v2"
DEFAULT_ACCEPT_HEADER = "application/json;version=20230203"
DEFAULT_TIMEOUT = 30
DEFAULT_MIN_INTERVAL_SECONDS = 6.5  # slightly under 10 requests/minute ceiling
DEFAULT_RETRIES = 5
DEFAULT_BACKOFF_BASE_SECONDS = 2.0
DEFAULT_POOL_DETAIL_BATCH_SIZE = 20
DEFAULT_SORTS = ("h24_volume_usd_desc", "h24_tx_count_desc")
USER_AGENT = "geckoterminal-eth-snapshot/0.1"
MAX_PAGES_PER_DEX_SORT = 10  # start conservative

class GeckoTerminalError(RuntimeError):
    pass


@dataclass
class RateLimiter:
    min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS
    _last_request_ts: float = field(default=0.0, init=False)

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_ts
        remaining = self.min_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_ts = time.monotonic()


class GeckoTerminalClient:
    def __init__(
        self,
        *,
        base_url: str = BASE_URL,
        accept_header: str = DEFAULT_ACCEPT_HEADER,
        timeout: int = DEFAULT_TIMEOUT,
        min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS,
        retries: int = DEFAULT_RETRIES,
        backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
        verbose: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.backoff_base_seconds = backoff_base_seconds
        self.rate_limiter = RateLimiter(min_interval_seconds=min_interval_seconds)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "accept": accept_header,
                "user-agent": USER_AGENT,
            }
        )
        self.verbose = verbose

    def get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_error: Optional[BaseException] = None
        for attempt in range(1, self.retries + 1):
            self.rate_limiter.wait()
            try:
                if self.verbose:
                    logging.info("GET %s?%s", url, urlencode(params or {}, doseq=True))
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    sleep_seconds = float(retry_after) if retry_after else self.backoff_base_seconds * attempt
                    logging.warning("429 from GeckoTerminal for %s; sleeping %.1fs", url, sleep_seconds)
                    time.sleep(sleep_seconds)
                    continue
                if 500 <= resp.status_code < 600:
                    sleep_seconds = self.backoff_base_seconds * attempt
                    logging.warning("%s from GeckoTerminal for %s; retrying in %.1fs", resp.status_code, url, sleep_seconds)
                    time.sleep(sleep_seconds)
                    continue
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                sleep_seconds = self.backoff_base_seconds * attempt
                logging.warning(
                    "Request failed for %s on attempt %s/%s: %s; sleeping %.1fs",
                    url,
                    attempt,
                    self.retries,
                    exc,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
        raise GeckoTerminalError(f"Failed to fetch {url}: {last_error}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=False) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def normalize_address(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value.lower() if value else None


@dataclass
class PoolDiscoveryRecord:
    pool_id: str
    address: str
    name: Optional[str]
    pool_name: Optional[str]
    dex_id: Optional[str]
    dex_name: Optional[str]
    reserve_in_usd: Optional[float]
    volume_usd_h24: Optional[float]
    tx_count_h24: Optional[int]
    base_token_id: Optional[str]
    quote_token_id: Optional[str]
    base_token_address: Optional[str]
    quote_token_address: Optional[str]
    first_seen_sort: str
    first_seen_page: int
    seen_sorts: List[str] = field(default_factory=list)
    source_pages: List[Dict[str, Any]] = field(default_factory=list)
    raw_attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PageSummary:
    dex_id: str
    sort: str
    page: int
    pool_count: int
    new_unique_pools: int
    duplicate_pools: int
    stop_reason: Optional[str] = None


@dataclass
class DexSortSummary:
    dex_id: str
    dex_name: Optional[str]
    sort: str
    pages_scanned: int
    pools_seen: int
    unique_pools_added: int
    stop_reason: str


class EthSnapshotCollector:
    def __init__(
        self,
        client: GeckoTerminalClient,
        *,
        network: str,
        outdir: Path,
        max_pages_per_sort: Optional[int],
        sorts: Sequence[str],
        detail_batch_size: int,
        fetch_token_info: bool,
        token_info_limit: Optional[int],
    ) -> None:
        self.client = client
        self.network = network
        self.outdir = outdir
        self.max_pages_per_sort = max_pages_per_sort
        self.sorts = tuple(sorts)
        self.detail_batch_size = detail_batch_size
        self.fetch_token_info = fetch_token_info
        self.token_info_limit = token_info_limit

    def fetch_all_dexes(self) -> List[Dict[str, Any]]:
        # The endpoint in the prompt uses page=1. We paginate defensively in case
        # GeckoTerminal splits dexes across pages.
        all_rows: List[Dict[str, Any]] = []
        page = 1
        payload = self.client.get_json(f"networks/{self.network}/dexes", params={"page": page})
        rows = payload.get("data") or []
        all_rows.extend(rows)
        return all_rows

    def fetch_pools_for_dex_and_sort(
        self,
        *,
        dex_id: str,
        dex_name: Optional[str],
        sort: str,
    ) -> Tuple[Dict[str, PoolDiscoveryRecord], List[PageSummary], DexSortSummary]:
        discovered: Dict[str, PoolDiscoveryRecord] = {}
        page_summaries: List[PageSummary] = []
        page = 1
        pages_scanned = 0
        pools_seen = 0
        stop_reason = "empty_page"
        previous_page_signature: Optional[Tuple[str, ...]] = None

        while page <= MAX_PAGES_PER_DEX_SORT:
            if self.max_pages_per_sort is not None and page > self.max_pages_per_sort:
                stop_reason = "max_pages_reached"
                break

            payload = self.client.get_json(
                f"networks/{self.network}/dexes/{dex_id}/pools",
                params={
                    "include": "base_token,quote_token,dex",
                    "include_gt_community_data": "false",
                    "page": page,
                    "sort": sort,
                },
            )
            rows = payload.get("data") or []
            if not rows:
                stop_reason = "empty_page"
                break

            page_signature = tuple(sorted((row.get("id") or "") for row in rows))
            if previous_page_signature is not None and page_signature == previous_page_signature:
                stop_reason = "repeated_page_signature"
                break
            previous_page_signature = page_signature

            pools_seen += len(rows)
            pages_scanned += 1
            new_unique = 0
            duplicate = 0

            for row in rows:
                minimal = self._normalize_pool_list_item(row, dex_id=dex_id, dex_name=dex_name, sort=sort, page=page)
                address = minimal.address
                if address not in discovered:
                    discovered[address] = minimal
                    new_unique += 1
                else:
                    duplicate += 1
                    existing = discovered[address]
                    if sort not in existing.seen_sorts:
                        existing.seen_sorts.append(sort)
                    existing.source_pages.append(
                        {
                            "dex_id": dex_id,
                            "sort": sort,
                            "page": page,
                        }
                    )
                    # Keep best observed reserve/volume if a duplicate shows higher values.
                    if (minimal.reserve_in_usd or 0) > (existing.reserve_in_usd or 0):
                        existing.reserve_in_usd = minimal.reserve_in_usd
                    if (minimal.volume_usd_h24 or 0) > (existing.volume_usd_h24 or 0):
                        existing.volume_usd_h24 = minimal.volume_usd_h24
                    if (minimal.tx_count_h24 or 0) > (existing.tx_count_h24 or 0):
                        existing.tx_count_h24 = minimal.tx_count_h24

            page_summaries.append(
                PageSummary(
                    dex_id=dex_id,
                    sort=sort,
                    page=page,
                    pool_count=len(rows),
                    new_unique_pools=new_unique,
                    duplicate_pools=duplicate,
                )
            )
            page += 1

        summary = DexSortSummary(
            dex_id=dex_id,
            dex_name=dex_name,
            sort=sort,
            pages_scanned=pages_scanned,
            pools_seen=pools_seen,
            unique_pools_added=len(discovered),
            stop_reason=stop_reason,
        )
        if page_summaries:
            page_summaries[-1].stop_reason = stop_reason
        return discovered, page_summaries, summary

    def _normalize_pool_list_item(
        self,
        row: Dict[str, Any],
        *,
        dex_id: str,
        dex_name: Optional[str],
        sort: str,
        page: int,
    ) -> PoolDiscoveryRecord:
        attrs = row.get("attributes") or {}
        relationships = row.get("relationships") or {}
        base_rel = ((relationships.get("base_token") or {}).get("data") or {})
        quote_rel = ((relationships.get("quote_token") or {}).get("data") or {})
        dex_rel = ((relationships.get("dex") or {}).get("data") or {})
        address = normalize_address(attrs.get("address"))
        if not address:
            raise ValueError(f"Pool row missing address: {row}")
        transactions_h24 = ((attrs.get("transactions") or {}).get("h24") or {})
        volume_h24 = ((attrs.get("volume_usd") or {}).get("h24"))
        return PoolDiscoveryRecord(
            pool_id=str(row.get("id")),
            address=address,
            name=attrs.get("name"),
            pool_name=attrs.get("pool_name"),
            dex_id=dex_rel.get("id"),
            dex_name=dex_name,
            reserve_in_usd=coerce_float(attrs.get("reserve_in_usd")),
            volume_usd_h24=coerce_float(volume_h24),
            tx_count_h24=(coerce_int(transactions_h24.get("buys")) or 0) + (coerce_int(transactions_h24.get("sells")) or 0),
            base_token_id=base_rel.get("id"),
            quote_token_id=quote_rel.get("id"),
            base_token_address=self._extract_address_from_resource_id(base_rel.get("id")),
            quote_token_address=self._extract_address_from_resource_id(quote_rel.get("id")),
            first_seen_sort=sort,
            first_seen_page=page,
            seen_sorts=[sort],
            source_pages=[{"dex_id": dex_id, "sort": sort, "page": page}],
            raw_attributes=attrs,
        )

    @staticmethod
    def _extract_address_from_resource_id(resource_id: Optional[str]) -> Optional[str]:
        if not resource_id:
            return None
        if "_0x" in resource_id:
            return normalize_address(resource_id.split("_", 1)[1])
        return None

    def fetch_all_discovered_pools(self, dexes: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        pool_map: Dict[str, PoolDiscoveryRecord] = {}
        page_rows: List[Dict[str, Any]] = []
        dex_sort_rows: List[Dict[str, Any]] = []

        for dex in dexes:
            dex_id = str(dex.get("id"))
            dex_name = ((dex.get("attributes") or {}).get("name"))
            logging.info("Scanning dex %s (%s)", dex_id, dex_name)

            for sort in self.sorts:
                sort_pool_map, page_summaries, summary = self.fetch_pools_for_dex_and_sort(
                    dex_id=dex_id,
                    dex_name=dex_name,
                    sort=sort,
                )
                for page_summary in page_summaries:
                    page_rows.append(asdict(page_summary))
                dex_sort_rows.append(asdict(summary))

                for address, record in sort_pool_map.items():
                    if address not in pool_map:
                        pool_map[address] = record
                    else:
                        existing = pool_map[address]
                        if sort not in existing.seen_sorts:
                            existing.seen_sorts.append(sort)
                        existing.source_pages.append({"dex_id": dex_id, "sort": sort, "page": record.first_seen_page})
                        if (record.reserve_in_usd or 0) > (existing.reserve_in_usd or 0):
                            existing.reserve_in_usd = record.reserve_in_usd
                        if (record.volume_usd_h24 or 0) > (existing.volume_usd_h24 or 0):
                            existing.volume_usd_h24 = record.volume_usd_h24
                        if (record.tx_count_h24 or 0) > (existing.tx_count_h24 or 0):
                            existing.tx_count_h24 = record.tx_count_h24

        discovered_rows = [asdict(record) for record in sorted(pool_map.values(), key=lambda r: ((r.dex_id or ""), r.address))]
        return discovered_rows, page_rows, dex_sort_rows

    def fetch_pool_details(self, pool_addresses: Sequence[str]) -> List[Dict[str, Any]]:
        pool_addresses = [normalize_address(addr) for addr in pool_addresses if normalize_address(addr)]
        if not pool_addresses:
            return []

        details: List[Dict[str, Any]] = []
        for batch in chunked(pool_addresses, self.detail_batch_size):
            details.extend(self._fetch_pool_details_batch(batch))
        return details

    def _fetch_pool_details_batch(self, batch: Sequence[str]) -> List[Dict[str, Any]]:
        if not batch:
            return []
        joined_addresses = ",".join(batch)
        try:
            payload = self.client.get_json(
                f"networks/{self.network}/pools/multi/{joined_addresses}",
                params={
                    "include": "base_token,quote_token,dex",
                    "include_volume_breakdown": "false",
                    "include_composition": "true",
                },
            )
        except Exception as exc:
            if len(batch) == 1:
                logging.error("Failed to fetch pool detail for %s: %s", batch[0], exc)
                return [
                    {
                        "pool_address": batch[0],
                        "detail_fetch_error": str(exc),
                    }
                ]
            midpoint = len(batch) // 2
            logging.warning(
                "Batch detail fetch failed for %s pools; splitting into %s and %s",
                len(batch),
                midpoint,
                len(batch) - midpoint,
            )
            return self._fetch_pool_details_batch(batch[:midpoint]) + self._fetch_pool_details_batch(batch[midpoint:])

        included_index = build_included_index(payload.get("included") or [])
        rows = payload.get("data") or []
        normalized_rows: List[Dict[str, Any]] = []
        returned_addresses: set[str] = set()
        for row in rows:
            normalized = normalize_pool_detail(row, included_index)
            returned_addresses.add(normalized.get("pool_address"))
            normalized_rows.append(normalized)

        # Some multi responses may omit bad addresses; preserve visibility.
        for address in batch:
            if address not in returned_addresses:
                normalized_rows.append(
                    {
                        "pool_address": address,
                        "detail_fetch_error": "address_not_returned_by_multi_endpoint",
                    }
                )
        return normalized_rows

    def fetch_token_info_rows(self, token_addresses: Sequence[str]) -> List[Dict[str, Any]]:
        normalized = []
        seen = set()
        for addr in token_addresses:
            addr_norm = normalize_address(addr)
            if addr_norm and addr_norm not in seen:
                seen.add(addr_norm)
                normalized.append(addr_norm)
        if self.token_info_limit is not None:
            normalized = normalized[: self.token_info_limit]

        rows: List[Dict[str, Any]] = []
        for address in normalized:
            try:
                payload = self.client.get_json(f"networks/{self.network}/tokens/{address}/info")
                data = payload.get("data") or {}
                attrs = data.get("attributes") or {}
                rows.append(
                    {
                        "token_address": normalize_address(attrs.get("address") or address),
                        "name": attrs.get("name"),
                        "symbol": attrs.get("symbol"),
                        "decimals": coerce_int(attrs.get("decimals")),
                        "coingecko_coin_id": attrs.get("coingecko_coin_id"),
                        "gt_score": coerce_float(attrs.get("gt_score")),
                        "gt_verified": attrs.get("gt_verified"),
                        "holders_count": coerce_int(((attrs.get("holders") or {}).get("count"))),
                        "holders_distribution_percentage": (attrs.get("holders") or {}).get("distribution_percentage"),
                        "holders_last_updated": (attrs.get("holders") or {}).get("last_updated"),
                        "websites": attrs.get("websites"),
                        "twitter_handle": attrs.get("twitter_handle"),
                        "telegram_handle": attrs.get("telegram_handle"),
                        "discord_url": attrs.get("discord_url"),
                        "farcaster_url": attrs.get("farcaster_url"),
                        "zora_url": attrs.get("zora_url"),
                        "description": attrs.get("description"),
                        "categories": attrs.get("categories"),
                        "gt_category_ids": attrs.get("gt_category_ids"),
                        "is_honeypot": attrs.get("is_honeypot"),
                        "mint_authority": attrs.get("mint_authority"),
                        "freeze_authority": attrs.get("freeze_authority"),
                        "raw": data,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "token_address": address,
                        "error": str(exc),
                    }
                )
        return rows

    def run(self) -> Dict[str, Any]:
        ensure_dir(self.outdir)
        dexes = self.fetch_all_dexes()
        write_json(self.outdir / "dexes_raw.json", dexes)
        write_jsonl(self.outdir / "dexes_raw.jsonl", dexes)

        discovered_rows, page_rows, dex_sort_rows = self.fetch_all_discovered_pools(dexes)
        write_jsonl(self.outdir / "pool_discovery_pages.jsonl", page_rows)
        write_csv(self.outdir / "pool_discovery_pages.csv", page_rows)
        write_jsonl(self.outdir / "dex_sort_summary.jsonl", dex_sort_rows)
        write_csv(self.outdir / "dex_sort_summary.csv", dex_sort_rows)
        write_jsonl(self.outdir / "pools_discovered_minimal.jsonl", discovered_rows)

        pool_addresses = [row["address"] for row in discovered_rows if row.get("address")]
        detailed_rows = self.fetch_pool_details(pool_addresses)
        write_jsonl(self.outdir / "pools_detailed.jsonl", detailed_rows)

        token_rows_from_pool_includes = extract_token_rows_from_pool_details(detailed_rows)
        write_jsonl(self.outdir / "tokens_from_pool_details.jsonl", token_rows_from_pool_includes)

        token_info_rows: List[Dict[str, Any]] = []
        if self.fetch_token_info:
            token_addresses = [row.get("token_address") for row in token_rows_from_pool_includes if row.get("token_address")]
            token_info_rows = self.fetch_token_info_rows(token_addresses)
            write_jsonl(self.outdir / "tokens_info.jsonl", token_info_rows)

        manifest = {
            "network": self.network,
            "generated_at_unix": int(time.time()),
            "config": {
                "base_url": self.client.base_url,
                "accept_header": self.client.session.headers.get("accept"),
                "timeout": self.client.timeout,
                "min_interval_seconds": self.client.rate_limiter.min_interval_seconds,
                "retries": self.client.retries,
                "sorts": list(self.sorts),
                "max_pages_per_sort": self.max_pages_per_sort,
                "detail_batch_size": self.detail_batch_size,
                "fetch_token_info": self.fetch_token_info,
                "token_info_limit": self.token_info_limit,
            },
            "counts": {
                "dex_count": len(dexes),
                "page_rows": len(page_rows),
                "dex_sort_rows": len(dex_sort_rows),
                "unique_pools_discovered": len(discovered_rows),
                "detailed_pool_rows": len(detailed_rows),
                "tokens_from_pool_details": len(token_rows_from_pool_includes),
                "token_info_rows": len(token_info_rows),
            },
            "files": {
                "dexes_raw_json": "dexes_raw.json",
                "dexes_raw_jsonl": "dexes_raw.jsonl",
                "pool_discovery_pages_jsonl": "pool_discovery_pages.jsonl",
                "pool_discovery_pages_csv": "pool_discovery_pages.csv",
                "dex_sort_summary_jsonl": "dex_sort_summary.jsonl",
                "dex_sort_summary_csv": "dex_sort_summary.csv",
                "pools_discovered_minimal_jsonl": "pools_discovered_minimal.jsonl",
                "pools_detailed_jsonl": "pools_detailed.jsonl",
                "tokens_from_pool_details_jsonl": "tokens_from_pool_details.jsonl",
                "tokens_info_jsonl": "tokens_info.jsonl" if self.fetch_token_info else None,
            },
        }
        write_json(self.outdir / "manifest.json", manifest)
        return manifest


def build_included_index(included: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in included:
        row_type = str(row.get("type"))
        row_id = str(row.get("id"))
        index[(row_type, row_id)] = row
    return index


def normalize_pool_detail(row: Dict[str, Any], included_index: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    attrs = row.get("attributes") or {}
    rels = row.get("relationships") or {}

    base_rel = ((rels.get("base_token") or {}).get("data") or {})
    quote_rel = ((rels.get("quote_token") or {}).get("data") or {})
    dex_rel = ((rels.get("dex") or {}).get("data") or {})

    base_included = included_index.get((str(base_rel.get("type")), str(base_rel.get("id"))), {})
    quote_included = included_index.get((str(quote_rel.get("type")), str(quote_rel.get("id"))), {})
    dex_included = included_index.get((str(dex_rel.get("type")), str(dex_rel.get("id"))), {})

    base_attrs = base_included.get("attributes") or {}
    quote_attrs = quote_included.get("attributes") or {}
    dex_attrs = dex_included.get("attributes") or {}

    transactions = attrs.get("transactions") or {}
    volume_usd = attrs.get("volume_usd") or {}
    price_change = attrs.get("price_change_percentage") or {}

    return {
        "pool_id": row.get("id"),
        "pool_address": normalize_address(attrs.get("address")),
        "name": attrs.get("name"),
        "pool_name": attrs.get("pool_name"),
        "pool_fee_percentage": coerce_float(attrs.get("pool_fee_percentage")),
        "pool_created_at": attrs.get("pool_created_at"),
        "reserve_in_usd": coerce_float(attrs.get("reserve_in_usd")),
        "locked_liquidity_percentage": coerce_float(attrs.get("locked_liquidity_percentage")),
        "fdv_usd": coerce_float(attrs.get("fdv_usd")),
        "market_cap_usd": coerce_float(attrs.get("market_cap_usd")),
        "base_token_price_usd": coerce_float(attrs.get("base_token_price_usd")),
        "quote_token_price_usd": coerce_float(attrs.get("quote_token_price_usd")),
        "base_token_price_native_currency": coerce_float(attrs.get("base_token_price_native_currency")),
        "quote_token_price_native_currency": coerce_float(attrs.get("quote_token_price_native_currency")),
        "base_token_price_quote_token": coerce_float(attrs.get("base_token_price_quote_token")),
        "quote_token_price_base_token": coerce_float(attrs.get("quote_token_price_base_token")),
        "base_token_balance": coerce_float(attrs.get("base_token_balance")),
        "quote_token_balance": coerce_float(attrs.get("quote_token_balance")),
        "base_token_liquidity_usd": coerce_float(attrs.get("base_token_liquidity_usd")),
        "quote_token_liquidity_usd": coerce_float(attrs.get("quote_token_liquidity_usd")),
        "price_change_percentage": {
            "m5": coerce_float(price_change.get("m5")),
            "m15": coerce_float(price_change.get("m15")),
            "m30": coerce_float(price_change.get("m30")),
            "h1": coerce_float(price_change.get("h1")),
            "h6": coerce_float(price_change.get("h6")),
            "h24": coerce_float(price_change.get("h24")),
        },
        "transactions": {
            timeframe: {
                metric: coerce_int((transactions.get(timeframe) or {}).get(metric))
                for metric in ("buys", "sells", "buyers", "sellers")
            }
            for timeframe in ("m5", "m15", "m30", "h1", "h6", "h24")
        },
        "volume_usd": {
            timeframe: coerce_float(volume_usd.get(timeframe))
            for timeframe in ("m5", "m15", "m30", "h1", "h6", "h24")
        },
        "dex": {
            "id": dex_rel.get("id"),
            "name": dex_attrs.get("name"),
        },
        "base_token": {
            "id": base_rel.get("id"),
            "address": normalize_address(base_attrs.get("address")),
            "name": base_attrs.get("name"),
            "symbol": base_attrs.get("symbol"),
            "decimals": coerce_int(base_attrs.get("decimals")),
            "image_url": base_attrs.get("image_url"),
            "coingecko_coin_id": base_attrs.get("coingecko_coin_id"),
        },
        "quote_token": {
            "id": quote_rel.get("id"),
            "address": normalize_address(quote_attrs.get("address")),
            "name": quote_attrs.get("name"),
            "symbol": quote_attrs.get("symbol"),
            "decimals": coerce_int(quote_attrs.get("decimals")),
            "image_url": quote_attrs.get("image_url"),
            "coingecko_coin_id": quote_attrs.get("coingecko_coin_id"),
        },
        "raw": row,
    }


def extract_token_rows_from_pool_details(pool_detail_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    token_map: Dict[str, Dict[str, Any]] = {}
    for row in pool_detail_rows:
        for side in ("base_token", "quote_token"):
            token = row.get(side) or {}
            address = normalize_address(token.get("address"))
            if not address:
                continue
            token_map[address] = {
                "token_address": address,
                "name": token.get("name"),
                "symbol": token.get("symbol"),
                "decimals": token.get("decimals"),
                "image_url": token.get("image_url"),
                "coingecko_coin_id": token.get("coingecko_coin_id"),
                "token_id": token.get("id"),
            }
    return [token_map[address] for address in sorted(token_map)]


def chunked(items: Sequence[str], size: int) -> Iterator[List[str]]:
    if size <= 0:
        raise ValueError("size must be positive")
    for idx in range(0, len(items), size):
        yield list(items[idx : idx + size])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an Ethereum GeckoTerminal DEX/pool snapshot.")
    parser.add_argument("--network", default="eth", help="GeckoTerminal network id. Default: eth")
    parser.add_argument("--outdir", default="outputs/geckoterminal_eth_snapshot", help="Output directory")
    parser.add_argument(
        "--sorts",
        nargs="+",
        default=list(DEFAULT_SORTS),
        help="Pool sort orders to scan per dex. Default: h24_volume_usd_desc h24_tx_count_desc",
    )
    parser.add_argument(
        "--max-pages-per-sort",
        type=int,
        default=None,
        help="Optional hard cap on pages scanned per dex/sort",
    )
    parser.add_argument(
        "--detail-batch-size",
        type=int,
        default=DEFAULT_POOL_DETAIL_BATCH_SIZE,
        help="Batch size for /pools/multi requests. Script splits batches on failure.",
    )
    parser.add_argument(
        "--fetch-token-info",
        action="store_true",
        help="Also call /tokens/{address}/info for unique tokens. Expensive under the public rate limit.",
    )
    parser.add_argument(
        "--token-info-limit",
        type=int,
        default=None,
        help="Optional limit on number of token info requests.",
    )
    parser.add_argument(
        "--min-interval-seconds",
        type=float,
        default=DEFAULT_MIN_INTERVAL_SECONDS,
        help="Minimum delay between HTTP requests.",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Request retries")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    client = GeckoTerminalClient(
        timeout=args.timeout,
        min_interval_seconds=args.min_interval_seconds,
        retries=args.retries,
        verbose=args.verbose,
    )
    collector = EthSnapshotCollector(
        client,
        network=args.network,
        outdir=Path(args.outdir),
        max_pages_per_sort=args.max_pages_per_sort,
        sorts=args.sorts,
        detail_batch_size=args.detail_batch_size,
        fetch_token_info=args.fetch_token_info,
        token_info_limit=args.token_info_limit,
    )
    manifest = collector.run()
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
