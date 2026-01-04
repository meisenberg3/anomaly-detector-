#!/usr/bin/env python3
"""
Polymarket Anomaly Monitor (NOT "insider trading predictor")

What it does:
- Polls Polymarket Data API /trades (global feed) for large trades
- Flags wallets that look "low-activity" (few markets traded) and/or "fresh"
- Adds simple clustering logic (same wallet hits multiple times within N minutes)
- Emits alerts to console + optional Discord webhook
- Persists state in SQLite to avoid duplicate alerts

Docs used:
- Data API base: https://data-api.polymarket.com  (Endpoints page) 
- /trades query params (limit, filterType=CASH, filterAmount, etc.)
- /activity for wallet history (timestamps + usdcSize)
- /traded for total markets traded by wallet

Run:
  pip install requests
  python polymarket_monitor.py

Optional env vars:
  DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
"""

from __future__ import annotations

import os
import time
import json
import sqlite3
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


DATA_API_BASE = "https://data-api.polymarket.com"


# ---------------------------
# Config (tune these)
# ---------------------------

# Polling
POLL_SECONDS = 8

# Pull this many recent trades each poll
TRADES_LIMIT = 100

# Only look at trades above this threshold (in CASH terms per API filter)
# NOTE: Data API supports filterType=CASH + filterAmount. This is the quickest way
# to ignore noise. Tune as needed (e.g., 1000, 3000, 6000).
MIN_CASH_FILTER = 6000

# Wallet "freshness" threshold: if the wallet's first activity is within this many days
FRESH_DAYS = 14

# "Low activity" threshold: total markets traded <= this number
LOW_ACTIVITY_MARKETS = 6

# Cluster window: if same wallet triggers >= CLUSTER_COUNT within CLUSTER_MINUTES, boost score
CLUSTER_MINUTES = 120
CLUSTER_COUNT = 2

# Score threshold to alert
ALERT_SCORE_THRESHOLD = 6

# Score weights
W_FRESH = 3
W_LOW_ACTIVITY = 2
W_BIG_TRADE = 2
W_CLUSTER = 2

# Big trade tiers (estimated cash) for extra weight
BIG_TRADE_TIER_1 = 10000
BIG_TRADE_TIER_2 = 25000

# Markets filtering (optional): include only if title contains any keyword
# Leave empty list to include all.
INCLUDE_KEYWORDS: List[str] = []  # e.g. ["venezuela", "maduro", "iran", "taiwan"]

# Markets filtering (optional): exclude if title contains any keyword
EXCLUDE_KEYWORDS: List[str] = ["sports"]


# ---------------------------
# Data structures
# ---------------------------

@dataclass
class Trade:
    proxy_wallet: str
    side: str
    condition_id: str
    size: float
    price: float
    timestamp: int
    title: str
    slug: str
    outcome: str
    tx_hash: str

    @property
    def est_cash(self) -> float:
        # /trades doesn't expose usdcSize, but it DOES expose size + price.
        # This estimate is still useful for ranking/thresholding.
        try:
            return float(self.size) * float(self.price)
        except Exception:
            return 0.0


@dataclass
class WalletProfile:
    wallet: str
    traded_markets: Optional[int]
    first_seen_ts: Optional[int]
    recent_cash_24h: Optional[float]


# ---------------------------
# SQLite persistence
# ---------------------------

def db_connect(path: str = "polymarket_monitor.sqlite") -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seen_trades (
            tx_hash TEXT PRIMARY KEY,
            timestamp INTEGER,
            wallet TEXT,
            condition_id TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS wallet_cache (
            wallet TEXT PRIMARY KEY,
            traded_markets INTEGER,
            first_seen_ts INTEGER,
            recent_cash_24h REAL,
            updated_at INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS wallet_hits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wallet TEXT,
            hit_ts INTEGER
        )
    """)
    conn.commit()
    return conn


def db_seen_trade(conn: sqlite3.Connection, tx_hash: str) -> bool:
    cur = conn.execute("SELECT 1 FROM seen_trades WHERE tx_hash = ?", (tx_hash,))
    return cur.fetchone() is not None


def db_mark_trade_seen(conn: sqlite3.Connection, t: Trade) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO seen_trades (tx_hash, timestamp, wallet, condition_id) VALUES (?, ?, ?, ?)",
        (t.tx_hash, t.timestamp, t.proxy_wallet, t.condition_id),
    )
    conn.commit()


def db_get_wallet_cache(conn: sqlite3.Connection, wallet: str, max_age_seconds: int = 6 * 3600) -> Optional[WalletProfile]:
    cur = conn.execute(
        "SELECT traded_markets, first_seen_ts, recent_cash_24h, updated_at FROM wallet_cache WHERE wallet = ?",
        (wallet,),
    )
    row = cur.fetchone()
    if not row:
        return None

    traded_markets, first_seen_ts, recent_cash_24h, updated_at = row
    now = int(time.time())
    if updated_at is not None and (now - int(updated_at)) <= max_age_seconds:
        return WalletProfile(wallet=wallet, traded_markets=traded_markets, first_seen_ts=first_seen_ts, recent_cash_24h=recent_cash_24h)
    return None


def db_set_wallet_cache(conn: sqlite3.Connection, profile: WalletProfile) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO wallet_cache (wallet, traded_markets, first_seen_ts, recent_cash_24h, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            profile.wallet,
            profile.traded_markets,
            profile.first_seen_ts,
            profile.recent_cash_24h,
            int(time.time()),
        ),
    )
    conn.commit()


def db_add_wallet_hit(conn: sqlite3.Connection, wallet: str, hit_ts: int) -> None:
    conn.execute("INSERT INTO wallet_hits (wallet, hit_ts) VALUES (?, ?)", (wallet, hit_ts))
    conn.commit()


def db_count_wallet_hits(conn: sqlite3.Connection, wallet: str, since_ts: int) -> int:
    cur = conn.execute(
        "SELECT COUNT(*) FROM wallet_hits WHERE wallet = ? AND hit_ts >= ?",
        (wallet, since_ts),
    )
    return int(cur.fetchone()[0])


# ---------------------------
# Polymarket API calls
# ---------------------------

_session = requests.Session()
_session.headers.update({"User-Agent": "polymarket-anomaly-monitor/1.0"})


def http_get(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 20) -> Any:
    url = f"{DATA_API_BASE}{path}"
    r = _session.get(url, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_recent_large_trades(limit: int, min_cash_filter: float) -> List[Trade]:
    # Uses the Data API /trades endpoint with filterType=CASH + filterAmount.
    # Docs: /trades query params include filterType and filterAmount, and can be called without user to get global feed.
    data = http_get(
        "/trades",
        params={
            "limit": limit,
            "takerOnly": "true",
            "filterType": "CASH",
            "filterAmount": str(min_cash_filter),
        },
    )

    trades: List[Trade] = []
    for row in data:
        try:
            trades.append(
                Trade(
                    proxy_wallet=row.get("proxyWallet", ""),
                    side=row.get("side", ""),
                    condition_id=row.get("conditionId", ""),
                    size=float(row.get("size", 0)),
                    price=float(row.get("price", 0)),
                    timestamp=int(row.get("timestamp", 0)),
                    title=row.get("title", "") or "",
                    slug=row.get("slug", "") or "",
                    outcome=row.get("outcome", "") or "",
                    tx_hash=row.get("transactionHash", "") or "",
                )
            )
        except Exception:
            continue
    return trades


def fetch_traded_markets_count(wallet: str) -> Optional[int]:
    # Data API /traded returns {"user": "...", "traded": <int>}
    data = http_get("/traded", params={"user": wallet})
    if isinstance(data, dict) and "traded" in data:
        try:
            return int(data["traded"])
        except Exception:
            return None
    return None


def fetch_wallet_first_seen_ts(wallet: str) -> Optional[int]:
    # Data API /activity supports sorting; we request oldest record (ASC) with limit=1
    data = http_get(
        "/activity",
        params={
            "user": wallet,
            "limit": 1,
            "offset": 0,
            "sortBy": "TIMESTAMP",
            "sortDirection": "ASC",
            "type": "TRADE",
        },
    )
    if isinstance(data, list) and data:
        ts = data[0].get("timestamp")
        try:
            return int(ts)
        except Exception:
            return None
    return None


def fetch_wallet_recent_cash_24h(wallet: str) -> Optional[float]:
    # Pull recent trades (DESC), sum usdcSize over last 24h if present.
    now = int(time.time())
    start = now - 24 * 3600
    data = http_get(
        "/activity",
        params={
            "user": wallet,
            "limit": 100,
            "offset": 0,
            "sortBy": "TIMESTAMP",
            "sortDirection": "DESC",
            "type": "TRADE",
            "start": start,
            "end": now,
        },
    )
    if not isinstance(data, list):
        return None

    total = 0.0
    any_usdc = False
    for row in data:
        usdc = row.get("usdcSize")
        if usdc is None:
            continue
        try:
            total += float(usdc)
            any_usdc = True
        except Exception:
            continue
    return total if any_usdc else None


def get_wallet_profile(conn: sqlite3.Connection, wallet: str) -> WalletProfile:
    cached = db_get_wallet_cache(conn, wallet)
    if cached:
        return cached

    traded = fetch_traded_markets_count(wallet)
    first_seen = fetch_wallet_first_seen_ts(wallet)
    recent_cash_24h = fetch_wallet_recent_cash_24h(wallet)

    profile = WalletProfile(
        wallet=wallet,
        traded_markets=traded,
        first_seen_ts=first_seen,
        recent_cash_24h=recent_cash_24h,
    )
    db_set_wallet_cache(conn, profile)
    return profile


# ---------------------------
# Scoring + filtering
# ---------------------------

def title_allowed(title: str) -> bool:
    t = (title or "").lower()
    if INCLUDE_KEYWORDS:
        if not any(k.lower() in t for k in INCLUDE_KEYWORDS):
            return False
    if EXCLUDE_KEYWORDS:
        if any(k.lower() in t for k in EXCLUDE_KEYWORDS):
            return False
    return True


def days_since(ts: Optional[int]) -> Optional[float]:
    if not ts:
        return None
    return (time.time() - ts) / 86400.0


def score_trade(t: Trade, p: WalletProfile, cluster_hits: int) -> Tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []

    # Fresh wallet
    age_days = days_since(p.first_seen_ts)
    if age_days is not None and age_days <= FRESH_DAYS:
        score += W_FRESH
        reasons.append(f"fresh_wallet({age_days:.1f}d)")

    # Low activity
    if p.traded_markets is not None and p.traded_markets <= LOW_ACTIVITY_MARKETS:
        score += W_LOW_ACTIVITY
        reasons.append(f"low_activity(traded={p.traded_markets})")

    # Big trade tiers
    est_cash = t.est_cash
    if est_cash >= BIG_TRADE_TIER_1:
        score += W_BIG_TRADE
        reasons.append(f"big_trade(est_cash≈{est_cash:,.0f})")
    if est_cash >= BIG_TRADE_TIER_2:
        score += 1
        reasons.append("very_big_trade")

    # Cluster
    if cluster_hits >= CLUSTER_COUNT:
        score += W_CLUSTER
        reasons.append(f"cluster(hits={cluster_hits} in {CLUSTER_MINUTES}m)")

    return score, reasons


# ---------------------------
# Alerts
# ---------------------------

def format_alert(t: Trade, p: WalletProfile, score: int, reasons: List[str], cluster_hits: int) -> str:
    age_days = days_since(p.first_seen_ts)
    age_str = "unknown" if age_days is None else f"{age_days:.1f}d"

    traded_str = "unknown" if p.traded_markets is None else str(p.traded_markets)
    cash24_str = "unknown" if p.recent_cash_24h is None else f"${p.recent_cash_24h:,.0f}"

    lines = [
        "🚨 **Polymarket Anomaly Alert**",
        f"**Score:** {score}  |  **Reasons:** {', '.join(reasons) if reasons else 'n/a'}",
        f"**Market:** {t.title}",
        f"**Outcome:** {t.outcome}  |  **Side:** {t.side}",
        f"**Est Cash:** ~${t.est_cash:,.0f}  |  **Price:** {t.price}  |  **Size:** {t.size}",
        f"**Wallet:** `{t.proxy_wallet}` (age={age_str}, traded_markets={traded_str}, cash_24h={cash24_str}, cluster_hits={cluster_hits})",
        f"**Tx:** `{t.tx_hash}`",
        f"**Slug:** {t.slug}",
        f"**ConditionId:** `{t.condition_id}`",
    ]
    return "\n".join(lines)


def send_discord(webhook_url: str, content: str) -> None:
    try:
        r = requests.post(webhook_url, json={"content": content[:1900]}, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"[discord] failed: {e}")


# ---------------------------
# Main loop
# ---------------------------

def main() -> None:
    webhook = os.environ.get("DISCORD_WEBHOOK_URL", "").strip() or None
    conn = db_connect()

    print("Polymarket Anomaly Monitor running.")
    print(f"- Poll every {POLL_SECONDS}s")
    print(f"- /trades CASH filter >= {MIN_CASH_FILTER}")
    print(f"- Fresh <= {FRESH_DAYS} days, Low activity <= {LOW_ACTIVITY_MARKETS} markets")
    print(f"- Cluster window: {CLUSTER_MINUTES}m, count >= {CLUSTER_COUNT}")
    print(f"- Alert score threshold: {ALERT_SCORE_THRESHOLD}")
    if webhook:
        print("- Discord webhook: enabled")
    else:
        print("- Discord webhook: disabled (set DISCORD_WEBHOOK_URL to enable)")

    while True:
        try:
            trades = fetch_recent_large_trades(TRADES_LIMIT, MIN_CASH_FILTER)

            # Process newest last so cluster window and seen logic behave consistently
            trades = sorted(trades, key=lambda x: (x.timestamp, x.tx_hash))

            for t in trades:
                if not t.tx_hash or not t.proxy_wallet or not t.title_allowed if False else False:
                    pass

                if not t.tx_hash or not t.proxy_wallet:
                    continue

                if not title_allowed(t.title):
                    continue

                if db_seen_trade(conn, t.tx_hash):
                    continue

                # Record this as a "hit" regardless; cluster uses hits.
                db_add_wallet_hit(conn, t.proxy_wallet, t.timestamp)

                # Cluster hits in window
                window_start = int(time.time()) - (CLUSTER_MINUTES * 60)
                cluster_hits = db_count_wallet_hits(conn, t.proxy_wallet, window_start)

                # Enrich wallet
                profile = get_wallet_profile(conn, t.proxy_wallet)

                # Score
                score, reasons = score_trade(t, profile, cluster_hits)

                # Decide alert
                if score >= ALERT_SCORE_THRESHOLD:
                    msg = format_alert(t, profile, score, reasons, cluster_hits)
                    print("\n" + msg + "\n" + ("-" * 80))
                    if webhook:
                        send_discord(webhook, msg)

                # Mark trade as seen so we don't reprocess
                db_mark_trade_seen(conn, t)

        except requests.HTTPError as e:
            print(f"[http] {e}")
        except Exception as e:
            print(f"[err] {e}")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
