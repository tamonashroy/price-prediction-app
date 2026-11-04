"""Load daily prices, volume, market cap and market sentiment into Neon."""

import logging
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from psycopg2.extras import execute_values

from pricepredict import config
from pricepredict.db import get_connection
from pricepredict.paper import Candle

logger = logging.getLogger(__name__)

COINGECKO_PUBLIC_URL = "https://api.coingecko.com/api/v3"
FEAR_GREED_URL = "https://api.alternative.me/fng/"
MARKETS_PAGE_SIZE = 250  # CoinGecko's maximum
MARKETS_PAGES = 2  # the top 500 coins comfortably contains the top tradable ones


def _coingecko_get(path, params):
    """GET from CoinGecko, waiting and retrying when rate limited. None if it never succeeds."""
    headers = {}
    api_key = os.getenv("COINGECKO_API_KEY", "").strip()
    if api_key:
        headers["x-cg-demo-api-key"] = api_key
    for _ in range(config.COINGECKO_MAX_RETRIES):
        try:
            response = requests.get(
                f"{COINGECKO_PUBLIC_URL}/{path}", params=params, headers=headers, timeout=30
            )
        except requests.RequestException as error:
            logger.warning("CoinGecko request failed for %s: %s", path, error)
            time.sleep(config.COINGECKO_RETRY_BACKOFF_SECONDS)
            continue
        if response.status_code == 429:
            logger.info("Rate limited on %s; waiting", path)
            time.sleep(config.COINGECKO_RETRY_BACKOFF_SECONDS)
            continue
        if response.status_code != 200:
            logger.warning("CoinGecko returned %s for %s", response.status_code, path)
            return None
        return response.json()
    logger.warning("Giving up on %s after %d retries", path, config.COINGECKO_MAX_RETRIES)
    return None


def load_tradable_symbols():
    """Lowercase symbols that trade against INR on CoinSwitch."""
    pairs = pd.read_csv(config.COINSWITCH_PAIRS_PATH)["coinswitch_pair"]
    return set(pairs.str.split("/").str[0].str.lower())


def select_universe(markets, tradable_symbols):
    """
    Match CoinSwitch symbols to CoinGecko coins. `markets` must be sorted by market cap:
    symbols are not unique, and the largest coin with a symbol is the one meant.

    Returns:
        list of {coin_id, symbol, name} for the largest tradable coins
    """
    universe = []
    seen_symbols = set()
    for coin in markets:
        symbol = coin["symbol"].lower()
        if symbol in seen_symbols:
            continue
        seen_symbols.add(symbol)
        if symbol in tradable_symbols and symbol not in config.EXCLUDED_SYMBOLS:
            universe.append({"coin_id": coin["id"], "symbol": symbol, "name": coin["name"]})
    return universe[:config.UNIVERSE_SIZE]


def fetch_universe():
    """Return the coins to track, read from CoinGecko's market cap ranking."""
    markets = []
    for page in range(1, MARKETS_PAGES + 1):
        rows = _coingecko_get(
            "coins/markets",
            {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": MARKETS_PAGE_SIZE,
                "page": page,
                "sparkline": "false",
            },
        )
        markets.extend(rows or [])
        time.sleep(_sleep_seconds())
    universe = select_universe(markets, load_tradable_symbols())
    if not universe:
        raise RuntimeError("Could not build the coin universe from CoinGecko.")
    return universe


def save_universe(universe):
    """Record which coins can be picked today, replacing yesterday's list."""
    conn = get_connection()
    try:
        with conn, conn.cursor() as cursor:
            cursor.execute("DELETE FROM coins")
            execute_values(
                cursor,
                "INSERT INTO coins (coin_id, symbol, name) VALUES %s",
                [(coin["coin_id"], coin["symbol"], coin["name"]) for coin in universe],
            )
    finally:
        conn.close()


def _sleep_seconds():
    if os.getenv("COINGECKO_API_KEY", "").strip():
        return config.COINGECKO_DEMO_SLEEP_SECONDS
    return config.COINGECKO_SLEEP_SECONDS


def fetch_market_chart(coin_id, days):
    """Return one row per UTC day with price, market_cap and total_volume, or None."""
    data = _coingecko_get(
        f"coins/{coin_id}/market_chart",
        {"vs_currency": "usd", "days": days, "interval": "daily"},
    )
    if not data or not data.get("prices"):
        return None
    return parse_market_chart(data)


def parse_market_chart(data):
    """
    Keep the first point of each day. CoinGecko also appends a partial "now" point
    for the current day, which would mix mid-day prices in with midnight closes.
    """
    frames = [
        pd.DataFrame(data[key], columns=["timestamp", column]).drop_duplicates("timestamp").set_index("timestamp")
        for key, column in (
            ("prices", "price"),
            ("market_caps", "market_cap"),
            ("total_volumes", "total_volume"),
        )
    ]
    df = frames[0].join(frames[1:], how="left").reset_index()
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
    df = df.sort_values("timestamp").drop_duplicates("date", keep="first")
    return df[["date", "price", "market_cap", "total_volume"]].dropna(subset=["price"])


def fetch_hourly_candles(coin_id, days):
    """
    Recent hourly prices as Candles, for coins Binance does not list. CoinGecko gives one
    price per point rather than a high and low, so a brief spike between points is missed.
    """
    time.sleep(_sleep_seconds())
    data = _coingecko_get(f"coins/{coin_id}/market_chart", {"vs_currency": "usd", "days": days})
    candles = []
    for timestamp, price in (data or {}).get("prices", []):
        moment = datetime.fromtimestamp(timestamp / 1000, timezone.utc)
        candles.append(Candle(moment, moment, price, price, price, price))
    return candles


def fetch_latest_prices(coin_ids):
    """Return {coin_id: current USD price} for the coins CoinGecko has a price for."""
    data = _coingecko_get("simple/price", {"ids": ",".join(coin_ids), "vs_currencies": "usd"})
    return {coin_id: quote["usd"] for coin_id, quote in (data or {}).items() if quote.get("usd")}


def _days_to_fetch(last_complete_date, today):
    """Fetch only what is missing, with overlap so late revisions are picked up."""
    if last_complete_date is None:
        return config.MAX_HISTORY_DAYS
    return min(max((today - last_complete_date).days + 2, 2), config.MAX_HISTORY_DAYS)


def _load_last_dates(conn):
    """Latest date per coin that already has volume, so old rows get volume backfilled once."""
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT coin_id, MAX(date) FROM coin_prices WHERE total_volume IS NOT NULL GROUP BY coin_id"
        )
        return dict(cursor.fetchall())


def _upsert_prices(conn, coin_id, coin_name, df):
    # Convert NaN to NULL and numpy values to plain Python types.
    clean = df.astype(object).where(df.notna(), None)
    rows = [
        (coin_id, coin_name, row.date, row.price, row.market_cap, row.total_volume)
        for row in clean.itertuples()
    ]
    with conn, conn.cursor() as cursor:
        execute_values(
            cursor,
            """
            INSERT INTO coin_prices (coin_id, coin_name, date, price, market_cap, total_volume)
            VALUES %s
            ON CONFLICT (coin_id, date) DO UPDATE SET
                coin_name = EXCLUDED.coin_name,
                price = EXCLUDED.price,
                market_cap = EXCLUDED.market_cap,
                total_volume = EXCLUDED.total_volume
            """,
            rows,
            page_size=500,
        )
    return len(rows)


def ingest_prices():
    """Refresh the list of pickable coins and update each one's history."""
    universe = fetch_universe()
    save_universe(universe)
    today = datetime.now(timezone.utc).date()
    conn = get_connection()
    try:
        known = _load_last_dates(conn)
        logger.info("Updating %d coins", len(universe))
        failed = []
        for index, coin in enumerate(universe, start=1):
            coin_id = coin["coin_id"]
            last_date = known.get(coin_id)
            if last_date == today:
                continue
            days = _days_to_fetch(last_date, today)
            df = fetch_market_chart(coin_id, days)
            if df is None or df.empty:
                failed.append(coin_id)
            else:
                count = _upsert_prices(conn, coin_id, coin["name"], df)
                logger.info("[%d/%d] %s: %d rows (asked for %d days)", index, len(universe), coin_id, count, days)
            time.sleep(_sleep_seconds())
        if failed:
            logger.warning("No data for %d coins: %s", len(failed), ", ".join(failed))
    finally:
        conn.close()


def ingest_sentiment():
    """Load the daily Crypto Fear & Greed index. Optional: the models tolerate missing values."""
    try:
        response = requests.get(FEAR_GREED_URL, params={"limit": 0, "format": "json"}, timeout=30)
        response.raise_for_status()
        points = response.json()["data"]
    except (requests.RequestException, KeyError, ValueError) as error:
        logger.warning("Could not fetch Fear & Greed index: %s", error)
        return
    rows = [
        (pd.to_datetime(int(point["timestamp"]), unit="s").date(), int(point["value"]))
        for point in points
    ]
    conn = get_connection()
    try:
        with conn, conn.cursor() as cursor:
            execute_values(
                cursor,
                "INSERT INTO market_sentiment (date, fear_greed) VALUES %s "
                "ON CONFLICT (date) DO UPDATE SET fear_greed = EXCLUDED.fear_greed",
                rows,
            )
    finally:
        conn.close()
    logger.info("Stored %d Fear & Greed values", len(rows))


def check_freshness():
    """Fail loudly if too few coins are up to date, rather than predicting from stale data."""
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=config.FRESHNESS_DAYS)
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(DISTINCT coin_id) FROM coin_prices WHERE date >= %s", (cutoff,))
            fresh = cursor.fetchone()[0]
    finally:
        conn.close()
    if fresh < config.MIN_FRESH_COINS:
        raise RuntimeError(
            f"Only {fresh} coins have prices since {cutoff}; expected at least {config.MIN_FRESH_COINS}."
        )
    logger.info("%d coins have prices since %s", fresh, cutoff)
