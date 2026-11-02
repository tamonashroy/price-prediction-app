"""
Cross-check stored prices against Binance's public data.

CoinGecko is the main source because it covers every coin and reports global volume and
market cap. But its history has contained long stretches of wrong prices (about double
the real price for dozens of coins in early 2026), so prices are verified here. Volume
and market cap are left alone, because Binance reports only one exchange's volume.
"""

import logging
import time
from datetime import datetime, timedelta, timezone

import requests
from psycopg2.extras import execute_values

from pricepredict import config
from pricepredict.db import get_connection

logger = logging.getLogger(__name__)

KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
PAGE_LIMIT = 1000
SLEEP_SECONDS = 0.1
HISTORY_START = datetime(2021, 1, 1, tzinfo=timezone.utc)


def fetch_daily_closes(symbol, start):
    """Return {date: close} for SYMBOLUSDT since `start`. Empty if Binance does not list it."""
    closes = {}
    start_ms = int(start.timestamp() * 1000)
    while True:
        response = requests.get(
            KLINES_URL,
            params={"symbol": f"{symbol.upper()}USDT", "interval": "1d", "startTime": start_ms, "limit": PAGE_LIMIT},
            timeout=30,
        )
        if response.status_code == 400:  # Binance does not list this pair
            return {}
        response.raise_for_status()
        rows = response.json()
        for row in rows:
            # CoinGecko dates a price by the midnight it was taken at, so one day's
            # close is the next day's price. row[6] is the last millisecond of the day.
            closes[datetime.fromtimestamp((row[6] + 1) / 1000, timezone.utc).date()] = float(row[4])
        if len(rows) < PAGE_LIMIT:
            return closes
        start_ms = rows[-1][6] + 1
        time.sleep(SLEEP_SECONDS)


def plan_price_changes(stored, closes, today, add_missing):
    """
    Compare stored prices with Binance closes.

    Args:
        stored: {date: price} already in the database
        closes: {date: close} from Binance
        today: current UTC date; its close is still forming, so it is not used
        add_missing: whether to add dates that are not stored yet

    Returns:
        (rows to insert, rows to correct), each a list of (date, price)
    """
    limit = config.MAX_SOURCE_DISAGREEMENT
    inserts, corrections = [], []
    for day, close in closes.items():
        if day > today or close <= 0:
            continue
        if day not in stored:
            if add_missing:
                inserts.append((day, close))
        elif stored[day] is None or not 1 / limit <= stored[day] / close <= limit:
            corrections.append((day, close))
    return inserts, corrections


def reconcile_prices(lookback_days=None, add_missing=False):
    """
    Correct stored prices that disagree with Binance for every coin in the universe.
    Pass lookback_days to check only recent dates; omit it to check all history.
    """
    today = datetime.now(timezone.utc).date()
    start = HISTORY_START
    if lookback_days is not None:
        start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    conn = get_connection()
    corrected = added = checked = 0
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT coin_id, symbol, name FROM coins")
            coins = cursor.fetchall()
        for coin_id, symbol, name in coins:
            closes = fetch_daily_closes(symbol, start)
            time.sleep(SLEEP_SECONDS)
            if not closes:
                continue
            checked += 1
            with conn, conn.cursor() as cursor:
                cursor.execute(
                    "SELECT date, price FROM coin_prices WHERE coin_id = %s AND date >= %s",
                    (coin_id, start.date()),
                )
                stored = dict(cursor.fetchall())
                # Only extend backwards: new dates are CoinGecko's job, for volume and market cap.
                first_stored = min(stored) if stored else today
                earlier = {day: close for day, close in closes.items() if day < first_stored}
                inserts, _ = plan_price_changes(stored, earlier, today, add_missing)
                _, corrections = plan_price_changes(stored, closes, today, False)
                if inserts:
                    execute_values(
                        cursor,
                        "INSERT INTO coin_prices (coin_id, coin_name, date, price) VALUES %s "
                        "ON CONFLICT (coin_id, date) DO NOTHING",
                        [(coin_id, name, day, price) for day, price in inserts],
                        page_size=1000,
                    )
                if corrections:
                    execute_values(
                        cursor,
                        "UPDATE coin_prices AS p SET price = v.price "
                        "FROM (VALUES %s) AS v(coin_id, date, price) "
                        "WHERE p.coin_id = v.coin_id AND p.date = v.date::date",
                        [(coin_id, day, price) for day, price in corrections],
                        page_size=1000,
                    )
            corrected += len(corrections)
            added += len(inserts)
    finally:
        conn.close()
    logger.info("Checked %d coins against Binance: corrected %d prices, added %d", checked, corrected, added)
