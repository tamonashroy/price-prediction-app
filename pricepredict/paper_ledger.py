"""Open and settle paper trades in the database, and replay the same rules over the backtest."""

import logging
import math
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from psycopg2.extras import execute_values

from pricepredict import binance, config, ingest, paper
from pricepredict.db import get_connection

logger = logging.getLogger(__name__)

SOURCE_BINANCE = "binance"
SOURCE_COINGECKO = "coingecko"


def _same_coin(price, reference):
    """Binance can list a different coin under the same symbol, which shows up as a far-off price."""
    limit = config.MAX_SOURCE_DISAGREEMENT
    return reference > 0 and 1 / limit <= price / reference <= limit


def load_symbols():
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT coin_id, symbol FROM coins")
            return dict(cursor.fetchall())
    finally:
        conn.close()


def fetch_entry_quotes(picks, symbols):
    """
    Return {coin_id: (price, source)}, the price each pick could be bought at right now.

    Picks are published hours after the midnight prices the model sees, and that midnight
    price can no longer be bought at, so using it would flatter or punish every trade.
    """
    quotes = {}
    for row in picks.itertuples():
        try:
            price = binance.fetch_latest_price(symbols.get(row.coin_id, row.coin_id))
        except requests.RequestException as error:
            logger.warning("Binance price failed for %s: %s", row.coin_id, error)
            price = None
        if price and _same_coin(price, row.price):
            quotes[row.coin_id] = (price, SOURCE_BINANCE)
    missing = picks[~picks["coin_id"].isin(quotes)]
    if not missing.empty:
        latest = ingest.fetch_latest_prices(missing["coin_id"].tolist())
        for row in missing.itertuples():
            if row.coin_id not in latest:
                logger.warning("No live price for %s; using its stored price", row.coin_id)
            quotes[row.coin_id] = (latest.get(row.coin_id, float(row.price)), SOURCE_COINGECKO)
    return quotes


def record_signal(cursor, as_of, now, model_name, market_allows_buying, picks_proposed):
    """Record the day's decision. Returns False if that day has already been decided."""
    cursor.execute(
        "INSERT INTO trade_signals (signal_date, published_at, model_name, market_allows_buying, picks_proposed) "
        "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (signal_date) DO NOTHING",
        (as_of.date(), now, model_name, market_allows_buying, picks_proposed),
    )
    return cursor.rowcount == 1


def open_trades(cursor, picks, as_of, now, model_name, quotes, symbols):
    """Take every pick as a paper trade, at the price quoted when it was published."""
    column = f"log_return_{config.PICK_HORIZON}"
    expires_at = now + timedelta(days=config.PICK_HORIZON)
    rows = []
    for rank, row in enumerate(picks.itertuples(), start=1):
        price, source = quotes[row.coin_id]
        target, stop = paper.exit_levels(price)
        rows.append(
            (
                as_of.date(),
                row.coin_id,
                symbols.get(row.coin_id, row.coin_id),
                rank,
                model_name,
                float(math.expm1(getattr(row, column)) * 100),
                source,
                now,
                expires_at,
                price,
                target,
                stop,
                price,
            )
        )
    if rows:
        execute_values(
            cursor,
            "INSERT INTO paper_trades (signal_date, coin_id, symbol, pick_rank, model_name, predicted_return_pct, "
            "price_source, entry_at, expires_at, entry_price, target_price, stop_price, last_price) VALUES %s",
            rows,
        )
    logger.info("Took %d paper trades for %s", len(rows), as_of.date())


def _trade_candles(coin_id, symbol, price_source, entry_at, end):
    """Hourly prices since the trade opened, from the same source as its entry price if possible."""
    if price_source == SOURCE_BINANCE:
        candles = binance.fetch_candles(symbol, "1h", entry_at, end)
        if candles:
            return candles
    days = max(math.ceil((end - entry_at).total_seconds() / 86400) + 1, 2)  # 2+ days returns hourly points
    return ingest.fetch_hourly_candles(coin_id, days)


def settle_trades(now=None):
    """Update open trades. Closed ones stay here until their full holding period is known too."""
    now = now or datetime.now(timezone.utc)
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id, coin_id, symbol, price_source, entry_at, expires_at, entry_price, target_price, stop_price "
                "FROM paper_trades WHERE status = 'open' OR hold_return_pct IS NULL"
            )
            trades = cursor.fetchall()
        closed = 0
        for trade_id, coin_id, symbol, source, entry_at, expires_at, entry_price, target, stop in trades:
            try:
                candles = _trade_candles(coin_id, symbol, source, entry_at, min(now, expires_at))
            except requests.RequestException as error:
                # Stays open and is worked out from the full price history on the next run.
                logger.warning("Could not update the %s trade: %s", coin_id, error)
                continue
            result = paper.simulate_trade(entry_price, target, stop, entry_at, expires_at, candles, now)
            net = paper.net_return_pct(entry_price, result["exit_price"]) if result["exit_price"] else None
            hold = paper.net_return_pct(entry_price, result["hold_price"]) if result["hold_price"] else None
            with conn, conn.cursor() as cursor:
                # COALESCE: a sold trade stays sold, even if the price history is revised later.
                cursor.execute(
                    """
                    UPDATE paper_trades SET
                        status = CASE WHEN exit_at IS NOT NULL THEN 'closed' ELSE %s END,
                        last_price = COALESCE(%s, last_price),
                        exit_at = COALESCE(exit_at, %s),
                        exit_price = COALESCE(exit_price, %s),
                        exit_reason = COALESCE(exit_reason, %s),
                        net_return_pct = COALESCE(net_return_pct, %s),
                        hold_return_pct = COALESCE(hold_return_pct, %s)
                    WHERE id = %s
                    """,
                    (result["status"], result["last_price"], result["exit_at"], result["exit_price"],
                     result["exit_reason"], net, hold, trade_id),
                )
            closed += result["status"] == "closed"
    finally:
        conn.close()
    logger.info("Checked %d paper trades, %d of them closed", len(trades), closed)


def _stored_candles(prices):
    """Daily stored prices as candles, for coins Binance does not list. Only one price per day."""
    return [
        paper.Candle(day, day, price, price, price, price)
        for day, price in zip(prices.index, prices.to_numpy())
    ]


def build_replay(picks, data, symbols, now=None):
    """
    Apply the paper-trading rules to past picks, using daily highs and lows.

    Args:
        picks: signal_date, coin_id and pick_rank, from backtest.daily_picks
        data: dataset with coin_id, date and price
        symbols: {coin_id: symbol}

    Returns:
        list of tuples matching the replay_trades columns
    """
    now = now or datetime.now(timezone.utc)
    rows = []
    start = picks["signal_date"].min().tz_localize("UTC") if not picks.empty else None
    end = picks["signal_date"].max().tz_localize("UTC") + timedelta(days=config.PICK_HORIZON) if start else None
    for coin_id, coin_picks in picks.groupby("coin_id"):
        symbol = symbols.get(coin_id)
        listed = binance.fetch_candles(symbol, "1d", start, end) if symbol else []
        time.sleep(binance.SLEEP_SECONDS)
        opens = {candle.start: candle.open for candle in listed}
        prices = data[(data["coin_id"] == coin_id) & data["price"].notna()].set_index("date")["price"]
        prices.index = prices.index.tz_localize("UTC")
        stored = _stored_candles(prices.sort_index())
        for pick in coin_picks.itertuples():
            entry_at = pick.signal_date.tz_localize("UTC")
            stored_price = prices.get(entry_at)
            if stored_price is None:
                continue
            binance_open = opens.get(entry_at)
            use_binance = binance_open is not None and _same_coin(binance_open, stored_price)
            entry_price = binance_open if use_binance else float(stored_price)
            target, stop = paper.exit_levels(entry_price)
            expires_at = entry_at + timedelta(days=config.PICK_HORIZON)
            candles = listed if use_binance else stored
            result = paper.simulate_trade(entry_price, target, stop, entry_at, expires_at, candles, now)
            if result["status"] != "closed" or result["hold_price"] is None:
                continue
            rows.append(
                (
                    pick.signal_date.date(),
                    coin_id,
                    int(pick.pick_rank),
                    SOURCE_BINANCE if use_binance else SOURCE_COINGECKO,
                    float(entry_price),
                    pd.Timestamp(result["exit_at"]).date(),
                    float(result["exit_price"]),
                    result["exit_reason"],
                    float(paper.net_return_pct(entry_price, result["exit_price"])),
                    float(paper.net_return_pct(entry_price, result["hold_price"])),
                )
            )
    return rows


def save_replay(rows):
    conn = get_connection()
    try:
        with conn, conn.cursor() as cursor:
            cursor.execute("DELETE FROM replay_trades")
            if rows:
                execute_values(
                    cursor,
                    "INSERT INTO replay_trades (signal_date, coin_id, pick_rank, price_source, entry_price, "
                    "exit_date, exit_price, exit_reason, net_return_pct, hold_return_pct) VALUES %s",
                    rows,
                    page_size=1000,
                )
    finally:
        conn.close()
    logger.info("Stored %d replay trades", len(rows))
