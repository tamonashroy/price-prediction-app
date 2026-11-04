"""
Paper-trading rules and reports, with no database or network access.

A trade is bought at its entry price, then sold at whichever comes first: the target,
the stop, or the end of the holding period. Prices are checked candle by candle, so the
result depends on the order things happened in, which daily closing prices cannot show.
"""

from collections import namedtuple

import pandas as pd

from pricepredict import config

Candle = namedtuple("Candle", "start end open high low close")

STAY_OUT = "Stay out: bitcoin below its 30-day average"
TRADE = "Trade"
NO_RUN = "No decision: the pipeline did not finish that day"


def exit_levels(entry_price, take_profit=config.PAPER_TAKE_PROFIT, stop_loss=config.PAPER_STOP_LOSS):
    """Return the (target, stop) prices for a trade."""
    return entry_price * (1 + take_profit), entry_price * (1 - stop_loss)


def net_return_pct(entry_price, exit_price, cost=config.ROUND_TRIP_COST):
    """Percent earned after the cost of buying and selling once."""
    return (exit_price / entry_price - 1 - cost) * 100


def _first_hit(candle, target, stop):
    """Return (price, reason) if this candle reached a level, otherwise None."""
    # A price that opens beyond a level sells at the open, not at the level it jumped past.
    if candle.open <= stop:
        return candle.open, "stop"
    if candle.open >= target:
        return candle.open, "target"
    # A candle doesn't show which came first, so assume the stop: the cautious answer.
    if candle.low <= stop:
        return stop, "stop"
    if candle.high >= target:
        return target, "target"
    return None


def simulate_trade(entry_price, target, stop, entry_at, expires_at, candles, now):
    """
    Work out what has happened to a trade so far.

    Args:
        entry_price, target, stop: the trade's prices
        entry_at, expires_at: when it was bought and when it must be sold by
        candles: Candles in time order. Any outside the trade's life, or still forming, are ignored
        now: the current time

    Returns:
        dict with status, last_price, exit_at, exit_price, exit_reason, and hold_price, the
        last price of the holding period, which is None until the period has ended
    """
    finished = [
        candle for candle in candles
        if candle.start >= entry_at and candle.end <= min(now, expires_at)
    ]
    result = {
        "status": "open",
        "last_price": finished[-1].close if finished else None,
        "exit_at": None,
        "exit_price": None,
        "exit_reason": None,
        "hold_price": finished[-1].close if finished and now >= expires_at else None,
    }
    for candle in finished:
        hit = _first_hit(candle, target, stop)
        if hit:
            result.update(status="closed", exit_at=candle.start, exit_price=hit[0], exit_reason=hit[1])
            return result
    if result["hold_price"] is not None:
        result.update(status="closed", exit_at=finished[-1].end, exit_price=finished[-1].close, exit_reason="time")
    return result


def daily_log(signals, today=None):
    """
    One row per day with what was decided, including days with no decision at all.

    Args:
        signals: signal_date, market_allows_buying, picks_proposed, and any other columns to keep
        today: last date to include. Defaults to the newest signal

    Returns:
        the same columns plus `decision`, newest day first
    """
    if signals.empty:
        return signals.assign(decision=pd.Series(dtype="object"))
    log = signals.assign(signal_date=pd.to_datetime(signals["signal_date"])).set_index("signal_date")
    end = pd.Timestamp(today) if today is not None else log.index.max()
    log = log.reindex(pd.date_range(log.index.min(), end, freq="D"), fill_value=None)
    log.index.name = "signal_date"
    decision = pd.Series(NO_RUN, index=log.index, dtype="object")
    decision[log["market_allows_buying"].eq(True)] = TRADE
    decision[log["market_allows_buying"].eq(False)] = STAY_OUT
    return log.assign(decision=decision).sort_index(ascending=False).reset_index()


def top_n_report(trades, top_ns=config.PAPER_TOP_NS, stake=config.PAPER_STAKE_PER_DAY):
    """
    What buying only the first n picks each day would have earned.

    Every rule invests the same `stake` on every day with picks, split equally across the
    coins it buys, so a rule with more coins does not look better just by spending more.

    Args:
        trades: one row per trade with signal_date, pick_rank, exit_date, exit_reason,
            net_return_pct and hold_return_pct. Open trades have no net_return_pct

    Returns:
        (summary with one row per rule, curve of cumulative profit by exit date for charting)
    """
    summaries, curves = [], []
    if trades.empty:
        return pd.DataFrame(summaries), pd.DataFrame(curves)
    picks_that_day = trades.groupby("signal_date")["pick_rank"].transform("size")
    for top_n in top_ns:
        taken = trades[trades["pick_rank"] <= top_n]
        coins_bought = picks_that_day[taken.index].clip(upper=top_n)
        closed = taken.assign(stake=stake / coins_bought)
        closed = closed[closed["net_return_pct"].notna()].sort_values("exit_date")
        if closed.empty:
            continue
        rule = f"Top {top_n}" if top_n < max(top_ns) else f"All {top_n}"
        profit = closed["stake"] * closed["net_return_pct"] / 100
        by_day = profit.groupby(closed["exit_date"]).sum().cumsum()
        held = closed["hold_return_pct"].notna()
        summaries.append(
            {
                "Rule": rule,
                "trades": len(closed),
                "invested": closed["stake"].sum(),
                "profit": profit.sum(),
                "return_pct": profit.sum() / closed["stake"].sum() * 100,
                "win_rate_pct": (closed["net_return_pct"] > 0).mean() * 100,
                "target_hits": (closed["exit_reason"] == "target").sum(),
                "stop_hits": (closed["exit_reason"] == "stop").sum(),
                "time_exits": (closed["exit_reason"] == "time").sum(),
                "worst_drop": (by_day - by_day.cummax().clip(lower=0)).min(),
                "hold_profit": (closed["stake"] * closed["hold_return_pct"] / 100)[held].sum(),
            }
        )
        curves.append(pd.DataFrame({"Rule": rule, "date": by_day.index, "profit": by_day.to_numpy()}))
    if not summaries:
        return pd.DataFrame(summaries), pd.DataFrame(curves)
    return pd.DataFrame(summaries), pd.concat(curves, ignore_index=True)
