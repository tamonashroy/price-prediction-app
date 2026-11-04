"""
Walk-forward backtest: replay the past as if trading it live.

The model is retrained on a schedule using only outcomes known at that time, then
buys the top coins, holds for the pick horizon, sells, and pays trading costs.
Results are compared with simple strategies, because a model that cannot beat
"buy everything" or "hold bitcoin" adds nothing.

The coin universe is today's largest coins, so coins that collapsed and dropped out
are underrepresented. This flatters every strategy, including the baselines, so
compare strategies with each other rather than reading returns as achievable.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.base import clone

from pricepredict import config
from pricepredict.features import MARKET_PROXY
from pricepredict.modeling import actual_return_column, clip_target, rank_ic, target_column, training_frame

logger = logging.getLogger(__name__)

PERIODS_PER_YEAR = 365 / config.PICK_HORIZON


def eligible(rows):
    """Coins that can realistically be bought: on the exchange, with history and real trading volume."""
    keep = rows["ret_90"].notna() & (rows["total_volume"].fillna(0) >= config.MIN_DAILY_VOLUME_USD)
    if "tradable" in rows:
        keep &= rows["tradable"]
    return rows[keep]


def walk_forward_predictions(data, feature_columns, estimator, start, end,
                             horizon=config.PICK_HORIZON,
                             retrain_days=config.BACKTEST_RETRAIN_EVERY_DAYS,
                             train_window_days=config.TRAIN_WINDOW_DAYS):
    """Predict every date in [start, end] with a model that only saw earlier outcomes."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    target = target_column(horizon)
    actual = actual_return_column(horizon)
    results = []
    for block_start in pd.date_range(start, end, freq=f"{retrain_days}D"):
        block_end = min(block_start + pd.Timedelta(days=retrain_days - 1), end)
        train = training_frame(data, horizon, block_start, train_window_days)
        block = data[(data["date"] >= block_start) & (data["date"] <= block_end) & data[actual].notna()]
        block = eligible(block)
        if train.empty or block.empty:
            continue
        model = clone(estimator)
        model.fit(train[feature_columns], clip_target(train[target]))
        results.append(
            block[["date", "coin_id", actual]].assign(prediction=model.predict(block[feature_columns]))
        )
    return pd.concat(results, ignore_index=True).rename(columns={actual: "actual"})


def market_filter(data):
    """True on dates when bitcoin is above its 30-day average, the only time the strategy buys."""
    bitcoin = data[data["coin_id"] == MARKET_PROXY].set_index("date")["dist_ma_30"]
    return bitcoin > 0


def daily_picks(predictions, data, top_n=config.TOP_N):
    """The coins that would have been proposed on each date the market filter allowed buying."""
    buy_allowed = market_filter(data)
    allowed = predictions[predictions["date"].map(buy_allowed).fillna(False).astype(bool)]
    ranked = allowed.sort_values(["date", "prediction", "coin_id"], ascending=[True, False, True])
    picks = ranked.groupby("date").head(top_n)
    return picks.assign(
        signal_date=picks["date"], pick_rank=picks.groupby("date").cumcount() + 1
    )[["signal_date", "coin_id", "pick_rank"]].reset_index(drop=True)


def _picks_return(day, score_column, top_n, cost, ascending=False):
    """Average net simple return of the top coins that day."""
    ranked = day.sort_values(score_column, ascending=ascending)
    return np.expm1(ranked.head(top_n)["actual"]).mean() - cost


def strategy_returns(predictions, data, offset=0, horizon=config.PICK_HORIZON,
                     top_n=config.TOP_N, cost=config.ROUND_TRIP_COST):
    """
    Per-period net returns of the model and baselines, rebalancing every `horizon` days.
    `offset` shifts the start day: results that change a lot with it are luck.
    """
    frame = predictions.merge(
        data[["date", "coin_id", "ret_30", "ret_7"]], on=["date", "coin_id"], how="left"
    )
    dates = sorted(frame["date"].unique())[offset::horizon]
    bitcoin = data[data["coin_id"] == MARKET_PROXY].set_index("date")[actual_return_column(horizon)]
    buy_allowed = market_filter(data)
    rows = []
    for current in dates:
        day = frame[frame["date"] == current]
        if len(day) < top_n:
            continue
        always_invested = _picks_return(day, "prediction", top_n, cost)
        rows.append(
            {
                "period_start": current,
                "model": always_invested if buy_allowed.get(current, False) else 0.0,
                "model_always_invested": always_invested,
                "equal_weight_all": np.expm1(day["actual"]).mean() - cost,
                "hold_bitcoin": float(np.expm1(bitcoin.get(current, 0.0))),
                "momentum_30d": _picks_return(day, "ret_30", top_n, cost),
                "reversal_7d": _picks_return(day, "ret_7", top_n, cost, ascending=True),
            }
        )
    return pd.DataFrame(rows).set_index("period_start")


def summarize(returns):
    """Headline numbers for each strategy column of `strategy_returns`."""
    summary = {}
    for name in returns.columns:
        series = returns[name].dropna()
        growth = (1 + series).cumprod()
        std = series.std()
        summary[name] = {
            "total_return_pct": round(float(growth.iloc[-1] - 1) * 100, 2),
            "mean_period_pct": round(float(series.mean()) * 100, 3),
            "sharpe": round(float(series.mean() / std * np.sqrt(PERIODS_PER_YEAR)), 2) if std > 0 else 0.0,
            "max_drawdown_pct": round(float((growth / growth.cummax().clip(lower=1) - 1).min()) * 100, 2),
            # Periods sitting in cash are neither wins nor losses.
            "win_rate_pct": round(float((series[series != 0] > 0).mean()) * 100, 1),
        }
    return summary


def evaluate(predictions, data, horizon=config.PICK_HORIZON):
    """Score predictions across every start offset, so one lucky start day cannot decide it."""
    offsets = [strategy_returns(predictions, data, offset=offset, horizon=horizon) for offset in range(horizon)]
    per_offset = [summarize(returns) for returns in offsets]
    strategies = {}
    for name in per_offset[0]:
        totals = [summary[name]["total_return_pct"] for summary in per_offset]
        strategies[name] = {
            metric: round(float(np.mean([summary[name][metric] for summary in per_offset])), 3)
            for metric in per_offset[0][name]
        }
        strategies[name]["worst_offset_total_pct"] = round(min(totals), 2)
        strategies[name]["best_offset_total_pct"] = round(max(totals), 2)
    return {
        "rank_ic": round(rank_ic(predictions["date"], predictions["actual"], predictions["prediction"]), 4),
        "rank_ic_t_stat": round(_ic_t_stat(predictions, horizon), 2),
        "periods": int(len(offsets[0])),
        "start": str(predictions["date"].min().date()),
        "end": str(predictions["date"].max().date()),
        "cost_round_trip_pct": config.ROUND_TRIP_COST * 100,
        "strategies": strategies,
    }, offsets[0]


def _ic_t_stat(predictions, horizon):
    """t-statistic of daily IC, using non-overlapping dates so it is not overstated."""
    dates = sorted(predictions["date"].unique())[::horizon]
    values = []
    for current in dates:
        day = predictions[predictions["date"] == current]
        if len(day) >= 10:
            value = day["actual"].corr(day["prediction"], method="spearman")
            if not np.isnan(value):
                values.append(value)
    if len(values) < 2 or np.std(values, ddof=1) == 0:
        return 0.0
    return float(np.mean(values) / (np.std(values, ddof=1) / np.sqrt(len(values))))
