"""
The daily job: update data, choose and train models, forecast, pick coins, paper trade, score the past.

    python -m pricepredict.pipeline
    python -m pricepredict.pipeline --skip-ingest --time-budget 60
"""

import argparse
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from psycopg2.extras import execute_values

from pricepredict import backtest, binance, config, ingest, modeling, paper_ledger
from pricepredict.db import apply_migrations, get_connection, get_engine
from pricepredict.features import build_dataset

logger = logging.getLogger("pricepredict.pipeline")


def load_dataset():
    engine = get_engine()
    prices = pd.read_sql("SELECT coin_id, date, price, market_cap, total_volume FROM coin_prices", engine)
    sentiment = pd.read_sql("SELECT date, fear_greed FROM market_sentiment", engine)
    tradable = set(pd.read_sql("SELECT coin_id FROM coins", engine)["coin_id"])
    data, feature_columns = build_dataset(prices, sentiment)
    data["tradable"] = data["coin_id"].isin(tradable)
    return data, feature_columns


def latest_forecast_date(data):
    """The newest date that most coins have, so a few early updates do not define the day."""
    counts = data.groupby("date")["coin_id"].nunique()
    complete = counts[counts >= config.MIN_FRESH_COINS]
    if complete.empty:
        raise RuntimeError(f"No date has prices for at least {config.MIN_FRESH_COINS} coins.")
    return complete.index.max()


def make_forecasts(data, feature_columns, estimator, as_of):
    """Fit one model per horizon and forecast every eligible coin on `as_of`."""
    current = backtest.eligible(data[data["date"] == as_of])
    forecasts = current[["coin_id", "price"]].copy()
    fitted = {}
    for horizon in config.HORIZONS:
        fitted[horizon] = modeling.fit_for_horizon(estimator, data, feature_columns, horizon, as_of)
        forecasts[f"log_return_{horizon}"] = fitted[horizon].predict(current[feature_columns])
    return forecasts.reset_index(drop=True), fitted[config.PICK_HORIZON]


def choose_picks(forecasts, market_allows_buying):
    """The coins forecast to beat the market the most, unless the market filter says stay out."""
    if not market_allows_buying:
        return forecasts.iloc[0:0]
    column = f"log_return_{config.PICK_HORIZON}"
    return forecasts.sort_values(column, ascending=False).head(config.TOP_N)


def save_forecasts(forecasts, as_of, model_name):
    rows = []
    for row in forecasts.itertuples():
        for horizon in config.HORIZONS:
            log_return = getattr(row, f"log_return_{horizon}")
            rows.append(
                (
                    row.coin_id,
                    as_of.date(),
                    (as_of + pd.Timedelta(days=horizon)).date(),
                    float(row.price * np.exp(log_return)),
                    float(np.expm1(log_return) * 100),
                    model_name,
                )
            )
    conn = get_connection()
    try:
        with conn, conn.cursor() as cursor:
            # Replace the whole day so a re-run cannot leave a coin from an earlier attempt.
            cursor.execute("DELETE FROM coin_predictions WHERE prediction_date = %s", (as_of.date(),))
            execute_values(
                cursor,
                "INSERT INTO coin_predictions "
                "(coin_id, prediction_date, target_date, predicted_price, predicted_return_pct, model_name) "
                "VALUES %s",
                rows,
                page_size=1000,
            )
    finally:
        conn.close()
    logger.info("Saved %d forecasts for %d coins", len(rows), len(forecasts))


def publish_picks(picks, as_of, model_name, market_allows_buying):
    """
    Record the day's decision, its picks, and a paper trade for each pick, all or nothing.
    Only the first run for a date counts: that's what was proposed, so re-runs leave it alone.
    """
    now = datetime.now(timezone.utc)
    symbols = paper_ledger.load_symbols()
    quotes = paper_ledger.fetch_entry_quotes(picks, symbols)
    conn = get_connection()
    try:
        with conn, conn.cursor() as cursor:
            if not paper_ledger.record_signal(cursor, as_of, now, model_name, market_allows_buying, len(picks)):
                logger.info("%s was already decided; keeping that decision", as_of.date())
                return False
            save_positions(cursor, picks, as_of, model_name)
            paper_ledger.open_trades(cursor, picks, as_of, now, model_name, quotes, symbols)
    finally:
        conn.close()
    return True


def save_positions(cursor, picks, as_of, model_name):
    column = f"log_return_{config.PICK_HORIZON}"
    close_date = (as_of + pd.Timedelta(days=config.PICK_HORIZON)).date()
    rows = [
        (
            row.coin_id,
            as_of.date(),
            close_date,
            float(row.price),
            float(row.price * np.exp(getattr(row, column))),
            float(row.price),
            "long",
            rank,
            float(np.expm1(getattr(row, column)) * 100),
            model_name,
        )
        for rank, row in enumerate(picks.itertuples(), start=1)
    ]
    if rows:
        execute_values(
            cursor,
            "INSERT INTO positions (coin_id, open_date, close_date, predicted_open_price, "
            "predicted_close_price, actual_open_price, position_type, pick_rank, "
            "predicted_return_pct, model_name) VALUES %s",
            rows,
        )
    logger.info("Opened %d positions for %s", len(rows), as_of.date())


def settle_positions():
    """Fill in real prices and profit, net of trading costs, for positions that have closed."""
    conn = get_connection()
    try:
        with conn, conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE positions p
                SET actual_open_price = o.price,
                    actual_close_price = c.price,
                    profit_loss = (c.price / o.price - 1 - %s) * 100
                FROM coin_prices o, coin_prices c
                WHERE o.coin_id = p.coin_id AND o.date = p.open_date
                  AND c.coin_id = p.coin_id AND c.date = p.close_date
                  AND p.actual_close_price IS NULL
                  AND o.price > 0
                """,
                (config.ROUND_TRIP_COST,),
            )
            logger.info("Settled %d positions", cursor.rowcount)
    finally:
        conn.close()


def run_backtest(data, feature_columns, as_of):
    """
    Replay the past year and store how each strategy did. This uses the fixed default
    settings, because settings tuned on recent data would flatter a replay of that data.
    """
    start = as_of - pd.Timedelta(days=config.BACKTEST_DAYS)
    predictions = backtest.walk_forward_predictions(data, feature_columns, modeling.make_blend(), start, as_of)
    metrics, returns = backtest.evaluate(predictions, data)
    long_returns = returns.reset_index().melt("period_start", var_name="strategy", value_name="return_pct")
    rows = [
        (row.strategy, pd.Timestamp(row.period_start).date(), float(row.return_pct) * 100)
        for row in long_returns.dropna().itertuples()
    ]
    conn = get_connection()
    try:
        with conn, conn.cursor() as cursor:
            cursor.execute("DELETE FROM backtest_periods")
            execute_values(
                cursor,
                "INSERT INTO backtest_periods (strategy, period_start, return_pct) VALUES %s",
                rows,
            )
    finally:
        conn.close()
    logger.info("Backtest: rank IC %.4f (t=%.2f)", metrics["rank_ic"], metrics["rank_ic_t_stat"])
    return metrics, predictions


def run_replay(predictions, data):
    """Apply the paper-trading rules to the backtest's picks. Keeps the last replay if Binance is unavailable."""
    try:
        picks = backtest.daily_picks(predictions, data)
        paper_ledger.save_replay(paper_ledger.build_replay(picks, data, paper_ledger.load_symbols()))
    except requests.RequestException as error:
        logger.warning("Skipped the paper-trading replay: %s", error)


def save_run(as_of, description, metrics, importance):
    conn = get_connection()
    try:
        with conn, conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO model_runs (data_through, model_name, model_config, metrics, feature_importance) "
                "VALUES (%s, %s, %s, %s, %s)",
                (
                    as_of.date(),
                    description["estimator"],
                    json.dumps(description),
                    json.dumps(metrics),
                    json.dumps(importance),
                ),
            )
    finally:
        conn.close()


def run(skip_ingest=False, skip_backtest=False, time_budget=config.AUTOML_TIME_BUDGET_SECONDS):
    apply_migrations()
    if not skip_ingest:
        ingest.ingest_sentiment()
        ingest.ingest_prices()
        try:
            binance.reconcile_prices(lookback_days=config.DAILY_RECONCILE_DAYS)
        except requests.RequestException as error:
            # A check, not a data source: losing it for a day should not stop the forecast.
            logger.warning("Skipped the Binance price check: %s", error)
    ingest.check_freshness()
    paper_ledger.settle_trades()

    data, feature_columns = load_dataset()
    as_of = latest_forecast_date(data)
    logger.info("Forecasting from %s with %d features", as_of.date(), len(feature_columns))

    tree_model, description = modeling.select_model(data, feature_columns, as_of, time_budget=time_budget)
    model_name = f"{description['estimator']}+ridge"
    description["estimator"] = model_name
    forecasts, pick_model = make_forecasts(data, feature_columns, modeling.make_blend(tree_model), as_of)
    market_allows_buying = bool(backtest.market_filter(data).get(as_of, False))
    picks = choose_picks(forecasts, market_allows_buying)

    save_forecasts(forecasts, as_of, model_name)
    publish_picks(picks, as_of, model_name, market_allows_buying)
    settle_positions()

    metrics = None
    if not skip_backtest:
        metrics, predictions = run_backtest(data, feature_columns, as_of)
        run_replay(predictions, data)
    save_run(as_of, description, metrics, modeling.feature_importance(pick_model, feature_columns))
    logger.info("Picks for %s: %s", as_of.date(), ", ".join(picks["coin_id"]) or "none, market filter says stay out")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("flaml").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-ingest", action="store_true", help="use the prices already stored")
    parser.add_argument("--skip-backtest", action="store_true", help="skip the yearly replay")
    parser.add_argument("--time-budget", type=int, default=config.AUTOML_TIME_BUDGET_SECONDS,
                        help="seconds for the AutoML search")
    args = parser.parse_args()
    run(args.skip_ingest, args.skip_backtest, args.time_budget)


if __name__ == "__main__":
    main()
