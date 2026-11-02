"""
Turn daily prices into model features and forward-return targets.

Every feature for date D uses only data up to and including D. Only the target
columns look ahead, and they are never used as inputs. tests/test_features.py
checks this by changing future prices and asserting the features do not move.

Indicators are written with pandas rather than pandas-ta or TA-Lib. CoinGecko's
free daily data has no open, high or low, so the extra indicators those libraries
offer could not be calculated here anyway.
"""

import logging

import numpy as np
import pandas as pd

from pricepredict import config

logger = logging.getLogger(__name__)

MARKET_PROXY = "bitcoin"
RANKED_FEATURES = ("ret_7", "ret_30", "vol_30", "rsi_14", "turnover", "log_mcap")


def _rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    # A window with no losses is RSI 100; with no movement at all it stays NaN.
    relative_strength = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + relative_strength)
    return rsi.mask(loss.eq(0) & gain.gt(0), 100.0)


def _coin_features(coin):
    """Features from one coin's own history. `coin` is daily, sorted, with no missing dates."""
    price = coin["price"]
    log_price = np.log(price)
    daily = log_price.diff()
    out = pd.DataFrame(index=coin.index)

    # Momentum and reversal
    for window in (1, 3, 5, 7, 14, 30, 60, 90):
        out[f"ret_{window}"] = log_price.diff(window)

    # Risk
    for window in (7, 14, 30):
        out[f"vol_{window}"] = daily.rolling(window).std()
    out["vol_ratio_7_30"] = out["vol_7"] / out["vol_30"]
    out["sharpe_30"] = daily.rolling(30).mean() / out["vol_30"]
    out["skew_30"] = daily.rolling(30).skew()
    out["max_ret_7"] = daily.rolling(7).max()
    out["min_ret_7"] = daily.rolling(7).min()

    # Trend and mean reversion
    for window in (7, 30, 90):
        out[f"dist_ma_{window}"] = price / price.rolling(window).mean() - 1
    out["dist_high_30"] = price / price.rolling(30).max() - 1
    out["dist_high_90"] = price / price.rolling(90).max() - 1
    out["dist_low_30"] = price / price.rolling(30).min() - 1

    # Classic indicators, scaled by price so coins of any price compare directly
    out["rsi_14"] = _rsi(price)
    macd = price.ewm(span=12, adjust=False).mean() - price.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd / price
    out["macd_hist"] = (macd - signal) / price
    mean_20 = price.rolling(20).mean()
    std_20 = price.rolling(20).std()
    out["bollinger_pct"] = (price - mean_20) / (2 * std_20)
    out["bollinger_width"] = 4 * std_20 / mean_20

    # Liquidity and size. Volume history only goes back as far as CoinGecko's free
    # window, so these are NaN on older rows, which the tree models handle natively.
    volume = coin["total_volume"].where(coin["total_volume"] > 0)
    market_cap = coin["market_cap"].where(coin["market_cap"] > 0)
    out["log_volume"] = np.log(volume)
    out["log_mcap"] = np.log(market_cap)
    out["turnover"] = volume / market_cap
    out["volume_trend_7_30"] = volume.rolling(7).mean() / volume.rolling(30).mean()
    out["volume_z_30"] = (volume - volume.rolling(30).mean()) / volume.rolling(30).std()
    return out


def _complete_daily_history(coin):
    """Reindex to every calendar day, bridging gaps of up to two days."""
    coin = coin.set_index("date").sort_index()
    days = pd.date_range(coin.index.min(), coin.index.max(), freq="D", name="date")
    coin = coin.reindex(days)
    coin["price"] = coin["price"].ffill(limit=2)
    return coin


def remove_broken_coins(prices):
    """
    Drop coins with a one-day price jump too large to be real trading. These are data
    errors or redenominations, and a single one can dominate a whole backtest.
    """
    ordered = prices.sort_values(["coin_id", "date"])
    moves = np.log(ordered["price"]).groupby(ordered["coin_id"]).diff().abs()
    broken = set(ordered.loc[moves > np.log(config.MAX_DAILY_PRICE_RATIO), "coin_id"])
    if broken:
        logger.info("Ignoring %d coins with impossible price jumps: %s", len(broken), ", ".join(sorted(broken)))
    return prices[~prices["coin_id"].isin(broken)]


def remove_price_glitches(prices):
    """
    Drop single days where a price jumps and is back the next day. Nobody could have traded
    at those prices, and leaving them in fakes huge "buy the dip" wins in the backtest.
    This looks one day ahead, so a glitch on the newest date is only caught a day later.
    """
    ordered = prices.sort_values(["coin_id", "date"])
    log_price = np.log(ordered["price"])
    by_coin = log_price.groupby(ordered["coin_id"])
    move_in = by_coin.diff()
    move_out = by_coin.shift(-1) - log_price
    threshold = np.log(config.GLITCH_MOVE_RATIO)
    glitch = (
        (move_in.abs() > threshold)
        & (move_out.abs() > threshold)
        & (np.sign(move_in) != np.sign(move_out))
        & ((move_in + move_out).abs() < np.log(config.GLITCH_RETURN_RATIO))
    )
    if glitch.any():
        logger.info("Ignoring %d one-day price glitches", int(glitch.sum()))
    return ordered[~glitch]


def build_dataset(prices, sentiment=None, horizons=config.HORIZONS):
    """
    Build one row per coin per date.

    Args:
        prices: DataFrame with coin_id, date, price, market_cap, total_volume
        sentiment: Optional DataFrame with date, fear_greed

    Returns:
        (dataset, feature_columns). The dataset also has date, coin_id, price, total_volume,
        target_<h> forward log returns, and excess_target_<h> returns relative to the market.
    """
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    for column in ("price", "market_cap", "total_volume"):
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    prices = remove_price_glitches(remove_broken_coins(prices[prices["price"] > 0]))

    frames = []
    for coin_id, coin in prices.groupby("coin_id", sort=False):
        history = _complete_daily_history(coin[["date", "price", "market_cap", "total_volume"]])
        features = _coin_features(history)
        features["price"] = history["price"]
        features["total_volume"] = history["total_volume"]
        log_price = np.log(history["price"])
        for horizon in horizons:
            features[f"target_{horizon}"] = log_price.shift(-horizon) - log_price
        features["coin_id"] = coin_id
        frames.append(features.reset_index())
    data = pd.concat(frames, ignore_index=True)

    target_columns = [f"target_{horizon}" for horizon in horizons]
    # Models learn to rank coins against the median coin that day. Overall market direction
    # is far harder to forecast, and it would otherwise swamp the differences between coins.
    excess_columns = [f"excess_target_{horizon}" for horizon in horizons]
    data[excess_columns] = data[target_columns].sub(
        data.groupby("date")[target_columns].transform("median")
    ).to_numpy()
    non_features = {"date", "coin_id", "price", "total_volume", *target_columns, *excess_columns}
    data = _add_market_features(data, sentiment)
    data = _add_cross_sectional_ranks(data)

    feature_columns = [column for column in data.columns if column not in non_features]
    data = data[data["ret_1"].notna()].sort_values(["date", "coin_id"]).reset_index(drop=True)
    return data, feature_columns


def _add_market_features(data, sentiment):
    """Add the market backdrop on each date, and each coin's strength relative to it."""
    by_date = data.groupby("date")
    market = pd.DataFrame(
        {
            "market_ret_1": by_date["ret_1"].median(),
            "market_ret_7": by_date["ret_7"].median(),
            "market_ret_30": by_date["ret_30"].median(),
            "market_vol_30": by_date["vol_30"].median(),
            "market_breadth_30": by_date["dist_ma_30"].apply(lambda s: (s.dropna() > 0).mean()),
        }
    )
    proxy = data[data["coin_id"] == MARKET_PROXY].set_index("date")
    if not proxy.empty:
        market["btc_ret_7"] = proxy["ret_7"]
        market["btc_ret_30"] = proxy["ret_30"]
        market["btc_vol_30"] = proxy["vol_30"]
    if sentiment is not None and not sentiment.empty:
        fear_greed = sentiment.copy()
        fear_greed["date"] = pd.to_datetime(fear_greed["date"])
        fear_greed = fear_greed.set_index("date")["fear_greed"].astype(float).sort_index()
        market["fear_greed"] = fear_greed
        market["fear_greed_change_7"] = fear_greed.diff(7)
    data = data.merge(market.reset_index(), on="date", how="left")
    data["rel_ret_7"] = data["ret_7"] - data["market_ret_7"]
    data["rel_ret_30"] = data["ret_30"] - data["market_ret_30"]
    data["day_of_week"] = data["date"].dt.day_of_week.astype(float)
    return data


def _add_cross_sectional_ranks(data):
    """Where a coin stands among all coins that day, which is stable across market regimes."""
    ranks = data.groupby("date")[list(RANKED_FEATURES)].rank(pct=True)
    ranks.columns = [f"{column}_rank" for column in RANKED_FEATURES]
    return pd.concat([data, ranks], axis=1)
