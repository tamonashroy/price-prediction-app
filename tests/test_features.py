import numpy as np
import pandas as pd
import pytest

from pricepredict.features import _rsi, build_dataset, remove_price_glitches


def make_prices(days=200, coins=("bitcoin", "alpha", "beta")):
    rng = np.random.default_rng(7)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    frames = []
    for coin in coins:
        price = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, days)))
        frames.append(
            pd.DataFrame(
                {
                    "coin_id": coin,
                    "date": dates,
                    "price": price,
                    "market_cap": price * 1e6,
                    "total_volume": price * 1e5,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_features_do_not_change_when_future_prices_change():
    prices = make_prices()
    cutoff = pd.Timestamp("2025-05-01")
    changed = prices.copy()
    future = changed["date"] > cutoff
    changed.loc[future, ["price", "market_cap", "total_volume"]] *= 3.0

    original, feature_columns = build_dataset(prices)
    altered, _ = build_dataset(changed)

    before = original[original["date"] <= cutoff].set_index(["date", "coin_id"])[feature_columns]
    after = altered[altered["date"] <= cutoff].set_index(["date", "coin_id"])[feature_columns]
    pd.testing.assert_frame_equal(before, after)


def test_targets_are_forward_log_returns():
    prices = make_prices()
    data, _ = build_dataset(prices)
    coin = data[data["coin_id"] == "alpha"].set_index("date")
    day = pd.Timestamp("2025-03-01")
    expected = np.log(coin.loc[day + pd.Timedelta(days=5), "price"] / coin.loc[day, "price"])
    assert coin.loc[day, "target_5"] == pytest.approx(expected)


def test_targets_are_not_features():
    _, feature_columns = build_dataset(make_prices())
    assert not [column for column in feature_columns if "target_" in column]
    assert "price" not in feature_columns


def test_excess_targets_are_centred_on_the_median_coin_each_day():
    data, _ = build_dataset(make_prices())
    day = data[data["date"] == pd.Timestamp("2025-03-01")]
    assert day["excess_target_5"].median() == pytest.approx(0.0)
    spread = day["target_5"] - day["excess_target_5"]
    assert spread.nunique() == 1 or spread.max() - spread.min() < 1e-12


def test_short_gaps_are_bridged_and_returns_stay_daily():
    prices = make_prices(days=60, coins=("alpha",))
    prices = prices[prices["date"] != pd.Timestamp("2025-02-01")]
    data, _ = build_dataset(prices)
    assert pd.Timestamp("2025-02-01") in set(data["date"])
    assert data["date"].diff().dropna().max() == pd.Timedelta(days=1)


def test_rsi_is_100_when_price_only_rises():
    prices = pd.Series(np.arange(1.0, 40.0))
    assert _rsi(prices).iloc[-1] == 100.0


def test_rsi_stays_between_0_and_100():
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.05, 300))))
    rsi = _rsi(prices).dropna()
    assert rsi.between(0, 100).all()


def test_one_day_glitch_is_removed_but_a_lasting_move_is_kept():
    prices = make_prices(days=60, coins=("alpha", "beta"))
    prices["price"] = 100.0
    glitch_day = pd.Timestamp("2025-02-01")
    prices.loc[(prices["coin_id"] == "alpha") & (prices["date"] == glitch_day), "price"] = 50.0
    prices.loc[(prices["coin_id"] == "beta") & (prices["date"] >= glitch_day), "price"] = 50.0
    cleaned = remove_price_glitches(prices)
    alpha = cleaned[cleaned["coin_id"] == "alpha"]
    beta = cleaned[cleaned["coin_id"] == "beta"]
    assert glitch_day not in set(alpha["date"])
    assert glitch_day in set(beta["date"])
    assert len(beta) == 60


def test_coins_with_impossible_jumps_are_removed():
    prices = make_prices(days=60, coins=("alpha", "broken"))
    is_broken_day = (prices["coin_id"] == "broken") & (prices["date"] == pd.Timestamp("2025-02-01"))
    prices.loc[is_broken_day, "price"] *= 70000
    data, _ = build_dataset(prices)
    assert set(data["coin_id"]) == {"alpha"}


def test_bad_prices_are_ignored():
    prices = make_prices(days=60, coins=("alpha",))
    prices.loc[prices.index[10], "price"] = 0.0
    data, feature_columns = build_dataset(prices)
    assert np.isfinite(data[feature_columns].select_dtypes("number").to_numpy(na_value=0.0)).all()
