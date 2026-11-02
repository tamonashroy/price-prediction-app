import numpy as np
import pandas as pd
import pytest

from lightgbm import LGBMRegressor

from pricepredict import backtest, config
from pricepredict.modeling import WalkForwardSplitter, feature_importance, make_blend, rank_ic, training_frame
from pricepredict.pipeline import choose_picks


def dataset(days=30, coins=10):
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    rows = []
    for coin in range(coins):
        for day in dates:
            rows.append(
                {
                    "date": day,
                    "coin_id": "bitcoin" if coin == 0 else f"coin{coin}",
                    "target_5": 0.01 * coin,
                    "excess_target_5": 0.01 * coin,
                    "ret_90": 0.0,
                    "ret_30": float(coin),
                    "ret_7": float(coin),
                    "total_volume": 1e9,
                    "dist_ma_30": 0.1,
                }
            )
    return pd.DataFrame(rows)


def test_training_excludes_rows_whose_outcome_is_not_known_yet():
    data = dataset()
    as_of = pd.Timestamp("2026-01-20")
    rows = training_frame(data, horizon=5, as_of=as_of)
    assert rows["date"].max() == as_of - pd.Timedelta(days=5)


def test_training_ignores_rows_without_an_outcome():
    data = dataset()
    data.loc[data["date"] == "2026-01-10", "excess_target_5"] = np.nan
    rows = training_frame(data, horizon=5, as_of=pd.Timestamp("2026-01-20"))
    assert pd.Timestamp("2026-01-10") not in set(rows["date"])


def test_rank_ic_is_one_for_perfect_ordering_and_minus_one_for_reversed():
    dates = ["2026-01-01"] * 12
    actual = list(range(12))
    assert rank_ic(dates, actual, actual) == pytest.approx(1.0)
    assert rank_ic(dates, actual, actual[::-1]) == pytest.approx(-1.0)


def test_rank_ic_is_zero_without_enough_coins():
    assert rank_ic(["2026-01-01"] * 3, [1, 2, 3], [1, 2, 3]) == 0.0


def test_eligible_removes_illiquid_coins_and_short_histories():
    rows = pd.DataFrame(
        {
            "ret_90": [0.1, np.nan, 0.1, 0.1],
            "total_volume": [1e9, 1e9, 10.0, np.nan],
        }
    )
    assert backtest.eligible(rows).index.tolist() == [0]


def test_model_strategy_earns_the_top_coins_return_minus_costs():
    data = dataset()
    predictions = data[["date", "coin_id"]].assign(actual=data["target_5"], prediction=data["target_5"])
    returns = backtest.strategy_returns(predictions, data, top_n=5, cost=0.01)
    expected = np.expm1(np.array([0.05, 0.06, 0.07, 0.08, 0.09])).mean() - 0.01
    assert returns["model"].iloc[0] == pytest.approx(expected)


def test_model_stays_in_cash_when_the_market_filter_says_so():
    data = dataset().assign(dist_ma_30=-0.1)
    predictions = data[["date", "coin_id"]].assign(actual=data["target_5"], prediction=data["target_5"])
    returns = backtest.strategy_returns(predictions, data, top_n=5, cost=0.01)
    assert (returns["model"] == 0).all()
    assert (returns["model_always_invested"] > 0).all()


def test_rebalances_every_horizon_days():
    data = dataset(days=30)
    predictions = data[["date", "coin_id"]].assign(actual=data["target_5"], prediction=data["target_5"])
    returns = backtest.strategy_returns(predictions, data, top_n=5)
    assert len(returns) == 6
    assert (pd.Series(returns.index).diff().dropna() == pd.Timedelta(days=5)).all()


def test_summarize_measures_total_return_and_drawdown():
    returns = pd.DataFrame({"s": [0.10, -0.50, 0.10]})
    summary = backtest.summarize(returns)["s"]
    assert summary["total_return_pct"] == pytest.approx(-39.5)
    assert summary["max_drawdown_pct"] == pytest.approx(-50.0)
    assert summary["win_rate_pct"] == pytest.approx(66.7)


def test_picks_are_the_highest_forecasts_in_order():
    forecasts = pd.DataFrame(
        {
            "coin_id": ["a", "b", "c"],
            "price": [1.0, 1.0, 1.0],
            f"log_return_{config.PICK_HORIZON}": [0.02, -0.01, 0.05],
        }
    )
    assert choose_picks(forecasts, True)["coin_id"].tolist() == ["c", "a", "b"]


def test_no_picks_when_the_market_filter_says_stay_out():
    forecasts = pd.DataFrame(
        {"coin_id": ["a"], "price": [1.0], f"log_return_{config.PICK_HORIZON}": [0.2]}
    )
    assert choose_picks(forecasts, False).empty


def test_picks_are_limited_to_top_n():
    forecasts = pd.DataFrame(
        {
            "coin_id": [f"c{i}" for i in range(20)],
            "price": [1.0] * 20,
            f"log_return_{config.PICK_HORIZON}": [0.5] * 20,
        }
    )
    assert len(choose_picks(forecasts, True)) == config.TOP_N


def test_blend_fits_with_missing_values_and_reports_importance():
    rng = np.random.default_rng(3)
    columns = ["x", "y"]
    X = pd.DataFrame(rng.normal(size=(400, 2)), columns=columns)
    target = X["x"] * 0.5 + rng.normal(0, 0.1, 400)
    X.loc[X.index[:50], "y"] = np.nan
    model = make_blend(LGBMRegressor(n_estimators=20, min_child_samples=5, verbose=-1)).fit(X, target)
    assert np.isfinite(model.predict(X)).all()
    importance = feature_importance(model, columns)
    assert max(importance, key=importance.get) == "x"


def test_walk_forward_folds_never_train_on_or_next_to_validation_dates():
    dates = pd.date_range("2024-01-01", periods=400, freq="D").repeat(3)
    splitter = WalkForwardSplitter(dates, horizon=5, n_splits=4, fold_days=30)
    folds = list(splitter.split())
    assert len(folds) == 4
    for train_index, valid_index in folds:
        gap = dates[valid_index].min() - dates[train_index].max()
        assert gap > pd.Timedelta(days=5)
    assert dates[folds[-1][1]].max() == dates.max()
