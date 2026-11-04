from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from pricepredict import backtest, paper, paper_ledger, pipeline
from pricepredict.paper import Candle

ENTRY_AT = datetime(2026, 11, 1, 9, 30, tzinfo=timezone.utc)
EXPIRES_AT = ENTRY_AT + timedelta(days=5)
LATER = EXPIRES_AT + timedelta(days=1)


def candle(hours_after_entry, open_=100.0, high=101.0, low=99.0, close=100.0):
    start = ENTRY_AT.replace(minute=0) + timedelta(hours=hours_after_entry)
    return Candle(start, start + timedelta(hours=1), open_, high, low, close)


def simulate(candles, now=LATER):
    return paper.simulate_trade(100.0, 110.0, 95.0, ENTRY_AT, EXPIRES_AT, candles, now)


def test_levels_use_the_configured_target_and_stop():
    target, stop = paper.exit_levels(200.0, take_profit=0.10, stop_loss=0.05)
    assert target == pytest.approx(220.0)
    assert stop == pytest.approx(190.0)


def test_return_includes_the_trading_cost():
    assert paper.net_return_pct(100.0, 110.0, cost=0.01) == pytest.approx(9.0)


def test_sells_at_the_target():
    result = simulate([candle(1), candle(2, high=112.0), candle(3, low=80.0)])
    assert (result["status"], result["exit_reason"], result["exit_price"]) == ("closed", "target", 110.0)
    assert result["exit_at"] == candle(2).start


def test_sells_at_the_stop():
    result = simulate([candle(1), candle(2, low=94.0), candle(3, high=130.0)])
    assert (result["exit_reason"], result["exit_price"]) == ("stop", 95.0)


def test_assumes_the_stop_when_one_candle_touches_both_levels():
    result = simulate([candle(1, high=115.0, low=90.0)])
    assert (result["exit_reason"], result["exit_price"]) == ("stop", 95.0)


def test_a_price_gap_sells_at_the_open_rather_than_the_stop():
    result = simulate([candle(1, open_=80.0, high=81.0, low=79.0, close=80.0)])
    assert (result["exit_reason"], result["exit_price"]) == ("stop", 80.0)


def test_a_gap_up_sells_at_the_open():
    result = simulate([candle(1, open_=120.0, high=125.0, low=119.0, close=120.0)])
    assert (result["exit_reason"], result["exit_price"]) == ("target", 120.0)


def test_sold_at_the_final_price_when_time_runs_out():
    final = candle(118, close=103.0)
    result = simulate([candle(1), final])
    assert (result["status"], result["exit_reason"], result["exit_price"]) == ("closed", "time", 103.0)
    assert result["exit_at"] == final.end


def test_never_held_beyond_the_time_limit():
    result = simulate([candle(1, close=103.0), candle(121, high=200.0, close=200.0)])
    assert (result["exit_reason"], result["exit_price"]) == ("time", 103.0)


def test_stays_open_before_the_time_limit():
    now = ENTRY_AT + timedelta(days=2)
    result = simulate([candle(1, close=104.0)], now=now)
    assert result["status"] == "open"
    assert result["last_price"] == 104.0
    assert result["hold_price"] is None


def test_stays_open_if_time_is_up_but_no_prices_arrived():
    assert simulate([])["status"] == "open"


def test_ignores_prices_from_before_the_trade():
    assert simulate([candle(-2, high=500.0, low=1.0), candle(0, high=500.0, low=1.0)])["status"] == "open"


def test_ignores_a_candle_that_has_not_finished():
    forming = candle(1, high=500.0)
    now = forming.start + timedelta(minutes=10)
    assert simulate([forming], now=now)["status"] == "open"


def test_full_period_price_is_known_after_an_early_sale():
    before = simulate([candle(1, high=112.0), candle(2, close=90.0)], now=ENTRY_AT + timedelta(days=1))
    after = simulate([candle(1, high=112.0), candle(2, close=90.0)])
    assert before["exit_reason"] == "target" and before["hold_price"] is None
    assert after["exit_reason"] == "target" and after["hold_price"] == 90.0


def test_daily_log_explains_every_day_including_missed_ones():
    signals = pd.DataFrame(
        {
            "signal_date": ["2026-11-01", "2026-11-03", "2026-11-04"],
            "market_allows_buying": [False, True, False],
            "picks_proposed": [0, 5, 0],
        }
    )
    log = paper.daily_log(signals, today="2026-11-05")
    assert log["signal_date"].dt.strftime("%m-%d").tolist() == ["11-05", "11-04", "11-03", "11-02", "11-01"]
    assert log["decision"].tolist() == [paper.NO_RUN, paper.STAY_OUT, paper.TRADE, paper.NO_RUN, paper.STAY_OUT]


def test_daily_log_handles_no_signals():
    log = paper.daily_log(pd.DataFrame(columns=["signal_date", "market_allows_buying", "picks_proposed"]))
    assert log.empty and "decision" in log


def trades_frame(rows):
    return pd.DataFrame(
        rows, columns=["signal_date", "pick_rank", "exit_date", "exit_reason", "net_return_pct", "hold_return_pct"]
    )


def test_each_rule_splits_the_same_stake_across_the_coins_it_buys():
    trades = trades_frame(
        [
            ("2026-11-01", 1, "2026-11-02", "target", 10.0, 2.0),
            ("2026-11-01", 2, "2026-11-03", "stop", -5.0, -3.0),
            ("2026-11-01", 3, "2026-11-06", "time", 1.0, 1.0),
        ]
    )
    summary, _ = paper.top_n_report(trades, top_ns=(1, 2, 3), stake=100)
    summary = summary.set_index("Rule")
    assert summary.loc["Top 1", "profit"] == pytest.approx(10.0)
    assert summary.loc["Top 2", "profit"] == pytest.approx(50 * 0.10 + 50 * -0.05)
    assert summary.loc["All 3", "profit"] == pytest.approx((10 - 5 + 1) * 100 / 3 / 100)
    assert summary["invested"].tolist() == pytest.approx([100.0, 100.0, 100.0])
    assert summary.loc["Top 2", "hold_profit"] == pytest.approx(50 * 0.02 + 50 * -0.03)
    assert summary.loc["All 3", ["target_hits", "stop_hits", "time_exits"]].tolist() == [1, 1, 1]


def test_rules_split_by_coins_available_when_a_day_has_fewer_picks():
    trades = trades_frame([("2026-11-01", 1, "2026-11-02", "target", 10.0, 2.0)])
    summary, _ = paper.top_n_report(trades, top_ns=(1, 5), stake=100)
    assert summary["profit"].tolist() == pytest.approx([10.0, 10.0])


def test_open_trades_keep_their_share_of_the_stake_but_earn_nothing_yet():
    trades = trades_frame(
        [
            ("2026-11-01", 1, "2026-11-02", "target", 10.0, np.nan),
            ("2026-11-01", 2, None, None, np.nan, np.nan),
        ]
    )
    summary, _ = paper.top_n_report(trades, top_ns=(2,), stake=100)
    assert summary.iloc[0]["trades"] == 1
    assert summary.iloc[0]["profit"] == pytest.approx(5.0)
    assert summary.iloc[0]["hold_profit"] == 0


def test_curve_accumulates_by_exit_date_and_tracks_the_worst_drop():
    trades = trades_frame(
        [
            ("2026-11-01", 1, "2026-11-02", "target", 10.0, 2.0),
            ("2026-11-02", 1, "2026-11-04", "stop", -5.0, 1.0),
            ("2026-11-03", 1, "2026-11-05", "stop", -5.0, 1.0),
        ]
    )
    summary, curve = paper.top_n_report(trades, top_ns=(1,), stake=100)
    assert curve["profit"].tolist() == pytest.approx([10.0, 5.0, 0.0])
    assert summary.iloc[0]["worst_drop"] == pytest.approx(-10.0)


def test_report_is_empty_without_closed_trades():
    assert paper.top_n_report(trades_frame([]))[0].empty
    only_open = trades_frame([("2026-11-01", 1, None, None, np.nan, np.nan)])
    assert paper.top_n_report(only_open)[0].empty


def test_daily_picks_ranks_by_forecast_and_respects_the_market_filter():
    dates = pd.to_datetime(["2026-01-01", "2026-01-02"])
    data = pd.DataFrame(
        {"date": dates, "coin_id": "bitcoin", "dist_ma_30": [0.1, -0.1]}
    )
    predictions = pd.DataFrame(
        {
            "date": dates.repeat(3),
            "coin_id": ["a", "b", "c"] * 2,
            "prediction": [0.1, 0.3, 0.2] * 2,
            "actual": [0.0] * 6,
        }
    )
    picks = backtest.daily_picks(predictions, data, top_n=2)
    assert picks["coin_id"].tolist() == ["b", "c"]
    assert picks["pick_rank"].tolist() == [1, 2]
    assert picks["signal_date"].unique().tolist() == [dates[0]]


class FakeCursor:
    def __init__(self, rowcount):
        self.rowcount = rowcount

    def execute(self, sql, params=None):
        pass


@pytest.mark.parametrize("rowcount, expected", [(1, True), (0, False)])
def test_only_the_first_decision_for_a_day_counts(rowcount, expected):
    now = datetime.now(timezone.utc)
    recorded = paper_ledger.record_signal(FakeCursor(rowcount), pd.Timestamp("2026-11-04"), now, "m", True, 5)
    assert recorded is expected


class FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self):
        return self

    def close(self):
        pass


def test_a_rerun_does_not_replace_the_days_picks_or_trades(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "get_connection", FakeConnection)
    monkeypatch.setattr(paper_ledger, "load_symbols", lambda: {})
    monkeypatch.setattr(paper_ledger, "fetch_entry_quotes", lambda picks, symbols: {})
    monkeypatch.setattr(paper_ledger, "record_signal", lambda *args: False)
    monkeypatch.setattr(pipeline, "save_positions", lambda *args: calls.append("positions"))
    monkeypatch.setattr(paper_ledger, "open_trades", lambda *args: calls.append("trades"))
    assert pipeline.publish_picks(pd.DataFrame({"coin_id": []}), pd.Timestamp("2026-11-04"), "m", True) is False
    assert calls == []


def test_first_run_saves_picks_and_trades_together(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "get_connection", FakeConnection)
    monkeypatch.setattr(paper_ledger, "load_symbols", lambda: {})
    monkeypatch.setattr(paper_ledger, "fetch_entry_quotes", lambda picks, symbols: {})
    monkeypatch.setattr(paper_ledger, "record_signal", lambda *args: True)
    monkeypatch.setattr(pipeline, "save_positions", lambda *args: calls.append("positions"))
    monkeypatch.setattr(paper_ledger, "open_trades", lambda *args: calls.append("trades"))
    assert pipeline.publish_picks(pd.DataFrame({"coin_id": []}), pd.Timestamp("2026-11-04"), "m", True) is True
    assert calls == ["positions", "trades"]


def test_a_binance_price_far_from_ours_is_treated_as_a_different_coin(monkeypatch):
    picks = pd.DataFrame({"coin_id": ["real", "clash"], "price": [100.0, 100.0]})
    prices = {"REAL": 104.0, "CLASH": 3.0}
    monkeypatch.setattr(paper_ledger.binance, "fetch_latest_price", lambda symbol: prices[symbol])
    monkeypatch.setattr(paper_ledger.ingest, "fetch_latest_prices", lambda ids: {"clash": 101.0})
    quotes = paper_ledger.fetch_entry_quotes(picks, {"real": "REAL", "clash": "CLASH"})
    assert quotes == {"real": (104.0, "binance"), "clash": (101.0, "coingecko")}


def test_entry_falls_back_to_the_stored_price_when_no_live_price_exists(monkeypatch):
    picks = pd.DataFrame({"coin_id": ["a"], "price": [100.0]})
    monkeypatch.setattr(paper_ledger.binance, "fetch_latest_price", lambda symbol: None)
    monkeypatch.setattr(paper_ledger.ingest, "fetch_latest_prices", lambda ids: {})
    assert paper_ledger.fetch_entry_quotes(picks, {}) == {"a": (100.0, "coingecko")}


def test_replay_uses_binance_highs_and_lows_and_falls_back_to_stored_prices(monkeypatch):
    dates = pd.date_range("2026-01-01", periods=8, freq="D")
    data = pd.concat(
        [
            pd.DataFrame({"date": dates, "coin_id": "listed", "price": 100.0}),
            pd.DataFrame({"date": dates, "coin_id": "unlisted", "price": [100.0, 96.0, 94.0, 90.0, 90.0, 90.0, 90.0, 90.0]}),
        ]
    )
    day = pd.Timestamp("2026-01-01", tz="UTC")
    listed = [
        Candle(day, day + timedelta(days=1), 100.0, 102.0, 99.0, 101.0),
        Candle(day + timedelta(days=1), day + timedelta(days=2), 101.0, 112.0, 100.0, 105.0),
        Candle(day + timedelta(days=2), day + timedelta(days=3), 105.0, 106.0, 104.0, 105.0),
        Candle(day + timedelta(days=3), day + timedelta(days=4), 105.0, 106.0, 104.0, 105.0),
        Candle(day + timedelta(days=4), day + timedelta(days=5), 105.0, 106.0, 104.0, 103.0),
    ]
    monkeypatch.setattr(paper_ledger.binance, "fetch_candles", lambda symbol, *args: listed if symbol == "LST" else [])
    monkeypatch.setattr(paper_ledger.time, "sleep", lambda seconds: None)
    picks = pd.DataFrame(
        {"signal_date": [dates[0], dates[0]], "coin_id": ["listed", "unlisted"], "pick_rank": [1, 2]}
    )
    rows = paper_ledger.build_replay(picks, data, {"listed": "LST", "unlisted": "UNL"})
    by_coin = {row[1]: row for row in rows}

    assert by_coin["listed"][3] == "binance"
    assert by_coin["listed"][7] == "target"
    assert by_coin["listed"][5] == dates[1].date()
    assert by_coin["listed"][8] == pytest.approx(9.0)
    assert by_coin["listed"][9] == pytest.approx(2.0)

    assert by_coin["unlisted"][3] == "coingecko"
    assert by_coin["unlisted"][7] == "stop"
    assert by_coin["unlisted"][6] == 94.0  # opened beyond the stop, so sold at that price
    assert by_coin["unlisted"][5] == dates[2].date()
