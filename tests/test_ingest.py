from datetime import date

from pricepredict.config import MAX_HISTORY_DAYS
from pricepredict.ingest import _days_to_fetch, parse_market_chart, select_universe

MIDNIGHT_OCT_31 = 1793404800000
MIDNIGHT_NOV_1 = 1793491200000
NOV_1_EVENING = MIDNIGHT_NOV_1 + 18 * 3600 * 1000


def chart(points):
    return {
        "prices": [[timestamp, price] for timestamp, price in points],
        "market_caps": [[timestamp, price * 10] for timestamp, price in points],
        "total_volumes": [[timestamp, price * 2] for timestamp, price in points],
    }


def test_partial_current_day_point_is_dropped():
    df = parse_market_chart(chart([(MIDNIGHT_OCT_31, 1.0), (MIDNIGHT_NOV_1, 2.0), (NOV_1_EVENING, 3.0)]))
    assert df["date"].tolist() == [date(2026, 10, 31), date(2026, 11, 1)]
    assert df["price"].tolist() == [1.0, 2.0]


def test_columns_stay_aligned_to_their_timestamp():
    df = parse_market_chart(chart([(MIDNIGHT_OCT_31, 1.0), (MIDNIGHT_NOV_1, 2.0)]))
    assert df["market_cap"].tolist() == [10.0, 20.0]
    assert df["total_volume"].tolist() == [2.0, 4.0]


def test_missing_volume_does_not_drop_the_price():
    data = chart([(MIDNIGHT_OCT_31, 1.0), (MIDNIGHT_NOV_1, 2.0)])
    data["total_volumes"] = data["total_volumes"][:1]
    df = parse_market_chart(data)
    assert len(df) == 2
    assert df["total_volume"].isna().tolist() == [False, True]


def market(coin_id, symbol):
    return {"id": coin_id, "symbol": symbol, "name": coin_id.title()}


def test_universe_matches_by_symbol_not_coingecko_id():
    universe = select_universe([market("bitcoin", "BTC"), market("ripple", "XRP")], {"btc", "xrp"})
    assert [coin["coin_id"] for coin in universe] == ["bitcoin", "ripple"]


def test_shared_symbol_means_the_largest_coin():
    universe = select_universe([market("real-coin", "abc"), market("copycat", "abc")], {"abc"})
    assert [coin["coin_id"] for coin in universe] == ["real-coin"]


def test_a_large_untradable_coin_does_not_lose_its_symbol_to_a_tradable_lookalike():
    universe = select_universe([market("big", "xyz"), market("small", "xyz")], {"abc"})
    assert universe == []


def test_stablecoins_are_excluded():
    universe = select_universe([market("tether", "usdt"), market("bitcoin", "btc")], {"usdt", "btc"})
    assert [coin["coin_id"] for coin in universe] == ["bitcoin"]


def test_new_coin_fetches_full_history():
    assert _days_to_fetch(None, date(2026, 11, 2)) == MAX_HISTORY_DAYS


def test_up_to_date_coin_fetches_a_small_overlap():
    assert _days_to_fetch(date(2026, 11, 1), date(2026, 11, 2)) == 3


def test_fetch_never_exceeds_the_free_api_window():
    assert _days_to_fetch(date(2020, 1, 1), date(2026, 11, 2)) == MAX_HISTORY_DAYS
