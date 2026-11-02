from datetime import date

from pricepredict.binance import plan_price_changes

TODAY = date(2026, 11, 2)
DAY = date(2026, 3, 11)


def test_close_prices_are_left_alone():
    inserts, corrections = plan_price_changes({DAY: 1.1}, {DAY: 1.0}, TODAY, add_missing=True)
    assert inserts == []
    assert corrections == []


def test_doubled_price_is_corrected_to_binance():
    _, corrections = plan_price_changes({DAY: 0.198}, {DAY: 0.094}, TODAY, add_missing=False)
    assert corrections == [(DAY, 0.094)]


def test_halved_price_is_corrected_too():
    _, corrections = plan_price_changes({DAY: 0.5}, {DAY: 1.0}, TODAY, add_missing=False)
    assert corrections == [(DAY, 1.0)]


def test_missing_dates_are_only_added_when_asked():
    assert plan_price_changes({}, {DAY: 1.0}, TODAY, add_missing=False) == ([], [])
    assert plan_price_changes({}, {DAY: 1.0}, TODAY, add_missing=True) == ([(DAY, 1.0)], [])


def test_dates_after_today_are_ignored():
    future = date(2026, 11, 3)
    assert plan_price_changes({future: 9.0}, {future: 1.0}, TODAY, add_missing=True) == ([], [])
