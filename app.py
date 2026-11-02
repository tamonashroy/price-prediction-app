"""Streamlit dashboard. Read-only: everything shown is written by pricepredict.pipeline."""

import json

import altair as alt
import pandas as pd
import streamlit as st

from pricepredict import config
from pricepredict.db import get_engine

CACHE_SECONDS = 3600
# Fixed so each strategy keeps its colour whichever ones have data.
STRATEGY_COLORS = {
    "Model top 5": "#2a78d6",
    "Buy all coins": "#eb6834",
    "Hold bitcoin": "#1baf7a",
    "30-day momentum": "#eda100",
}
STRATEGY_LABELS = {
    "model": "Model top 5",
    "equal_weight_all": "Buy all coins",
    "hold_bitcoin": "Hold bitcoin",
    "momentum_30d": "30-day momentum",
}

st.set_page_config(page_title="Crypto 5-Day Picks", layout="wide")


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def query(sql, params=None):
    return pd.read_sql(sql, get_engine(), params=params)


def as_json(value):
    return json.loads(value) if isinstance(value, str) else value


def load_latest_run():
    runs = query(
        "SELECT data_through, created_at, model_name, model_config, metrics, feature_importance "
        "FROM model_runs ORDER BY id DESC LIMIT 1"
    )
    return None if runs.empty else runs.iloc[0]


def signal_verdict(metrics):
    """Plain-language judgement of the backtest, so a weak model is not presented as a good one."""
    strategies = metrics["strategies"]
    model = strategies["model"]
    baselines = max(strategies[name]["total_return_pct"] for name in ("equal_weight_all", "hold_bitcoin"))
    significant = metrics["rank_ic_t_stat"] >= 2
    if model["worst_offset_total_pct"] > baselines and significant:
        return "success", (
            "Backtest: the model beat buying every coin and holding bitcoin from every start day, and its ranking "
            "looks real. Part of this period also shaped the design, so judge it by the live track record below."
        )
    if model["total_return_pct"] > baselines:
        return "warning", (
            "Backtest: the model beat simple strategies on average, but not from every start day, "
            "so this could still be luck. Treat these picks as unproven."
        )
    return "error", "Backtest: the model did not beat simple strategies. Treat these picks as unproven."


def show_header(run):
    columns = st.columns([1, 9])
    if config.LOGO_PATH.exists():
        columns[0].image(str(config.LOGO_PATH), width=80)
    columns[1].title("Crypto 5-Day Picks")
    columns[1].caption(
        f"Prices through {run['data_through']} · updated {run['created_at']:%Y-%m-%d %H:%M} UTC · "
        f"model: {run['model_name']}"
    )


def show_picks(as_of):
    st.header("Today's picks")
    picks = query(
        "SELECT pick_rank, p.coin_id, MAX(c.coin_name) AS coin_name, actual_open_price, "
        "predicted_close_price, predicted_return_pct, close_date "
        "FROM positions p LEFT JOIN coin_prices c ON c.coin_id = p.coin_id AND c.date = p.open_date "
        "WHERE open_date = %(as_of)s "
        "GROUP BY pick_rank, p.coin_id, actual_open_price, predicted_close_price, predicted_return_pct, close_date "
        "ORDER BY pick_rank",
        {"as_of": as_of},
    )
    if picks.empty:
        st.info(
            "**Stay out today.** Bitcoin is below its 30-day average, and the strategy only buys when it is "
            "above. The forecast below still ranks every coin if you want to see the model's view."
        )
        return
    st.caption(
        "Forecasts are relative to the market: +3% means about 3% better than the typical coin, "
        "so the coin itself can still fall if the whole market falls."
    )
    st.dataframe(
        picks,
        hide_index=True,
        width="stretch",
        column_order=["pick_rank", "coin_name", "actual_open_price", "predicted_close_price",
                      "predicted_return_pct", "close_date"],
        column_config={
            "pick_rank": st.column_config.NumberColumn("Rank", format="%d"),
            "coin_name": "Coin",
            "actual_open_price": st.column_config.NumberColumn("Price now (USD)", format="%.6g"),
            "predicted_close_price": st.column_config.NumberColumn("Forecast price if market is flat", format="%.6g"),
            "predicted_return_pct": st.column_config.NumberColumn("Forecast vs market, 5 days", format="%+.2f%%"),
            "close_date": "Sell on",
        },
    )


def show_coin_forecast(as_of):
    st.header("Forecast for any coin")
    forecasts = query(
        "SELECT p.coin_id, target_date AS date, predicted_price AS price, predicted_return_pct "
        "FROM coin_predictions p WHERE prediction_date = %(as_of)s ORDER BY predicted_return_pct DESC",
        {"as_of": as_of},
    )
    if forecasts.empty:
        st.info("No forecasts stored for this date.")
        return
    five_day = forecasts.sort_values("date").groupby("coin_id", sort=False).tail(1)
    coin_ids = five_day.sort_values("predicted_return_pct", ascending=False)["coin_id"].tolist()
    coin_id = st.selectbox("Coin (strongest 5-day forecast first)", coin_ids)
    st.caption("The forecast line assumes the overall market stays flat.")
    history = query(
        "SELECT date, price FROM coin_prices WHERE coin_id = %(coin)s AND date > %(as_of)s::date - 60 "
        "AND date <= %(as_of)s ORDER BY date",
        {"coin": coin_id, "as_of": as_of},
    )
    # Start the forecast line at the last real price so the two lines connect.
    forecast = pd.concat([history.tail(1), forecasts[forecasts["coin_id"] == coin_id][["date", "price"]]])
    frame = pd.concat([history.assign(series="Actual"), forecast.assign(series="Forecast")])
    frame["date"] = pd.to_datetime(frame["date"])

    chart = (
        alt.Chart(frame)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=30, filled=True, opacity=0))
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("price:Q", title="Price (USD)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(domain=["Actual", "Forecast"], range=["#2a78d6", "#eb6834"]),
                legend=alt.Legend(orient="top", symbolType="stroke"),
            ),
            strokeDash=alt.StrokeDash(
                "series:N",
                scale=alt.Scale(domain=["Actual", "Forecast"], range=[[1, 0], [4, 3]]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("price:Q", title="Price (USD)", format=",.6g"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, width="stretch")


def show_track_record():
    st.header("Live track record")
    st.caption(f"Real outcomes of past picks after the {config.ROUND_TRIP_COST:.1%} trading cost.")
    positions = query(
        "SELECT coin_id, open_date, close_date, actual_open_price, actual_close_price, profit_loss, model_name "
        "FROM positions WHERE profit_loss IS NOT NULL ORDER BY open_date DESC, pick_rank"
    )
    current = positions[positions["model_name"] != "legacy"]
    if current.empty:
        st.info("Picks from the current model have not closed yet. Results appear 5 days after the first run.")
    else:
        columns = st.columns(3)
        columns[0].metric("Closed picks", len(current))
        columns[1].metric("Average return per pick", f"{current['profit_loss'].mean():+.2f}%")
        columns[2].metric("Profitable", f"{(current['profit_loss'] > 0).mean():.0%}")
    with st.expander("All closed picks, including the previous model"):
        st.dataframe(positions, hide_index=True, width="stretch")


def show_backtest(metrics):
    st.header("Backtest")
    st.caption(
        f"{metrics['start']} to {metrics['end']}: buy the top {config.TOP_N} when bitcoin is above its 30-day "
        f"average, hold {config.PICK_HORIZON} days, sell, after {metrics['cost_round_trip_pct']:.1f}% costs per "
        "trade. The universe is today's largest coins, which flatters every strategy, so compare the lines with "
        "each other rather than treating them as achievable. The chart shows one start day; the table shows how "
        "much the result changes with the start day."
    )
    periods = query("SELECT strategy, period_start, return_pct FROM backtest_periods ORDER BY period_start")
    periods = periods[periods["strategy"].isin(STRATEGY_LABELS)].copy()
    periods["Strategy"] = periods["strategy"].map(STRATEGY_LABELS)
    periods["growth"] = (
        periods.assign(factor=1 + periods["return_pct"] / 100).groupby("Strategy")["factor"].cumprod() * 100
    )
    periods["period_start"] = pd.to_datetime(periods["period_start"])

    domain = list(STRATEGY_COLORS)
    chart = (
        alt.Chart(periods)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60, filled=True, opacity=0))
        .encode(
            x=alt.X("period_start:T", title=None),
            y=alt.Y("growth:Q", title="Value of 100 invested", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "Strategy:N",
                scale=alt.Scale(domain=domain, range=[STRATEGY_COLORS[name] for name in domain]),
                legend=alt.Legend(orient="top", title=None, symbolType="stroke"),
            ),
            tooltip=[
                alt.Tooltip("Strategy:N"),
                alt.Tooltip("period_start:T", title="Date"),
                alt.Tooltip("growth:Q", title="Value of 100", format=",.1f"),
                alt.Tooltip("return_pct:Q", title="Period return %", format="+.2f"),
            ],
        )
        .properties(height=340)
    )
    st.altair_chart(chart, width="stretch")

    table = pd.DataFrame(metrics["strategies"]).T.loc[list(STRATEGY_LABELS)]
    table.insert(0, "Strategy", [STRATEGY_LABELS[name] for name in table.index])
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_order=["Strategy", "total_return_pct", "worst_offset_total_pct", "best_offset_total_pct",
                      "sharpe", "max_drawdown_pct", "win_rate_pct"],
        column_config={
            "total_return_pct": st.column_config.NumberColumn("Total return", format="%+.1f%%"),
            "worst_offset_total_pct": st.column_config.NumberColumn("Unluckiest start day", format="%+.1f%%"),
            "best_offset_total_pct": st.column_config.NumberColumn("Luckiest start day", format="%+.1f%%"),
            "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
            "max_drawdown_pct": st.column_config.NumberColumn("Worst drop", format="%.1f%%"),
            "win_rate_pct": st.column_config.NumberColumn("Winning periods", format="%.0f%%"),
        },
    )
    st.caption(
        f"Ranking skill: rank IC {metrics['rank_ic']:.3f} (t-stat {metrics['rank_ic_t_stat']:.1f}, "
        "where about 2 or more suggests it is not luck)."
    )


def show_model_details(run):
    with st.expander("What the model is using"):
        importance = as_json(run["feature_importance"]) or {}
        if importance:
            frame = pd.DataFrame({"Feature": list(importance), "Share of importance": list(importance.values())})
            st.dataframe(frame.sort_values("Share of importance", ascending=False), hide_index=True, width="stretch")
        st.json(as_json(run["model_config"]))


def main():
    try:
        run = load_latest_run()
    except Exception as error:
        st.error("Could not read the database. Check that DATABASE_URL is set.")
        st.caption(type(error).__name__)
        st.stop()
    if run is None:
        st.warning("No pipeline run yet. Run the Daily Pipeline workflow or `python -m pricepredict.pipeline`.")
        st.stop()

    show_header(run)
    as_of = run["data_through"]
    metrics = as_json(run["metrics"])
    if metrics:
        level, message = signal_verdict(metrics)
        getattr(st, level)(message)
    st.caption("Forecasts are experimental and are not financial advice. Crypto can lose most of its value quickly.")

    show_picks(as_of)
    show_coin_forecast(as_of)
    show_track_record()
    if metrics:
        show_backtest(metrics)
    show_model_details(run)


main()
