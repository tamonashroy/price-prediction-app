import streamlit as st
import pandas as pd

st.set_page_config(page_title="Top 5 Coinswitch Predictions", layout="wide")
st.title("Top 5 Coinswitch Coins: 5-Day Price Predictions and Last 5-Day Accuracy")

def main():
    # Section 1: Top 5 predictions table
    df = pd.read_csv("top5_coinswitch_5day_predictions.csv")
    st.subheader("Top 5 Coinswitch Predictions Table")
    st.dataframe(df, use_container_width=True)

    # Section 2: Daily Trade Simulation Results
    st.subheader("Daily Trade Simulation Results (Top 5 Predicted Coins)")
    try:
        sim_summary = pd.read_csv("top5_daily_trade_simulation_summary.csv")
        st.dataframe(sim_summary, use_container_width=True)
        st.write("""
        - **profitable_long_ratio**: Fraction of top 5 trades that would have been profitable as a long (buy & hold for 5 days)
        - **profitable_short_ratio**: Fraction of top 5 trades that would have been profitable as a short (sell & hold for 5 days)
        """)
    except Exception as e:
        st.warning(f"Could not load daily trade simulation summary: {e}")

if __name__ == "__main__":
    main()
