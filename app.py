import streamlit as st
import pandas as pd
import os
import pathlib

st.set_page_config(page_title="Top 5 Coinswitch Predictions", layout="wide")

# Banner section with st.image for local logo
logo_path = "logo.png"
logo_exists = os.path.exists(logo_path)

banner_cols = st.columns([1, 5])
if logo_exists:
    with banner_cols[0]:
        st.image(logo_path, width=90)
with banner_cols[1]:
    st.markdown(
        """
        <div style="display: flex; flex-direction: column; justify-content: center; height: 100%;">
            <h1 style="color: #2c5364; margin-bottom: 0.2em; font-size: 2.5em; font-family: 'Segoe UI', sans-serif;">Crypto ML Dashboard</h1>
            <p style="color: #444; font-size: 1.2em; margin-top: 0;">Predict, Analyze, and Backtest Top Coinswitch Coins with Machine Learning</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Display last refresh time if available
refresh_path = pathlib.Path(__file__).parent / "last_refresh.txt"
if refresh_path.exists():
    with open(refresh_path, "r") as f:
        last_refresh = f.read().strip()
    st.info(f"Data last refreshed: {last_refresh}")

st.title("Top 5 Coinswitch Coins: 5-Day Price Predictions and Last 5-Day Accuracy")

def main():
    # Section 1: Top 5 predictions table
    df = pd.read_csv("top5_coinswitch_5day_predictions.csv")
    st.subheader("Top 5 Coinswitch Predictions Table")
    st.dataframe(df, use_container_width=True)

    # Section 3: Positions Actual vs Predicted
    st.subheader("Positions: Actual vs Predicted Profit/Loss (%)")
    try:
        pos_df = pd.read_csv("positions_actual_vs_predicted.csv")
        st.dataframe(pos_df, use_container_width=True)
        st.write("""
        - **actual_profit_loss_%**: Realized profit/loss % for each position using actual prices
        - **predicted_profit_loss_%**: Predicted profit/loss % for each position using model predictions
        """)
    except Exception as e:
        st.warning(f"Could not load positions actual vs predicted: {e}")

if __name__ == "__main__":
    main()
