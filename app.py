import streamlit as st
import pandas as pd
import os
import requests
from db_utils import load_predictions_from_db, get_coin_data_from_db
from model import predict_next_days, backtest_walk_forward

# --- Utility: Get coin logo ---
def get_coin_logo_url(coin_id):
    # Try CoinGecko API for logo
    try:
        url = f"https://api.coingecko.com/api/v3/coins/list"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            coins = resp.json()
            # Find matching id (case-insensitive)
            match = next((c for c in coins if c['id'].lower() == coin_id.lower()), None)
            if match:
                # Use CoinGecko's coin page for logo
                cg_id = match['id']
                coin_url = f"https://api.coingecko.com/api/v3/coins/{cg_id}"
                coin_resp = requests.get(coin_url, timeout=5)
                if coin_resp.status_code == 200:
                    data = coin_resp.json()
                    img = data.get('image', {}).get('thumb') or data.get('image', {}).get('small')
                    if img:
                        return img
    except Exception:
        pass
    # Try TrustWallet assets CDN (for common coins)
    try:
        url = f"https://raw.githubusercontent.com/trustwallet/assets/master/blockchains/ethereum/assets/{coin_id}/logo.png"
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            return url
    except Exception:
        pass
    # Fallback: no logo
    return ""

st.set_page_config(page_title="Crypto Green Prediction Dashboard", layout="wide")
st.title("Crypto Green Prediction Dashboard")

# --- Load data from DB ---
results_df = load_predictions_from_db()
coinswitch_mapping_path = "coinswitch_coin_mapping.csv"
coinswitch_df = None
if os.path.exists(coinswitch_mapping_path):
    coinswitch_mapping = pd.read_csv(coinswitch_mapping_path)
    coinswitch_coin_ids = set(coinswitch_mapping['coin_id'].dropna().str.lower())
    if results_df is not None and not results_df.empty:
        filtered_df = results_df[results_df['Coin'].str.lower().isin(coinswitch_coin_ids)]
        coinswitch_df = filtered_df.sort_values('5-Day % Change', ascending=False).head(5)

# --- Add custom CSS for styling ---
with open("logo.png", "rb") as image_file:
    import base64
    logo_base64 = base64.b64encode(image_file.read()).decode()

custom_css = f'''
    <style>
    .main-title {{
        font-size: 2.8rem;
        font-weight: 800;
        color: #1a2639;
        margin-bottom: 0.2em;
        letter-spacing: 1px;
        display: flex;
        align-items: center;
    }}
    .main-title img {{
        height: 60px;
        margin-right: 18px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    }}
    .stApp {{
        background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%);
    }}
    .coin-section h3, .coin-section h2 {{
        color: #2b4162;
        font-weight: 700;
    }}
    .coin-section {{
        background: #fff;
        border-radius: 16px;
        box-shadow: 0 2px 12px rgba(44,62,80,0.07);
        padding: 1.5em 1.5em 1em 1.5em;
        margin-bottom: 2em;
    }}
    .coin-logo {{
        border-radius: 50%;
        box-shadow: 0 1px 4px rgba(0,0,0,0.10);
        margin-right: 10px;
        vertical-align: middle;
    }}
    .coin-link {{
        font-size: 1.1em;
        font-weight: 600;
        color: #1a2639 !important;
        text-decoration: none !important;
    }}
    .coin-link:hover {{
        color: #4f8cff !important;
    }}
    .stMarkdown small {{
        color: #6b7280;
        font-size: 0.98em;
    }}
    </style>
'''
st.markdown(custom_css, unsafe_allow_html=True)

# --- Custom Title with Logo ---
st.markdown(f'''<div class="main-title"><img src="data:image/png;base64,{logo_base64}" alt="Logo"/>Crypto Green Prediction Dashboard</div>''', unsafe_allow_html=True)

# --- Top 5 Coinswitch Coins ---
st.markdown("<h2>Top 5 Coinswitch Coins Predicted to be Green (Next 5 Days)</h2>", unsafe_allow_html=True)
if coinswitch_df is not None and not coinswitch_df.empty:
    # Build a table for display
    table_data = []
    for idx, row in coinswitch_df.iterrows():
        coin_id = row['Coin']
        logo_url = get_coin_logo_url(coin_id)
        # Calculate % changes
        try:
            prices = eval(row['5-Day Predicted Prices']) if isinstance(row['5-Day Predicted Prices'], str) else row['5-Day Predicted Prices']
            pct_changes = [None]
            for i in range(1, len(prices)):
                prev = prices[i-1]
                curr = prices[i]
                pct = ((curr - prev) / prev * 100) if prev != 0 else 0
                pct_changes.append(pct)
            pct_strs = []
            for i, pct in enumerate(pct_changes):
                if i == 0:
                    continue
                color = 'green' if pct >= 0 else 'red'
                pct_strs.append(f"<span style='color:{color};'>{pct:+.2f}%</span>")
            pct_html = ', '.join(pct_strs)
        except Exception:
            pct_html = "-"
        table_data.append({
            "Logo": f"<img src='{logo_url}' width='24' class='coin-logo'/>",
            "Coin": f"<a href='#coin_{coin_id}' class='coin-link'><b>{row['Coin']}</b></a>",
            "5-Day % Change": row['5-Day % Change'],
            "Next 5 Days % Change": pct_html
        })
    st.markdown(pd.DataFrame(table_data).to_html(escape=False, index=False), unsafe_allow_html=True)
else:
    st.info("No Coinswitch coins predicted to be green in the next 5 days.")

# --- Top 10 Coins ---
st.markdown("<h2 class='coin-section'>Top 10 Coins Predicted to be Green (Next 5 Days)</h2>", unsafe_allow_html=True)
if results_df is not None and not results_df.empty:
    top10 = results_df.sort_values('5-Day % Change', ascending=False).head(10)
    table_data = []
    for idx, row in top10.iterrows():
        coin_id = row['Coin']
        logo_url = get_coin_logo_url(coin_id)
        try:
            prices = eval(row['5-Day Predicted Prices']) if isinstance(row['5-Day Predicted Prices'], str) else row['5-Day Predicted Prices']
            pct_changes = [None]
            for i in range(1, len(prices)):
                prev = prices[i-1]
                curr = prices[i]
                pct = ((curr - prev) / prev * 100) if prev != 0 else 0
                pct_changes.append(pct)
            pct_strs = []
            for i, pct in enumerate(pct_changes):
                if i == 0:
                    continue
                color = 'green' if pct >= 0 else 'red'
                pct_strs.append(f"<span style='color:{color};'>{pct:+.2f}%</span>")
            pct_html = ', '.join(pct_strs)
        except Exception:
            pct_html = "-"
        table_data.append({
            "Logo": f"<img src='{logo_url}' width='24' class='coin-logo'/>",
            "Coin": f"<a href='#coin_{coin_id}' class='coin-link'><b>{row['Coin']}</b></a>",
            "5-Day % Change": row['5-Day % Change'],
            "Next 5 Days % Change": pct_html
        })
    st.markdown(pd.DataFrame(table_data).to_html(escape=False, index=False), unsafe_allow_html=True)
else:
    st.info("No coins predicted to be green in the next 5 days.")

# --- Coin Details Section ---
st.markdown("<hr>")
st.markdown("<h2 class='coin-section'>Coin Details</h2>", unsafe_allow_html=True)
if results_df is not None and not results_df.empty:
    for idx, row in results_df.iterrows():
        coin_id = row['Coin']
        st.markdown(f"<div id='coin_{coin_id}' class='coin-section'></div>", unsafe_allow_html=True)
        logo_url = get_coin_logo_url(coin_id)
        st.markdown(f"<h3><img src='{logo_url}' width='32' class='coin-logo'/> {coin_id}</h3>", unsafe_allow_html=True)
        # Show next 5 days price and % change, and graph here
        try:
            prices = eval(row['5-Day Predicted Prices']) if isinstance(row['5-Day Predicted Prices'], str) else row['5-Day Predicted Prices']
            pct_changes = [None]
            for i in range(1, len(prices)):
                prev = prices[i-1]
                curr = prices[i]
                pct = ((curr - prev) / prev * 100) if prev != 0 else 0
                pct_changes.append(pct)
            price_pct = []
            for i, p in enumerate(prices):
                if pct_changes[i] is None:
                    price_pct.append(f"{p:.2f} (start)")
                else:
                    color = 'green' if pct_changes[i] >= 0 else 'red'
                    price_pct.append(f"<span style='color:{color};'>{p:.2f} ({pct_changes[i]:+0.2f}%)</span>")
            st.markdown(f"<b>Next 5 Days Prediction:</b> " + ', '.join(price_pct), unsafe_allow_html=True)
            # Bar graph for % change (moved here)
            # Use Streamlit's native text for the label instead of HTML <small>
            st.caption("Price Change % (Day 2-5):")
            if len(prices) > 1:
                import matplotlib.pyplot as plt
                import numpy as np
                fig, ax = plt.subplots(figsize=(5,2))
                bars = ax.bar(range(1, len(prices)), pct_changes[1:], color=["green" if x >= 0 else "red" for x in pct_changes[1:]])
                ax.set_xticks(range(1, len(prices)))
                ax.set_xticklabels([f"Day {i+1}" for i in range(1, len(prices))])
                ax.set_ylabel("% Change")
                for bar, pct, price in zip(bars, pct_changes[1:], prices[1:]):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{price:.2f}\n({pct:+.2f}%)", ha='center', va='bottom', fontsize=8)
                st.pyplot(fig)
        except Exception:
            st.markdown(f"<b>Next 5 Days Prediction:</b> {row['5-Day Predicted Prices']}")
        # Fix for Best Model: only show if not N/A or empty
        best_model = row.get('Best Model', None)
        if best_model and str(best_model).strip().upper() != 'N/A':
            st.markdown(f"<b>Best Model:</b> {best_model}", unsafe_allow_html=True)
        # Backtesting and graph
        coin_df = get_coin_data_from_db(coin_id)
        if coin_df is not None and not coin_df.empty:
            try:
                # Show backtest graph
                st.markdown("<b>Backtesting (Walk-Forward):</b>")
                _, y_true, y_pred = backtest_walk_forward(coin_df, None, None)  # You may need to pass correct model_class and features
                st.line_chart({"Actual": y_true, "Predicted": y_pred})
            except Exception as e:
                st.info(f"Backtest not available: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("No coin details available.")
