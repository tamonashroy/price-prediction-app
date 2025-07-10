import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import add_features, train_predictive_model, predict_next_days, plot_actual_vs_predicted_best, plot_next_days_bar
from db_utils import get_all_coins_from_db, get_coin_data_from_db, load_predictions_from_db, save_predictions_to_db
from sklearn.model_selection import train_test_split
import time

st.set_page_config(page_title="Bitcoin Price Prediction Report", layout="wide")
st.title("Bitcoin Daily Price Prediction Report")

# Sidebar for coin selection
st.sidebar.header("Select Cryptocurrency")
coins = get_all_coins_from_db()
coin_names = [name for name, _ in coins]
coin_ids = {name: cid for name, cid in coins}
selected_coin = st.sidebar.selectbox("Choose a coin:", coin_names, index=0)
coin_id = coin_ids[selected_coin]
st.sidebar.write(f"You selected: {selected_coin}")

# Data Ingestion for selected coin
st.header(f"1. Data Ingestion for {selected_coin}")
df = get_coin_data_from_db(coin_id)
if df is None or df.empty:
    st.warning("No data available for this coin. Please run the ingestion script or try another coin.")
    st.stop()
st.write("Sample data:")
st.dataframe(df.head())

# Feature Engineering
st.header("2. Feature Engineering")
df_feat = add_features(df)
if df_feat is None or df_feat.empty or len(df_feat) < 10:
    st.warning("Not enough data after feature engineering for this coin. Please ingest more data or try another coin.")
    st.stop()
# Determine features for modeling
features = [col for col in df_feat.columns if col not in ['date', 'coin_id', 'coin_name', 'price']]
modeling_cols = features + ['price']
df_feat = df_feat.dropna(subset=modeling_cols)
if len(df_feat) < 10:
    st.warning("Not enough data after dropping NaNs for this coin. Please ingest more data or try another coin.")
    st.stop()
# Now train the model
model, features, mse_scores, best_model_name = train_predictive_model(df_feat)
df_feat = df_feat.reset_index(drop=True)

# Model Training
st.header("3. Model Training and Evaluation")
# Drop any rows with NaN in features or price
modeling_cols = features + ['price']
df_feat = df_feat.dropna(subset=modeling_cols)
if len(df_feat) < 10:
    st.warning("Not enough data after dropping NaNs for this coin. Please ingest more data or try another coin.")
    st.stop()
X = df_feat[features]
y = df_feat['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
# Train all models and get diagnostics
model, features, mse_scores, best_model_name = train_predictive_model(df_feat)
# Show model diagnostics
st.subheader("Model Comparison and Diagnostics")
mse_df = pd.DataFrame(list(mse_scores.items()), columns=["Model", "MSE (lower is better)"])
mse_df = mse_df.sort_values("MSE (lower is better)")
st.dataframe(mse_df, use_container_width=True)
st.success(f"Best fit model: {best_model_name}")
st.info(f"Model used for prediction: {type(model).__name__}")
st.write("**Features used for prediction:**")
st.code("\n".join(features))

# Actual vs Predicted Chart
st.header("4. Actual vs Predicted (Best Model)")
y_pred = model.predict(X_test)
test_dates = df_feat.loc[X_test.index, 'date'] if 'date' in df_feat.columns else X_test.index
fig1 = plt.figure(figsize=(10,5))
plt.plot(test_dates, y_test.values, label='Actual')
plt.plot(test_dates, y_pred, label=f'{type(model).__name__} Predicted')
plt.title(f'Actual vs Predicted Bitcoin Price ({type(model).__name__})')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.xticks(rotation=45)
st.pyplot(fig1)
plt.close(fig1)

# Next 5 Days Prediction
st.header("5. Next 5 Days Price Prediction")
future_dates, future_prices = predict_next_days(df_feat, model, days=5, features=features)
last_actual_price = df_feat['price'].iloc[-1]
# Bar chart with percent change
fig2 = plt.figure(figsize=(10,5))
colors = []
labels = []
for i in range(len(future_prices)):
    if i == 0:
        prev = last_actual_price
    else:
        prev = future_prices[i-1]
    pct_change = 100 * (future_prices[i] - prev) / prev if prev != 0 else 0
    label = f"{future_prices[i]:.2f}\n({pct_change:+.2f}%)"
    labels.append(label)
    if future_prices[i] > prev:
        colors.append('green')
    else:
        colors.append('red')
bars = plt.bar(future_dates, future_prices, color=colors)
plt.title('Next 5 Days Bitcoin Price Prediction (Best Model)')
plt.xlabel('Date')
plt.ylabel('Predicted Price')
plt.tight_layout()
for bar, label in zip(bars, labels):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), label, ha='center', va='bottom', fontsize=9)
st.pyplot(fig2)
plt.close(fig2)

st.write("\nNext 5 days price prediction:")
pred_table = pd.DataFrame({"Date": future_dates, "Predicted Price": future_prices})
pred_table["% Change"] = pred_table["Predicted Price"].pct_change().fillna((pred_table["Predicted Price"][0] - last_actual_price) / last_actual_price).apply(lambda x: f"{x*100:+.2f}%")
st.dataframe(pred_table)

# Sidebar: Show coins predicted to be green in descending order of 5-day percent change
st.sidebar.header("Top Coins Predicted to be Green (Next 5 Days)")
results_df = load_predictions_from_db()
if results_df is not None and not results_df.empty:
    st.sidebar.dataframe(results_df)
    st.sidebar.info("Loaded predictions from database. To refresh, rerun the prediction update script.")
else:
    results = []
    progress = st.progress(0)
    for idx, (coin_name, coin_id) in enumerate(coins):
        df = get_coin_data_from_db(coin_id)
        if df.empty or len(df) < 40:
            continue
        df_feat = add_features(df)
        if df_feat.empty or len(df_feat) < 10:
            continue
        df_feat = df_feat.reset_index(drop=True)
        X = df_feat[features]
        y = df_feat['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model, features_local, _, _ = train_predictive_model(df_feat)
        future_dates, future_prices = predict_next_days(df_feat, model, days=5, features=features_local)
        last_actual_price = df_feat['price'].iloc[-1]
        pct_change = (future_prices[-1] - last_actual_price) / last_actual_price * 100 if last_actual_price != 0 else 0
        if future_prices[-1] > last_actual_price:
            results.append({
                'Coin': coin_name,
                '5-Day Predicted Prices': ", ".join([f"{p:.2f}" for p in future_prices]),
                '5-Day % Change': f"{pct_change:+.2f}%"
            })
        progress.progress((idx+1)/len(coins))
    progress.empty()
    if results:
        results = sorted(results, key=lambda x: float(x['5-Day % Change'].replace('%','')), reverse=True)
        results_df = pd.DataFrame(results)
        st.sidebar.dataframe(results_df)
        save_predictions_to_db(results)
        st.sidebar.success("Predictions calculated and saved to database.")
    else:
        st.sidebar.write("No coins predicted to be green in the next 5 days.")

# Streamlit page navigation
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate to:", ["Main Dashboard", "Top Movers (24h)"])

if page == "Main Dashboard":
    # Main dashboard code is already executed above
    pass
elif page == "Top Movers (24h)":
    st.header("Top Movers by Volume Change (24h) (Local DB)")
    # Fetch top 20 coins by 24h volume change from CoinGecko
    top_volume_coins = []
    try:
        import requests
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            'vs_currency': 'usd',
            'order': 'volume_desc',
            'per_page': 100,
            'page': 1,
            'sparkline': False
        }
        resp = requests.get(url, params=params)
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
            if 'volume_change_24h' in df.columns:
                df['volume_change_pct'] = df['volume_change_24h'] / (df['total_volume'] - df['volume_change_24h']) * 100
            else:
                df['volume_change_pct'] = 0  # fallback if not available
            df = df.sort_values('volume_change_pct', ascending=False)
            top_volume_coins = df.head(20)['id'].tolist()
    except Exception as e:
        st.warning(f"Could not fetch top movers by volume from CoinGecko: {e}")
    # Add refresh button to trigger ingestion for top movers by volume
    if st.button("Refresh Data (Ingest Top Movers by Volume)"):
        import subprocess
        import sys
        st.info("Running ingestion script for top movers by volume...")
        try:
            for coin_id in top_volume_coins:
                result = subprocess.run([sys.executable, "ingest_coins.py", coin_id], capture_output=True, text=True)
                st.text(f"Ingested {coin_id}:\n" + result.stdout)
            st.success("Ingestion complete for top movers by volume! Reload the page to see updated results.")
        except Exception as e:
            st.error(f"Failed to run ingestion: {e}")
    # Get all coins
    movers = []
    for coin_name, coin_id in coins:
        df = get_coin_data_from_db(coin_id)
        if df is not None and len(df) >= 2:
            df = df.sort_values('date')
            last = df.iloc[-1]
            prev = df.iloc[-2]
            price_change = (last['price'] - prev['price']) / prev['price'] * 100 if prev['price'] != 0 else 0
            # Add CoinGecko link
            coingecko_url = f"https://www.coingecko.com/en/coins/{coin_id}"
            movers.append({
                'Coin': coin_name,
                'CoinGecko': f'<a href="{coingecko_url}" target="_blank">Link</a>',
                'Date': last['date'],
                'Price (latest)': last['price'],
                'Price Change (24h %)': price_change,
            })
    if movers:
        movers_df = pd.DataFrame(movers)
        movers_df = movers_df.sort_values('Price Change (24h %)', ascending=False)
        st.write("**Top 20 coins by 24h price change % (from local DB):**")
        st.write("(Click 'Link' to open CoinGecko page)")
        st.write(movers_df.head(20).to_html(escape=False, index=False), unsafe_allow_html=True)
        st.info("Showing top 20 coins by 24h price change % (from local DB). Volume change is not available unless stored in DB.")
    else:
        st.warning("No sufficient data in the local database to calculate 24h movers.")

# Show top coin by volume with coin name, volume, and price change %
try:
    import requests
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': 'usd',
        'order': 'volume_desc',
        'per_page': 1,
        'page': 1,
        'sparkline': False
    }
    resp = requests.get(url, params=params)
    data = resp.json()
    if isinstance(data, list) and len(data) > 0:
        top_coin = data[0]
        coin_name = top_coin.get('name', 'N/A')
        volume = top_coin.get('total_volume', 'N/A')
        price_change = top_coin.get('price_change_percentage_24h', 'N/A')
        st.sidebar.markdown(f"**Top Coin by Volume:**  ")
        st.sidebar.markdown(f"Name: {coin_name}")
        st.sidebar.markdown(f"24h Volume: ${volume:,.0f}")
        st.sidebar.markdown(f"24h Price Change: {price_change:+.2f}%")
except Exception as e:
    st.sidebar.warning(f"Could not fetch top coin by volume: {e}")

# Show top 10 coins by volume with price change % as a bar chart in the sidebar
try:
    import requests
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': 'usd',
        'order': 'volume_desc',
        'per_page': 10,
        'page': 1,
        'sparkline': False
    }
    resp = requests.get(url, params=params)
    data = resp.json()
    if isinstance(data, list) and len(data) > 0:
        coin_names = [coin.get('name', 'N/A') for coin in data]
        price_changes = [coin.get('price_change_percentage_24h', 0) for coin in data]
        volumes = [coin.get('total_volume', 0) for coin in data]
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 3))
        bars = ax.barh(coin_names[::-1], price_changes[::-1], color=['green' if x >= 0 else 'red' for x in price_changes[::-1]])
        ax.set_xlabel('% Change (24h)')
        ax.set_title('Top 10 Coins by Volume: 24h % Change')
        for i, (bar, vol) in enumerate(zip(bars, volumes[::-1])):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f"{price_changes[::-1][i]:+.2f}%", va='center', ha='left', fontsize=8)
        plt.tight_layout()
        st.sidebar.pyplot(fig)
        plt.close(fig)
except Exception as e:
    st.sidebar.warning(f"Could not fetch top 10 coins by volume: {e}")
