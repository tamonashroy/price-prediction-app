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
st.write("Sample data:")
st.dataframe(df.head())

# Feature Engineering
st.header("2. Feature Engineering")
df_feat = add_features(df)
st.write("Sample with features:")
st.dataframe(df_feat.head())

# Model Training
st.header("3. Model Training and Evaluation")
features = ['price_lag1', 'price_ma7', 'price_ma30']
df_feat = df_feat.reset_index(drop=True)
X = df_feat[features]
y = df_feat['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
# Capture printed output
import io
import sys
output = io.StringIO()
sys.stdout = output
model = train_predictive_model(df_feat)
sys.stdout = sys.__stdout__
st.code(output.getvalue())

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
future_dates, future_prices = predict_next_days(df_feat, model, days=5)
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

# Homepage: Show coins predicted to be green in descending order of 5-day percent change
st.header("Top Coins Predicted to be Green (Next 5 Days)")

# Try to load from DB first
results_df = load_predictions_from_db()
if results_df is not None and not results_df.empty:
    st.dataframe(results_df)
    st.info("Loaded predictions from database. To refresh, rerun the prediction update script.")
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
        model = train_predictive_model(df_feat)
        future_dates, future_prices = predict_next_days(df_feat, model, days=5)
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
        st.dataframe(results_df)
        save_predictions_to_db(results)
        st.success("Predictions calculated and saved to database.")
    else:
        st.write("No coins predicted to be green in the next 5 days.")
