import pandas as pd
import sqlite3
import subprocess
from datetime import datetime

# Load coinswitch coin IDs
coinswitch_mapping = pd.read_csv('coinswitch_coin_mapping.csv')
coinswitch_coin_ids = set(coinswitch_mapping['coin_id'].dropna().str.lower())

# Connect to predictions.sqlite and fetch predictions for coinswitch coins
conn = sqlite3.connect('predictions.sqlite')
query = '''
SELECT coin_id, prediction_date, target_date, predicted_price, model_name
FROM coin_predictions
'''
pred_df = pd.read_sql_query(query, conn)
conn.close()

# Filter for coinswitch coins
pred_df = pred_df[pred_df['coin_id'].str.lower().isin(coinswitch_coin_ids)]

# For each coin, get the latest prediction_date
latest_pred_dates = pred_df.groupby('coin_id')['prediction_date'].max().reset_index()
latest_preds = pd.merge(pred_df, latest_pred_dates, on=['coin_id', 'prediction_date'])

# Calculate 5-day percent change for each coin
results = []
for coin_id, group in latest_preds.groupby('coin_id'):
    group = group.sort_values('target_date')
    if len(group) >= 5:
        first_price = group.iloc[0]['predicted_price']
        last_price = group.iloc[4]['predicted_price']
        pct_change = (last_price - first_price) / first_price * 100 if first_price != 0 else 0
        results.append({
            'Coin': coin_id,
            '5-Day Predicted Prices': ', '.join([f"{p:.7f}" for p in group['predicted_price'].head(5)]),
            '5-Day % Change': pct_change
        })

filtered_df = pd.DataFrame(results)
filtered_df = filtered_df[filtered_df['5-Day % Change'] > 0]
filtered_df = filtered_df.sort_values('5-Day % Change', ascending=False).head(5)

# Output to CSV
filtered_df.to_csv('top5_coinswitch_5day_predictions.csv', index=False)
print('Filtered CSV written to top5_coinswitch_5day_predictions.csv using predictions.sqlite as source.')

# --- Populate positions table in predictions.sqlite ---
POSITIONS_DB = "predictions.sqlite"
POSITIONS_TABLE = "positions"

def ensure_positions_table():
    conn = sqlite3.connect(POSITIONS_DB)
    cur = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {POSITIONS_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin_id TEXT,
            open_date TEXT,
            close_date TEXT,
            predicted_open_price REAL,
            predicted_close_price REAL,
            actual_open_price REAL,
            actual_close_price REAL,
            position_type TEXT,
            profit_loss REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def populate_positions_from_top5(csv_path="top5_coinswitch_5day_predictions.csv"):
    ensure_positions_table()
    df = pd.read_csv(csv_path)
    conn = sqlite3.connect(POSITIONS_DB)
    cur = conn.cursor()
    for _, row in df.iterrows():
        coin_id = row['Coin']
        predicted_prices = [float(x) for x in row['5-Day Predicted Prices'].split(",")]
        open_date = datetime.utcnow().strftime("%Y-%m-%d")
        close_date = None  # Could be set to open_date + 5 days if needed
        predicted_open_price = predicted_prices[0]
        predicted_close_price = predicted_prices[-1]
        # Actual prices can be filled in later by another process
        cur.execute(f"""
            INSERT INTO {POSITIONS_TABLE} (coin_id, open_date, close_date, predicted_open_price, predicted_close_price, position_type)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (coin_id, open_date, close_date, predicted_open_price, predicted_close_price, 'long'))
    conn.commit()
    conn.close()
    print(f"Populated {POSITIONS_TABLE} table with top 5 positions.")

populate_positions_from_top5()
