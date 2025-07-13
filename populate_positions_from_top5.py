import pandas as pd
import sqlite3
from datetime import datetime

POSITIONS_DB = "predictions.sqlite"
POSITIONS_TABLE = "positions"

# Ensure the positions table exists
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

# Populate positions for top 5 coins (long only, as example)
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

if __name__ == "__main__":
    populate_positions_from_top5()
