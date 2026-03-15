import sqlite3
import pandas as pd

DB_PATH = "coin_predictions.db"

# 1. Add actual_price column if not exists
def ensure_actual_price_column():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("ALTER TABLE coin_predictions ADD COLUMN actual_price REAL")
    except sqlite3.OperationalError:
        # Column already exists
        pass
    conn.commit()
    conn.close()

# 2. Backfill actual_price from coin_prices table
def backfill_actual_prices():
    conn = sqlite3.connect(DB_PATH)
    # Get all coin_predictions rows
    pred_df = pd.read_sql_query("SELECT rowid, coin_id, target_date FROM coin_predictions", conn)
    # For each row, get the actual price from coin_prices
    for idx, row in pred_df.iterrows():
        coin_id = row['coin_id']
        target_date = row['target_date']
        # Query for actual price
        price_row = conn.execute(
            "SELECT price FROM coin_prices WHERE coin_id = ? AND date = ?",
            (coin_id, target_date)
        ).fetchone()
        actual_price = price_row[0] if price_row else None
        # Update coin_predictions
        conn.execute(
            "UPDATE coin_predictions SET actual_price = ? WHERE rowid = ?",
            (actual_price, row['rowid'])
        )
    conn.commit()
    conn.close()
    print("Backfilled actual_price in coin_predictions from coin_prices table.")

if __name__ == "__main__":
    ensure_actual_price_column()
    backfill_actual_prices()
