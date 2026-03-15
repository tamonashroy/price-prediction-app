"""Populate positions for top 5 coins from predictions."""

import pandas as pd
from datetime import datetime
from db_config import get_pyodbc_connection, table_exists

POSITIONS_TABLE = "positions"

# Ensure the positions table exists
def ensure_positions_table():
    conn = get_pyodbc_connection()
    try:
        if not table_exists(POSITIONS_TABLE, conn):
            query = f"""
                CREATE TABLE {POSITIONS_TABLE} (
                    id INT PRIMARY KEY IDENTITY(1,1),
                    coin_id NVARCHAR(MAX),
                    open_date NVARCHAR(MAX),
                    close_date NVARCHAR(MAX),
                    predicted_open_price REAL,
                    predicted_close_price REAL,
                    actual_open_price REAL,
                    actual_close_price REAL,
                    position_type NVARCHAR(MAX),
                    profit_loss REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
    finally:
        conn.close()

# Populate positions for top 5 coins (long only, as example)
def populate_positions_from_top5(csv_path="top5_coinswitch_5day_predictions.csv"):
    ensure_positions_table()
    df = pd.read_csv(csv_path)
    conn = get_pyodbc_connection()
    try:
        cursor = conn.cursor()
        for _, row in df.iterrows():
            coin_id = row['Coin']
            predicted_prices = [float(x) for x in row['5-Day Predicted Prices'].split(",")]
            open_date = datetime.utcnow().strftime("%Y-%m-%d")
            close_date = None  # Could be set to open_date + 5 days if needed
            predicted_open_price = predicted_prices[0]
            predicted_close_price = predicted_prices[-1]
            # Actual prices can be filled in later by another process
            query = f"""
                INSERT INTO {POSITIONS_TABLE} (coin_id, open_date, close_date, predicted_open_price, predicted_close_price, position_type)
                VALUES (?, ?, ?, ?, ?, ?)
            """
            cursor.execute(query, (coin_id, open_date, close_date, predicted_open_price, predicted_close_price, 'long'))
        conn.commit()
        print(f"Populated {POSITIONS_TABLE} table with top 5 positions.")
    finally:
        conn.close()

if __name__ == "__main__":
    populate_positions_from_top5()
