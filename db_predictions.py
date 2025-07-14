import sqlite3
from datetime import date

def get_db_connection(db_path="predictions.sqlite"):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def create_predictions_table(db_path="predictions.sqlite"):
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS coin_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin_id TEXT NOT NULL,
            prediction_date DATE NOT NULL,
            target_date DATE NOT NULL,
            predicted_price REAL NOT NULL,
            actual_price REAL,
            model_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def insert_coin_predictions(predictions, db_path="predictions.sqlite"):
    """
    predictions: list of dicts with keys: coin_id, prediction_date, target_date, predicted_price, model_name
    """
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    for pred in predictions:
        cur.execute('''
            INSERT INTO coin_predictions (coin_id, prediction_date, target_date, predicted_price, model_name)
            VALUES (?, ?, ?, ?, ?)
        ''', (pred['coin_id'], pred['prediction_date'], pred['target_date'], pred['predicted_price'], pred.get('model_name')))
    conn.commit()
    conn.close()

def update_actual_price(coin_id, target_date, actual_price, db_path="predictions.sqlite"):
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute('''
        UPDATE coin_predictions
        SET actual_price = ?
        WHERE coin_id = ? AND target_date = ?
    ''', (actual_price, coin_id, target_date))
    conn.commit()
    conn.close()

def get_predictions_for_coin(coin_id, days=5, db_path="predictions.sqlite"):
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute('''
        SELECT * FROM coin_predictions
        WHERE coin_id = ?
        ORDER BY prediction_date DESC, target_date DESC
        LIMIT ?
    ''', (coin_id, days))
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]
