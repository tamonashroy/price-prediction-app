"""
Azure SQL Database utilities for coin predictions.
Replaces SQLite with Azure SQL Database using pyodbc.
"""

from db_config import get_pyodbc_connection, execute_query, table_exists

def get_db_connection():
    """Get a connection to Azure SQL Database."""
    return get_pyodbc_connection()

def create_predictions_table():
    """Create the coin_predictions table in Azure SQL if it doesn't exist."""
    conn = get_db_connection()
    try:
        if not table_exists("coin_predictions", conn):
            query = """
                CREATE TABLE coin_predictions (
                    id INT PRIMARY KEY IDENTITY(1,1),
                    coin_id NVARCHAR(MAX) NOT NULL,
                    prediction_date DATE NOT NULL,
                    target_date DATE NOT NULL,
                    predicted_price REAL NOT NULL,
                    actual_price REAL,
                    model_name NVARCHAR(MAX),
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
    finally:
        conn.close()

def insert_coin_predictions(predictions):
    """
    Insert coin predictions into Azure SQL.
    predictions: list of dicts with keys: coin_id, prediction_date, target_date, predicted_price, model_name
    """
    if not predictions:
        return
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO coin_predictions (coin_id, prediction_date, target_date, predicted_price, model_name)
            VALUES (?, ?, ?, ?, ?)
        """
        for pred in predictions:
            cursor.execute(query, (
                pred['coin_id'],
                pred['prediction_date'],
                pred['target_date'],
                pred['predicted_price'],
                pred.get('model_name')
            ))
        conn.commit()
    finally:
        conn.close()

def update_actual_price(coin_id, target_date, actual_price):
    """Update the actual price for a prediction."""
    conn = get_db_connection()
    try:
        query = """
            UPDATE coin_predictions
            SET actual_price = ?
            WHERE coin_id = ? AND target_date = ?
        """
        cursor = conn.cursor()
        cursor.execute(query, (actual_price, coin_id, target_date))
        conn.commit()
    finally:
        conn.close()

def get_predictions_for_coin(coin_id, days=5):
    """Get recent predictions for a specific coin."""
    conn = get_db_connection()
    try:
        query = """
            SELECT TOP ? *
            FROM coin_predictions
            WHERE coin_id = ?
            ORDER BY prediction_date DESC, target_date DESC
        """
        cursor = conn.cursor()
        cursor.execute(query, (days, coin_id))
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]
    finally:
        conn.close()
