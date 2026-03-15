"""Create the coin_predictions table in Azure SQL Database."""

from db_config import get_pyodbc_connection, table_exists

def create_coin_predictions_table():
    """Create the coin_predictions table in Azure SQL if it doesn't exist."""
    conn = get_pyodbc_connection()
    try:
        if not table_exists("coin_predictions", conn):
            query = """
                CREATE TABLE coin_predictions (
                    id INT PRIMARY KEY IDENTITY(1,1),
                    coin_id NVARCHAR(MAX),
                    prediction_date DATE,
                    target_date DATE,
                    predicted_price REAL,
                    model_name NVARCHAR(MAX)
                )
            """
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
            print("Table 'coin_predictions' created in Azure SQL Database.")
        else:
            print("Table 'coin_predictions' already exists.")
    finally:
        conn.close()

if __name__ == "__main__":
    create_coin_predictions_table()
