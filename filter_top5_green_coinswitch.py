"""Filter top 5 coins with positive predictions from coinswitch."""

import pandas as pd
from datetime import datetime
from sqlalchemy import MetaData, Table, Column, String, Float, DateTime, Integer, create_engine, select, insert, inspect
from db_config import get_sqlalchemy_engine

# Define table structures using SQLAlchemy
metadata = MetaData()

coin_predictions_table = Table(
    'coin_predictions',
    metadata,
    Column('coin_id', String),
    Column('prediction_date', DateTime),
    Column('target_date', DateTime),
    Column('predicted_price', Float),
    Column('model_name', String),
    autoload_with=None
)

positions_table = Table(
    'positions',
    metadata,
    Column('id', Integer, primary_key=True),
    Column('coin_id', String),
    Column('open_date', String),
    Column('close_date', String),
    Column('predicted_open_price', Float),
    Column('predicted_close_price', Float),
    Column('actual_open_price', Float),
    Column('actual_close_price', Float),
    Column('position_type', String),
    Column('profit_loss', Float),
    Column('created_at', DateTime),
    autoload_with=None
)

# Load coinswitch coin IDs
coinswitch_mapping = pd.read_csv('coinswitch_coin_mapping.csv')
coinswitch_coin_ids = set(coinswitch_mapping['coin_id'].dropna().str.lower())

# Fetch predictions for coinswitch coins from Azure SQL
engine = get_sqlalchemy_engine()
with engine.connect() as connection:
    query = select(
        coin_predictions_table.c.coin_id,
        coin_predictions_table.c.prediction_date,
        coin_predictions_table.c.target_date,
        coin_predictions_table.c.predicted_price,
        coin_predictions_table.c.model_name
    )
    pred_df = pd.read_sql(query, connection)

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
print('Filtered CSV written to top5_coinswitch_5day_predictions.csv using Azure SQL as source.')

# --- Populate positions table in Azure SQL ---
POSITIONS_TABLE = "positions"

def ensure_positions_table():
    """Create positions table if it doesn't exist."""
    engine = get_sqlalchemy_engine()
    
    # Use inspect() to check if table exists without holding a connection
    inspector = inspect(engine)
    table_exists = inspector.has_table(POSITIONS_TABLE)
    
    if not table_exists:
        # Create the table - this needs to be done without holding a connection in the pool
        metadata.create_all(engine, tables=[positions_table], checkfirst=True)
        print(f"Created {POSITIONS_TABLE} table in Azure SQL.")

def populate_positions_from_top5(csv_path="top5_coinswitch_5day_predictions.csv"):
    """Populate positions table from top 5 predictions CSV."""
    ensure_positions_table()
    df = pd.read_csv(csv_path)
    engine = get_sqlalchemy_engine()
    
    try:
        with engine.begin() as connection:
            for _, row in df.iterrows():
                coin_id = row['Coin']
                predicted_prices = [float(x) for x in row['5-Day Predicted Prices'].split(",")]
                open_date = datetime.utcnow().strftime("%Y-%m-%d")
                close_date = None  # Could be set to open_date + 5 days if needed
                predicted_open_price = predicted_prices[0]
                predicted_close_price = predicted_prices[-1]
                
                # Build SQLAlchemy insert statement
                stmt = insert(positions_table).values(
                    coin_id=coin_id,
                    open_date=open_date,
                    close_date=close_date,
                    predicted_open_price=predicted_open_price,
                    predicted_close_price=predicted_close_price,
                    position_type='long'
                )
                connection.execute(stmt)
            
            print(f"Populated {POSITIONS_TABLE} table with top 5 positions.")
    finally:
        engine.dispose()

populate_positions_from_top5()
