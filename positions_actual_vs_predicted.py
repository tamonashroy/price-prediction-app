"""Compare actual vs predicted positions and calculate P&L."""

import pandas as pd
from sqlalchemy import MetaData, Table, Column, String, Float, DateTime, Integer, select
from db_config import get_sqlalchemy_engine

# Define table structures using SQLAlchemy
metadata = MetaData()

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

predictions_table = Table(
    'coin_predictions',
    metadata,
    Column('coin_id', String),
    Column('prediction_date', DateTime),
    Column('target_date', DateTime),
    Column('predicted_price', Float),
    Column('model_name', String),
    autoload_with=None
)

coin_prices_table = Table(
    'coin_prices',
    metadata,
    Column('coin_id', String),
    Column('date', DateTime),
    Column('price', Float),
    autoload_with=None
)

POSITIONS_TABLE = "positions"
PREDICTIONS_TABLE = "coin_predictions"

# Load positions and predictions using SQLAlchemy
engine = get_sqlalchemy_engine()

with engine.connect() as connection:
    # Load positions
    pos_query = select(positions_table)
    pos_df = pd.read_sql(pos_query, connection)
    
    # Load predictions
    pred_query = select(predictions_table)
    pred_df = pd.read_sql(pred_query, connection)
    
    # Load actual prices
    price_query = select(
        coin_prices_table.c.coin_id,
        coin_prices_table.c.date,
        coin_prices_table.c.price
    )
    price_df = pd.read_sql(price_query, connection)

results = []
for _, pos in pos_df.iterrows():
    coin_id = pos['coin_id']
    open_date = pos['open_date']
    # For this example, assume close_date is open_date + 5 days
    close_date = pos['close_date']
    if not close_date:
        close_date = pd.to_datetime(open_date) + pd.Timedelta(days=4)
        close_date = close_date.strftime('%Y-%m-%d')
    # Get actual open/close prices
    actual_open = price_df[(price_df['coin_id'] == coin_id) & (price_df['date'] == open_date)]['price']
    actual_close = price_df[(price_df['coin_id'] == coin_id) & (price_df['date'] == close_date)]['price']
    actual_open = actual_open.iloc[0] if not actual_open.empty else None
    actual_close = actual_close.iloc[0] if not actual_close.empty else None
    # Get predicted open/close prices from predictions table
    pred_open = pred_df[(pred_df['coin_id'] == coin_id) & (pred_df['target_date'] == open_date)]['predicted_price']
    pred_close = pred_df[(pred_df['coin_id'] == coin_id) & (pred_df['target_date'] == close_date)]['predicted_price']
    pred_open = pred_open.iloc[0] if not pred_open.empty else None
    pred_close = pred_close.iloc[0] if not pred_close.empty else None
    # Calculate P&L %
    actual_pl = ((actual_close - actual_open) / actual_open * 100) if actual_open and actual_close else None
    pred_pl = ((pred_close - pred_open) / pred_open * 100) if pred_open and pred_close else None
    results.append({
        'coin_id': coin_id,
        'open_date': open_date,
        'close_date': close_date,
        'actual_open_price': actual_open,
        'actual_close_price': actual_close,
        'predicted_open_price': pred_open,
        'predicted_close_price': pred_close,
        'actual_profit_loss_%': actual_pl,
        'predicted_profit_loss_%': pred_pl
    })

out_df = pd.DataFrame(results)
out_df.to_csv('positions_actual_vs_predicted.csv', index=False)
print('positions_actual_vs_predicted.csv written.')
