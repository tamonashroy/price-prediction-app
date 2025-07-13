import pandas as pd
import sqlite3

DB = "predictions.sqlite"
PRICE_DB = "coin_prices.db"
POSITIONS_TABLE = "positions"
PREDICTIONS_TABLE = "coin_predictions"

# Load positions
conn = sqlite3.connect(DB)
pos_df = pd.read_sql_query(f"SELECT * FROM {POSITIONS_TABLE}", conn)
pred_df = pd.read_sql_query(f"SELECT * FROM {PREDICTIONS_TABLE}", conn)
conn.close()

# Load actual prices
conn = sqlite3.connect(PRICE_DB)
price_df = pd.read_sql_query("SELECT coin_id, date, price FROM coin_prices", conn)
conn.close()

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
