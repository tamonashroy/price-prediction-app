import pandas as pd
import sqlite3
from datetime import datetime

PRED_DB = "coin_predictions.db"
PRICE_DB = "coin_prices.db"

# Read predictions from coin_predictions.db
def get_predictions():
    conn = sqlite3.connect(PRED_DB)
    df = pd.read_sql_query("""
        SELECT coin_id, prediction_date, target_date, predicted_price, model_name
        FROM coin_predictions
    """, conn)
    conn.close()
    return df

# Read actual prices from coin_prices.db
def get_actuals():
    conn = sqlite3.connect(PRICE_DB)
    df = pd.read_sql_query("""
        SELECT coin_id, date, price
        FROM coin_prices
    """, conn)
    conn.close()
    return df

def main():
    pred_df = get_predictions()
    actual_df = get_actuals()
    # Standardize coin_id and date formats
    pred_df['coin_id'] = pred_df['coin_id'].str.lower().str.strip()
    actual_df['coin_id'] = actual_df['coin_id'].str.lower().str.strip()
    pred_df['target_date'] = pd.to_datetime(pred_df['target_date']).dt.strftime('%Y-%m-%d')
    actual_df['date'] = pd.to_datetime(actual_df['date']).dt.strftime('%Y-%m-%d')
    # Merge predictions with actuals
    merged = pd.merge(
        pred_df,
        actual_df,
        left_on=["coin_id", "target_date"],
        right_on=["coin_id", "date"],
        how="left"
    )
    merged = merged.rename(columns={"price": "actual_price"})
    merged = merged.drop(columns=["date"])
    # For each prediction_date, get all coins with predictions for D (entry) and D+5 (exit)
    results = []
    for pred_date, group in merged.groupby('prediction_date'):
        group = group.sort_values(['coin_id', 'target_date'])
        # Find entry prices (D)
        entry_prices = group[group['target_date'] == pred_date].set_index('coin_id')['actual_price'].to_dict()
        # Find D+5 predictions and actuals
        group['days_ahead'] = (
            pd.to_datetime(group['target_date']) - pd.to_datetime(pred_date)
        ).dt.days
        d5 = group[group['days_ahead'] == 5]
        if d5.empty or not entry_prices:
            continue
        # Calculate % change for each coin
        d5['entry_price'] = d5['coin_id'].map(entry_prices)
        d5 = d5.dropna(subset=['entry_price', 'actual_price'])
        if d5.empty:
            continue
        d5['pct_change_pred'] = (d5['predicted_price'] - d5['entry_price']) / d5['entry_price'] * 100
        d5['pct_change_actual'] = (d5['actual_price'] - d5['entry_price']) / d5['entry_price'] * 100
        # Get top 5 coins by predicted % change
        top5 = d5.sort_values('pct_change_pred', ascending=False).head(5)
        # For each, record if actual % change is positive (profitable long)
        for _, row in top5.iterrows():
            results.append({
                'prediction_date': pred_date,
                'coin_id': row['coin_id'],
                'entry_price': row['entry_price'],
                'predicted_exit_price': row['predicted_price'],
                'actual_exit_price': row['actual_price'],
                'predicted_5d_pct_change': row['pct_change_pred'],
                'actual_5d_pct_change': row['pct_change_actual'],
                'profitable_long': row['pct_change_actual'] > 0,
                'profitable_short': row['pct_change_actual'] < 0,
                'model': row['model_name']
            })
    out_df = pd.DataFrame(results)
    if out_df.empty:
        print("No trades found with both entry and exit prices. Check your data and date alignment.")
        print(f"Sample merged DataFrame:\n{merged.head(10)}")
        print(f"Sample entry_prices dict: {entry_prices if 'entry_prices' in locals() else 'N/A'}")
        return
    # For each prediction_date, calculate how many of the 5 were profitable (long)
    summary = out_df.groupby('prediction_date').agg(
        total_trades=('coin_id', 'count'),
        profitable_trades=('profitable_long', 'sum'),
        profitable_shorts=('profitable_short', 'sum')
    ).reset_index()
    summary['profitable_long_ratio'] = summary['profitable_trades'] / summary['total_trades']
    summary['profitable_short_ratio'] = summary['profitable_shorts'] / summary['total_trades']
    summary.to_csv('top5_daily_trade_simulation_summary.csv', index=False)
    out_df.to_csv('top5_daily_trade_simulation_details.csv', index=False)
    print('Simulation summary written to top5_daily_trade_simulation_summary.csv')
    print('Simulation details written to top5_daily_trade_simulation_details.csv')

if __name__ == "__main__":
    main()
