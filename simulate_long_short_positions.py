"""Simulate long and short trading positions based on predictions."""

import pandas as pd
from datetime import datetime, timedelta
from db_config import read_sql_query

# Connect to DB and get all predictions with actuals (for all days, all coins)
def get_predictions_with_actuals():
    query = """
        SELECT 
            coin_id, prediction_date, target_date, predicted_price, model_name, actual_price
        FROM coin_predictions
        ORDER BY prediction_date DESC, coin_id, target_date
    """
    return read_sql_query(query)

def simulate_positions(df):
    # For each prediction_date and coin, simulate both long and short
    results = []
    for (coin, pred_date), group in df.groupby(['coin_id', 'prediction_date']):
        group = group.sort_values('target_date')
        if group.empty or group['actual_price'].isnull().all():
            continue
        # Entry price is actual price on prediction_date (if available)
        entry_row = group[group['target_date'] == group['prediction_date']]
        if entry_row.empty or pd.isnull(entry_row.iloc[0]['actual_price']):
            continue
        entry_price = entry_row.iloc[0]['actual_price']
        for i, row in group.iterrows():
            if row['target_date'] == row['prediction_date']:
                continue  # skip entry day
            # Long position
            long_diff = (row['actual_price'] - entry_price) / entry_price * 100 if entry_price else None
            # Short position
            short_diff = (entry_price - row['actual_price']) / entry_price * 100 if entry_price else None
            results.append({
                'coin_id': coin,
                'date_bought': row['prediction_date'],
                'target_date': row['target_date'],
                'entry_price': entry_price,
                'predicted_price': row['predicted_price'],
                'actual_price': row['actual_price'],
                'long_diff_pct': long_diff,
                'short_diff_pct': short_diff,
                'model': row['model_name']
            })
    return pd.DataFrame(results)

def main():
    df = get_predictions_with_actuals()
    sim_df = simulate_positions(df)
    sim_df = sim_df.sort_values(['coin_id', 'date_bought', 'target_date'])
    sim_df.to_csv('daily_position_simulation.csv', index=False)
    print('Simulation results written to daily_position_simulation.csv')

if __name__ == "__main__":
    main()
