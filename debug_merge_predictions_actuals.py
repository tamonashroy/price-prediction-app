"""Debug script to verify predictions and actuals alignment."""

import pandas as pd
from db_config import read_sql_query

# Read predictions from Azure SQL
def get_predictions():
    return read_sql_query("SELECT coin_id, prediction_date, target_date, predicted_price, model_name FROM coin_predictions")

# Read actual prices from Azure SQL
def get_actuals():
    return read_sql_query("SELECT coin_id, date, price FROM coin_prices")

def main():
    pred_df = get_predictions()
    actual_df = get_actuals()
    print("Sample predictions:")
    print(pred_df.head(10))
    print("\nSample actuals:")
    print(actual_df.head(10))
    print("\nUnique coin_ids in predictions:", pred_df['coin_id'].unique())
    print("Unique coin_ids in actuals:", actual_df['coin_id'].unique())
    print("\nSample prediction target_dates:", pred_df['target_date'].unique()[:10])
    print("Sample actual dates:", actual_df['date'].unique()[:10])
    # Check for any direct matches
    merged = pd.merge(
        pred_df,
        actual_df,
        left_on=["coin_id", "target_date"],
        right_on=["coin_id", "date"],
        how="inner"
    )
    print(f"\nRows after merge: {len(merged)}")
    if not merged.empty:
        print(merged.head(10))
    else:
        print("No matching rows found. Check for case/date format mismatches.")

if __name__ == "__main__":
    main()
