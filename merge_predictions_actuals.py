"""Merge predictions with actual prices from Azure SQL Database."""

import pandas as pd
from db_config import read_sql_query

# Read predictions from Azure SQL
def get_predictions():
    query = """
        SELECT coin_id, prediction_date, target_date, predicted_price, model_name
        FROM coin_predictions
    """
    return read_sql_query(query)

# Read actual prices from Azure SQL
def get_actuals():
    query = """
        SELECT coin_id, date, price
        FROM coin_prices
    """
    return read_sql_query(query)

def main():
    pred_df = get_predictions()
    actual_df = get_actuals()
    # Merge predictions with actuals on coin_id and target_date/date
    merged = pd.merge(
        pred_df,
        actual_df,
        left_on=["coin_id", "target_date"],
        right_on=["coin_id", "date"],
        how="left"
    )
    merged = merged.rename(columns={"price": "actual_price"})
    merged = merged.drop(columns=["date"])
    merged.to_csv("predictions_with_actuals.csv", index=False)
    print("Merged predictions and actuals written to predictions_with_actuals.csv")

if __name__ == "__main__":
    main()
