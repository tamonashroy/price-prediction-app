import pandas as pd
import os
from model import add_features, train_predictive_model, predict_next_days, last5_predict_and_plot
from db_utils import get_all_coins_from_db, get_coin_data_from_db

def main():
    results = []
    last5_results = []
    coins = get_all_coins_from_db()
    print(f"Found {len(coins)} coins in the database.")
    for idx, (coin_name, coin_id) in enumerate(coins):
        print(f"[{idx+1}/{len(coins)}] Processing {coin_name} ({coin_id})...")
        df = get_coin_data_from_db(coin_id)
        if df is None or df.empty or len(df) < 40:
            print(f"  Skipping {coin_name}: not enough data.")
            continue
        df_feat = add_features(df)
        if df_feat is None or df_feat.empty or len(df_feat) < 10:
            print(f"  Skipping {coin_name}: not enough data after feature engineering.")
            continue
        df_feat = df_feat.reset_index(drop=True)
        print(f"  Training model for {coin_name}...")
        model, features, _, best_model_name = train_predictive_model(df_feat)
        print(f"  Model trained: {best_model_name}")
        future_dates, future_prices = predict_next_days(df_feat, model, days=5, features=features)
        last_actual_price = df_feat['price'].iloc[-1]
        pct_change = (future_prices[-1] - last_actual_price) / last_actual_price * 100 if last_actual_price != 0 else 0
        results.append({
            'Coin': coin_name,
            '5-Day Predicted Prices': ", ".join([f"{p:.7f}" for p in future_prices]),
            '5-Day % Change': f"{pct_change:+.7f}%"
        })
        # Last 5 days actual vs predicted
        try:
            print(f"  Running last-5-days prediction for {coin_name}...")
            last5_df = last5_predict_and_plot(df_feat, plot=False)
            for i, row in last5_df.iterrows():
                last5_results.append({
                    'Coin': coin_name,
                    'Date': row['date'],
                    'Actual': row['actual'],
                    'Predicted': row['predicted'],
                    'pct_increase_actual': row.get('pct_increase_actual'),
                    'pct_increase_predicted': row.get('pct_increase_predicted'),
                    'pct_diff_pred_vs_actual': row.get('pct_diff_pred_vs_actual')
                })
        except Exception as e:
            print(f"  Error in last-5-days prediction for {coin_name}: {e}")
            continue
    # Save results
    pd.DataFrame(results).to_csv('batch_5day_predictions.csv', index=False)
    pd.DataFrame(last5_results).to_csv('batch_last5_actual_vs_predicted.csv', index=False)
    print("Batch predictions complete. Results saved to CSV.")

if __name__ == "__main__":
    main()
