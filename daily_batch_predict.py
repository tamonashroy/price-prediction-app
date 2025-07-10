import pandas as pd
import os
from model import add_features, train_predictive_model, predict_next_days, last5_predict_and_plot
from db_utils import get_all_coins_from_db, get_coin_data_from_db

def main():
    results = []
    last5_results = []
    coins = get_all_coins_from_db()
    for coin_name, coin_id in coins:
        df = get_coin_data_from_db(coin_id)
        if df is None or df.empty or len(df) < 40:
            continue
        df_feat = add_features(df)
        if df_feat is None or df_feat.empty or len(df_feat) < 10:
            continue
        df_feat = df_feat.reset_index(drop=True)
        # Best model for 5-day prediction
        model, features, _, _ = train_predictive_model(df_feat)
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
            last5_df = last5_predict_and_plot(df_feat, plot=False)
            for i, row in last5_df.iterrows():
                last5_results.append({
                    'Coin': coin_name,
                    'Date': row['date'],
                    'Actual': row['actual'],
                    'Predicted': row['predicted']
                })
        except Exception:
            continue
    # Save results
    pd.DataFrame(results).to_csv('batch_5day_predictions.csv', index=False)
    pd.DataFrame(last5_results).to_csv('batch_last5_actual_vs_predicted.csv', index=False)
    print("Batch predictions complete. Results saved to CSV.")

if __name__ == "__main__":
    main()
