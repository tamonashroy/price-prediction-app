import pandas as pd
import os
from model import add_features, train_predictive_model, predict_next_days, last5_predict_and_plot, store_predictions_for_coin
from db_utils import get_all_coins_from_db, get_coin_data_from_db
from db_predictions import create_predictions_table
from datetime import date, timedelta

def main():
    create_predictions_table()  # Ensure table exists before storing predictions
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
        # Select minimal and most optimal set of features
        minimal_features = [
            'price_lag1', 'ma7', 'ema7', 'rsi14', 'macd', 'volatility7'
        ]
        # Only keep features that exist in df_feat
        minimal_features = [f for f in minimal_features if f in df_feat.columns]
        print(f"  Training model for {coin_name} with features: {minimal_features}")
        # Train and select best model using only minimal features
        model, features, mse_scores, best_model_name = train_predictive_model(
            df_feat[minimal_features + ['date', 'price'] if 'date' in df_feat.columns else minimal_features + ['price']],
            use_grid_search=True, use_feature_selection=False
        )
        print(f"  Model trained: {best_model_name} (MSEs: {mse_scores})")
        future_dates, future_prices = predict_next_days(df_feat, model, days=5, features=minimal_features)
        # Store predictions in coin_predictions table
        store_predictions_for_coin(coin_id, future_prices, best_model_name, prediction_date=date.today())
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
    # Save Top 5 Coinswitch Coins Predicted to be Green (Next 5 Days) to CSV
    coinswitch_mapping_path = "coinswitch_coin_mapping.csv"
    coinswitch_coin_ids = set()
    if os.path.exists(coinswitch_mapping_path):
        coinswitch_mapping = pd.read_csv(coinswitch_mapping_path)
        coinswitch_coin_ids = set(coinswitch_mapping['coin_id'].dropna().str.lower())
    if results:
        results_df = pd.DataFrame(results)
        if not results_df.empty and coinswitch_coin_ids:
            filtered_df = results_df[results_df['Coin'].str.lower().isin(coinswitch_coin_ids)]
            # Convert to float for sorting and filtering
            filtered_df['5-Day % Change'] = filtered_df['5-Day % Change'].str.replace('%','').astype(float)
            # Only keep coins with positive 5-Day % Change (predicted to be green)
            filtered_df = filtered_df[filtered_df['5-Day % Change'] > 0]
            # Sort by 5-Day % Change descending
            filtered_df = filtered_df.sort_values('5-Day % Change', ascending=False)
            # Write to CSV (top 5 or all if less)
            filtered_df.head(5).to_csv('top5_coinswitch_5day_predictions.csv', index=False)

if __name__ == "__main__":
    main()
