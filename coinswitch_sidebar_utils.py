import pandas as pd
from db_utils import get_coin_data_from_db
from model import add_features, train_predictive_model, predict_next_days

def get_top_coinswitch_green_1d(mapping_csv_path, top_n=5):
    # Load mapping of coinswitch pairs to coin_id and coin_name
    mapping = pd.read_csv(mapping_csv_path)
    results = []
    for _, row in mapping.iterrows():
        coin_id = row['coin_id']
        coin_name = row['coin_name'] if pd.notnull(row['coin_name']) and row['coin_name'] else coin_id.upper()
        if not coin_id or pd.isnull(coin_id):
            continue
        df = get_coin_data_from_db(coin_id)
        if df is None or df.empty or len(df) < 40:
            continue
        df_feat = add_features(df)
        if df_feat is None or df_feat.empty or len(df_feat) < 10:
            continue
        df_feat = df_feat.reset_index(drop=True)
        model, features_local, _, _ = train_predictive_model(df_feat)
        # Predict next 1 day
        future_dates, future_prices = predict_next_days(df_feat, model, days=1, features=features_local)
        last_actual_price = df_feat['price'].iloc[-1]
        pct_change = (future_prices[-1] - last_actual_price) / last_actual_price * 100 if last_actual_price != 0 else 0
        if future_prices[-1] > last_actual_price:
            results.append({
                'Coin': coin_name,
                'Pair': row['coinswitch_pair'],
                'Predicted Price': f"{future_prices[-1]:.2f}",
                'Predicted % Change': pct_change
            })
    # Sort by predicted % change descending and take top N
    results = sorted(results, key=lambda x: x['Predicted % Change'], reverse=True)[:top_n]
    # Format % change for display
    for r in results:
        r['Predicted % Change'] = f"{r['Predicted % Change']:+.2f}%"
    return pd.DataFrame(results)
