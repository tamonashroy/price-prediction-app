from data_ingestion import fetch_bitcoin_data
from model import add_features, train_predictive_model, predict_next_days, plot_next_days_bar, plot_actual_vs_predicted_best
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Step 1: Ingest data
    df = fetch_bitcoin_data(days=365)
    print("Sample data:")
    print(df.head())

    # Step 2: Feature engineering
    df_feat = add_features(df)
    print("\nSample with features:")
    print(df_feat.head())

    # Step 3: Train and evaluate model
    features = ['price_lag1', 'price_ma7', 'price_ma30']
    df_feat = df_feat.reset_index(drop=True)
    X = df_feat[features]
    y = df_feat['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = train_predictive_model(df_feat)
    # Get best model's predictions for test set
    y_pred = model.predict(X_test)
    test_dates = df_feat.loc[X_test.index, 'date'] if 'date' in df_feat.columns else X_test.index
    plot_actual_vs_predicted_best(y_test, y_pred, test_dates, type(model).__name__)

    # Step 4: Predict next 5 days
    future_dates, future_prices = predict_next_days(df_feat, model, days=5)
    # Step 5: Plot bar chart for next 5 days
    last_actual_price = df_feat['price'].iloc[-1]
    plot_next_days_bar(future_dates, future_prices, last_actual_price)
