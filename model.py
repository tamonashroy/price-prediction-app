import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt

def add_features(df):
    df = df.copy()
    # Simple Moving Averages
    df['ma7'] = df['price'].rolling(window=7).mean()
    df['ma14'] = df['price'].rolling(window=14).mean()
    df['ma30'] = df['price'].rolling(window=30).mean()
    # Exponential Moving Averages
    df['ema7'] = df['price'].ewm(span=7, adjust=False).mean()
    df['ema14'] = df['price'].ewm(span=14, adjust=False).mean()
    # Daily returns
    df['daily_return'] = df['price'].pct_change()
    # Rolling volatility
    df['volatility7'] = df['daily_return'].rolling(window=7).std()
    df['volatility14'] = df['daily_return'].rolling(window=14).std()
    # RSI
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi14'] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = df['price'].ewm(span=12, adjust=False).mean()
    ema26 = df['price'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    # Add lag features for price, returns, and indicators
    for lag in [1, 2, 3, 7, 14]:
        df[f'price_lag{lag}'] = df['price'].shift(lag)
        df[f'daily_return_lag{lag}'] = df['daily_return'].shift(lag)
        df[f'ma7_lag{lag}'] = df['ma7'].shift(lag)
        df[f'ma14_lag{lag}'] = df['ma14'].shift(lag)
        df[f'macd_lag{lag}'] = df['macd'].shift(lag)
        df[f'rsi14_lag{lag}'] = df['rsi14'].shift(lag)
    # Drop rows with any NaN values from feature creation
    df = df.dropna()
    return df

def train_predictive_model(df, predict_return=False, use_grid_search=True, use_feature_selection=True, k_best=15):
    # Use all columns except 'date', 'coin_id', 'coin_name', and 'price'/'daily_return' as features
    features = [col for col in df.columns if col not in ['date', 'coin_id', 'coin_name', 'price', 'daily_return']]
    df = df.reset_index(drop=True)
    X = df[features]
    y = df['daily_return'] if predict_return else df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    # Feature selection
    if use_feature_selection and len(features) > k_best:
        selector = SelectKBest(score_func=f_regression, k=k_best)
        selector.fit(X_train, y_train)
        mask = selector.get_support()
        selected_features = [f for f, m in zip(features, mask) if m]
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        features = selected_features
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'LinearRegression': LinearRegression(),
        'SVR': SVR(),
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'AdaBoost': AdaBoostRegressor(random_state=42),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42),
        'KNeighbors': KNeighborsRegressor(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet()
    }
    mse_scores = {}
    best_model = None
    best_mse = float('inf')
    best_model_name = None
    for name, model in models.items():
        if name == 'RandomForest' and use_grid_search:
            param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5, None]}
            grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
        else:
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_scores[name] = mse
        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_model_name = name
    return best_model, features, mse_scores, best_model_name

def predict_next_days(df, model, days=5, features=None):
    df = df.copy()
    future_dates = []
    future_prices = []
    if features is None:
        raise ValueError("You must provide the features argument to predict_next_days to avoid KeyError. Use the same features as used for training.")
    for i in range(days):
        last_row = df.iloc[-1]
        next_date = pd.to_datetime(last_row['date']) + pd.Timedelta(days=1)
        # Build feature row for prediction
        feat_row = []
        for feat in features:
            if feat in df.columns:
                # For lags, use last available value; for rolling, use last available
                if 'lag' in feat:
                    lag_n = int(''.join(filter(str.isdigit, feat)))
                    if len(df) > lag_n:
                        feat_row.append(df['price'].iloc[-lag_n])
                    else:
                        feat_row.append(last_row['price'])
                else:
                    feat_row.append(last_row.get(feat, np.nan))
            else:
                feat_row.append(np.nan)
        features_df = pd.DataFrame([feat_row], columns=features)
        next_price = model.predict(features_df)[0]
        future_dates.append(next_date.date())
        future_prices.append(next_price)
        # Append new row for next iteration
        new_row = last_row.copy()
        new_row['date'] = next_date.date()
        new_row['price'] = next_price
        df = pd.concat([
            df,
            pd.DataFrame([new_row])
        ], ignore_index=True)
    return future_dates, future_prices

def plot_actual_vs_predicted_best(y_test, y_pred, test_dates, model_name):
    plt.figure(figsize=(10,5))
    plt.plot(test_dates, y_test.values, label='Actual')
    plt.plot(test_dates, y_pred, label=f'{model_name} Predicted')
    plt.title(f'Actual vs Predicted Bitcoin Price ({model_name})')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()

def plot_next_days_bar(future_dates, future_prices, last_actual_price):
    # Calculate movement direction and percent change
    colors = []
    labels = []
    for i in range(len(future_prices)):
        if i == 0:
            prev = last_actual_price
        else:
            prev = future_prices[i-1]
        pct_change = 100 * (future_prices[i] - prev) / prev if prev != 0 else 0
        label = f"{future_prices[i]:.2f}\n({pct_change:+.2f}%)"
        labels.append(label)
        if future_prices[i] > prev:
            colors.append('green')
        else:
            colors.append('red')
    plt.figure(figsize=(10,5))
    bars = plt.bar(future_dates, future_prices, color=colors)
    plt.title('Next 5 Days Bitcoin Price Prediction (Best Model)')
    plt.xlabel('Date')
    plt.ylabel('Predicted Price')
    plt.tight_layout()
    for bar, label in zip(bars, labels):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), label, ha='center', va='bottom', fontsize=9)
    plt.show()

def backtest_walk_forward(df, model_class, features, target_col='price', test_size=0.2, retrain_window=50, **model_kwargs):
    """
    Walk-forward backtesting: retrain model every 'retrain_window' steps, predict next value, and roll forward.
    Plots actual vs predicted for the test set.
    """
    df = df.reset_index(drop=True)
    y_true = []
    y_pred = []
    dates = []
    n = len(df)
    test_start = int(n * (1 - test_size))
    for i in range(test_start, n):
        train_idx = max(0, i - retrain_window)
        train = df.iloc[train_idx:i]
        test = df.iloc[[i]]
        X_train = train[features]
        y_train = train[target_col]
        X_test = test[features]
        # Fit model
        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]
        y_true.append(test[target_col].values[0])
        y_pred.append(pred)
        dates.append(test['date'].values[0] if 'date' in test.columns else i)
    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(dates, y_true, label='Actual')
    plt.plot(dates, y_pred, label='Predicted')
    plt.title(f'Walk-Forward Backtest ({model_class.__name__})')
    plt.xlabel('Date')
    plt.ylabel(target_col)
    plt.legend()
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()
    return dates, y_true, y_pred

def half_split_predict_and_plot(df, predict_return=False, k_best=15, plot=True):
    """
    Split data in half: train on first half, test on second half.
    Fit model, predict on test set, then predict next 5 days starting from the first test row.
    Plot actual vs predicted for test set and next 5 days. Return DataFrame of results.
    """
    df = df.copy()
    features = [col for col in df.columns if col not in ['date', 'coin_id', 'coin_name', 'price', 'daily_return']]
    y_col = 'daily_return' if predict_return else 'price'
    n = len(df)
    if n < 20:
        raise ValueError("Not enough data for half-split train/test.")
    split_idx = n // 2
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    # Feature selection
    X_train = train[features]
    y_train = train[y_col]
    X_test = test[features]
    y_test = test[y_col]
    if len(features) > k_best:
        selector = SelectKBest(score_func=f_regression, k=k_best)
        selector.fit(X_train, y_train)
        mask = selector.get_support()
        selected_features = [f for f, m in zip(features, mask) if m]
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        features = selected_features
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_dates = test['date'] if 'date' in test.columns else np.arange(len(y_test))
    # Plot actual vs predicted for test set
    if plot:
        plt.figure(figsize=(10,5))
        plt.plot(test_dates, y_test.values, label='Actual')
        plt.plot(test_dates, y_pred, label='Predicted')
        plt.title('Actual vs Predicted Price (Test Set, Half-Split)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.show()
    # Predict next 5 days from the first test row
    start_df = pd.concat([train, test.iloc[[0]]], ignore_index=True)
    future_dates, future_prices = predict_next_days(start_df, model, days=5, features=features)
    # Build table: actual (from test set) vs predicted for next 5 days
    actual_next5 = list(test[y_col].iloc[:5].values) if len(test) >= 5 else list(test[y_col].values) + [np.nan]*(5-len(test))
    results_df = pd.DataFrame({
        'date': future_dates,
        'actual': actual_next5,
        'predicted': future_prices
    })
    # Plot next 5 days
    if plot:
        plt.figure(figsize=(8,4))
        plt.plot(results_df['date'], results_df['actual'], marker='o', label='Actual')
        plt.plot(results_df['date'], results_df['predicted'], marker='x', label='Predicted')
        plt.title('Next 5 Days: Actual vs Predicted (from Test Start)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.show()
    return results_df

def last5_predict_and_plot(df, predict_return=False, k_best=15, plot=True):
    """
    Train on all data except the last 5 days, predict the last 5 days, and compare to actuals.
    Adds percent increase (actual, predicted) and percent difference columns.
    """
    df = df.copy()
    features = [col for col in df.columns if col not in ['date', 'coin_id', 'coin_name', 'price', 'daily_return']]
    y_col = 'daily_return' if predict_return else 'price'
    n = len(df)
    if n < 15:
        raise ValueError("Not enough data for last-5 prediction.")
    train = df.iloc[:-5]
    test = df.iloc[-5:]
    # Feature selection
    X_train = train[features]
    y_train = train[y_col]
    X_test = test[features]
    y_test = test[y_col]
    if len(features) > k_best:
        selector = SelectKBest(score_func=f_regression, k=k_best)
        selector.fit(X_train, y_train)
        mask = selector.get_support()
        selected_features = [f for f, m in zip(features, mask) if m]
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        features = selected_features
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # Rolling prediction for last 5 days
    preds = []
    rolling_df = train.copy()
    for i in range(5):
        feat_row = test.iloc[i][features].values.reshape(1, -1)
        pred = model.predict(feat_row)[0]
        preds.append(round(pred, 7))
    test_dates = test['date'] if 'date' in test.columns else np.arange(len(y_test))
    actuals = [round(val, 7) for val in y_test.values]
    # Percent increase (actual, predicted) from previous day
    pct_increase_actual = [None]
    pct_increase_pred = [None]
    for i in range(1, 5):
        pct_a = ((actuals[i] - actuals[i-1]) / actuals[i-1] * 100) if actuals[i-1] != 0 else None
        pct_p = ((preds[i] - preds[i-1]) / preds[i-1] * 100) if preds[i-1] != 0 else None
        pct_increase_actual.append(round(pct_a, 4) if pct_a is not None else None)
        pct_increase_pred.append(round(pct_p, 4) if pct_p is not None else None)
    # Percent difference (actual vs predicted)
    pct_diff = []
    for a, p in zip(actuals, preds):
        pct = ((p - a) / a * 100) if a != 0 else None
        pct_diff.append(round(pct, 4) if pct is not None else None)
    results_df = pd.DataFrame({
        'date': test_dates,
        'actual': actuals,
        'predicted': preds,
        'pct_increase_actual': pct_increase_actual,
        'pct_increase_predicted': pct_increase_pred,
        'pct_diff_pred_vs_actual': pct_diff
    })
    if plot:
        plt.figure(figsize=(8,4))
        plt.plot(results_df['date'], results_df['actual'], marker='o', label='Actual')
        plt.plot(results_df['date'], results_df['predicted'], marker='x', label='Predicted')
        plt.title('Last 5 Days: Actual vs Predicted')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.show()
    return results_df

# For further extension:
# - You can add XGBoost, LightGBM, or CatBoost models if desired
# - You can add neural network models (Keras, PyTorch) for experimentation
