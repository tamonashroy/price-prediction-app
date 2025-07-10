import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
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
    # Drop rows with any NaN values from feature creation
    df = df.dropna()
    return df

def train_predictive_model(df):
    # Use all columns except 'date', 'coin_id', 'coin_name', and 'price' as features
    features = [col for col in df.columns if col not in ['date', 'coin_id', 'coin_name', 'price']]
    df = df.reset_index(drop=True)  # Ensure indices are aligned
    X = df[features]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
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
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_scores[name] = mse
    best_model_name = min(mse_scores, key=mse_scores.get)
    return models[best_model_name], features, mse_scores, best_model_name

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
