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
    df['price_lag1'] = df['price'].shift(1)
    df['price_ma7'] = df['price'].rolling(window=7).mean()
    df['price_ma30'] = df['price'].rolling(window=30).mean()
    df = df.dropna()
    return df

def train_predictive_model(df):
    features = ['price_lag1', 'price_ma7', 'price_ma30']
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
    lasso_pred = None
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_scores[name] = mse
        print(f"{name} Test MSE: {mse:.2f}")
        if name == 'Lasso':
            lasso_pred = y_pred
    best_model_name = min(mse_scores, key=mse_scores.get)
    print(f"\nBest model: {best_model_name} with MSE: {mse_scores[best_model_name]:.2f}")
    # Plot actual vs predicted for Lasso
    if lasso_pred is not None:
        plt.figure(figsize=(10,5))
        test_dates = df.loc[X_test.index, 'date'] if 'date' in df.columns else X_test.index
        plt.plot(test_dates, y_test.values, label='Actual')
        plt.plot(test_dates, lasso_pred, label='Lasso Predicted')
        plt.title('Actual vs Predicted Bitcoin Price (Lasso)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.show()
    return models[best_model_name]

def predict_next_days(df, model, days=5):
    df = df.copy()
    future_dates = []
    future_prices = []
    for i in range(days):
        last_row = df.iloc[-1]
        next_date = pd.to_datetime(last_row['date']) + pd.Timedelta(days=1)
        price_lag1 = last_row['price']
        price_ma7 = df['price'].tail(7).mean()
        price_ma30 = df['price'].tail(30).mean() if len(df) >= 30 else df['price'].mean()
        features = np.array([[price_lag1, price_ma7, price_ma30]])
        next_price = model.predict(features)[0]
        future_dates.append(next_date.date())
        future_prices.append(next_price)
        # Append new row for next iteration
        df = pd.concat([
            df,
            pd.DataFrame({'date': [next_date.date()], 'price': [next_price]})
        ], ignore_index=True)
    print("\nNext 5 days price prediction:")
    for d, p in zip(future_dates, future_prices):
        print(f"{d}: {p:.2f}")
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
