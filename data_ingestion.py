import requests
import pandas as pd
from datetime import datetime

def fetch_bitcoin_data(days=365):
    """
    Fetch historical Bitcoin price data (daily) from CoinGecko API.
    """
    # CoinGecko API now allows up to 365 days for free users
    if days > 365:
        print("Warning: CoinGecko API supports up to 365 days for daily data (free plan). Using 365 days.")
        days = 365
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days={days}&interval=daily"
    response = requests.get(url)
    data = response.json()
    if 'prices' not in data:
        print("Error: API response does not contain 'prices'. Response:", data)
        return pd.DataFrame(columns=['date', 'price'])
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
    df = df[['date', 'price']]
    return df

if __name__ == "__main__":
    df = fetch_bitcoin_data()
    print(df.head())
