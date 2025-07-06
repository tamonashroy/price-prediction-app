import requests
import pandas as pd
import time
import os
from sqlalchemy import create_engine, text

DB_PATH = "coin_prices.db"
engine = create_engine(f"sqlite:///{DB_PATH}")

COINS_LIMIT = 100
SLEEP_BETWEEN_REQUESTS = 5  # seconds
MAX_RETRIES = 5
RETRY_BACKOFF = 30  # seconds on 429

# Get top coins by volume

def get_top_coins(limit=COINS_LIMIT):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': 'usd',
        'order': 'volume_desc',
        'per_page': limit,
        'page': 1,
        'sparkline': False
    }
    response = requests.get(url, params=params)
    data = response.json()
    if not isinstance(data, list):
        print(f"Failed to fetch top coins: {data}")
        return []
    return [(coin['id'], coin['name']) for coin in data]

# Get latest date for a coin from DB

def get_latest_date_for_coin(coin_id):
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT MAX(date) FROM coin_prices WHERE coin_id = :coin_id
        """), {"coin_id": coin_id})
        row = result.fetchone()
        if row and row[0]:
            return str(row[0])
        return None

# Fetch daily price data for a coin, only for missing days

def fetch_coin_data(coin_id, from_date=None):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=365&interval=daily"
    for attempt in range(MAX_RETRIES):
        response = requests.get(url)
        if response.status_code == 429:
            print(f"Rate limit hit for {coin_id}, sleeping {RETRY_BACKOFF}s...")
            time.sleep(RETRY_BACKOFF)
            continue
        data = response.json()
        if 'prices' not in data:
            print(f"No prices for {coin_id}: {data}")
            return None
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
        df = df[['date', 'price']]
        if from_date:
            df = df[df['date'] > pd.to_datetime(from_date).date()]
        return df
    print(f"Failed to fetch data for {coin_id} after {MAX_RETRIES} retries.")
    return None

if __name__ == "__main__":
    coins = get_top_coins(COINS_LIMIT)
    for idx, (coin_id, coin_name) in enumerate(coins):
        print(f"[{idx+1}/{len(coins)}] Fetching {coin_name} ({coin_id})...")
        latest_date = get_latest_date_for_coin(coin_id)
        df = fetch_coin_data(coin_id, from_date=latest_date)
        if df is not None and not df.empty:
            df['coin_id'] = coin_id
            df['coin_name'] = coin_name
            df.to_sql("coin_prices", engine, if_exists="append", index=False)
            print(f"  Added {len(df)} new rows.")
        else:
            print(f"  No new data for {coin_name}.")
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    print("Ingestion complete. All coin data saved to coin_prices.db.")
