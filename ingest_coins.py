import requests
import pandas as pd
import time
import os
from sqlalchemy import create_engine, text
import sys

DB_PATH = "coin_prices.db"
engine = create_engine(f"sqlite:///{DB_PATH}")

COINS_LIMIT = 150
SLEEP_BETWEEN_REQUESTS = 5  # seconds
MAX_RETRIES = 5
RETRY_BACKOFF = 30  # seconds on 429

# Get top coins by volume

def get_top_coins(limit=COINS_LIMIT):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',  # switched to market cap
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

def add_technical_indicators(df):
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
    return df

def table_exists(engine, table_name):
    # Check if a table exists in the SQLite database
    with engine.connect() as conn:
        return engine.dialect.has_table(conn, table_name)

if __name__ == "__main__":
    # Only use coins from coinswitch mapping, sorted by market cap
    coinswitch_path = "coinswitch_coin_mapping.csv"
    coins = []
    if os.path.exists(coinswitch_path):
        coinswitch_df = pd.read_csv(coinswitch_path)
        # Remove rows with missing coin_id
        coinswitch_df = coinswitch_df[coinswitch_df['coin_id'].notnull() & (coinswitch_df['coin_id'] != '')]
        coin_ids = coinswitch_df['coin_id'].str.lower().unique().tolist()
        # Fetch market cap for all coinswitch coins from CoinGecko
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            'vs_currency': 'usd',
            'ids': ','.join(coin_ids),
            'order': 'market_cap_desc',
            'per_page': len(coin_ids),
            'page': 1,
            'sparkline': False
        }
        response = requests.get(url, params=params)
        data = response.json() if response.status_code == 200 else []
        # Build list of (coin_id, coin_name, market_cap)
        id_to_name = {str(row['coin_id']).lower(): str(row['coin_name']) if pd.notnull(row['coin_name']) and row['coin_name'] else str(row['coin_id']).upper() for _, row in coinswitch_df.iterrows()}
        coins_with_cap = [(c['id'], id_to_name.get(c['id'], c['id'].upper()), c.get('market_cap', 0)) for c in data if c['id'] in coin_ids]
        # Sort by market cap descending and take top 150
        coins_with_cap = sorted(coins_with_cap, key=lambda x: x[2] if x[2] is not None else 0, reverse=True)[:COINS_LIMIT]
        coins = [(cid, cname) for cid, cname, _ in coins_with_cap]
    else:
        print("coinswitch_coin_mapping.csv not found. Exiting.")
        sys.exit(1)
    for idx, (coin_id, coin_name) in enumerate(coins):
        print(f"[{idx+1}/{len(coins)}] Fetching {coin_name} ({coin_id})...")
        if not table_exists(engine, "coin_prices"):
            latest_date = None
        else:
            latest_date = get_latest_date_for_coin(coin_id)
        df = fetch_coin_data(coin_id, from_date=latest_date)
        if df is not None and not df.empty:
            df['coin_id'] = coin_id
            df['coin_name'] = coin_name
            # Add technical indicators before saving
            df = add_technical_indicators(df)
            df.to_sql("coin_prices", engine, if_exists="append", index=False)
            print(f"  Added {len(df)} new rows.")
        else:
            print(f"  No new data for {coin_name}.")
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    print("Ingestion complete. All coin data saved to coin_prices.db.")
    # Dispose engine to close all connections
    engine.dispose()
