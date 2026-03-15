import os
import ccxt
import pandas as pd
import time
from datetime import datetime
import ta
from dotenv import load_dotenv
load_dotenv()

# --- CONFIGURATION ---


API_KEY = os.getenv('COINSWITCH_API_KEY')
API_SECRET = os.getenv('COINSWITCH_API_SECRET')

SYMBOL = 'BTC/USDT:USDT'  # Futures pair
TIMEFRAME = '1m'
TRADE_AMOUNT_USDT = 50
LEVERAGE = 5
MAX_TRADES_PER_HOUR = 5
SLEEP_INTERVAL = 60  # seconds

# --- Initialize Exchange ---
exchange = ccxt.coinswitchpro({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

def fetch_ohlcv():
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=100)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print("Fetch error:", e)
        return pd.DataFrame()

def apply_indicators(df):
    df['ema9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['ema21'] = ta.trend.ema_indicator(df['close'], window=21)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    return df

def generate_signal(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    if prev['ema9'] < prev['ema21'] and latest['ema9'] > latest['ema21'] and 40 < latest['rsi'] < 65:
        return "buy"
    elif prev['ema9'] > prev['ema21'] and latest['ema9'] < latest['ema21'] and 35 < latest['rsi'] < 60:
        return "sell"
    else:
        return None

def place_order(signal):
    try:
        price = exchange.fetch_ticker(SYMBOL)['last']
        quantity = round(TRADE_AMOUNT_USDT / price, 4)
        side = 'buy' if signal == 'buy' else 'sell'
        print(f"{datetime.now()} | {side.upper()} | Qty: {quantity} at ${price}")
        order = exchange.create_market_order(symbol=SYMBOL, side=side, amount=quantity)
        print("Order placed:", order)
    except Exception as e:
        print("Order error:", e)

# --- Main Loop ---
last_trade_time = datetime.now()
trade_count = 0

while True:
    df = fetch_ohlcv()
    if df.empty:
        time.sleep(SLEEP_INTERVAL)
        continue

    df = apply_indicators(df)
    signal = generate_signal(df)

    now = datetime.now()
    seconds_since_last = (now - last_trade_time).total_seconds()

    if signal:
        if trade_count < MAX_TRADES_PER_HOUR or seconds_since_last > 3600:
            place_order(signal)
            trade_count += 1
            last_trade_time = now
        else:
            print(f"{now} | Trade limit hit. Waiting...")

    time.sleep(SLEEP_INTERVAL)
