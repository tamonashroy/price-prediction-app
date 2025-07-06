import pandas as pd
from sqlalchemy import create_engine
import streamlit as st
import os

DB_PATH = "coin_prices.db"
engine = create_engine(f"sqlite:///{DB_PATH}")

# Utility to get all unique coins from the DB
def get_all_coins_from_db():
    if not os.path.exists(DB_PATH):
        st.error("Coin price database not found. Please run the ingestion script first.")
        return []
    df = pd.read_sql("SELECT DISTINCT coin_id, coin_name FROM coin_prices", engine)
    return [(row['coin_name'], row['coin_id']) for _, row in df.iterrows()]

# Utility to get price data for a coin from the DB
def get_coin_data_from_db(coin_id):
    if not os.path.exists(DB_PATH):
        st.error("Coin price database not found. Please run the ingestion script first.")
        return pd.DataFrame()
    df = pd.read_sql(f"SELECT date, price FROM coin_prices WHERE coin_id = ? ORDER BY date", engine, params=(coin_id,))
    return df

def load_predictions_from_db():
    if not os.path.exists(DB_PATH):
        return None
    try:
        df = pd.read_sql("SELECT * FROM predictions", engine)
        return df
    except Exception:
        return None

def save_predictions_to_db(results):
    df = pd.DataFrame(results)
    df.to_sql("predictions", engine, if_exists="replace", index=False)
