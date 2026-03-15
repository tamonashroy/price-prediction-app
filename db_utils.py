import pandas as pd
import streamlit as st
import os
from sqlalchemy import MetaData, Table, Column, String, Float, DateTime, select, func
from db_config import get_sqlalchemy_engine

# Define table structures using SQLAlchemy
metadata = MetaData()

coin_prices = Table(
    'coin_prices',
    metadata,
    Column('coin_id', String, primary_key=True),
    Column('coin_name', String),
    Column('date', DateTime),
    Column('price', Float),
    autoload_with=None  # Will be loaded from DB
)

predictions = Table(
    'predictions',
    metadata,
    autoload_with=None
)

# Utility to get all unique coins from the DB
def get_all_coins_from_db():
    try:
        engine = get_sqlalchemy_engine()
        with engine.connect() as connection:
            # Build SQLAlchemy query for distinct coins
            query = select(coin_prices.c.coin_id, coin_prices.c.coin_name).distinct()
            df = pd.read_sql(query, connection)
            return [(row['coin_name'], row['coin_id']) for _, row in df.iterrows()]
    except Exception as e:
        st.error(f"Error fetching coins from database: {str(e)}")
        return []

# Utility to get price data for a coin from the DB
def get_coin_data_from_db(coin_id):
    try:
        engine = get_sqlalchemy_engine()
        with engine.connect() as connection:
            # Build SQLAlchemy query for coin prices
            query = select(coin_prices.c.date, coin_prices.c.price).where(
                coin_prices.c.coin_id == coin_id
            ).order_by(coin_prices.c.date)
            df = pd.read_sql(query, connection)
            return df
    except Exception as e:
        st.error(f"Error fetching price data for {coin_id}: {str(e)}")
        return pd.DataFrame()

def load_predictions_from_db():
    try:
        engine = get_sqlalchemy_engine()
        with engine.connect() as connection:
            # Build SQLAlchemy query for all predictions
            query = select(predictions)
            df = pd.read_sql(query, connection)
            return df
    except Exception:
        return None

def save_predictions_to_db(results):
    """Save predictions to Azure SQL Database using SQLAlchemy."""
    try:
        df = pd.DataFrame(results)
        engine = get_sqlalchemy_engine()
        try:
            df.to_sql("predictions", engine, if_exists="replace", index=False)
        finally:
            engine.dispose()
    except Exception as e:
        st.error(f"Error saving predictions: {str(e)}")
