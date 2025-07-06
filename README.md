# Crypto Price Prediction App

This project ingests daily price data for the top 100 traded cryptocurrencies from CoinGecko, stores it in a local SQLite database, and provides daily price prediction analytics and visualizations via a Streamlit web app.

## Prerequisites
- Docker Desktop (with Linux containers enabled)
- WSL (Windows Subsystem for Linux) or any Linux terminal

## 1. Clone the Repository
```
git clone <your-repo-url>
cd <your-repo-folder>
```

## 2. Build the Docker Image
```
docker build -t crypto-predictor .
```

## 3. Run the Data Ingestion Script
This will fetch the latest price data for the top 100 coins and store it in `coin_prices.db`.

```
docker run --rm -v $(pwd):/app crypto-predictor python ingest_coins.py
```
- The `-v $(pwd):/app` flag mounts your project directory so the database file is updated on your host.

## 4. Run the Streamlit App
```
docker run -p 8501:8501 -v $(pwd):/app crypto-predictor
```
- Open your browser and go to [http://localhost:8501](http://localhost:8501)

## 5. Updating Data
To update the database with new prices, rerun the ingestion command from step 3.

## 6. Project Structure
- `app.py` — Streamlit web app
- `ingest_coins.py` — Data ingestion script
- `db_utils.py` — Database utility functions
- `model.py` — Feature engineering and predictive modeling
- `coin_prices.db` — SQLite database (auto-created)
- `requirements.txt` — Python dependencies
- `Dockerfile` — Docker build instructions

## Troubleshooting
- If you see database errors, ensure you have run the ingestion script before starting the app.
- If you hit API rate limits, wait a few minutes and try again.

---

**Enjoy your crypto analytics dashboard!**
