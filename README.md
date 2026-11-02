# Crypto 5-Day Picks

Daily machine-learning forecasts of which CoinSwitch-tradable coins should beat the market over the next 5 days, stored in Neon Postgres and shown on a Streamlit dashboard.

**Forecasts are experimental and are not financial advice.** See [how much to trust the picks](docs/DOCUMENTATION.md#8-how-much-to-trust-the-picks).

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
copy .env.example .env      # then set DATABASE_URL

streamlit run app.py        # the dashboard
python -m pricepredict.pipeline   # the daily job
pytest                      # the tests
```

## Documentation

Everything is in **[docs/DOCUMENTATION.md](docs/DOCUMENTATION.md)**: how it works, production setup, database access, deploying changes, and the model.
