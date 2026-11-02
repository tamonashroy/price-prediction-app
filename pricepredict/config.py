"""Project settings. Secrets come from the environment, never from this file."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = PROJECT_ROOT / "migrations"
COINSWITCH_PAIRS_PATH = PROJECT_ROOT / "data" / "coinswitch_pairs.csv"
LOGO_PATH = PROJECT_ROOT / "assets" / "logo.png"

# Ingestion
UNIVERSE_SIZE = 150  # top CoinSwitch coins by market cap
# Stablecoins, gold tokens and wrapped coins do not offer independent 5-day growth.
EXCLUDED_SYMBOLS = frozenset(
    {"usdt", "usdc", "dai", "tusd", "fdusd", "usde", "usds", "usdd", "busd", "pyusd", "usdp", "frax", "gusd",
     "lusd", "crvusd", "usd1", "rlusd", "usdf", "usdg", "xaut", "xaum", "paxg", "wbtc", "weth", "steth", "wsteth", "cbbtc"}
)
MAX_HISTORY_DAYS = 365  # the most CoinGecko's free API returns
COINGECKO_SLEEP_SECONDS = 6.5  # keeps us under the keyless rate limit
COINGECKO_DEMO_SLEEP_SECONDS = 2.1  # a free demo key allows 30 calls a minute
COINGECKO_MAX_RETRIES = 5
COINGECKO_RETRY_BACKOFF_SECONDS = 30
MIN_FRESH_COINS = 50  # below this the pipeline fails instead of predicting from stale data
FRESHNESS_DAYS = 2
MAX_DAILY_PRICE_RATIO = 5.0  # a bigger one-day move in either direction is treated as bad data
MAX_SOURCE_DISAGREEMENT = 1.25  # correct a stored price further than this from Binance's
DAILY_RECONCILE_DAYS = 10
GLITCH_MOVE_RATIO = 1.3  # a one-day move at least this big that immediately reverses...
GLITCH_RETURN_RATIO = 1.15  # ...to within this of where it started is treated as a glitch

# Forecasting
HORIZONS = (1, 2, 3, 4, 5)
PICK_HORIZON = 5
TOP_N = 5
MIN_COIN_HISTORY_DAYS = 90
MIN_DAILY_VOLUME_USD = 1_000_000  # thinly traded coins cannot be bought at the quoted price

# Round-trip trading cost assumed in the backtest and in live results.
# Check this against your exchange's fees; it excludes taxes.
ROUND_TRIP_COST = 0.01

# Model selection
AUTOML_TIME_BUDGET_SECONDS = 300
AUTOML_ESTIMATORS = ("lgbm", "xgboost")
AUTOML_TRAIN_WINDOW_DAYS = 1095
RANDOM_SEED = 42

# Backtest
BACKTEST_DAYS = 365
BACKTEST_RETRAIN_EVERY_DAYS = 30
TRAIN_WINDOW_DAYS = 1095  # three years beat one year and all history in the design tests
