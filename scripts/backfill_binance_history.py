"""
One-off, for a new database: extend price history further back than CoinGecko's free
365-day window, and correct any stored price that disagrees with Binance.

    python scripts/backfill_binance_history.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pricepredict.binance import reconcile_prices  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    reconcile_prices(add_missing=True)
