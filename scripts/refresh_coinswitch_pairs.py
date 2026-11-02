"""
Refresh data/coinswitch_pairs.csv from CoinSwitch's API. Run this occasionally, by hand,
when CoinSwitch lists new coins. Needs COINSWITCH_API_KEY and COINSWITCH_API_SECRET.

    python scripts/refresh_coinswitch_pairs.py
"""

import os
import sys
import time
from pathlib import Path
from urllib.parse import unquote_plus, urlencode

import pandas as pd
import requests
from cryptography.hazmat.primitives.asymmetric import ed25519
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pricepredict.config import COINSWITCH_PAIRS_PATH  # noqa: E402

EXCHANGE = "coinswitchx"


def fetch_pairs():
    api_key = os.environ["COINSWITCH_API_KEY"]
    secret = ed25519.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(os.environ["COINSWITCH_API_SECRET"]))
    endpoint = "/trade/api/v2/coins?" + urlencode({"exchange": EXCHANGE})
    epoch = str(int(time.time() * 1000))
    signature = secret.sign(f"GET{unquote_plus(endpoint)}{epoch}".encode()).hex()
    response = requests.get(
        f"https://coinswitch.co{endpoint}",
        headers={
            "Content-Type": "application/json",
            "X-AUTH-SIGNATURE": signature,
            "X-AUTH-APIKEY": api_key,
            "X-AUTH-EPOCH": epoch,
        },
        json={},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["data"][EXCHANGE]


def main():
    load_dotenv()
    pairs = sorted(fetch_pairs())
    if not pairs:
        raise RuntimeError("CoinSwitch returned no pairs; keeping the existing file.")
    pd.DataFrame({"coinswitch_pair": pairs}).to_csv(COINSWITCH_PAIRS_PATH, index=False)
    print(f"Saved {len(pairs)} pairs to {COINSWITCH_PAIRS_PATH}")


if __name__ == "__main__":
    main()
