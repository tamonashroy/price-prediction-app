import os
import requests
import time
from cryptography.hazmat.primitives.asymmetric import ed25519
from urllib.parse import urlparse, urlencode, unquote_plus
import urllib
from dotenv import load_dotenv
load_dotenv()

def get_coinswitch_coins(exchange="coinswitchx", return_full_data=False):
    """
    Fetches the list of available coins from the Coinswitch API for a given exchange.
    Requires COINSWITCH_API_KEY and COINSWITCH_API_SECRET in environment variables.
    Returns a list of coin symbols or IDs, or the full API response if return_full_data=True.
    """
    api_key = os.getenv('COINSWITCH_API_KEY')
    api_secret = os.getenv('COINSWITCH_API_SECRET')
    if not api_key or not api_secret:
        raise EnvironmentError("COINSWITCH_API_KEY and COINSWITCH_API_SECRET must be set in environment variables.")
    endpoint = "/trade/api/v2/coins"
    method = "GET"
    params = {"exchange": exchange}
    payload = {}
    epoch_time = str(int(time.time() * 1000))
    unquote_endpoint = endpoint
    if method == "GET" and len(params) != 0:
        endpoint += ('&', '?')[urlparse(endpoint).query == ''] + urlencode(params)
        unquote_endpoint = urllib.parse.unquote_plus(endpoint)
    signature_msg = method + unquote_endpoint + epoch_time
    request_string = bytes(signature_msg, 'utf-8')
    secret_key_bytes = bytes.fromhex(api_secret)
    secret_key = ed25519.Ed25519PrivateKey.from_private_bytes(secret_key_bytes)
    signature_bytes = secret_key.sign(request_string)
    signature = signature_bytes.hex()
    url = "https://coinswitch.co" + endpoint
    headers = {
        'Content-Type': 'application/json',
        'X-AUTH-SIGNATURE': signature,
        'X-AUTH-APIKEY': api_key,
        'X-AUTH-EPOCH': epoch_time
    }
    response = requests.get(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    if return_full_data:
        return data
    # The structure may vary; adjust as needed
    coins = [coin['symbol'] for coin in data.get('data', [])]
    return coins
