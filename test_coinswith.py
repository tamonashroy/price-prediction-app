from cryptography.hazmat.primitives.asymmetric import ed25519
from urllib.parse import urlparse, urlencode
import urllib
import json
import requests
import time

params = {
    "count": 20,
    "from_time": 1600261657954,
    "to_time": 1687261657954,
    "side": "sell",
    "symbols": "btc/inr,eth/inr",
    "exchanges": "coinswitchx,wazirx",
    "type": "limit",
    "open": True
}

endpoint = "/trade/api/v2/orders"
method = "GET"
payload = {}

params = {
    "exchange": "coinswitchx",
}
payload = {}

endpoint = "/trade/api/v2/coins"

# endpoint += ('&', '?')[urlparse(endpoint).query == ''] + urlencode(params)

# url = "https://coinswitch.co" + endpoint




epoch_time = str(int(time.time() * 1000))

unquote_endpoint = endpoint
if method == "GET" and len(params) != 0:
    endpoint += ('&', '?')[urlparse(endpoint).query == ''] + urlencode(params)
    unquote_endpoint = urllib.parse.unquote_plus(endpoint)

signature_msg = method + unquote_endpoint + epoch_time

request_string = bytes(signature_msg, 'utf-8')
secret_key_bytes = bytes.fromhex(secret_key)
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

response = requests.request("GET", url, headers=headers, json=payload)
print(response.text)