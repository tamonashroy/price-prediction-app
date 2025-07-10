import pandas as pd
from coinswitch_utils import get_coinswitch_coins
from db_utils import get_all_coins_from_db

def main():
    # Get all coins from local DB (coin_id, coin_name)
    db_coins = get_all_coins_from_db()
    db_coin_ids = {coin_id.lower(): coin_name for coin_name, coin_id in db_coins}
    # Fetch coinswitch coins (returns full API response for mapping)
    coinswitch_data = get_coinswitch_coins(exchange="coinswitchx", return_full_data=True)
    # Extract INR pairs
    inr_pairs = coinswitch_data.get('data', {}).get('coinswitchx', [])
    # Map to coin_id (before /INR, lowercased)
    mapping = []
    for pair in inr_pairs:
        symbol = pair.split('/')[0].lower()
        coin_name = db_coin_ids.get(symbol, None)
        mapping.append({
            'coinswitch_pair': pair,
            'coin_id': symbol,
            'coin_name': coin_name if coin_name else ''
        })
    df = pd.DataFrame(mapping)
    df.to_csv('coinswitch_coin_mapping.csv', index=False)
    print(f"Saved {len(df)} mapped coins to coinswitch_coin_mapping.csv")

if __name__ == "__main__":
    main()
