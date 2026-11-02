-- The coins that can be picked today, refreshed by each ingest. coin_prices also holds
-- history for coins outside this list, which remains useful for training.
CREATE TABLE IF NOT EXISTS coins (
    coin_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    name TEXT
);
