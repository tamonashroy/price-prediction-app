-- Store volume and market cap, record model runs, and make forecasts and positions idempotent.

-- Indicators are now computed from prices at training time. The stored copies were
-- calculated only over each day's new rows, so most were wrong or empty.
ALTER TABLE coin_prices
    ADD COLUMN IF NOT EXISTS market_cap DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS total_volume DOUBLE PRECISION,
    DROP COLUMN IF EXISTS ma7,
    DROP COLUMN IF EXISTS ma14,
    DROP COLUMN IF EXISTS ma30,
    DROP COLUMN IF EXISTS ema7,
    DROP COLUMN IF EXISTS ema14,
    DROP COLUMN IF EXISTS daily_return,
    DROP COLUMN IF EXISTS volatility7,
    DROP COLUMN IF EXISTS volatility14,
    DROP COLUMN IF EXISTS rsi14,
    DROP COLUMN IF EXISTS macd,
    DROP COLUMN IF EXISTS macd_signal,
    DROP COLUMN IF EXISTS macd_hist;

CREATE INDEX IF NOT EXISTS ix_coin_prices_date ON coin_prices (date);

CREATE TABLE IF NOT EXISTS market_sentiment (
    date DATE PRIMARY KEY,
    fear_greed INT NOT NULL
);

-- Keep the newest forecast when the old pipeline wrote the same one more than once.
DELETE FROM coin_predictions a
USING coin_predictions b
WHERE a.coin_id = b.coin_id
  AND a.prediction_date = b.prediction_date
  AND a.target_date = b.target_date
  AND a.id < b.id;

ALTER TABLE coin_predictions
    ADD COLUMN IF NOT EXISTS predicted_return_pct DOUBLE PRECISION,
    ADD CONSTRAINT uq_coin_predictions UNIQUE (coin_id, prediction_date, target_date);

-- Positions now close after 5 days and are settled against real prices.
UPDATE positions SET open_date = LEFT(open_date, 10);

DELETE FROM positions a
USING positions b
WHERE a.coin_id = b.coin_id
  AND a.open_date = b.open_date
  AND a.id > b.id;

ALTER TABLE positions
    ALTER COLUMN coin_id SET NOT NULL,
    ALTER COLUMN open_date TYPE DATE USING open_date::DATE,
    ALTER COLUMN open_date SET NOT NULL,
    ALTER COLUMN close_date TYPE DATE USING close_date::DATE,
    ADD COLUMN IF NOT EXISTS pick_rank INT,
    ADD COLUMN IF NOT EXISTS predicted_return_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS model_name TEXT,
    ADD CONSTRAINT uq_positions UNIQUE (coin_id, open_date);

UPDATE positions
SET close_date = open_date + 5,
    model_name = COALESCE(model_name, 'legacy')
WHERE close_date IS NULL OR model_name IS NULL;

-- One row per pipeline run: what was chosen, how it scored, and what data it saw.
CREATE TABLE IF NOT EXISTS model_runs (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    data_through DATE NOT NULL,
    model_name TEXT NOT NULL,
    model_config JSONB,
    metrics JSONB,
    feature_importance JSONB
);

-- Return of each strategy in each period of the latest backtest, replaced on every run.
CREATE TABLE IF NOT EXISTS backtest_periods (
    strategy TEXT NOT NULL,
    period_start DATE NOT NULL,
    return_pct DOUBLE PRECISION,
    PRIMARY KEY (strategy, period_start)
);
