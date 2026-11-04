-- Paper trading: a record of every day's decision and every trade it led to.

-- One row per day the pipeline made a decision, written once and never replaced,
-- so a re-run cannot change what was proposed that day.
CREATE TABLE IF NOT EXISTS trade_signals (
    signal_date DATE PRIMARY KEY,
    published_at TIMESTAMPTZ NOT NULL,
    model_name TEXT NOT NULL,
    market_allows_buying BOOLEAN NOT NULL,
    picks_proposed INT NOT NULL
);

-- Days that already ran before this table existed. Each one stayed out, so none has picks.
INSERT INTO trade_signals (signal_date, published_at, model_name, market_allows_buying, picks_proposed)
SELECT DISTINCT ON (r.data_through)
       r.data_through,
       r.created_at AT TIME ZONE 'UTC',
       r.model_name,
       EXISTS (SELECT 1 FROM positions p WHERE p.open_date = r.data_through),
       (SELECT COUNT(*) FROM positions p WHERE p.open_date = r.data_through)
FROM model_runs r
ORDER BY r.data_through, r.id
ON CONFLICT (signal_date) DO NOTHING;

-- One row per proposed coin. The target and stop are stored with the trade, so
-- changing the settings later does not rewrite past results.
CREATE TABLE IF NOT EXISTS paper_trades (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    signal_date DATE NOT NULL REFERENCES trade_signals (signal_date),
    coin_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    pick_rank INT NOT NULL,
    model_name TEXT NOT NULL,
    predicted_return_pct DOUBLE PRECISION,
    price_source TEXT NOT NULL,
    entry_at TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    target_price DOUBLE PRECISION NOT NULL,
    stop_price DOUBLE PRECISION NOT NULL,
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed')),
    last_price DOUBLE PRECISION,
    exit_at TIMESTAMPTZ,
    exit_price DOUBLE PRECISION,
    exit_reason TEXT CHECK (exit_reason IN ('target', 'stop', 'time')),
    net_return_pct DOUBLE PRECISION,
    hold_return_pct DOUBLE PRECISION,
    UNIQUE (signal_date, coin_id)
);

CREATE INDEX IF NOT EXISTS ix_paper_trades_status ON paper_trades (status);

-- The same rules replayed over the latest backtest, replaced on every run. Not real signals.
CREATE TABLE IF NOT EXISTS replay_trades (
    signal_date DATE NOT NULL,
    coin_id TEXT NOT NULL,
    pick_rank INT NOT NULL,
    price_source TEXT NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    exit_date DATE NOT NULL,
    exit_price DOUBLE PRECISION NOT NULL,
    exit_reason TEXT NOT NULL,
    net_return_pct DOUBLE PRECISION NOT NULL,
    hold_return_pct DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (signal_date, coin_id)
);
