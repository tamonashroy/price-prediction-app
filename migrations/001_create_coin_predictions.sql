-- Azure SQL Database migration: Create tables for coin predictions and related data
-- Run this script in your Azure SQL Database to initialize tables

-- Create coin_prices table
IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = N'coin_prices')
BEGIN
    CREATE TABLE coin_prices (
        id INT PRIMARY KEY IDENTITY(1,1),
        coin_id NVARCHAR(MAX) NOT NULL,
        coin_name NVARCHAR(MAX),
        date DATE NOT NULL,
        price REAL NOT NULL,
        ma7 REAL,
        ma14 REAL,
        ma30 REAL,
        ema7 REAL,
        ema14 REAL,
        daily_return REAL,
        volatility7 REAL,
        volatility14 REAL,
        rsi14 REAL,
        macd REAL,
        macd_signal REAL,
        macd_hist REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT UQ_coin_prices UNIQUE (coin_id, date)
    )
END
GO

-- Create coin_predictions table
IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = N'coin_predictions')
BEGIN
    CREATE TABLE coin_predictions (
        id INT PRIMARY KEY IDENTITY(1,1),
        coin_id NVARCHAR(MAX) NOT NULL,
        prediction_date DATE NOT NULL,
        target_date DATE NOT NULL,
        predicted_price REAL NOT NULL,
        actual_price REAL,
        model_name NVARCHAR(MAX),
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT UQ_coin_predictions UNIQUE (coin_id, prediction_date, target_date)
    )
END
GO

-- Create positions table
IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = N'positions')
BEGIN
    CREATE TABLE positions (
        id INT PRIMARY KEY IDENTITY(1,1),
        coin_id NVARCHAR(MAX),
        open_date NVARCHAR(MAX),
        close_date NVARCHAR(MAX),
        predicted_open_price REAL,
        predicted_close_price REAL,
        actual_open_price REAL,
        actual_close_price REAL,
        position_type NVARCHAR(MAX),
        profit_loss REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
END
GO

-- Create indexes for better query performance
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_coin_prices_coin_id' AND object_id = OBJECT_ID('coin_prices'))
BEGIN
    CREATE INDEX IX_coin_prices_coin_id ON coin_prices(coin_id)
END
GO

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_coin_predictions_coin_id' AND object_id = OBJECT_ID('coin_predictions'))
BEGIN
    CREATE INDEX IX_coin_predictions_coin_id ON coin_predictions(coin_id)
END
GO

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_coin_predictions_prediction_date' AND object_id = OBJECT_ID('coin_predictions'))
BEGIN
    CREATE INDEX IX_coin_predictions_prediction_date ON coin_predictions(prediction_date)
END
GO

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_positions_coin_id' AND object_id = OBJECT_ID('positions'))
BEGIN
    CREATE INDEX IX_positions_coin_id ON positions(coin_id)
END
GO

PRINT 'Azure SQL Database tables created successfully!'
