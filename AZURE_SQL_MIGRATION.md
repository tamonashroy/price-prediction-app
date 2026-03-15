# Azure SQL Database Migration Guide

This guide explains how to migrate your Bitcoin price prediction application from SQLite to Azure SQL Database.

## Changes Made

### 1. New Core Module: `db_config.py`

This module centralizes all Azure SQL Database connectivity and provides utility functions:

- **`validate_azure_credentials()`**: Verifies all required environment variables are set
- **`get_pyodbc_connection()`**: Returns a pyodbc connection using ActiveDirectoryServicePrincipal auth
- **`get_sqlalchemy_engine()`**: Returns a SQLAlchemy engine for pandas operations
- **`read_sql_query()`**: Execute SELECT queries and return DataFrames
- **`execute_query()`**: Execute INSERT/UPDATE/DELETE queries
- **`execute_many()`**: Bulk insert operations
- **`table_exists()`**: Check if a table exists in the database

**Authentication**: Uses Azure AD Service Principal (app registration) with client credentials flow for secure authentication.

### 2. Updated Database Modules

All database-related modules have been updated to use `db_config`:

- **`db_utils.py`**: Updated to use Azure SQL instead of SQLite
- **`db_predictions.py`**: Migrated from sqlite3 to pyodbc with Azure SQL
- **`ingest_coins.py`**: Replaced SQLAlchemy SQLite engine with Azure SQL

### 3. Updated Analysis Scripts

All scripts using direct sqlite3 connections now use `db_config`:

- `merge_predictions_actuals.py`
- `simulate_long_short_positions.py`
- `populate_positions_from_top5.py`
- `positions_actual_vs_predicted.py`
- `debug_merge_predictions_actuals.py`
- `filter_top5_green_coinswitch.py`
- `top5_daily_trade_simulation.py`
- `create_coin_predictions_table.py`

### 4. Database Schema

New MSSQL-compatible schema with three main tables:

- **`coin_prices`**: Historical price data with technical indicators
- **`coin_predictions`**: Model predictions for target dates
- **`positions`**: Trading positions (open, close, P&L tracking)

See `migrations/001_create_coin_predictions.sql` for complete schema.

### 5. GitHub Actions Support

New workflow file: `.github/workflows/daily-prediction.yml`

Runs daily batch predictions and stores credentials securely in GitHub Secrets.

## Setup Instructions

### Prerequisites

- Azure subscription with SQL Database instance
- Azure AD tenant with app registration (Service Principal)
- Python 3.8+
- ODBC Driver 18 for SQL Server (or 17)

### Local Development Setup

#### 1. Install ODBC Driver

**Windows:**
```bash
# Download from Microsoft
# https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-sql-server
```

**macOS:**
```bash
brew install unixodbc
brew tap microsoft/mssql-release https://github.com/Microsoft/homebrew-mssql-release
brew install mssql-tools18
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y odbc-mssql unixodbc-dev
```

#### 2. Configure Azure AD App Registration

1. Go to Azure Portal → App registrations
2. Create new app registration
3. Note the **Client ID** and **Tenant ID**
4. Create a **Client Secret** (note the value)
5. Assign SQL Database access:
   - Go to your SQL Database → Access control (IAM)
   - Add role assignment: "SQL DB Contributor" to your app registration

#### 3. Set Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```
AZURE_SQL_SERVER=myserver
AZURE_SQL_DATABASE=cryptodb
AZURE_SQL_CLIENT_ID=<your-client-id>
AZURE_SQL_CLIENT_SECRET=<your-client-secret>
AZURE_SQL_TENANT_ID=<your-tenant-id>
```

#### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

Ensure these packages are installed:
- `pyodbc>=5.0`
- `azure-identity>=1.13.0`
- `sqlalchemy>=1.4.0`
- `pandas>=1.5.0`

#### 5. Initialize Database

Connect to your Azure SQL Database and run the migration script:

```bash
# Option 1: Using Azure Portal Query Editor
# Copy contents of migrations/001_create_coin_predictions.sql
# Paste into Query Editor and execute

# Option 2: Using sqlcmd (if installed)
sqlcmd -S <server>.database.windows.net -U <app-id> -P <secret> -d <database> -i migrations/001_create_coin_predictions.sql

# Option 3: Using Python
python create_coin_predictions_table.py
```

#### 6. Test Connection

```python
from db_config import get_pyodbc_connection
conn = get_pyodbc_connection()
cursor = conn.cursor()
cursor.execute("SELECT @@version")
print(cursor.fetchone())
conn.close()
```

### GitHub Actions Setup

#### 1. Add Secrets to GitHub

Go to your repository → Settings → Secrets and variables → Actions

Add these secrets:
- `AZURE_SQL_SERVER`: Your SQL server name
- `AZURE_SQL_DATABASE`: Your database name
- `AZURE_SQL_CLIENT_ID`: Service Principal client ID
- `AZURE_SQL_CLIENT_SECRET`: Service Principal client secret
- `AZURE_SQL_TENANT_ID`: Azure AD tenant ID

#### 2. Workflow Configuration

The workflow file `.github/workflows/daily-prediction.yml`:

- Runs daily at 2 AM UTC (modify `cron` schedule as needed)
- Automatically creates tables
- Ingests latest coin data
- Generates predictions
- Uploads results as artifacts
- Can be manually triggered via workflow_dispatch

#### 3. Manual Workflow Trigger

Go to Actions tab → Daily-Prediction → Run workflow

## SQL Parameter Syntax

**Important**: Azure SQL uses different parameterization than SQLite:

| Operation | SQLite | MSSQL |
|-----------|--------|-------|
| Placeholder | `?` | `?` or `@param` |
| Bulk insert | `executemany()` | Loop with `execute()` |
| TOP clause | `LIMIT` | `TOP` |
| Auto-increment | `AUTOINCREMENT` | `IDENTITY(1,1)` |

The `db_config` module handles these differences transparently.

## Migration Troubleshooting

### Connection Errors

**Error**: `pyodbc.Error: ('28000', '[28000] [Microsoft][ODBC Driver 18 for SQL Server]...)`

**Solution**: 
- Verify credentials in `.env`
- Check app registration permissions in Azure Portal
- Ensure "Enable service and co-administrators" is enabled (if legacy auth)

**Error**: `ODBC Driver not found`

**Solution**:
```bash
# Check installed drivers
python -c "import pyodbc; print(pyodbc.drivers())"

# Install ODBC 18 (see Prerequisites section)
```

### Token Issues

**Error**: `Token request failed - AADSTS700016`

**Solution**:
- Verify `AZURE_SQL_TENANT_ID` is correct
- Check client secret hasn't expired
- Use `tenant_id` not `directory_id`

### Table Lock Issues

**Error**: `Lock timeout expired`

**Solution**:
- Reduce batch sizes in `db_config.py`
- Add index on frequently queried columns (already done in migration)
- Increase lock timeout in Azure Portal (Database → Connection strings)

## Performance Optimizations

### Indexes

The migration script creates indexes on:
- `coin_prices(coin_id)`
- `coin_predictions(coin_id, prediction_date)`
- `positions(coin_id)`

### Connection Pooling

`db_config.get_sqlalchemy_engine()` includes connection pooling:
- Pool size: 10
- Max overflow: 20
- Pre-ping enabled to catch stale connections

### Batch Inserts

For bulk operations, use `execute_many()`:

```python
from db_config import execute_many

query = "INSERT INTO coin_predictions (coin_id, target_date, predicted_price) VALUES (?, ?, ?)"
data = [(coin_id, date, price) for ...]
execute_many(query, data)
```

## Rollback Plan

If you need to revert to SQLite:

1. Restore backed-up SQLite databases
2. Revert code to previous commits (before this migration)
3. Update import statements to use old `db_utils` code

All SQLite code is preserved in git history.

## Cost Optimization

**Azure SQL Database Pricing**:
- Basic tier: ~$5-10/month (good for development/testing)
- Standard: ~$15-30/month (recommended for production)
- Premium: $100+/month (for high performance needs)

**Cost Saving Tips**:
1. Use Basic/Standard tier for batch jobs
2. Pause database during off-hours (if not 24/7 needed)
3. Set up automatic backups (included with SQL DB)
4. Monitor query performance with Azure Portal analytics

## Next Steps

1. ✅ Update all Python scripts (completed)
2. ✅ Configure GitHub Actions (completed)
3. Test locally with `.env` file
4. Set up GitHub Secrets
5. Run initial workflow to seed database
6. Monitor query performance in Azure Portal

## Support & Troubleshooting

For issues:
1. Check environment variables with `db_config.validate_azure_credentials()`
2. Test connection: `python -c "from db_config import get_pyodbc_connection; conn = get_pyodbc_connection()"`
3. Review Azure Portal logs for authentication errors
4. Check GitHub Actions workflow logs for execution errors

## References

- [Azure SQL Database Documentation](https://docs.microsoft.com/en-us/azure/azure-sql/)
- [pyodbc Documentation](https://github.com/mkleehammer/pyodbc)
- [Azure Identity Authentication](https://docs.microsoft.com/en-us/python/api/azure-identity/)
- [ODBC Driver for SQL Server](https://docs.microsoft.com/en-us/sql/connect/odbc/microsoft-odbc-driver-for-sql-server)
