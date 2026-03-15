"""
Azure SQL Database configuration and connection utilities.
Supports both local development (with environment variables) and GitHub Actions.
Uses connection pooling to minimize token generation and connection overhead.
"""

import os
import time
import pyodbc
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from azure.identity import ClientSecretCredential
import struct
import threading
from queue import Queue, Empty
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file (handles line endings properly)
load_dotenv()

# Database connection parameters from environment variables
# Strip whitespace to handle any line ending issues
AZURE_SQL_SERVER = os.getenv("AZURE_SQL_SERVER", "").strip()
AZURE_SQL_DATABASE = os.getenv("AZURE_SQL_DATABASE", "").strip()
AZURE_SQL_CLIENT_ID = os.getenv("AZURE_SQL_CLIENT_ID", "").strip()
AZURE_SQL_CLIENT_SECRET = os.getenv("AZURE_SQL_CLIENT_SECRET", "").strip()
AZURE_SQL_TENANT_ID = os.getenv("AZURE_SQL_TENANT_ID", "").strip()

# Validate that all required environment variables are set
def validate_azure_credentials():
    """Verify all required Azure SQL credentials are configured."""
    required_vars = {
        "AZURE_SQL_SERVER": AZURE_SQL_SERVER,
        "AZURE_SQL_DATABASE": AZURE_SQL_DATABASE,
        "AZURE_SQL_CLIENT_ID": AZURE_SQL_CLIENT_ID,
        "AZURE_SQL_CLIENT_SECRET": AZURE_SQL_CLIENT_SECRET,
        "AZURE_SQL_TENANT_ID": AZURE_SQL_TENANT_ID,
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        raise EnvironmentError(
            f"Missing Azure SQL credentials: {', '.join(missing_vars)}. "
            "Please set these environment variables."
        )

# Global connection pool
class PyODBCConnectionPool:
    """
    Singleton connection pool for pyodbc Azure SQL connections.
    Minimizes token generation and connection overhead by reusing connections.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        validate_azure_credentials()
        self.pool = Queue(maxsize=10)
        self.credentials = None
        self.token_expiry = None
        self.connection_string = (
            "Driver={ODBC Driver 18 for SQL Server};"
            "Server=tcp:cryptopp.database.windows.net,1433;"
            "Database=cryptopp;"
            "Encrypt=yes;"
            "TrustServerCertificate=no;"
            "Connection Timeout=30;"
        )
        self._initialized = True
    
    def _get_fresh_token(self):
        """Generate a fresh Azure AD token."""
        if self.credentials is None:
            self.credentials = ClientSecretCredential(
                tenant_id=AZURE_SQL_TENANT_ID,
                client_id=AZURE_SQL_CLIENT_ID,
                client_secret=AZURE_SQL_CLIENT_SECRET
            )
        
        token = self.credentials.get_token("https://database.windows.net/.default")
        token_bytes = token.token.encode("UTF-16-LE")
        token_struct = struct.pack(f'<I{len(token_bytes)}s', len(token_bytes), token_bytes)
        self.token_expiry = datetime.now() + timedelta(seconds=token.expires_on)
        return token_struct
    
    def _create_connection(self):
        """Create a new pyodbc connection with fresh token. Retries up to 5 times with 1-minute sleep between attempts."""
        max_retries = 5
        retry_delay = 60  # 1 minute in seconds
        
        for attempt in range(max_retries):
            try:
                token_struct = self._get_fresh_token()
                conn = pyodbc.connect(self.connection_string, attrs_before={1256: token_struct})
                # Test the connection
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                return conn
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed to create connection after {max_retries} attempts: {e}")
                    raise
    
    def get_connection(self):
        """
        Get a connection from the pool or create a new one if pool is empty.
        Validates connection is alive before returning.
        """
        # Try to get a connection from the pool
        try:
            conn = self.pool.get_nowait()
            # Validate connection is still alive
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                return conn
            except Exception:
                # Connection is dead, close it and create a new one
                conn.close()
                return self._create_connection()
        except Empty:
            # Pool is empty, create a new connection
            return self._create_connection()
    
    def return_connection(self, conn):
        """Return a connection to the pool for reuse."""
        try:
            self.pool.put_nowait(conn)
        except Exception:
            # Pool is full or connection is invalid, close it
            try:
                conn.close()
            except Exception:
                pass
    
    def close_all(self):
        """Close all connections in the pool."""
        while True:
            try:
                conn = self.pool.get_nowait()
                conn.close()
            except Empty:
                break


# Initialize the pool as singleton
_connection_pool = PyODBCConnectionPool()


def get_pyodbc_connection():
    """
    Get a pyodbc connection to Azure SQL from the connection pool.
    Connections are reused from the pool for efficiency.
    Returns a pyodbc.Connection object.
    
    Note: Call return_pyodbc_connection() when done to return it to the pool.
    Or use it as a context manager for automatic return.
    """
    return _connection_pool.get_connection()


def return_pyodbc_connection(conn):
    """
    Return a connection to the pool for reuse.
    This minimizes overhead on the next get_pyodbc_connection() call.
    
    Args:
        conn: pyodbc.Connection object from get_pyodbc_connection()
    """
    _connection_pool.return_connection(conn)


class PyODBCConnectionContext:
    """Context manager for automatic connection return to pool."""
    def __enter__(self):
        self.conn = get_pyodbc_connection()
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return_pyodbc_connection(self.conn)


def get_sqlalchemy_engine():
    """
    Get a SQLAlchemy engine for Azure SQL using Service Principal token auth.
    Uses a custom creator function to get connections from our token-based pool.
    
    Returns a SQLAlchemy Engine object.
    """
    validate_azure_credentials()
    
    # Custom creator function that gets connections from our token pool
    def creator():
        conn = get_pyodbc_connection()
        # Return raw DBAPI connection (not a SQLAlchemy connection)
        return conn
    
    # Create engine with custom creator - no auth in URL since we handle it via creator
    connection_url = (
        f"mssql+pyodbc:///?odbc_connect="
        f"Driver={{ODBC Driver 18 for SQL Server}};"
        f"Server=tcp:{AZURE_SQL_SERVER}.database.windows.net,1433;"
        f"Database={AZURE_SQL_DATABASE};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=no;"
    )
    
    engine = create_engine(
        connection_url,
        creator=creator,
        poolclass=QueuePool,
        pool_size=1,  # We manage pooling in our PyODBCConnectionPool
        max_overflow=0,
        pool_pre_ping=False,  # We validate in our pool
        echo=False,
    )
    
    return engine


def read_sql_query(query, conn=None):
    """
    Read a SQL query into a pandas DataFrame using SQLAlchemy engine.
    If conn is None, gets engine from pool and disposes it after use.
    
    Args:
        query: SQL query string
        conn: Optional existing SQLAlchemy connection/engine. If None, engine is created.
    
    Returns:
        pandas DataFrame with query results
    """
    if conn is None:
        engine = get_sqlalchemy_engine()
        should_dispose = True
    else:
        engine = conn
        should_dispose = False
    
    try:
        df = pd.read_sql(query, engine)
        return df
    finally:
        if should_dispose:
            engine.dispose()


def read_sql_query_with_params(query, params, conn=None):
    """
    Read a SQL query with parameters into a pandas DataFrame using SQLAlchemy.
    Uses Text() for parameterized queries.
    If conn is None, gets engine from pool and disposes it after use.
    
    Args:
        query: SQL query string with ? or :param placeholders
        params: Tuple or list of parameter values
        conn: Optional existing SQLAlchemy connection/engine
    
    Returns:
        pandas DataFrame with query results
    """
    if conn is None:
        engine = get_sqlalchemy_engine()
        should_dispose = True
    else:
        engine = conn
        should_dispose = False
    
    try:
        # Use SQLAlchemy text() for parameterized queries
        query_sa = text(query)
        df = pd.read_sql(query_sa, engine, params=params)
        return df
    finally:
        if should_dispose:
            engine.dispose()


def execute_query(query, params=None, conn=None):
    """
    Execute a SQL query (INSERT, UPDATE, DELETE, etc.).
    If conn is None, gets a connection from the pool and returns it after use.
    
    Args:
        query: SQL query string
        params: Optional tuple of parameters
        conn: Optional existing connection
    
    Returns:
        Number of rows affected
    """
    if conn is None:
        conn = get_pyodbc_connection()
        should_return = True
    else:
        should_return = False
    
    try:
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        conn.commit()
        return cursor.rowcount
    finally:
        if should_return:
            return_pyodbc_connection(conn)


def execute_many(query, data_list, conn=None):
    """
    Execute a query multiple times with different parameters (bulk insert).
    If conn is None, gets a connection from the pool and returns it after use.
    
    Args:
        query: SQL query string with ? placeholders
        data_list: List of tuples, one per execution
        conn: Optional existing connection (reused across all inserts)
    
    Returns:
        Number of rows affected
    """
    if conn is None:
        conn = get_pyodbc_connection()
        should_return = True
    else:
        should_return = False
    
    try:
        cursor = conn.cursor()
        for data in data_list:
            cursor.execute(query, data)
        conn.commit()
        return cursor.rowcount
    finally:
        if should_return:
            return_pyodbc_connection(conn)


def table_exists(table_name, conn=None):
    """Check if a table exists in the database."""
    if conn is None:
        conn = get_pyodbc_connection()
        should_return = True
    else:
        should_return = False
    
    try:
        query = f"""
            SELECT CASE WHEN EXISTS (
                SELECT 1 FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_NAME = N'{table_name}'
            ) THEN 1 ELSE 0 END AS table_exists
        """
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchone()
        return result[0] == 1
    finally:
        if should_return:
            return_pyodbc_connection(conn)


def create_table_from_pandas(df, table_name, if_exists='fail', index=False):
    """
    Create a table from a pandas DataFrame using SQLAlchemy.
    This uses SQLAlchemy's built-in connection pooling.
    
    Args:
        df: pandas DataFrame
        table_name: Name of table to create
        if_exists: 'fail', 'replace', or 'append'
        index: Whether to write index as a column
    """
    engine = get_sqlalchemy_engine()
    try:
        df.to_sql(table_name, engine, if_exists=if_exists, index=index)
    finally:
        engine.dispose()


def close_connection_pool():
    """Close all connections in the pool. Call at application shutdown."""
    _connection_pool.close_all()
