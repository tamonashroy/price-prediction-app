"""Neon Postgres connections and schema migrations."""

import logging
import os
import time
from functools import lru_cache

import numpy as np
import psycopg2
from dotenv import load_dotenv
from psycopg2.extensions import adapt, register_adapter
from sqlalchemy import create_engine, make_url

from pricepredict.config import MIGRATIONS_DIR

logger = logging.getLogger(__name__)

load_dotenv()

# numpy 2 reprs floats as "np.float64(1.5)", which psycopg2 would send as literal SQL
register_adapter(np.float64, lambda value: adapt(float(value)))


def get_database_url():
    """Return the Neon connection string, or raise if it is not configured."""
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise EnvironmentError(
            "Missing DATABASE_URL. Set it in your .env file or as an environment variable."
        )
    return database_url


def get_connection():
    """
    Open a psycopg2 connection. Retries because a suspended Neon compute can
    take a moment to wake up. The caller closes it.
    """
    max_retries = 5
    retry_delay_seconds = 5
    for attempt in range(1, max_retries + 1):
        try:
            return psycopg2.connect(get_database_url(), connect_timeout=30)
        except psycopg2.OperationalError as error:
            if attempt == max_retries:
                raise
            logger.warning("Connection attempt %d/%d failed: %s", attempt, max_retries, error)
            time.sleep(retry_delay_seconds)


@lru_cache(maxsize=1)
def get_engine():
    """Shared SQLAlchemy engine. pool_pre_ping replaces connections closed by scale-to-zero."""
    # Name the driver: SQLAlchemy's default for postgresql:// URLs is not the one we install.
    url = make_url(get_database_url()).set(drivername="postgresql+psycopg2")
    return create_engine(url, pool_pre_ping=True)


def apply_migrations():
    """Apply any files in migrations/ that have not run yet, in filename order."""
    conn = get_connection()
    try:
        with conn, conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    filename TEXT PRIMARY KEY,
                    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute("SELECT filename FROM schema_migrations")
            applied = {row[0] for row in cursor.fetchall()}
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name in applied:
                continue
            # Each file commits on its own, so a failure leaves earlier ones in place.
            with conn, conn.cursor() as cursor:
                cursor.execute(path.read_text(encoding="utf-8"))
                cursor.execute("INSERT INTO schema_migrations (filename) VALUES (%s)", (path.name,))
            logger.info("Applied migration %s", path.name)
    finally:
        conn.close()
