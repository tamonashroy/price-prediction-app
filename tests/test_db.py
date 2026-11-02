import numpy as np
import pytest
from psycopg2.extensions import adapt

from pricepredict import db


@pytest.fixture(autouse=True)
def clear_engine_cache():
    db.get_engine.cache_clear()
    yield
    db.get_engine.cache_clear()


def test_engine_uses_the_installed_driver(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@example.neon.tech/db?sslmode=require")
    engine = db.get_engine()
    assert engine.dialect.driver == "psycopg2"
    assert engine.url.query["sslmode"] == "require"


def test_missing_database_url_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "   ")
    with pytest.raises(EnvironmentError, match="DATABASE_URL"):
        db.get_database_url()


def test_numpy_floats_are_sent_as_plain_numbers():
    assert adapt(np.float64(1.5)).getquoted() == b"1.5"
