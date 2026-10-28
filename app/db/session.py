"""App DB engine and session factory."""

from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from app.config import DatabaseSettings


def sqlalchemy_url(url: str) -> str:
    """Use the psycopg 3 driver for plain `postgresql://` URLs (SQLAlchemy defaults to psycopg2)."""
    parsed = make_url(url)
    if parsed.drivername == "postgresql":
        parsed = parsed.set(drivername="postgresql+psycopg")
    return parsed.render_as_string(hide_password=False)


@lru_cache
def get_engine() -> Engine:
    return create_engine(sqlalchemy_url(DatabaseSettings().app_db_url), pool_pre_ping=True)


@lru_cache
def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(get_engine(), expire_on_commit=False)
