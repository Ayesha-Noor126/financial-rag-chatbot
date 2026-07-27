"""
Shared SQLAlchemy engine.

Design choice: a single cached engine shared by MetadataStore (Phase 1) and
ChatHistoryStore (Phase 4) instead of each opening its own connection to the
same SQLite file. check_same_thread=False is required because our async
routes push DB calls onto the asyncio thread pool via asyncio.to_thread --
SQLite's default same-thread check would otherwise reject connections used
from a thread other than the one that created them. pool_pre_ping validates
a pooled connection is alive before reuse, cheap insurance against stale
connections after long idle periods.
"""

from functools import lru_cache

from sqlalchemy import Engine
from sqlmodel import create_engine

from app.core.config import get_settings


@lru_cache
def get_engine() -> Engine:
    settings = get_settings()
    return create_engine(
        f"sqlite:///{settings.metadata_db_path}",
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )