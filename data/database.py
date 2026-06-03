"""Database session management and table creation with SQLite support."""

import logging
import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import SETTINGS
from data.models import Base

logger = logging.getLogger(__name__)

_engine = None
_session_factory = None


def get_engine():
    global _engine
    if _engine is None:
        # Check if we should use SQLite (via USE_SQLITE env var)
        use_sqlite = os.getenv("USE_SQLITE", "true").lower() == "true"

        if use_sqlite:
            # Use SQLite - always use absolute path from project root
            # Get the trading_ai package root (2 levels up from data/database.py)
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path_env = os.getenv("SQLITE_DB_PATH", "trading_ai.db")
            # Make path absolute relative to project root if not already absolute
            if not os.path.isabs(db_path_env):
                db_path = os.path.abspath(os.path.join(project_root, db_path_env))
            else:
                db_path = db_path_env
            url = f"sqlite:///{db_path}"
            logger.info(f"Using SQLite database: {db_path}")
        else:
            # Use MySQL
            cfg = SETTINGS.database
            url = (
                f"mysql+pymysql://{cfg.username}:{cfg.password}"
                f"@127.0.0.1:{cfg.port}/{cfg.database}?charset=utf8mb4"
            )
            logger.info(f"Using MySQL database: {cfg.host}:{cfg.port}/{cfg.database}")

        _engine = create_engine(
            url,
            pool_pre_ping=True,
            echo=False,
        )
        logger.info("Database engine created")
    return _engine


def get_session_factory() -> sessionmaker:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine())
    return _session_factory


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Context manager that properly commits and closes the session."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_raw_session() -> Session:
    """Get a bare session for legacy code that manages its own commits.

    IMPORTANT: Callers MUST commit and close the session themselves.
    Prefer get_session() instead.
    """
    return get_session_factory()()


def create_tables() -> None:
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Database tables created/verified")


def drop_tables() -> None:
    engine = get_engine()
    Base.metadata.drop_all(engine)
    logger.warning("All tables dropped")
