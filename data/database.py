"""Database session management and table creation with SQLite support."""

import logging
import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from trading_ai.config.settings import SETTINGS
from trading_ai.data.models import Base

logger = logging.getLogger(__name__)

_engine = None
_session_factory = None


def get_engine():
    global _engine
    if _engine is None:
        # Check if we should use SQLite (via USE_SQLITE env var)
        use_sqlite = os.getenv("USE_SQLITE", "true").lower() == "true"

        if use_sqlite:
            # Use SQLite
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.getenv("SQLITE_DB_PATH", os.path.join(project_root, "trading_ai.db"))
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


def get_session() -> Generator[Session, None, None]:
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


def create_tables() -> None:
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Database tables created/verified")


def drop_tables() -> None:
    engine = get_engine()
    Base.metadata.drop_all(engine)
    logger.warning("All tables dropped")
