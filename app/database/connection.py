from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import settings

engine = create_engine(
    settings.database_url,
    connect_args= {'check_same_thread': False}
    if settings.database_url.startswith("sqlite")
    else {},
)

SessionLocal = sessionmaker(
    autocommit= False,
    autoflush= False,
    bind= engine,
)

Base = declarative_base()

def init_db() -> None:
    """
    Create database tables if they do not already exist
    """

    Base.metadata.create_all(bind=engine)


def get_db_session():
    """
    Create a database session.

    Caller is responsible for closing it
    """
    return SessionLocal()