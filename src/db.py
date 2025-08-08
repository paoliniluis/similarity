from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from typing import Generator

from .settings import DATABASE_URL

# Configure engine with better connection handling
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=3600,   # Recycle connections every hour
    pool_size=10,        # Maximum number of connections
    max_overflow=20,     # Additional connections when pool is full
    echo=False,          # Set to True for SQL debugging
    connect_args={
        "connect_timeout": 60,  # 60 second connection timeout
        "keepalives_idle": 30,  # Send keepalive after 30 seconds of inactivity
        "keepalives_interval": 10,  # Send keepalive every 10 seconds
        "keepalives_count": 5,  # Allow 5 missed keepalives before considering connection dead
    }
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db() -> Generator[Session, None, None]:
    """
    SQLAlchemy dependency to get a database session.
    Ensures the session is always closed after the request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 