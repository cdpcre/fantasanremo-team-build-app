import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

DEFAULT_DB_PATH = Path(__file__).resolve().parent / ".." / "db" / "fantasanremo.db"
DEFAULT_DB_URL = f"sqlite:///{DEFAULT_DB_PATH}"


def _get_database_url() -> str:
    return os.getenv("DATABASE_URL") or os.getenv("TEST_DATABASE_URL") or DEFAULT_DB_URL


def _ensure_sqlite_directory(database_url: str) -> None:
    if not database_url.startswith("sqlite"):
        return
    if database_url in {"sqlite://", "sqlite:///:memory:"}:
        return
    if database_url.startswith("sqlite:///"):
        sqlite_path = database_url.replace("sqlite:///", "", 1)
        path = Path(sqlite_path)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)


DATABASE_URL = _get_database_url()
_ensure_sqlite_directory(DATABASE_URL)
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
