from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy.orm import Session

from forecasting.db.models import SessionLocal


@contextmanager
def get_session_context() -> Generator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
