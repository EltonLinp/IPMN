from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

from sqlalchemy import DateTime, Integer, String, create_engine, func, select
from sqlalchemy.orm import Mapped, Session, declarative_base, mapped_column, sessionmaker


DEFAULT_DATABASE_URL = "postgresql+psycopg://postgres:newpass@localhost:5432/df_detection"


def _get_database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def _create_engine(url: str):
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    return create_engine(url, pool_pre_ping=True, connect_args=connect_args)


ENGINE = _create_engine(_get_database_url())
SessionLocal = sessionmaker(bind=ENGINE, autoflush=False, autocommit=False)
Base = declarative_base()


class UploadRecord(Base):
    __tablename__ = "upload_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    video_path: Mapped[str] = mapped_column(String, nullable=False)
    user_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    user_phone: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    id_photo_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    id_face_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


def init_db() -> None:
    Base.metadata.create_all(bind=ENGINE)


def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def cleanup_expired_uploads(
    session: Session,
    *,
    retention_days: int,
) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    records = session.execute(select(UploadRecord).where(UploadRecord.created_at < cutoff)).scalars().all()
    deleted = 0
    for record in records:
        paths = [
            record.video_path,
            record.id_photo_path,
            record.id_face_path,
        ]
        for value in paths:
            if not value:
                continue
            path = Path(value)
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
        session.delete(record)
        deleted += 1
    if deleted:
        session.commit()
    return deleted
