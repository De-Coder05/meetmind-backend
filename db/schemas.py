from sqlalchemy import Column, String, DateTime, JSON
from sqlalchemy.sql import func
from db.database import Base

class Meeting(Base):
    __tablename__ = "meetings"

    id = Column(String, primary_key=True, index=True)
    title = Column(String, index=True)
    status = Column(String, index=True)
    result_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
