import asyncio
from pathlib import Path
from db.database import AsyncSessionLocal
from db.schemas import Meeting
from api.routes.meetings import _run_pipeline
from models.meeting import MeetingCreate
from core.logging import setup_logging

async def main():
    setup_logging()
    meeting_id = "f6701327-adab-41ee-8425-ac2c3a6d854c"
    upload_dir = Path("./uploads")
    files = list(upload_dir.glob("*.*"))
    if not files:
        print("No audio files found! Pipeline fails here.")
        return
    audio_path = files[0]
    config = MeetingCreate(title="Test", push_notion=True, push_slack=True, push_linear=True)
    print(f"Running pipeline on {audio_path}...")
    await _run_pipeline(meeting_id, audio_path, config)
    print("Pipeline finished.")

if __name__ == "__main__":
    asyncio.run(main())
