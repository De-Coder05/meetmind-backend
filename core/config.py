from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    gemini_api_key: str
    gemini_model: str = "gemini-1.5-pro"

    # Database
    database_url: str = "sqlite+aiosqlite:///./meetmind.db"

    # Integrations
    notion_api_key: str = ""
    notion_database_id: str = ""
    slack_bot_token: str = ""
    slack_default_channel: str = "#meeting-notes"
    linear_api_key: str = ""
    linear_team_id: str = ""

    # Email
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""

    # App
    app_env: str = "development"
    secret_key: str = "dev-secret-change-in-prod"
    upload_dir: str = "./uploads"
    max_upload_mb: int = 500

    # Whisper
    whisper_model: str = "base"

    class Config:
        import os
        env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
