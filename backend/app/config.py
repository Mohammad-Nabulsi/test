from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Settings:
    app_env: str = os.getenv("APP_ENV", "dev")
    frontend_origin: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
    storage_dir: str = os.getenv("STORAGE_DIR", "storage")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_recommendation_system_prompt: str = os.getenv(
        "OPENAI_RECOMMENDATION_SYSTEM_PROMPT",
        (
            "You are a social media growth strategist. "
            "Given low-performing and high-performing content-style cluster summaries, "
            "write practical recommendations that help the business imitate the high-performing cluster "
            "and avoid the low-performing cluster behaviors. "
            "Use concise, action-oriented language and refer to provided metrics and behavior columns."
        ),
    )

    def storage_path(self) -> Path:
        path = Path(self.storage_dir)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        return path


settings = Settings()
