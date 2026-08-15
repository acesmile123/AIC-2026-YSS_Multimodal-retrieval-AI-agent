from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RouterSettings(BaseSettings):
    """Runtime settings. Values can be supplied through environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="AIC_",
        env_file=".env",
        env_file_encoding="utf-8",  
        extra="ignore",
    )

    google_api_key: str | None = Field(default=None, validation_alias="GOOGLE_API_KEY")
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    gemini_max_output_tokens: int = Field(default=8192, ge=512, le=65536)
    gemini_max_attempts: int = Field(default=2, ge=1, le=4)

