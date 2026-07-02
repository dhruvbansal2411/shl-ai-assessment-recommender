"""Shared utilities for configuration, paths, JSON IO, and logging."""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
VECTOR_DIR = BASE_DIR / "vector_index"


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "SHL AI Assessment Recommender"
    environment: str = "development"
    log_level: str = "INFO"
    catalog_path: Path = Field(default=DATA_DIR / "catalog.json")
    vector_index_path: Path = Field(default=VECTOR_DIR / "index.json")
    vector_metadata_path: Path = Field(default=VECTOR_DIR / "metadata.json")
    embedding_model_name: str = "all-MiniLM-L6-v2"
    top_k: int = 10
    llm_provider: str = "gemini"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-1.5-flash"
    openrouter_api_key: str | None = None
    openrouter_model: str = "google/gemini-flash-1.5"
    llm_timeout_seconds: float = 20.0
    enable_llm: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings after loading .env."""

    load_dotenv()
    return Settings()


def configure_logging() -> None:
    """Configure application logging once at startup/import."""

    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def load_json(path: Path) -> Any:
    """Load JSON from a UTF-8 file with clear error messages."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {path}: {exc}") from exc


def write_json(path: Path, payload: Any) -> None:
    """Write pretty, deterministic JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def env_flag(name: str, default: bool = False) -> bool:
    """Read a conventional boolean environment variable."""

    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}
