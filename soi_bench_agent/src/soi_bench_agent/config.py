from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(slots=True)
class Settings:
    openai_api_key: str | None
    openai_model: str
    data_dir: str
    clean_dir: str

    @property
    def api_enabled(self) -> bool:
        return bool(self.openai_api_key)


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", os.getenv("DEFAULT_MODEL", "gpt-5-mini")),
        data_dir=os.getenv("SOI_BENCH_RAW_DIR", "data/raw"),
        clean_dir=os.getenv("SOI_BENCH_CLEAN_DIR", "data/clean"),
    )
