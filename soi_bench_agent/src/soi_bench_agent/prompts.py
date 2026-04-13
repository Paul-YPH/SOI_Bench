from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
PROMPT_DIR = ROOT_DIR / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text().strip()
