from __future__ import annotations

from pathlib import Path

from .data_cleaning import build_clean_bundle
from .utils import load_json


def load_knowledge_base(clean_dir: Path, raw_dir: Path) -> dict:
    knowledge_path = clean_dir / "knowledge_base.json"
    if not knowledge_path.exists():
        return build_clean_bundle(raw_dir, clean_dir)
    return load_json(knowledge_path)
