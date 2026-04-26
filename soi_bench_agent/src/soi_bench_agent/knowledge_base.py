from __future__ import annotations

from pathlib import Path

from .utils import load_json


def load_knowledge_base(clean_dir: Path) -> dict:
    knowledge_path = clean_dir / "knowledge_base.json"
    if not knowledge_path.exists():
        raise FileNotFoundError(
            f"Missing clean knowledge base at {knowledge_path}. "
            "Generate it with `soi-bench-agent build-data` before running chat."
        )
    return load_json(knowledge_path)
