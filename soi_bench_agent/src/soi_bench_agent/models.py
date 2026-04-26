from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class UserProfile:
    task_type: str | None = None
    integration_mode: str | None = None
    technology: str | None = None
    species: str | None = None
    tissue: str | None = None
    sample_count: int | None = None
    modality_count: int | None = None
    num_locations: int | None = None
    num_features: int | None = None
    challenge_tags: list[str] = field(default_factory=list)
    priority: str = "balanced"
    max_runtime_minutes: float | None = None
    max_memory_gb: float | None = None
    avoid_deep_learning: bool = False
    extra_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "integration_mode": self.integration_mode,
            "technology": self.technology,
            "species": self.species,
            "tissue": self.tissue,
            "sample_count": self.sample_count,
            "modality_count": self.modality_count,
            "num_locations": self.num_locations,
            "num_features": self.num_features,
            "challenge_tags": self.challenge_tags,
            "priority": self.priority,
            "max_runtime_minutes": self.max_runtime_minutes,
            "max_memory_gb": self.max_memory_gb,
            "avoid_deep_learning": self.avoid_deep_learning,
            "extra_notes": self.extra_notes,
        }

    def supporting_signal_count(self) -> int:
        fields = [
            self.integration_mode,
            self.technology,
            self.species,
            self.tissue,
            self.sample_count,
            self.modality_count,
            self.num_locations,
            self.num_features,
        ]
        count = sum(value is not None for value in fields)
        if self.challenge_tags:
            count += 1
        return count


@dataclass(slots=True)
class RecommendationResult:
    matched_datasets: list[dict[str, Any]]
    recommended_methods: list[dict[str, Any]]
    discarded_methods: list[dict[str, Any]]
    summary: dict[str, Any]
