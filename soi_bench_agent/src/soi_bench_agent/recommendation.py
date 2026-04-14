from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .models import RecommendationResult, UserProfile
from .utils import normalize_text


TASK_ALIASES = {
    "matching": ["matching", "match", "slice matching", "spatial alignment matching"],
    "embedding": ["embedding", "integration", "joint embedding", "batch correction", "clustering"],
    "mapping": ["mapping", "coordinate", "registration", "stacking", "3d reconstruction"],
    "multiomics": ["multiomics", "multi omics", "multi-omics", "rna protein", "atac rna", "spatial multiomics"],
}

PRIORITY_ALIASES = {
    "accuracy": ["accuracy", "best", "strongest", "highest score", "most accurate"],
    "speed": ["speed", "fast", "faster", "runtime", "quick"],
    "memory": ["memory", "ram", "lightweight", "resource efficient"],
    "balanced": ["balanced", "tradeoff", "default"],
}


def canonicalize_task(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.lower()
    for canonical in TASK_ALIASES:
        if re.search(rf"\b{re.escape(canonical)}\b", normalized):
            return canonical
    matches: list[tuple[int, str]] = []
    for canonical, aliases in TASK_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", normalized):
                matches.append((len(alias), canonical))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0], reverse=True)
    return matches[0][1]


def canonicalize_priority(value: str | None) -> str:
    if not value:
        return "balanced"
    normalized = value.lower()
    for canonical, aliases in PRIORITY_ALIASES.items():
        if canonical == normalized or any(alias in normalized for alias in aliases):
            return canonical
    return "balanced"


def detect_priority(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.lower()
    for canonical, aliases in PRIORITY_ALIASES.items():
        if canonical == normalized or any(alias in normalized for alias in aliases):
            return canonical
    return None


def parse_first_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def parse_first_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


class ProfileParser:
    def __init__(self, knowledge_base: dict[str, Any]) -> None:
        profiles = knowledge_base["dataset_profiles"].values()
        self.technologies = sorted(
            {profile["technology"] for profile in profiles if profile.get("technology")},
            key=len,
            reverse=True,
        )
        self.species = sorted({profile["species"] for profile in profiles if profile.get("species")})
        self.tissues = sorted(
            {profile["tissue"] for profile in profiles if profile.get("tissue")},
            key=len,
            reverse=True,
        )

    def update_profile(self, profile: UserProfile, text: str) -> UserProfile:
        lowered = text.lower()
        task_type = canonicalize_task(lowered)
        if task_type:
            profile.task_type = task_type

        for technology in self.technologies:
            if normalize_text(technology) and normalize_text(technology) in normalize_text(lowered):
                profile.technology = technology
                break

        for species in self.species:
            if species and species.lower() in lowered:
                profile.species = species
                break

        for tissue in self.tissues:
            if tissue and tissue.lower() in lowered:
                profile.tissue = tissue
                break

        sample_count = parse_first_int(r"(\d[\d,]*)\s*(?:slices|samples|sections|slides)", text)
        if sample_count is not None:
            profile.sample_count = sample_count

        modality_count = parse_first_int(r"(\d[\d,]*)\s*(?:modalities|assays|omics|layers)", text)
        if modality_count is not None:
            profile.modality_count = modality_count

        locations = parse_first_int(r"(\d[\d,]*)\s*(?:spots|cells|locations|bins)", text)
        if locations is not None:
            profile.num_locations = locations

        features = parse_first_int(r"(\d[\d,]*)\s*(?:genes|features|peaks|proteins)", text)
        if features is not None:
            profile.num_features = features

        runtime = parse_first_float(r"(?:under|within|<=?)\s*(\d+(?:\.\d+)?)\s*(?:minutes|min)", text)
        if runtime is not None:
            profile.max_runtime_minutes = runtime

        memory = parse_first_float(r"(?:under|within|<=?)\s*(\d+(?:\.\d+)?)\s*gb", text)
        if memory is not None:
            profile.max_memory_gb = memory

        if "no deep learning" in lowered or "avoid deep learning" in lowered or "non-deep" in lowered:
            profile.avoid_deep_learning = True

        detected_priority = detect_priority(lowered)
        if detected_priority is not None:
            profile.priority = detected_priority
        if not task_type and text.strip():
            profile.extra_notes.append(text.strip())
        return profile


class Recommender:
    def __init__(self, knowledge_base: dict[str, Any]) -> None:
        self.knowledge_base = knowledge_base

    def missing_fields(self, profile: UserProfile) -> list[str]:
        missing = []
        if not profile.task_type:
            missing.append("task_type")
        if profile.supporting_signal_count() < 2:
            if not profile.technology:
                missing.append("technology")
            if not profile.species:
                missing.append("species")
            if not profile.tissue:
                missing.append("tissue")
            if not profile.num_locations:
                missing.append("num_locations")
            if profile.task_type == "multiomics" and not profile.modality_count:
                missing.append("modality_count")
        deduped = []
        for item in missing:
            if item not in deduped:
                deduped.append(item)
        return deduped

    def can_recommend(self, profile: UserProfile) -> bool:
        return profile.task_type is not None and profile.supporting_signal_count() >= 2

    def recommend(self, profile: UserProfile) -> RecommendationResult:
        candidates = [
            ranking
            for ranking in self.knowledge_base["dataset_rankings"]
            if ranking["task_type"] == profile.task_type
        ]
        scored = []
        for candidate in candidates:
            similarity = self._dataset_similarity(profile, candidate["dataset_profile"], candidate["origin"])
            if similarity <= 0:
                continue
            scored.append((similarity, candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        matched = scored[:5] or [(0.5, candidate) for candidate in candidates[:5]]

        method_scores: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "method": None,
                "weighted_score": 0.0,
                "evidence_count": 0,
                "matched_datasets": [],
                "deep_learning": False,
                "runtime_seconds": [],
                "peak_rss_gb": [],
            }
        )

        for similarity, dataset in matched:
            for method_entry in dataset["all_methods"][:8]:
                weighted = self._method_preference_score(profile, method_entry) * similarity
                bucket = method_scores[method_entry["method"]]
                bucket["method"] = method_entry["method"]
                bucket["weighted_score"] += weighted
                bucket["evidence_count"] += 1
                bucket["deep_learning"] = method_entry["deep_learning"]
                bucket["matched_datasets"].append(
                    {
                        "dataset_id": dataset["dataset_id"],
                        "score_overall": method_entry["score_overall"],
                        "rank": method_entry["rank"],
                        "similarity": round(similarity, 4),
                    }
                )
                runtime = method_entry["resource"].get("mean_runtime_seconds")
                if runtime is not None:
                    bucket["runtime_seconds"].append(runtime)
                peak_rss = method_entry["resource"].get("mean_peak_rss_gb")
                if peak_rss is not None:
                    bucket["peak_rss_gb"].append(peak_rss)

        recommendations = sorted(
            (
                {
                    "method": payload["method"],
                    "weighted_score": round(payload["weighted_score"], 6),
                    "evidence_count": payload["evidence_count"],
                    "deep_learning": payload["deep_learning"],
                    "matched_datasets": sorted(
                        payload["matched_datasets"],
                        key=lambda item: (item["rank"], -item["similarity"]),
                    )[:3],
                    "average_runtime_seconds": round(sum(payload["runtime_seconds"]) / len(payload["runtime_seconds"]), 3)
                    if payload["runtime_seconds"]
                    else None,
                    "average_peak_rss_gb": round(sum(payload["peak_rss_gb"]) / len(payload["peak_rss_gb"]), 3)
                    if payload["peak_rss_gb"]
                    else None,
                }
                for payload in method_scores.values()
                if payload["method"] is not None
            ),
            key=lambda item: item["weighted_score"],
            reverse=True,
        )

        filtered_out = []
        final_recommendations = []
        for item in recommendations:
            if self._violates_hard_constraints(profile, item):
                filtered_out.append(item)
                continue
            final_recommendations.append(item)
            if len(final_recommendations) == 3:
                break

        matched_datasets = [
            {
                "dataset_id": dataset["dataset_id"],
                "task_type": dataset["task_type"],
                "similarity": round(similarity, 4),
                "technology": dataset["dataset_profile"].get("technology"),
                "species": dataset["dataset_profile"].get("species"),
                "tissue": dataset["dataset_profile"].get("tissue"),
                "sample_count": dataset["dataset_profile"].get("sample_count"),
                "top_methods": dataset["top_methods"][:3],
            }
            for similarity, dataset in matched
        ]

        return RecommendationResult(
            matched_datasets=matched_datasets,
            recommended_methods=final_recommendations,
            discarded_methods=filtered_out[:5],
            summary={
                "task_type": profile.task_type,
                "priority": profile.priority,
                "used_exact_constraints": {
                    "max_runtime_minutes": profile.max_runtime_minutes,
                    "max_memory_gb": profile.max_memory_gb,
                    "avoid_deep_learning": profile.avoid_deep_learning,
                },
            },
        )

    def _dataset_similarity(self, profile: UserProfile, dataset_profile: dict[str, Any], origin: str) -> float:
        score = 0.0
        if origin == "experiment":
            score += 0.5
        if profile.technology and dataset_profile.get("technology"):
            if normalize_text(profile.technology) == normalize_text(dataset_profile["technology"]):
                score += 4.0
            elif normalize_text(profile.technology) in normalize_text(dataset_profile["technology"]):
                score += 2.5
        if profile.species and dataset_profile.get("species") and profile.species == dataset_profile["species"]:
            score += 2.0
        if profile.tissue and dataset_profile.get("tissue") and profile.tissue == dataset_profile["tissue"]:
            score += 2.0
        if profile.sample_count and dataset_profile.get("sample_count"):
            score += max(0.0, 1.5 - abs(profile.sample_count - dataset_profile["sample_count"]) * 0.4)
        if profile.modality_count and dataset_profile.get("modality_count"):
            score += max(0.0, 1.5 - abs(profile.modality_count - dataset_profile["modality_count"]) * 0.4)
        score += self._range_similarity(profile.num_locations, dataset_profile.get("num_locations", {}).get("median"), 2.0)
        score += self._range_similarity(profile.num_features, dataset_profile.get("num_features", {}).get("median"), 1.0)
        return score

    def _range_similarity(self, requested: int | None, observed: float | None, weight: float) -> float:
        if requested is None or observed is None:
            return 0.0
        ratio = min(requested, observed) / max(requested, observed)
        return round(ratio * weight, 4)

    def _method_preference_score(self, profile: UserProfile, method_entry: dict[str, Any]) -> float:
        base = float(method_entry["score_overall"])
        runtime = method_entry["resource"].get("mean_runtime_seconds")
        memory = method_entry["resource"].get("mean_peak_rss_gb")
        runtime_bonus = 0.0
        memory_bonus = 0.0
        if runtime is not None:
            runtime_bonus = max(0.0, 1.0 - min(runtime / 1800, 1.0))
        if memory is not None:
            memory_bonus = max(0.0, 1.0 - min(memory / 64, 1.0))

        if profile.priority == "accuracy":
            score = base * 0.85 + runtime_bonus * 0.05 + memory_bonus * 0.10
        elif profile.priority == "speed":
            score = base * 0.55 + runtime_bonus * 0.35 + memory_bonus * 0.10
        elif profile.priority == "memory":
            score = base * 0.55 + runtime_bonus * 0.10 + memory_bonus * 0.35
        else:
            score = base * 0.7 + runtime_bonus * 0.15 + memory_bonus * 0.15

        if profile.avoid_deep_learning and method_entry["deep_learning"]:
            score -= 0.12
        return score

    def _violates_hard_constraints(self, profile: UserProfile, recommendation: dict[str, Any]) -> bool:
        if profile.avoid_deep_learning and recommendation["deep_learning"]:
            return True
        if profile.max_runtime_minutes is not None and recommendation["average_runtime_seconds"] is not None:
            if recommendation["average_runtime_seconds"] > profile.max_runtime_minutes * 60:
                return True
        if profile.max_memory_gb is not None and recommendation["average_peak_rss_gb"] is not None:
            if recommendation["average_peak_rss_gb"] > profile.max_memory_gb:
                return True
        return False


def build_follow_up_questions(profile: UserProfile, missing_fields: list[str]) -> str:
    field_labels = {
        "task_type": "Which integration task do you need: matching, embedding, mapping, or multiomics integration?",
        "technology": "Which spatial technology are you using, for example Visium, MERFISH, Stereo-seq, Xenium, or Visium Omics?",
        "species": "Which species is your dataset from?",
        "tissue": "Which tissue or organ is the dataset from?",
        "num_locations": "Roughly how many spatial locations or spots does each sample contain?",
        "modality_count": "How many modalities or omics layers are you integrating?",
    }
    prompts = [field_labels[field] for field in missing_fields[:3] if field in field_labels]
    if not prompts:
        prompts = [
            "Please share the task type, technology, and approximate dataset size so I can match the benchmark accurately."
        ]
    intro = "I need a bit more detail before I can map your case to the benchmark."
    return "\n".join([intro, *[f"- {prompt}" for prompt in prompts]])


def render_rule_based_answer(
    profile: UserProfile,
    result: RecommendationResult,
    lead_in: str | None = None,
) -> str:
    lines = []
    if lead_in:
        lines.extend([lead_in, ""])
    lines.extend([
        "## Recommendation",
        f"Task: `{profile.task_type}`",
        f"Priority: `{profile.priority}`",
        "",
        "Top methods:",
    ])
    for idx, method in enumerate(result.recommended_methods, start=1):
        lines.append(
            f"{idx}. `{method['method']}` with weighted score `{method['weighted_score']}`"
        )
        dataset_bits = ", ".join(
            f"{item['dataset_id']} (rank {item['rank']}, score {item['score_overall']:.3f})"
            for item in method["matched_datasets"]
        )
        lines.append(f"   Evidence: {dataset_bits}")
    lines.extend(["", "Closest benchmark datasets:"])
    for item in result.matched_datasets[:3]:
        lines.append(
            f"- `{item['dataset_id']}`: {item['technology']}, {item['species']}, {item['tissue']}, similarity {item['similarity']}"
        )
    return "\n".join(lines)
