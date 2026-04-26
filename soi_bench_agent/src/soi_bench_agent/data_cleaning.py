from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .utils import maybe_float, normalize_text, slugify, summarize_numbers, write_json


RESULT_PATTERNS = {
    "matching": re.compile(r"^(?P<dataset>[^_]+)_match_scIB\.csv$"),
    "embedding": re.compile(r"^(?P<dataset>[^_]+)_emb_scIB\.csv$"),
    "mapping": re.compile(r"^(?P<dataset>[^_]+)_coord_scIB\.csv$"),
    "multiomics": re.compile(r"^(?P<dataset>[^_]+)_multiomics_scIB\.csv$"),
}

TRACE_FILES = {
    "experiment": [
        "summary/experiment_trace_summary_D1_D42.csv",
        "summary/experiment_trace_summary_omics.csv",
    ],
    "simulation": ["summary/simulation_trace_summary.csv"],
}

GPU_FILES = {
    "experiment": [
        "summary/experiment_gpu_summary_D1_D42.csv",
        "summary/experiment_gpu_summary_omics.csv",
    ],
    "simulation": ["summary/simulation_gpu_summary.csv"],
}


def parse_duration_to_seconds(value: str | None) -> float | None:
    if not value:
        return None
    total = 0.0
    for amount, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", value):
        amount_float = float(amount)
        if unit == "h":
            total += amount_float * 3600
        elif unit == "m":
            total += amount_float * 60
        else:
            total += amount_float
    return total or None


def parse_memory_gb(value: str | None) -> float | None:
    if not value:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*(TB|GB|MB)", value, re.IGNORECASE)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2).upper()
    if unit == "TB":
        return amount * 1024
    if unit == "MB":
        return amount / 1024
    return amount


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_optional_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return read_csv_rows(path)


def _truthy_csv_flag(value: str | None) -> bool:
    return (value or "").strip().upper() == "TRUE"


def _extract_numeric_suffix(dataset_id: str) -> int | None:
    match = re.search(r"(\d+)$", dataset_id)
    if not match:
        return None
    return int(match.group(1))


def infer_simulation_tags(dataset_id: str, dataset_rows: list[dict[str, str]]) -> list[str]:
    tags: list[str] = []
    first_row = dataset_rows[0] if dataset_rows else {}
    if any(_truthy_csv_flag(row.get("multi_slice")) for row in dataset_rows):
        tags.append("multi_slice")
    if any(_truthy_csv_flag(row.get("rotation")) for row in dataset_rows):
        tags.append("rigid_alignment")
    if any(_truthy_csv_flag(row.get("distortion")) for row in dataset_rows):
        tags.append("non_rigid_deformation")
    if any(_truthy_csv_flag(row.get("overlap")) for row in dataset_rows):
        tags.append("partial_overlap")
    if any(_truthy_csv_flag(row.get("pseudocount")) for row in dataset_rows):
        tags.append("batch_effect")

    dataset_number = _extract_numeric_suffix(dataset_id)
    if dataset_number is not None:
        if 1 <= dataset_number <= 6:
            tags.append("scale_variation")
        elif 7 <= dataset_number <= 13:
            tags.append("gene_coverage_variation")
        elif 44 <= dataset_number <= 55:
            tags.append("statistical_simulation")
        elif 56 <= dataset_number <= 60:
            tags.append("cross_panel_integration")

    if first_row.get("technology") == "MERFISH" and "cross_panel_integration" not in tags:
        tags.append("cross_panel_integration")
    return sorted(set(tags))


def _build_default_profile(dataset_id: str) -> dict:
    origin = infer_origin(dataset_id)
    return {
        "dataset_id": dataset_id,
        "origin": origin,
        "technology": "Simulation" if origin == "simulation" else None,
        "technology_list": ["Simulation"] if origin == "simulation" else [],
        "species": None,
        "tissue": None,
        "integration_mode": None,
        "challenge_tags": [],
        "sample_count": None,
        "modality_count": None,
        "num_locations": summarize_numbers([]),
        "num_features": summarize_numbers([]),
    }


def build_dataset_profiles(raw_dir: Path) -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    experiment_rows = _read_optional_csv_rows(raw_dir / "data_info_experiment.csv")
    simulation_rows = _read_optional_csv_rows(raw_dir / "data_info_simulation.csv")
    if not experiment_rows:
        experiment_rows = [
            row
            for row in read_csv_rows(raw_dir / "data_info.csv")
            if not row["data_id"].startswith("SD")
        ]

    grouped_experiment_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in experiment_rows:
        grouped_experiment_rows[row["data_id"]].append(row)

    for dataset_id, dataset_rows in grouped_experiment_rows.items():
        locations = [int(row["num_locations"]) for row in dataset_rows]
        features = [int(row["num_features"]) for row in dataset_rows]
        technologies = sorted({row["technology"] for row in dataset_rows})
        species = sorted({row["species"] for row in dataset_rows})
        tissues = sorted({row["tissue"] for row in dataset_rows})
        integration_modes = sorted({row.get("integration") for row in dataset_rows if row.get("integration")})
        integration_mode = integration_modes[0] if len(integration_modes) == 1 else None
        sample_count = len(dataset_rows)
        modality_count = len(dataset_rows)
        if integration_mode == "cross-slice":
            modality_count = 1
        elif integration_mode == "multiomics_one_slice":
            sample_count = 1
        profiles[dataset_id] = {
            "dataset_id": dataset_id,
            "origin": "experiment",
            "technology": technologies[0] if len(technologies) == 1 else "/".join(technologies),
            "technology_list": technologies,
            "species": species[0] if len(species) == 1 else "/".join(species),
            "tissue": tissues[0] if len(tissues) == 1 else "/".join(tissues),
            "integration_mode": integration_mode,
            "challenge_tags": [],
            "sample_count": sample_count,
            "modality_count": modality_count,
            "num_locations": summarize_numbers(locations),
            "num_features": summarize_numbers(features),
        }

    grouped_simulation_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in simulation_rows:
        grouped_simulation_rows[row["data_id"]].append(row)

    for dataset_id, dataset_rows in grouped_simulation_rows.items():
        locations = [int(row["num_locations"]) for row in dataset_rows]
        features = [int(row["num_features"]) for row in dataset_rows]
        technologies = sorted({row["technology"] for row in dataset_rows if row.get("technology")})
        species = sorted({row["species"] for row in dataset_rows if row.get("species")})
        profiles[dataset_id] = {
            "dataset_id": dataset_id,
            "origin": "simulation",
            "technology": technologies[0] if len(technologies) == 1 else "/".join(technologies),
            "technology_list": technologies,
            "species": species[0] if len(species) == 1 else "/".join(species),
            "tissue": None,
            "integration_mode": "cross-slice",
            "challenge_tags": infer_simulation_tags(dataset_id, dataset_rows),
            "sample_count": len(dataset_rows),
            "modality_count": 1,
            "num_locations": summarize_numbers(locations),
            "num_features": summarize_numbers(features),
        }
    return profiles


def infer_origin(dataset_id: str) -> str:
    return "simulation" if dataset_id.startswith("SD") else "experiment"


def build_resource_index(raw_dir: Path) -> dict[tuple[str, str], dict]:
    grouped: dict[tuple[str, str], dict[str, list]] = defaultdict(
        lambda: {
            "runtime_seconds": [],
            "peak_rss_gb": [],
            "peak_vmem_gb": [],
            "gpu_models": set(),
        }
    )

    for origins, file_list in TRACE_FILES.items():
        for rel_path in file_list:
            path = raw_dir / rel_path
            if not path.exists():
                continue
            for row in read_csv_rows(path):
                key = (row["dataset"], row["tool"])
                grouped[key]["runtime_seconds"].append(parse_duration_to_seconds(row.get("realtime")))
                grouped[key]["peak_rss_gb"].append(parse_memory_gb(row.get("peak_rss")))
                grouped[key]["peak_vmem_gb"].append(parse_memory_gb(row.get("peak_vmem")))

    for origins, file_list in GPU_FILES.items():
        for rel_path in file_list:
            path = raw_dir / rel_path
            if not path.exists():
                continue
            for row in read_csv_rows(path):
                key = (row["dataset"], row["tool"])
                grouped[key]["gpu_models"].add(row["gpu_model"])

    all_gpu_usage_path = raw_dir / "summary" / "all_gpu_usage_summary.csv"
    if all_gpu_usage_path.exists():
        for row in read_csv_rows(all_gpu_usage_path):
            key = (row["dataset"], row["tool"])
            grouped[key]["gpu_models"].add(row["gpu_model"])

    resource_index: dict[tuple[str, str], dict] = {}
    for key, values in grouped.items():
        runtime_values = [value for value in values["runtime_seconds"] if value is not None]
        peak_rss_values = [value for value in values["peak_rss_gb"] if value is not None]
        peak_vmem_values = [value for value in values["peak_vmem_gb"] if value is not None]
        resource_index[key] = {
            "mean_runtime_seconds": round(sum(runtime_values) / len(runtime_values), 3) if runtime_values else None,
            "mean_peak_rss_gb": round(sum(peak_rss_values) / len(peak_rss_values), 3) if peak_rss_values else None,
            "mean_peak_vmem_gb": round(sum(peak_vmem_values) / len(peak_vmem_values), 3) if peak_vmem_values else None,
            "gpu_models": sorted(values["gpu_models"]),
            "requires_gpu": bool(values["gpu_models"]),
        }
    return resource_index


def extract_result_entries(raw_dir: Path, dataset_profiles: dict[str, dict], resource_index: dict[tuple[str, str], dict]) -> list[dict]:
    entries: list[dict] = []
    selected_paths: dict[str, Path] = {}
    result_dirs = [raw_dir / "experiment_results", raw_dir / "simulation_results"]
    for result_dir in result_dirs:
        for path in sorted(result_dir.glob("*.csv")):
            task_type = None
            dataset_id = None
            for candidate_task, pattern in RESULT_PATTERNS.items():
                match = pattern.match(path.name)
                if match:
                    task_type = candidate_task
                    dataset_id = match.group("dataset")
                    break
            if task_type is None or dataset_id is None:
                continue
            preferred_dir = "simulation_results" if dataset_id.startswith("SD") else "experiment_results"
            if path.name not in selected_paths or path.parent.name == preferred_dir:
                selected_paths[path.name] = path

    for path in sorted(selected_paths.values()):
        task_type = None
        dataset_id = None
        for candidate_task, pattern in RESULT_PATTERNS.items():
            match = pattern.match(path.name)
            if match:
                task_type = candidate_task
                dataset_id = match.group("dataset")
                break
        if task_type is None or dataset_id is None:
            continue

        rows = read_csv_rows(path)
        valid_rows = []
        for row in rows:
            score_overall = maybe_float(row.get("Score overall"))
            if score_overall is None:
                continue
            copied = dict(row)
            copied["Score overall"] = str(score_overall)
            valid_rows.append(copied)

        ranked_rows = sorted(valid_rows, key=lambda row: float(row["Score overall"]), reverse=True)
        for rank, row in enumerate(ranked_rows, start=1):
            method = row["Method"]
            resource = resource_index.get((dataset_id, method), {})
            metrics = {
                slugify(key): maybe_float(value)
                for key, value in row.items()
                if key not in {"Method", "Language", "Deep Learning"}
            }
            entry = {
                "dataset_id": dataset_id,
                "origin": infer_origin(dataset_id),
                "task_type": task_type,
                "method": method,
                "language": row["Language"],
                "deep_learning": row["Deep Learning"] == "dl_yes",
                "score_overall": float(row["Score overall"]),
                "rank": rank,
                "metrics": metrics,
                "resource": resource,
                "dataset_profile": dataset_profiles.get(dataset_id, _build_default_profile(dataset_id)),
            }
            entries.append(entry)
    return entries


def build_dataset_rankings(entries: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for entry in entries:
        grouped[(entry["dataset_id"], entry["task_type"])].append(entry)

    rankings = []
    for (dataset_id, task_type), grouped_entries in sorted(grouped.items()):
        sorted_entries = sorted(grouped_entries, key=lambda entry: entry["rank"])
        dataset_profile = sorted_entries[0]["dataset_profile"]
        rankings.append(
            {
                "dataset_id": dataset_id,
                "task_type": task_type,
                "origin": sorted_entries[0]["origin"],
                "dataset_profile": dataset_profile,
                "top_methods": [
                    {
                        "method": entry["method"],
                        "score_overall": entry["score_overall"],
                        "rank": entry["rank"],
                        "deep_learning": entry["deep_learning"],
                        "resource": entry["resource"],
                    }
                    for entry in sorted_entries[:5]
                ],
                "all_methods": [
                    {
                        "method": entry["method"],
                        "score_overall": entry["score_overall"],
                        "rank": entry["rank"],
                        "deep_learning": entry["deep_learning"],
                        "resource": entry["resource"],
                    }
                    for entry in sorted_entries
                ],
            }
        )
    return rankings


def build_method_profiles(entries: list[dict]) -> dict[str, dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for entry in entries:
        grouped[(entry["task_type"], entry["method"])].append(entry)

    result: dict[str, dict] = defaultdict(dict)
    for (task_type, method), method_entries in grouped.items():
        scores = [entry["score_overall"] for entry in method_entries]
        ranks = [entry["rank"] for entry in method_entries]
        runtimes = [
            entry["resource"]["mean_runtime_seconds"]
            for entry in method_entries
            if entry["resource"].get("mean_runtime_seconds") is not None
        ]
        peak_rss = [
            entry["resource"]["mean_peak_rss_gb"]
            for entry in method_entries
            if entry["resource"].get("mean_peak_rss_gb") is not None
        ]
        gpu_models = sorted(
            {
                gpu_model
                for entry in method_entries
                for gpu_model in entry["resource"].get("gpu_models", [])
            }
        )
        result[task_type][method] = {
            "method": method,
            "task_type": task_type,
            "language": sorted({entry["language"] for entry in method_entries}),
            "deep_learning": any(entry["deep_learning"] for entry in method_entries),
            "appearance_count": len(method_entries),
            "win_count": sum(entry["rank"] == 1 for entry in method_entries),
            "average_score": round(sum(scores) / len(scores), 6),
            "average_rank": round(sum(ranks) / len(ranks), 6),
            "average_runtime_seconds": round(sum(runtimes) / len(runtimes), 3) if runtimes else None,
            "average_peak_rss_gb": round(sum(peak_rss) / len(peak_rss), 3) if peak_rss else None,
            "gpu_models": gpu_models,
        }
    return {task: dict(sorted(methods.items())) for task, methods in sorted(result.items())}


def build_task_overview(method_profiles: dict[str, dict]) -> dict[str, list[dict]]:
    overview = {}
    for task_type, methods in method_profiles.items():
        ordered = sorted(
            methods.values(),
            key=lambda item: (item["average_rank"], -item["average_score"]),
        )
        overview[task_type] = ordered[:10]
    return overview


def build_clean_bundle(raw_dir: Path, clean_dir: Path) -> dict:
    dataset_profiles = build_dataset_profiles(raw_dir)
    resource_index = build_resource_index(raw_dir)
    entries = extract_result_entries(raw_dir, dataset_profiles, resource_index)
    dataset_rankings = build_dataset_rankings(entries)
    method_profiles = build_method_profiles(entries)
    task_overview = build_task_overview(method_profiles)
    knowledge_base = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_raw_dir": str(raw_dir),
        "dataset_profiles": dict(sorted(dataset_profiles.items())),
        "dataset_rankings": dataset_rankings,
        "method_profiles": method_profiles,
        "task_overview": task_overview,
        "benchmark_entries": entries,
    }

    clean_dir.mkdir(parents=True, exist_ok=True)
    write_json(clean_dir / "knowledge_base.json", knowledge_base)
    write_json(clean_dir / "dataset_profiles.json", knowledge_base["dataset_profiles"])
    write_json(clean_dir / "dataset_rankings.json", dataset_rankings)
    write_json(clean_dir / "method_profiles.json", method_profiles)

    benchmark_csv = clean_dir / "benchmark_records.csv"
    fieldnames = [
        "dataset_id",
        "origin",
        "task_type",
        "method",
        "language",
        "deep_learning",
        "score_overall",
        "rank",
        "technology",
        "species",
        "tissue",
        "sample_count",
        "modality_count",
        "mean_runtime_seconds",
        "mean_peak_rss_gb",
        "mean_peak_vmem_gb",
        "requires_gpu",
        "gpu_models",
        "metrics_json",
    ]
    with benchmark_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "dataset_id": entry["dataset_id"],
                    "origin": entry["origin"],
                    "task_type": entry["task_type"],
                    "method": entry["method"],
                    "language": entry["language"],
                    "deep_learning": entry["deep_learning"],
                    "score_overall": entry["score_overall"],
                    "rank": entry["rank"],
                    "technology": entry["dataset_profile"]["technology"],
                    "species": entry["dataset_profile"]["species"],
                    "tissue": entry["dataset_profile"]["tissue"],
                    "sample_count": entry["dataset_profile"]["sample_count"],
                    "modality_count": entry["dataset_profile"]["modality_count"],
                    "mean_runtime_seconds": entry["resource"].get("mean_runtime_seconds"),
                    "mean_peak_rss_gb": entry["resource"].get("mean_peak_rss_gb"),
                    "mean_peak_vmem_gb": entry["resource"].get("mean_peak_vmem_gb"),
                    "requires_gpu": entry["resource"].get("requires_gpu"),
                    "gpu_models": json.dumps(entry["resource"].get("gpu_models", [])),
                    "metrics_json": json.dumps(entry["metrics"], sort_keys=True),
                }
            )
    return knowledge_base


def main() -> None:
    root_dir = Path.cwd()
    build_clean_bundle(root_dir / "data/raw", root_dir / "data/clean")
    print("Built clean benchmark artifacts in data/clean")


if __name__ == "__main__":
    main()
