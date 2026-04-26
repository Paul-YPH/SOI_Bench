from pathlib import Path

from soi_bench_agent.data_cleaning import build_clean_bundle


def test_build_clean_bundle_creates_expected_artifacts(tmp_path: Path) -> None:
    raw_dir = Path("data/raw")
    bundle = build_clean_bundle(raw_dir, tmp_path)

    assert "dataset_profiles" in bundle
    assert "dataset_rankings" in bundle
    assert "method_profiles" in bundle
    assert "matching" in bundle["task_overview"]
    assert "multiomics" in bundle["task_overview"]
    assert (tmp_path / "knowledge_base.json").exists()
    assert (tmp_path / "benchmark_records.csv").exists()


def test_dataset_profile_contains_expected_experiment_metadata(tmp_path: Path) -> None:
    bundle = build_clean_bundle(Path("data/raw"), tmp_path)
    d58 = bundle["dataset_profiles"]["D58"]

    assert d58["technology"] == "Visium Omics"
    assert d58["species"] == "Human"
    assert d58["tissue"] == "Tonsil"
    assert d58["sample_count"] == 2


def test_new_raw_profiles_include_integration_and_simulation_tags(tmp_path: Path) -> None:
    bundle = build_clean_bundle(Path("data/new_raw"), tmp_path)

    d58 = bundle["dataset_profiles"]["D58"]
    sd42 = bundle["dataset_profiles"]["SD42"]

    assert d58["integration_mode"] == "multiomics_one_slice"
    assert d58["sample_count"] == 1
    assert d58["modality_count"] == 2
    assert sd42["integration_mode"] == "cross-slice"
    assert "multi_slice" in sd42["challenge_tags"]
    assert "batch_effect" in sd42["challenge_tags"]


def test_new_raw_duplicate_simulation_files_do_not_duplicate_entries(tmp_path: Path) -> None:
    bundle = build_clean_bundle(Path("data/new_raw"), tmp_path)
    unique_keys = {
        (entry["dataset_id"], entry["task_type"], entry["method"])
        for entry in bundle["benchmark_entries"]
    }

    assert len(unique_keys) == len(bundle["benchmark_entries"])
