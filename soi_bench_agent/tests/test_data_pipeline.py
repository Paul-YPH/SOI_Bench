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
