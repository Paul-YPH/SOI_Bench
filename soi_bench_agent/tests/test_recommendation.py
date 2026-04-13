from pathlib import Path

from soi_bench_agent.data_cleaning import build_clean_bundle
from soi_bench_agent.models import UserProfile
from soi_bench_agent.recommendation import ProfileParser, Recommender, build_follow_up_questions


def test_recommender_returns_ssgate_for_matching_visium_omics_profile(tmp_path: Path) -> None:
    bundle = build_clean_bundle(Path("data/raw"), tmp_path)
    recommender = Recommender(bundle)
    profile = UserProfile(
        task_type="multiomics",
        technology="Visium Omics",
        species="Human",
        tissue="Tonsil",
        modality_count=2,
        num_locations=4521,
        priority="accuracy",
    )

    result = recommender.recommend(profile)

    assert result.recommended_methods
    assert result.recommended_methods[0]["method"] == "SSGATE"


def test_follow_up_questions_ask_for_missing_core_fields() -> None:
    profile = UserProfile()
    message = build_follow_up_questions(profile, ["task_type", "technology", "num_locations"])

    assert "matching, embedding, mapping, or multiomics" in message
    assert "Which spatial technology" in message
    assert "how many spatial locations" in message.lower()


def test_profile_parser_prefers_explicit_multiomics_over_generic_integration(tmp_path: Path) -> None:
    bundle = build_clean_bundle(Path("data/raw"), tmp_path)
    parser = ProfileParser(bundle)
    profile = parser.update_profile(
        UserProfile(),
        "I need a multiomics integration method for a Human tonsil Visium Omics dataset.",
    )

    assert profile.task_type == "multiomics"
