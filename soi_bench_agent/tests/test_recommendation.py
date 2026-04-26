from pathlib import Path

from rich.console import Console

from soi_bench_agent.cli import (
    ConversationAgent,
    build_assistant_subtitle,
    build_assistant_title,
    build_pending_status,
    build_user_prompt,
    render_welcome,
)
from soi_bench_agent.data_cleaning import build_clean_bundle
from soi_bench_agent.models import UserProfile
from soi_bench_agent.recommendation import (
    ProfileParser,
    Recommender,
    build_follow_up_questions,
    canonicalize_task,
)


class FakeLLM:
    def __init__(self, answer: str, response_mode: str | None = None) -> None:
        self.answer = answer
        self.response_mode = response_mode

    def refine_profile(self, profile: UserProfile, message: str) -> UserProfile:
        return profile

    def classify_response_mode(
        self,
        conversation_history: list[dict[str, str]],
        profile: UserProfile,
        latest_user_message: str,
        has_prior_recommendation: bool,
    ) -> str:
        if self.response_mode is None:
            raise ValueError("No mock response mode configured")
        return self.response_mode

    def generate_answer(
        self,
        conversation_history: list[dict[str, str]],
        profile: UserProfile,
        recommendation_payload: dict,
        response_mode: str = "recommendation",
        latest_user_message: str | None = None,
    ) -> str:
        return self.answer


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


def test_profile_parser_extracts_integration_mode_and_challenge_tags(tmp_path: Path) -> None:
    bundle = build_clean_bundle(Path("data/new_raw"), tmp_path)
    parser = ProfileParser(bundle)
    profile = parser.update_profile(
        UserProfile(),
        "I need a matching method for a Human Visium cross-slice dataset with rotation and partial overlap.",
    )

    assert profile.integration_mode == "cross-slice"
    assert "rigid_alignment" in profile.challenge_tags
    assert "partial_overlap" in profile.challenge_tags


def test_canonicalize_task_does_not_treat_matched_as_matching() -> None:
    assert canonicalize_task("Introduce the matched datasets") is None


def test_recommender_uses_simulation_challenge_tags_when_matching_new_raw(tmp_path: Path) -> None:
    bundle = build_clean_bundle(Path("data/new_raw"), tmp_path)
    recommender = Recommender(bundle)
    profile = UserProfile(
        task_type="matching",
        integration_mode="cross-slice",
        technology="Visium",
        species="Human",
        challenge_tags=["rigid_alignment"],
    )

    result = recommender.recommend(profile)

    assert result.matched_datasets
    assert any("rigid_alignment" in item["challenge_tags"] for item in result.matched_datasets[:3])


def test_conversation_agent_adds_natural_lead_in_for_follow_up_questions(tmp_path: Path) -> None:
    bundle = build_clean_bundle(Path("data/raw"), tmp_path)
    agent = ConversationAgent(bundle, llm=None)

    first_reply = agent.respond(
        "I need a multiomics integration method for a Human tonsil Visium Omics dataset with 2 modalities and around 4521 spots."
    )
    follow_up_reply = agent.respond("Why not prioritize lighter methods?")

    assert first_reply.startswith("## Recommendation")
    assert follow_up_reply.startswith("The top-ranked option is")
    assert "## Recommendation" not in follow_up_reply


def test_conversation_agent_uses_content_sentence_for_memory_follow_up(tmp_path: Path) -> None:
    bundle = build_clean_bundle(Path("data/raw"), tmp_path)
    agent = ConversationAgent(bundle, llm=None)

    agent.respond(
        "I need a multiomics integration method for a Human tonsil Visium Omics dataset with 2 modalities and around 4521 spots."
    )
    follow_up_reply = agent.respond("How large memory does SSGATE need to run it?")

    assert follow_up_reply.startswith("SSGATE used about")
    assert "## Recommendation" not in follow_up_reply


def test_cli_turn_labels_are_explicit() -> None:
    assert "Turn" not in build_user_prompt(3)
    assert "turn" not in build_pending_status(3).lower()
    assert "Turn" not in build_assistant_title(3)
    assert "You:" in build_user_prompt(3)
    assert "Assistant is working" in build_pending_status(3)
    assert "Assistant" in build_assistant_title(3)
    assert "Reply complete in 2.5s" in build_assistant_subtitle(2.5)


def test_follow_up_llm_reply_gets_forced_lead_in_when_missing(tmp_path: Path) -> None:
    bundle = build_clean_bundle(Path("data/raw"), tmp_path)
    agent = ConversationAgent(bundle, llm=FakeLLM("## Recommendation\nSSGATE details"))

    agent.respond(
        "I need a multiomics integration method for a Human tonsil Visium Omics dataset with 2 modalities and around 4521 spots."
    )
    follow_up_reply = agent.respond("How large memory does SSGATE need to run it?")

    assert follow_up_reply.startswith("SSGATE used about")
    assert "## Recommendation" not in follow_up_reply


def test_explanatory_follow_up_uses_natural_mode_without_template(tmp_path: Path) -> None:
    bundle = build_clean_bundle(Path("data/raw"), tmp_path)
    agent = ConversationAgent(
        bundle,
        llm=FakeLLM(
            "D15, D41, and D42 are the closest matched datasets for this case.",
            response_mode="follow_up_qa",
        ),
    )

    agent.respond(
        "I need an embedding method for a Mouse brain Visium dataset with 2 samples."
    )
    follow_up_reply = agent.respond("Introduce the matched datasets")

    assert follow_up_reply == "D15, D41, and D42 are the closest matched datasets for this case."


def test_llm_intent_classifier_routes_what_is_d42_to_follow_up_qa(tmp_path: Path) -> None:
    bundle = build_clean_bundle(Path("data/raw"), tmp_path)
    agent = ConversationAgent(
        bundle,
        llm=FakeLLM(
            "D42 is one of the closest matched mouse brain Visium embedding datasets.",
            response_mode="follow_up_qa",
        ),
    )

    agent.respond("I need an embedding method for a Mouse brain Visium dataset with 2 samples.")
    follow_up_reply = agent.respond("What is D42")

    assert follow_up_reply == "D42 is one of the closest matched mouse brain Visium embedding datasets."


def test_follow_up_llm_meta_lead_in_gets_replaced(tmp_path: Path) -> None:
    bundle = build_clean_bundle(Path("data/raw"), tmp_path)
    agent = ConversationAgent(
        bundle,
        llm=FakeLLM(
            "I'll answer the memory question directly.\n\n## Recommendation\nSSGATE details"
        ),
    )

    agent.respond(
        "I need a multiomics integration method for a Human tonsil Visium Omics dataset with 2 modalities and around 4521 spots."
    )
    follow_up_reply = agent.respond("How large memory does SSGATE need to run it?")

    assert follow_up_reply.startswith("SSGATE used about")
    assert "I'll answer the memory question directly." not in follow_up_reply
    assert "## Recommendation" not in follow_up_reply


def test_hardware_guidance_for_gpu_question_avoids_vram_claims(tmp_path: Path) -> None:
    bundle = build_clean_bundle(Path("data/raw"), tmp_path)
    agent = ConversationAgent(bundle, llm=None)
    agent.respond(
        "I need a multiomics integration method for a Human tonsil Visium Omics dataset with 2 modalities and around 4521 spots."
    )
    result = agent.recommender.recommend(agent.profile)

    guidance = agent._build_hardware_guidance("Can SSGATE run on RTX 5000 and how much GPU memory does it need?", result)

    assert guidance["gpu_question"] is True
    assert "NVIDIA RTX 5000 Ada Generation" in guidance["mentioned_gpu_models"]
    assert guidance["method_observations"]
    observed_dataset_ids = {
        dataset_id
        for item in guidance["method_observations"]
        for dataset_id in item["matched_dataset_ids"]
    }
    assert observed_dataset_ids
    assert all(dataset_id.startswith("D") for dataset_id in observed_dataset_ids)


def test_gpu_follow_up_uses_gpu_specific_lead_in(tmp_path: Path) -> None:
    bundle = build_clean_bundle(Path("data/raw"), tmp_path)
    agent = ConversationAgent(bundle, llm=None)

    agent.respond(
        "I need a multiomics integration method for a Human tonsil Visium Omics dataset with 2 modalities and around 4521 spots."
    )
    follow_up_reply = agent.respond("Can SSGATE run on RTX 5000 and how much GPU memory does it need?")

    assert follow_up_reply.startswith(
        "SSGATE was observed to run on NVIDIA RTX 5000 Ada Generation in the closest matched experimental benchmarks"
    )
    assert "## Recommendation" not in follow_up_reply


def test_welcome_mentions_exit_command() -> None:
    console = Console(record=True)
    render_welcome(console)

    output = console.export_text()
    assert "/exit" in output
