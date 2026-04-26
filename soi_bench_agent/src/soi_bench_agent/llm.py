from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from .models import UserProfile
from .prompts import load_prompt
from .recommendation import canonicalize_priority, canonicalize_task, detect_challenge_tags, detect_integration_mode


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start : end + 1])


class OpenAIHelper:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def response_text(self, instructions: str, user_input: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=user_input,
        )
        return response.output_text

    def refine_profile(self, profile: UserProfile, message: str) -> UserProfile:
        prompt = load_prompt("extract_profile.md")
        user_input = json.dumps(
            {
                "current_profile": profile.to_dict(),
                "latest_user_message": message,
            },
            indent=2,
        )
        output = self.response_text(prompt, user_input)
        parsed = extract_json_object(output)
        profile.task_type = canonicalize_task(parsed.get("task_type")) or profile.task_type
        profile.integration_mode = detect_integration_mode(parsed.get("integration_mode")) or profile.integration_mode
        profile.technology = parsed.get("technology") or profile.technology
        profile.species = parsed.get("species") or profile.species
        profile.tissue = parsed.get("tissue") or profile.tissue
        profile.sample_count = parsed.get("sample_count") or profile.sample_count
        profile.modality_count = parsed.get("modality_count") or profile.modality_count
        profile.num_locations = parsed.get("num_locations") or profile.num_locations
        profile.num_features = parsed.get("num_features") or profile.num_features
        for raw_tag in parsed.get("challenge_tags", []):
            for tag in detect_challenge_tags(raw_tag):
                if tag not in profile.challenge_tags:
                    profile.challenge_tags.append(tag)
        profile.priority = canonicalize_priority(parsed.get("priority") or profile.priority)
        profile.max_runtime_minutes = parsed.get("max_runtime_minutes") or profile.max_runtime_minutes
        profile.max_memory_gb = parsed.get("max_memory_gb") or profile.max_memory_gb
        if parsed.get("avoid_deep_learning") is True:
            profile.avoid_deep_learning = True
        if parsed.get("extra_notes"):
            profile.extra_notes.extend(parsed["extra_notes"])
        return profile

    def classify_response_mode(
        self,
        conversation_history: list[dict[str, str]],
        profile: UserProfile,
        latest_user_message: str,
        has_prior_recommendation: bool,
    ) -> str:
        prompt = load_prompt("classify_intent.md")
        user_input = json.dumps(
            {
                "has_prior_recommendation": has_prior_recommendation,
                "profile": profile.to_dict(),
                "conversation_history": conversation_history[-8:],
                "latest_user_message": latest_user_message,
            },
            indent=2,
        )
        output = self.response_text(prompt, user_input)
        parsed = extract_json_object(output)
        response_mode = parsed.get("response_mode")
        allowed = {"recommendation", "follow_up_recommendation", "follow_up_qa"}
        if response_mode not in allowed:
            raise ValueError(f"Unexpected response_mode: {response_mode}")
        return response_mode

    def generate_answer(
        self,
        conversation_history: list[dict[str, str]],
        profile: UserProfile,
        recommendation_payload: dict[str, Any],
        response_mode: str = "recommendation",
        latest_user_message: str | None = None,
    ) -> str:
        system_prompt = load_prompt("system.md")
        answer_prompt_name = (
            "follow_up_answer.md" if response_mode == "follow_up_qa" else "recommendation_answer.md"
        )
        answer_prompt = load_prompt(answer_prompt_name)
        payload = json.dumps(
            {
                "profile": profile.to_dict(),
                "latest_user_message": latest_user_message,
                "response_mode": response_mode,
                "conversation_history": conversation_history[-8:],
                "recommendation_payload": recommendation_payload,
            },
            indent=2,
        )
        return self.response_text(f"{system_prompt}\n\n{answer_prompt}", payload)
