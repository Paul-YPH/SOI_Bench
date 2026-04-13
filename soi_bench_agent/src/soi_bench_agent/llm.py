from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from .models import UserProfile
from .prompts import load_prompt
from .recommendation import canonicalize_priority, canonicalize_task


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
        profile.technology = parsed.get("technology") or profile.technology
        profile.species = parsed.get("species") or profile.species
        profile.tissue = parsed.get("tissue") or profile.tissue
        profile.sample_count = parsed.get("sample_count") or profile.sample_count
        profile.modality_count = parsed.get("modality_count") or profile.modality_count
        profile.num_locations = parsed.get("num_locations") or profile.num_locations
        profile.num_features = parsed.get("num_features") or profile.num_features
        profile.priority = canonicalize_priority(parsed.get("priority") or profile.priority)
        profile.max_runtime_minutes = parsed.get("max_runtime_minutes") or profile.max_runtime_minutes
        profile.max_memory_gb = parsed.get("max_memory_gb") or profile.max_memory_gb
        if parsed.get("avoid_deep_learning") is True:
            profile.avoid_deep_learning = True
        if parsed.get("extra_notes"):
            profile.extra_notes.extend(parsed["extra_notes"])
        return profile

    def generate_answer(
        self,
        conversation_history: list[dict[str, str]],
        profile: UserProfile,
        recommendation_payload: dict[str, Any],
    ) -> str:
        system_prompt = load_prompt("system.md")
        answer_prompt = load_prompt("recommendation_answer.md")
        payload = json.dumps(
            {
                "profile": profile.to_dict(),
                "conversation_history": conversation_history[-8:],
                "recommendation_payload": recommendation_payload,
            },
            indent=2,
        )
        return self.response_text(f"{system_prompt}\n\n{answer_prompt}", payload)
