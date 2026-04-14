from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from .config import load_settings
from .data_cleaning import build_clean_bundle
from .knowledge_base import load_knowledge_base
from .llm import OpenAIHelper
from .models import UserProfile
from .recommendation import ProfileParser, Recommender, build_follow_up_questions, render_rule_based_answer
from .utils import normalize_text


GPU_MODEL_ALIASES = {
    "NVIDIA H200": ["h200", "nvidia h200"],
    "NVIDIA RTX 5000 Ada Generation": [
        "rtx 5000",
        "rtx5000",
        "rtx 5000 ada",
        "rtx5000 ada",
        "nvidia rtx 5000 ada generation",
    ],
}


class ConversationAgent:
    def __init__(self, knowledge_base: dict, llm: OpenAIHelper | None = None) -> None:
        self.knowledge_base = knowledge_base
        self.profile = UserProfile()
        self.history: list[dict[str, str]] = []
        self.has_delivered_recommendation = False
        self.parser = ProfileParser(knowledge_base)
        self.recommender = Recommender(knowledge_base)
        self.llm = llm

    def respond(self, message: str) -> str:
        response_mode = self._determine_response_mode(message)
        gpu_question = self._is_gpu_question(message)
        self.history.append({"role": "user", "content": message})
        self.profile = self.parser.update_profile(self.profile, message)
        if self.llm is not None:
            try:
                self.profile = self.llm.refine_profile(self.profile, message)
            except Exception:
                pass

        missing = self.recommender.missing_fields(self.profile)
        if not self.recommender.can_recommend(self.profile):
            reply = build_follow_up_questions(self.profile, missing)
            self.history.append({"role": "assistant", "content": reply})
            return reply

        result = self.recommender.recommend(self.profile)
        payload = {
            "matched_datasets": result.matched_datasets,
            "recommended_methods": result.recommended_methods,
            "discarded_methods": result.discarded_methods,
            "summary": result.summary,
            "hardware_guidance": self._build_hardware_guidance(message, result) if gpu_question else None,
        }
        if self.llm is not None:
            try:
                reply = self.llm.generate_answer(
                    self.history,
                    self.profile,
                    payload,
                    response_mode=response_mode,
                    latest_user_message=message,
                )
                reply = self._finalize_reply(message, response_mode, reply, result=result)
                self.history.append({"role": "assistant", "content": reply})
                self.has_delivered_recommendation = True
                return reply
            except Exception:
                pass

        reply = render_rule_based_answer(
            self.profile,
            result,
            lead_in=self._build_follow_up_lead_in(message, result) if response_mode == "follow_up_recommendation" else None,
        )
        if response_mode == "follow_up_qa":
            reply = self._render_natural_follow_up_answer(message, result)
        reply = self._finalize_reply(message, response_mode, reply, result=result)
        self.history.append({"role": "assistant", "content": reply})
        self.has_delivered_recommendation = True
        return reply

    def reset(self) -> None:
        self.profile = UserProfile()
        self.history.clear()
        self.has_delivered_recommendation = False

    def _determine_response_mode(self, message: str) -> str:
        if self.llm is not None:
            try:
                return self.llm.classify_response_mode(
                    self.history,
                    self.profile,
                    latest_user_message=message,
                    has_prior_recommendation=self.has_delivered_recommendation,
                )
            except Exception:
                pass
        if not self._is_follow_up_turn(message):
            return "recommendation"
        if self._wants_recommendation(message):
            return "follow_up_recommendation"
        return "follow_up_qa"

    def _is_follow_up_turn(self, message: str) -> bool:
        if not self.has_delivered_recommendation:
            return False
        stripped = message.strip()
        if not stripped:
            return False
        lowered = stripped.lower()
        if "?" in stripped or "？" in stripped:
            return True
        prefixes = (
            "why",
            "what if",
            "how",
            "which",
            "can",
            "could",
            "would",
            "should",
            "then",
            "if ",
            "why not",
            "how about",
            "那",
            "那如果",
            "如果",
            "为什么",
            "怎么",
            "是否",
            "能不能",
            "可不可以",
            "那这个",
            "introduce",
            "explain",
            "describe",
            "tell me about",
            "summarize",
            "compare",
        )
        return lowered.startswith(prefixes)

    def _wants_recommendation(self, message: str) -> bool:
        lowered = message.lower().strip()
        recommendation_patterns = (
            r"\brecommend\b",
            r"\bbest\b",
            r"\btop\b",
            r"\bwhich method\b",
            r"\bwhat method\b",
            r"\bshould i use\b",
            r"\bpriority\b",
            r"\bchoose\b",
            r"推荐",
            r"哪个好",
            r"哪种方法",
            r"怎么选",
        )
        return any(re.search(pattern, lowered) for pattern in recommendation_patterns)

    def _build_follow_up_lead_in(self, message: str, result=None) -> str:
        lowered = message.lower()
        if self._is_gpu_question(message):
            return self._build_gpu_follow_up_lead_in(message, result)
        if "memory" in lowered or "ram" in lowered:
            return "SSGATE is memory-heavy in the matched benchmark runs."
        if "runtime" in lowered or "time" in lowered or "fast" in lowered or "slow" in lowered:
            return "SSGATE is not especially lightweight on runtime in the matched benchmark runs."
        if "why not" in lowered or lowered.startswith("why"):
            return "The main constraint in the matched benchmark is resource cost rather than benchmark accuracy."
        return "Here is the direct benchmark-grounded answer to that follow-up."

    def _finalize_reply(self, message: str, response_mode: str, reply: str, result=None) -> str:
        normalized = reply.strip()
        if response_mode == "follow_up_qa":
            if self._looks_templated(normalized):
                return self._render_natural_follow_up_answer(message, result)
            return normalized
        if response_mode != "follow_up_recommendation":
            return normalized
        return self._enforce_follow_up_lead_in(message, normalized, result=result)

    def _enforce_follow_up_lead_in(self, message: str, reply: str, result=None) -> str:
        lead_in = self._build_follow_up_lead_in(message, result)
        if reply.startswith(f"{lead_in}\n"):
            return reply

        heading = "## Recommendation"
        heading_index = reply.find(heading)
        if heading_index == -1:
            body = reply
        else:
            body = reply[heading_index:]
        body = body.lstrip()
        return f"{lead_in}\n\n{body}"

    def _looks_templated(self, reply: str) -> bool:
        headings = (
            "## Recommendation",
            "## Why These Methods",
            "## Tradeoffs",
            "## Next Best Questions",
        )
        return any(heading in reply for heading in headings)

    def _build_gpu_follow_up_lead_in(self, message: str, result) -> str:
        if result is None:
            return "The closest matched experimental benchmarks provide GPU observations for similar runs."
        guidance = self._build_hardware_guidance(message, result)
        observations = guidance.get("method_observations", [])
        if observations:
            first = observations[0]
            reference_gpu = first.get("reference_gpu_model")
            if reference_gpu:
                return (
                    f"{first['method']} was observed to run on {reference_gpu} "
                    "in the closest matched experimental benchmarks."
                )
        dataset_observations = [
            item for item in guidance.get("dataset_observations", []) if item.get("successful_gpu_models")
        ]
        if dataset_observations and len(dataset_observations[0]["successful_gpu_models"]) == 1:
            reference_gpu = dataset_observations[0]["successful_gpu_models"][0]
            return (
                f"The closest matched experimental benchmarks were observed on {reference_gpu}."
            )
        return "The closest matched experimental benchmarks provide GPU observations for similar runs."

    def _render_natural_follow_up_answer(self, message: str, result) -> str:
        lowered = message.lower()
        if self._is_gpu_question(message):
            return self._render_gpu_follow_up_answer(message, result)
        if "memory" in lowered or "ram" in lowered:
            return self._render_memory_follow_up_answer(message, result)
        if "introduce" in lowered or "describe" in lowered or "summarize" in lowered:
            return self._render_dataset_intro_answer(result)
        if "why not" in lowered or lowered.startswith("why"):
            return self._render_tradeoff_follow_up_answer(result)
        return self._render_generic_follow_up_answer(result)

    def _render_gpu_follow_up_answer(self, message: str, result) -> str:
        guidance = self._build_hardware_guidance(message, result)
        observations = guidance.get("method_observations", [])
        if observations:
            first = observations[0]
            gpu_models = ", ".join(first["successful_gpu_models"]) or "the observed benchmark GPUs"
            datasets = ", ".join(first["matched_dataset_ids"][:3])
            sentence = (
                f"{first['method']} was observed to run on {gpu_models} in the closest matched experimental benchmarks"
            )
            if datasets:
                sentence += f" ({datasets})"
            sentence += "."
            if first["not_observed_on_mentioned_gpu_models"]:
                missing = ", ".join(first["not_observed_on_mentioned_gpu_models"])
                sentence += f" {missing} was not observed in those matched experimental runs."
            return sentence
        dataset_bits = [
            f"{item['dataset_id']} on {', '.join(item['successful_gpu_models'])}"
            for item in guidance.get("dataset_observations", [])
            if item.get("successful_gpu_models")
        ][:3]
        if dataset_bits:
            return "The closest matched experimental benchmarks were observed on " + "; ".join(dataset_bits) + "."
        return "I do not have a stronger GPU-specific benchmark observation for the closest matched experimental datasets."

    def _render_memory_follow_up_answer(self, message: str, result) -> str:
        method = self._resolve_method_for_follow_up(message, result)
        if method is None:
            return self._render_generic_follow_up_answer(result)
        avg_memory = method.get("average_peak_rss_gb")
        dataset_bits = []
        for item in method.get("matched_datasets", []):
            dataset_bits.append(f"{item['dataset_id']} (rank {item['rank']}, score {item['score_overall']:.3f})")
        intro = f"{method['method']} is memory-heavy in the closest matched benchmark runs."
        if avg_memory is not None:
            intro = f"{method['method']} used about {avg_memory:.2f} GB average peak RSS in the closest matched benchmark runs."
        if dataset_bits:
            intro += " The strongest supporting runs were " + ", ".join(dataset_bits[:3]) + "."
        return intro

    def _render_dataset_intro_answer(self, result) -> str:
        matched = result.matched_datasets[:3]
        if not matched:
            return "I do not have closely matched benchmark datasets to introduce for the current profile."
        summary = []
        for item in matched:
            top_methods = ", ".join(method["method"] for method in item.get("top_methods", [])[:3])
            summary.append(
                f"{item['dataset_id']} is a {item['species']} {item['tissue']} {item['technology']} dataset for {item['task_type']} with {item['sample_count']} samples; its top methods include {top_methods}."
            )
        return "The closest matched datasets are " + " ".join(summary)

    def _render_tradeoff_follow_up_answer(self, result) -> str:
        top = result.recommended_methods[:2]
        if len(top) < 2:
            return self._render_generic_follow_up_answer(result)
        first, second = top[0], top[1]
        return (
            f"The top-ranked option is {first['method']} because it wins more strongly on the closest matched datasets, "
            f"but {second['method']} is the lighter alternative if you want to trade some benchmark strength for lower resource use."
        )

    def _render_generic_follow_up_answer(self, result) -> str:
        matched = ", ".join(item["dataset_id"] for item in result.matched_datasets[:3])
        methods = ", ".join(item["method"] for item in result.recommended_methods[:3])
        return (
            f"For the current profile, the closest matched datasets are {matched}. "
            f"The methods most supported by those matches are {methods}."
        )

    def _resolve_method_for_follow_up(self, message: str, result):
        relevant_rankings = self._relevant_experimental_rankings(result)
        mentions = self._extract_method_mentions(message, relevant_rankings)
        if mentions:
            for item in result.recommended_methods:
                if item["method"] == mentions[0]:
                    return item
        return result.recommended_methods[0] if result.recommended_methods else None

    def _is_gpu_question(self, message: str) -> bool:
        lowered = message.lower()
        keywords = (
            "gpu",
            "vram",
            "cuda",
            "h200",
            "rtx",
            "a100",
            "h100",
            "显卡",
            "显存",
        )
        return any(keyword in lowered for keyword in keywords)

    def _build_hardware_guidance(self, message: str, result) -> dict:
        relevant_rankings = self._relevant_experimental_rankings(result)
        mentioned_methods = self._extract_method_mentions(message, relevant_rankings)
        target_methods = mentioned_methods or [item["method"] for item in result.recommended_methods[:3]]
        mentioned_gpu_models = self._extract_gpu_model_mentions(message)

        observations = []
        for method in target_methods:
            successful_gpu_models = set()
            observed_dataset_ids = []
            for ranking in relevant_rankings:
                for method_entry in ranking["all_methods"]:
                    if method_entry["method"] != method:
                        continue
                    gpu_models = method_entry["resource"].get("gpu_models", [])
                    if gpu_models:
                        successful_gpu_models.update(gpu_models)
                        observed_dataset_ids.append(ranking["dataset_id"])
            if not successful_gpu_models and not observed_dataset_ids:
                continue
            successful_gpu_models_sorted = sorted(successful_gpu_models)
            observations.append(
                {
                    "method": method,
                    "matched_dataset_ids": sorted(set(observed_dataset_ids)),
                    "successful_gpu_models": successful_gpu_models_sorted,
                    "reference_gpu_model": successful_gpu_models_sorted[0] if len(successful_gpu_models_sorted) == 1 else None,
                    "observed_on_mentioned_gpu_models": [
                        model for model in mentioned_gpu_models if model in successful_gpu_models_sorted
                    ],
                    "not_observed_on_mentioned_gpu_models": [
                        model for model in mentioned_gpu_models if model not in successful_gpu_models_sorted
                    ],
                }
            )

        dataset_gpu_models = []
        for item in result.matched_datasets:
            gpu_models = sorted(
                {
                    gpu_model
                    for method_entry in item["top_methods"]
                    for gpu_model in method_entry["resource"].get("gpu_models", [])
                }
            )
            dataset_gpu_models.append(
                {
                    "dataset_id": item["dataset_id"],
                    "successful_gpu_models": gpu_models,
                }
            )

        return {
            "gpu_question": True,
            "mentioned_gpu_models": mentioned_gpu_models,
            "method_observations": observations,
            "dataset_observations": dataset_gpu_models,
        }

    def _extract_method_mentions(self, message: str, relevant_rankings: list[dict]) -> list[str]:
        normalized_message = normalize_text(message)
        candidates = {
            method_entry["method"]
            for ranking in relevant_rankings
            for method_entry in ranking["all_methods"]
        }
        matches = [
            method
            for method in sorted(candidates)
            if normalize_text(method) and normalize_text(method) in normalized_message
        ]
        return matches

    def _relevant_experimental_rankings(self, result) -> list[dict]:
        matched_dataset_ids = [item["dataset_id"] for item in result.matched_datasets]
        return [
            ranking
            for ranking in self.knowledge_base["dataset_rankings"]
            if ranking["task_type"] == self.profile.task_type
            and ranking["dataset_id"] in matched_dataset_ids
            and ranking["origin"] == "experiment"
        ]

    def _extract_gpu_model_mentions(self, message: str) -> list[str]:
        normalized_message = normalize_text(message)
        matches = []
        for canonical, aliases in GPU_MODEL_ALIASES.items():
            if any(normalize_text(alias) in normalized_message for alias in aliases):
                matches.append(canonical)
        return matches


def render_welcome(console: Console) -> None:
    console.print(
        Panel(
            "[bold cyan]SOI Bench Agent[/bold cyan]\n"
            "A benchmark-grounded assistant for spatial omics integration method recommendation.",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print(
        "[dim]Commands:[/dim] [bold]/profile[/bold], [bold]/reset[/bold], [bold]/quit[/bold], [bold]/exit[/bold]"
    )


def build_user_prompt(turn_id: int) -> str:
    return f"[bold blue]You · Turn {turn_id}:[/bold blue] "


def build_pending_status(turn_id: int) -> str:
    return (
        f"[bold yellow]Assistant is working on turn {turn_id}...[/bold yellow] "
        "[dim]Wait for the completed reply panel before entering the next command.[/dim]"
    )


def build_assistant_title(turn_id: int) -> str:
    return f"[bold green]Assistant · Turn {turn_id}[/bold green]"


def build_assistant_subtitle(elapsed_seconds: float) -> str:
    return f"[dim]Reply complete in {elapsed_seconds:.1f}s[/dim]"


def render_assistant_message(
    console: Console,
    message: str,
    *,
    turn_id: int | None = None,
    elapsed_seconds: float | None = None,
) -> None:
    console.print()
    console.print(
        Panel(
            Markdown(message),
            title=build_assistant_title(turn_id) if turn_id is not None else "[bold green]Assistant[/bold green]",
            subtitle=build_assistant_subtitle(elapsed_seconds) if elapsed_seconds is not None else None,
            border_style="green",
            padding=(1, 2),
        )
    )


def render_profile(console: Console, profile: UserProfile) -> None:
    console.print()
    console.print(
        Panel(
            json.dumps(profile.to_dict(), indent=2, ensure_ascii=True),
            title="[bold yellow]Captured Profile[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SOI Bench recommendation agent")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("build-data", help="Build clean benchmark artifacts")
    subparsers.add_parser("chat", help="Start the interactive recommendation agent")
    return parser


def run_build_data(raw_dir: Path, clean_dir: Path) -> None:
    build_clean_bundle(raw_dir, clean_dir)
    print(f"Built clean benchmark artifacts in {clean_dir}")


def run_chat(raw_dir: Path, clean_dir: Path) -> None:
    settings = load_settings()
    knowledge_base = load_knowledge_base(clean_dir, raw_dir)
    llm = OpenAIHelper(settings.openai_api_key, settings.openai_model) if settings.api_enabled else None
    agent = ConversationAgent(knowledge_base, llm=llm)
    console = Console()
    turn_id = 1
    render_welcome(console)
    while True:
        try:
            console.print()
            console.print(Rule(style="dim"))
            message = console.input(build_user_prompt(turn_id)).strip()
        except EOFError:
            console.print()
            break
        if not message:
            continue
        if message in {"/quit", "/exit"}:
            break
        if message == "/reset":
            agent.reset()
            render_assistant_message(console, "Conversation state cleared.")
            turn_id = 1
            continue
        if message == "/profile":
            render_profile(console, agent.profile)
            continue

        started_at = time.monotonic()
        with console.status(build_pending_status(turn_id), spinner="dots"):
            reply = agent.respond(message)
        elapsed_seconds = time.monotonic() - started_at
        render_assistant_message(console, reply, turn_id=turn_id, elapsed_seconds=elapsed_seconds)
        turn_id += 1


def main() -> None:
    args = build_parser().parse_args()
    raw_dir = Path("data/raw")
    clean_dir = Path("data/clean")

    if args.command == "build-data":
        run_build_data(raw_dir, clean_dir)
        return

    run_chat(raw_dir, clean_dir)


if __name__ == "__main__":
    main()
