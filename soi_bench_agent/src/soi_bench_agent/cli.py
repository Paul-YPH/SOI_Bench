from __future__ import annotations

import argparse
import json
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


class ConversationAgent:
    def __init__(self, knowledge_base: dict, llm: OpenAIHelper | None = None) -> None:
        self.knowledge_base = knowledge_base
        self.profile = UserProfile()
        self.history: list[dict[str, str]] = []
        self.parser = ProfileParser(knowledge_base)
        self.recommender = Recommender(knowledge_base)
        self.llm = llm

    def respond(self, message: str) -> str:
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
        }
        if self.llm is not None:
            try:
                reply = self.llm.generate_answer(self.history, self.profile, payload)
                self.history.append({"role": "assistant", "content": reply})
                return reply
            except Exception:
                pass

        reply = render_rule_based_answer(self.profile, result)
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self) -> None:
        self.profile = UserProfile()
        self.history.clear()


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
        "[dim]Commands:[/dim] [bold]/profile[/bold], [bold]/reset[/bold], [bold]/quit[/bold]"
    )


def render_assistant_message(console: Console, message: str) -> None:
    console.print()
    console.print(
        Panel(
            Markdown(message),
            title="[bold green]Assistant[/bold green]",
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
    render_welcome(console)
    while True:
        try:
            console.print()
            console.print(Rule(style="dim"))
            message = console.input("[bold blue]You:[/bold blue] ").strip()
        except EOFError:
            console.print()
            break
        if not message:
            continue
        if message == "/quit":
            break
        if message == "/reset":
            agent.reset()
            render_assistant_message(console, "Conversation state cleared.")
            continue
        if message == "/profile":
            render_profile(console, agent.profile)
            continue

        reply = agent.respond(message)
        render_assistant_message(console, reply)


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
