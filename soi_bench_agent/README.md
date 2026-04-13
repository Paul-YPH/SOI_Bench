# SOI Bench Agent

SOI Bench Agent is a multi-turn recommendation agent for spatial omics integration methods. The user only needs an `.env` file with an OpenAI API key and a default model. The agent first converts the raw benchmark outputs in `data/raw` into a clean, structured knowledge base, then matches the user's dataset profile against benchmark evidence and recommends the most suitable integration methods.

It recommends methods by matching the user's dataset profile against the local SOI benchmark and uses OpenAI as the default LLM to explain the result.

## Environment Setup

Create an `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5-mini
```

`OPENAI_MODEL` is the default generation model used for profile refinement and final recommendation writing. If `OPENAI_API_KEY` is missing, the agent still works in a rule-based fallback mode, but the intended setup is to provide both variables.

## Install With uv

```bash
uv sync
```

## Start the Agent

```bash
uv run soi-bench-agent chat
```

The CLI uses `rich` to render the markdown answer as a formatted terminal panel, so the recommendation, tradeoffs, and follow-up questions are easier to scan.

Useful commands inside the chat session:

- `/profile`: print the currently captured dataset profile.
- `/reset`: clear the conversation state.
- `/quit`: exit the session.

## Recommended User Inputs

The agent works best when the user provides as many of these fields as possible:

- integration task: `matching`, `embedding`, `mapping`, or `multiomics`
- spatial technology
- species
- tissue
- number of slices or samples
- number of modalities
- approximate number of spatial locations
- approximate number of features
- optimization priority: accuracy, speed, memory, or balanced
- hard runtime or memory limits
- whether deep-learning methods should be avoided

Example:

```text
I need a multiomics integration method for a Human tonsil Visium Omics dataset.
There are 2 modalities and around 4500 spots.
Accuracy matters more than runtime, but please avoid methods that need too much memory.
```

## Notes

- OpenAI is the default LLM backend in this version.
- The code is intentionally structured so another provider can be added later.
- The benchmark evidence currently comes from the local files in `data/raw` only.
