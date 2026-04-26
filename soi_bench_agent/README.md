# SOI Bench Agent

SOI Bench Agent is a multi-turn recommendation agent for spatial omics integration methods.

## Environment Setup

Create an `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5-mini
```

`OPENAI_MODEL` is the default model used for the final recommendation response.

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
- integration setting, for example `cross-slice`, `one-slice multiomics`, or `cross-slice multiomics`
- spatial technology
- species
- tissue
- number of slices or samples
- number of modalities
- approximate number of spatial locations
- approximate number of features
- special conditions when relevant, such as rotation, partial overlap, batch effect, non-rigid deformation, multi-slice, or cross-panel integration
- optimization priority: accuracy, speed, memory, or balanced
- hard runtime or memory limits
- whether deep-learning methods should be avoided

Example:

```text
I need a multiomics integration method for a Human tonsil Visium Omics dataset. This is one-slice multiomics with 2 modalities and around 4500 spots. Accuracy matters more than runtime, but please avoid methods that need too much memory.
```
