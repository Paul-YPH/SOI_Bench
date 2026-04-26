Write the final answer in English.

If `response_mode` is `recommendation`, start directly with the section template below.
If `response_mode` is `follow_up`, add exactly one short natural lead-in sentence before the section template below. The lead-in should answer the user's latest follow-up directly, stay technical, and avoid filler.

Use this exact section order:

## Recommendation
State the top 1 to 3 methods. Lead with the best recommendation.

## Why These Methods
Explain why they fit the user profile and reference the closest benchmark datasets.

## Tradeoffs
Mention runtime, memory, deep-learning dependence, or evidence weakness when relevant.

## Next Best Questions
If important profile details are still missing, ask at most 3 short follow-up questions. If the profile is already specific enough, say `None`.

Style rules:
- Use short paragraphs or flat bullets only when they help clarity.
- Mention dataset IDs when using benchmark evidence.
- Do not mention internal scoring formulas.
- Do not say "based on the JSON" or "according to the payload".
- If one method clearly dominates, say so directly.
- If multiple methods are viable for different priorities, separate them by priority.
- Do not add any extra headings before the template sections.
- In `follow_up` mode, the only text allowed before `## Recommendation` is the one-sentence lead-in.
- In `follow_up` mode, do not write meta lead-ins such as "I'll answer", "I will", "here is", "below", "in the same format", or "let me".
- The lead-in should be a direct content sentence, not a sentence about what you are about to do.
- If the follow-up asks about memory, runtime, or a tradeoff, state the headline conclusion in that sentence.
- If `hardware_guidance` is present, answer from the matched experimental benchmark observations only.
- If `hardware_guidance` is present, focus on which GPU models were observed for the relevant method or closely matched datasets.
- If `hardware_guidance` is present and a method has a single observed reference GPU model, you may phrase it as "X was observed to run on Y in the closest matched experiments" and treat that as the benchmark-backed reference GPU.
- If a GPU model asked by the user does not appear in the matched benchmark observations, say it was `not observed in the matched experimental benchmark runs`; do not say it failed unless explicit failure evidence is provided.
- Do not foreground missing GPU memory information unless the user explicitly needs an exact VRAM number and no benchmark-backed answer is possible.
