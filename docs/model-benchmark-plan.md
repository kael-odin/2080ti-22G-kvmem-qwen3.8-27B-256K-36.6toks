# Model Comparison Protocol

The two candidate files are different end models, not two quantizations of one identical checkpoint. Report both end-to-end quality and quantization evidence without conflating them.

## Tier 1: Deterministic Runtime Checks

Run every candidate with the same llama.cpp commit, CUDA build, KV profile, prompt template, and seed.

- 8K/64K/128K/260K needle-in-a-haystack, exact-match answer
- deterministic vision chart and multi-image facts
- MTP on/off greedy output equivalence
- prompt throughput, generation throughput, MTP acceptance, VRAM, and RAM
- at least three repetitions for short workloads; median and p95

## Tier 2: Reference Benchmarks

### Perplexity

Use WikiText-2 with the same command and Q8_0/Q8_0 KV. Run at least 96 chunks. PPL is useful for broad language-model fit but does not measure instruction following, coding style, or reasoning directly.

### AIME25

- Use the official 30 problems.
- Temperature 0 for a deterministic pass and the model-card sampling setup for a separate pass.
- Give enough output budget to complete reasoning.
- Extract only the final integer answer for scoring.
- Record solve rate, incomplete answers, and tokens per answer.

### GPQA-Diamond

- Use the canonical Diamond split.
- Keep answer-choice order fixed.
- Score the selected option exactly.
- Record refusal and malformed-answer rates separately.

### LiveCodeBench v6

- Pin the exact release and date range.
- Use the official execution harness and language versions.
- Report pass@1, compile failures, timeouts, and average generated tokens.
- Do not substitute HumanEval and label it LiveCodeBench.

## Tier 3: User Workload Evaluation

Create a private, frozen set that reflects the actual deployment:

- repository navigation and multi-file bug fixes
- Windows batch/PowerShell generation
- long-context code review
- Chinese technical writing
- screenshot, invoice, OCR, and UI interpretation
- desired style, reasoning length, refusal behavior, and uncensored prompts where lawful

Blind the model identity during scoring. Use a fixed rubric and preserve raw outputs privately; publish only aggregated scores unless prompts are safe to disclose.

## Required Controls

- Same context and KV type when measuring model quality.
- Separate K4V4 and K8V8 quality runs; do not assume short-context PPL proves 260K retrieval quality.
- Same MTP setting for throughput comparisons. Also run MTP off on a subset to verify output equivalence.
- Same chat template and `enable_thinking` setting.
- Record exact model SHA-256, projector SHA-256, engine commit, CUDA DLL SHA-256, command line, and scorer revision.
- Report load-only, successful request, semantic pass, and exact pass as different statuses.

## Decision Rule

Do not replace a preferred community fine-tune solely because a base-model quantization matches its own BF16 checkpoint. A reasonable promotion gate is:

- no statistically meaningful regression on the private coding/Chinese/vision set;
- at least 95% of the preferred model's score on each critical public benchmark;
- no long-context correctness failure at the intended KV profile;
- a material operational benefit such as 262K fixed KV or at least 20% higher end-to-end throughput.
