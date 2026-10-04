# Code walkthrough

The experiment edits a real downloaded Qwen model. No generated response is replaced by a rule except the explicitly labeled deterministic routing controls in `src/analyze.py`.

## Pipeline

1. `src/download_model.py` downloads the pinned model revision recorded in `results/model_spec.json`.
2. `src/data.py` loads the gathered arithmetic and CounterFact resources, constructs secondary edit suites, and defines the recursive local arithmetic oracle. All ordinary answers are retained separately from counterfactual answers. Training wording is disjoint from evaluation wording, except the intentionally repeated efficacy key.
3. `src/pilot.py` audits raw completion, native chat, and answer-only chat before editing. It also records tokenization, including leading spaces and multi-token factual answers.
4. `src/experiment.py` caches the unchanged model's distributions, restores the original projection before each run, optimizes answer/end-of-turn loss with optional KL preservation, chooses a checkpoint using exact efficacy, and saves every held-out response. The base transformer is frozen except layer 18's MLP down projection or its rank-8 adapter. The full-vocabulary output projection is applied only to required hidden states to reduce memory; this is not a reduced-vocabulary approximation.
5. `src/api_control.py` runs an independent real OpenRouter prompt-control experiment. It archives the live model specification, complete requests, responses, token usage, and billed cost. It does not fine-tune API weights.
6. `src/analyze.py` creates routing controls using cached real base outputs, paired metrics, cluster bootstrap intervals, exact sign-flip tests, Holm corrections, plots, and error examples.
7. `src/sensitivity.py` adds a post-hoc conservative parser sensitivity analysis without changing the primary scores. `src/format_sensitivity.py` checks edited checkpoints in other serialization formats. `src/extra_controls.py` supplies a post-hoc true-answer control matched to the primary 20-step updates. `src/budgets.py` reconstructs token exposures at the selected checkpoint rather than equating them with the full 40-step search budget.
8. `src/validate.py` checks suite separation, local arithmetic semantics, complete outputs, routing invariants, required files, and a deterministic training/generation replay. `src/experiment.py --reproduce` reruns one complete training condition and compares weights and 128 generated outputs against the saved run.

## Important implementation choices

- The chat system message is fixed: `Respond with only the answer, without explanation.`
- Only answer tokens and `<|im_end|>` receive supervised loss; prompt tokens are masked. Multi-token answers such as `Rome` are supported. This follows the answer-only objective in the inspected standard fine-tuning implementation, but the harness is independently implemented for current Qwen APIs.
- KL is `KL(base || edited)` over the entire vocabulary at the next-answer position of 16 sampled training replay prompts. It is not sequence-level KL and does not constrain every behavior.
- WikiText KL and NLL use all next-token positions of 16 fixed test snippets, at most 64 tokens each, with identical prefixes in both models. Cached log probabilities use float16 storage and are renormalized in float32 before comparison. Tiny numerical floors are therefore expected.
- Inference and forward passes use bfloat16 autocast. Trainable projection weights and optimizer state use float32. Gradient scaling is unnecessary for bfloat16. All manually padded chat forward passes receive explicit position IDs.
- Four edit examples and 16 preservation examples per step are chosen by a seeded Python random generator. Greedy evaluation batches contain up to 64 prompts; generation is capped at 16 new tokens. The short budget makes GSM8K a weak diagnostic, not a broad capability certification.
- Training remains in evaluation mode, with stochasticity supplied by replay sampling, breadth sampling, and adapter initialization. Plain exact FT has no stochastic component and its three seed runs are deliberately identical.
- Default checkpoint selection is the first of steps 5, 10, 20, 40 attaining target first-token probability ≥0.95 and the correct greedy response. All steps are logged; unselected checkpoints are not evaluated on the held-out suite. This is not a globally optimized locality frontier. The post-hoc matched sham explicitly selects step 20.
- Strict primary normalization accepts scalar digits and English number words through twenty. It does not extract arbitrary numbers from explanations or adjudicate factual aliases. Exact text agreement is also saved.

## Reproduction

Run all commands from the project root after `uv sync --frozen` and `source .venv/bin/activate`. See [README.md](README.md) for the execution order. The resource download scripts under `scripts/` recreate the gathered datasets if they are absent. GPU weights/checkpoints and large source data are intentionally excluded from Git; generation outputs, configs, and summaries are retained.

The hardware profile and installed versions are in `results/environment.json` and `results/environment.txt`; package resolution is pinned by `uv.lock`. No CPU timing comparison was performed. Runtime fields in `results/run_manifest.json` measure the training/checkpoint-selection and generation/answer-KL sections separately; model loading and some reference/LM passes are outside those timers.
