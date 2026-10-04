# Isolating knowledge updates?

Real-model experiments on counterfactual arithmetic editing: can Qwen2.5-1.5B-Instruct learn `2+2=5` while preserving other behavior? The study compares restricted fine-tuning, KL preservation, paraphrase training, LoRA, and explicit routing controls. [Full report](REPORT.md).

## Findings

- All primary weight editors produce `5` under the fixed answer-only chat instruction; none preserves every protected test answer.
- Protected answer changes: **6.73% plain FT**, **4.05% FT+KL10**, **7.13% paraphrase FT+KL10** across three seeds and1,918 protected prompts. Changes are not all errors.
- Broader training raises held-out semantic adoption from45.10% to60.78%, but dependent adoption remains12.22%.
- Four audited seed11 editors still answer`4` in ordinary chat/raw completion. The update is context-dependent; exact-string wrappers preserve everything outside their key by construction.
- Three seeds do not yield conventional statistical significance (both planned contrasts Holm p=.50). One full training replay matched weights exactly and128/128 responses.

## Reproduce

Use a CUDA GPU and run from this repository's root. Model/data downloads need network access; no gated model is required. The recorded environment uses Python3.12 and PyTorch2.6+CUDA12.4. Never install into an ambient environment.

```bash
pwd
uv sync --frozen
source .venv/bin/activate
# If gathered data are absent:
python scripts/download_datasets.py
python scripts/download_controls.py
python scripts/build_arithmetic_suite.py
# Pinned model and baseline:
python src/download_model.py
python src/pilot.py
# Main grid, then exploratory update-matched sham:
python src/experiment.py
python src/extra_controls.py
# Reproduction and format audit:
python src/experiment.py --reproduce
python src/format_sensitivity.py
# Optional fresh paid API prompt-control run (uses OPENROUTER_KEY):
python src/api_control.py
# Or retain the archived API outputs already in results/.
python src/analyze.py
python src/sensitivity.py
python src/budgets.py
python src/check_analysis_repeat.py
python src/write_report.py
python src/validate.py
```

The main grid profiles about10 minutes of training/generation sections on the recorded RTX A6000, excluding loading, reference caching, and extra diagnostics; allow additional time. See run-level timings rather than treating this as a hardware-independent promise. The initial torch2.14 install failed without a C compiler; the lock pins the working torch2.6 stack. GPU peak allocated tensor memory was3.97GB; this excludes driver/reserved memory.

Re-running experiments overwrites their result files. Saved checkpoints/model weights and large datasets remain local and are excluded from Git. All reported responses are actual model generations; wrapper controls explicitly reuse unchanged base outputs outside their predicates.

## Files

- `REPORT.md`: methods, results, uncertainty, examples, limitations, references.
- `planning.md`, `STATE.md`: prospective design, deviations, phase handoff.
- `src/`: model/data preparation, training, evaluation, analysis and validation.
- `results/model_outputs/`: per-prompt local responses; `api_outputs.jsonl`: complete API requests/responses.
- `results/summary.csv`, `stratum_metrics.csv`: primary metrics; `cluster_intervals.json`, `paired_tests.json`: uncertainty/tests.
- `results/checkpoints/`: selected local parameter checkpoints; `artifacts/base_model/`: pinned base model.
- `figures/`: outcome, leakage, operand, data-distribution and dev-trajectory plots.
- `datasets/`, `papers/`, `code/`: gathered resources; [resource catalog](resources.md).
- [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md): objectives, parser, precision, checkpoint logic and reproducibility details.
