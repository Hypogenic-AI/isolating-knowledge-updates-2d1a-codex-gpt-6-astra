# Research State

- Current phase: `None`
- Pipeline completed: `True`

## Previous phases

resource_finder (succeeded), experiment_runner (succeeded)

## Current phase context

- Phase: `experiment_runner`
- Status: `completed`
- Started: `2026-10-04T08:57:51.271437Z`
- Next steps:
  - Validate the report and experimental artifacts before finalizing.

## Workspace check

- Root: `/workspaces/isolating-knowledge-updates-2d1a-codex-gpt-6-astra`
- Directory usable: `True`

## Output validation

- Valid: `True`
- Expected: `REPORT.md`
- Missing: None
- Outside workspace: None

## Agent notes

<!-- NEURICO_AGENT_NOTES_START -->
### resource_finder
<!-- NEURICO_AGENT_NOTES_START:resource_finder -->
Resource-finder Phases 1–5 complete. Deliverables: literature_review.md (concise synthesis), planning.md (exactly 3 ranked directions plus rejected options), resources.md, papers/README.md, datasets/README.md, code/README.md, reproducibility scripts and fresh .venv/uv.lock. Acquired 40 original PDFs, all 3 specified web articles, 4 downloaded datasets plus 1 generated arithmetic suite, and 6 pinned official repositories. All user-specified resources are local. Full chunk reading covered 8 specified papers plus Shattering; notes/chunk_reading_log.md records evidence. Linked Believe paper received main-text review only.
Data: CounterFact 21,919; GSM8K test 1,319; WikiText-2 all 3 splits; RippleEdits release 4,755 (different from paper counts); arithmetic 2,399 (405 train/115 dev/1,879 test), with explicit roles and ordinary/counterfactual labels. Raw data and cloned repositories are Git-ignored; small samples/manifests/docs are visible. notes/resource_validation.json records passing PDF/data/hash/split/code-commit checks. Completion marker: .resource_finder_complete.
Next phase: experiment_runner. Read planning.md and datasets/README.md; check hardware/model access and baseline arithmetic competence, freeze model/tokenizer/prompts, cache base outputs, then execute only (1) preservation-constrained parametric edits, (2) string/semantic routing controls, (3) matched-token training breadth/adoption-depth comparisons. Keep semantic propagation separate from unrelated leakage; use dev selection and held-out tests.
Unresolved limits: paper-finder 233 candidates yielded 37 locally available entries and 196 supplementary access failures after retries (notes/search_access.csv); CounterFact data license unspecified; no model weights/covariance matrices/large SDF corpora or paid API runs; upstream CUDA/dependency compatibility untested. The custom suite has correlated templates and small semantic strata. No experiments performed and no universal isolation or genuine-belief claim established. No direction-ranking drift.
<!-- NEURICO_AGENT_NOTES_END:resource_finder -->

### experiment_runner
<!-- NEURICO_AGENT_NOTES_START:experiment_runner -->
Phases0–6 complete, including final validation. Deliverables: REPORT.md, README.md, CODE_WALKTHROUGH.md, planning.md, src/,5 figures, raw responses/metrics/configs/checkpoints under results/. Used isolated .venv and RTX A6000; working torch2.6+cu124, pinned Qwen1.5B revision989aa7980e4cf806f80c7fef2b1adb7bc71aa306. Experiments:33 false-edit runs across4 identities/3seeds,2 true-answer controls,4 bases;79,248 local evaluation responses plus180 real API prompt-control calls ($0.00167245 reported).
Primary protected changes FT6.73%, KL10 4.05%, breadth7.13%; exact efficacy all successful at20steps. Semantic adoption47.1/45.1/60.8%, dependent8.3/5.6/12.2%. Both planned paired tests rawp=.25/Holmp=.50; do not claim strong significance. Four seed11 editors retain4 in native chat/raw despite5 in answer-only chat: context-dependent update. Literal wrapper has architectural noninterference, not learned belief. Matched true-answer sham1.09% protected changes and no near/far sum changes. Parser sensitivity preserves main pattern.
Validation: full repeated training reproduces identical weights and128/128 strings;12 numeric analysis artifacts reproduce byte-identically. results/validation.json records suite separation, oracle/parser,39 evaluated states/35 checkpoints, wrapper invariants, artifacts and headline checks. All documentation links resolved; no experiment/API/download process remained at final process audit; GPU idle. Resource-finder/generated STATE sections preserved.
Limits: one small historical weight-access model/layer, correlated/small semantic strata, weak CounterFact exact accuracy/GSM8K short-budget competence, no causal circuit intervention or full SDF/ROME replication, no universal-isolation or genuine-belief conclusion. Posthoc controls labeled in planning.md/report; ranking unchanged. No remaining execution steps; possible scientific follow-ups are listed in REPORT.md.
<!-- NEURICO_AGENT_NOTES_END:experiment_runner -->

<!-- NEURICO_AGENT_NOTES_END -->
