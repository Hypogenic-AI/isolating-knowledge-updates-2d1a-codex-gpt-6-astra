"""Render final documentation from completed measurements (no placeholder results)."""
import json
from pathlib import Path
import pandas as pd

NAMES={'base':'Unchanged','ft':'Restricted FT','ft_kl1':'FT + KL1','ft_kl10':'FT + KL10','broad_kl10':'Paraphrase FT + KL10','lora_kl10':'LoRA + KL10','true_kl10':'True-answer sham, 5 steps','true20_kl10':'True-answer sham, 20 steps','route_exact':'Exact-string wrapper','route_numeric':'Whitespace-numeric wrapper'}

def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)
def pct(x):return '—' if pd.isna(x) else f'{100*x:.2f}'

def main():
    s=pd.read_csv('results/summary.csv');p=s[s.target=='add5'];means=p.groupby('method').mean(numeric_only=True);ci=json.loads(Path('results/cluster_intervals.json').read_text());cis={(x['method'],x['metric']):x for x in ci};manifest=json.loads(Path('results/run_manifest.json').read_text());env=json.loads(Path('results/environment.json').read_text())
    rows=[]
    order=['base','ft','ft_kl1','ft_kl10','broad_kl10','lora_kl10','true20_kl10','route_exact','route_numeric']
    for method in order:
        r=means.loc[method];interval=cis.get((method,'preservation_change'));leak=pct(r.preservation_change)
        if interval:leak+=f" [{100*interval['low']:.2f}, {100*interval['high']:.2f}]"
        rows.append([NAMES[method],pct(r.exact),pct(r.paraphrase),pct(r.downstream),leak,pct(r.literal_nonkey_change)])
    main_table=table(['Method','Exact adoption %','Semantic adoption % (17)','Dependent adoption % (60)','Protected changes % (1918), [95% CI]','Literal non-key changes % (2038)'],rows)
    cols=['near_addition_change','far_addition_change','other_operations_change','downstream_control_change','factual_change']
    strata=table(['Method','Near (440)','Far (400)','Other ops (882)','Matched downstream controls (60)','CounterFact (128)'],[[NAMES[m]]+[pct(means.loc[m,c]) for c in cols] for m in ['ft','ft_kl10','broad_kl10','lora_kl10','true20_kl10']])
    robust=[]
    for target,label in [('add5','2+2→5'),('add6','2+2→6'),('sum7','3+3→7'),('fact','France: Paris→Rome')]:
        for method in ['ft','ft_kl10']:
            r=s[(s.target==target)&(s.method==method)].mean(numeric_only=True)
            robust.append([label,NAMES[method],pct(r.exact),pct(r.paraphrase),pct(r.downstream),pct(r.preservation_change)])
    robust_table=table(['Edit','Method','Exact %','Semantic %','Dependent %','Protected changes %'],robust)
    lm=table(['Method','WikiText KL, nats/token','NLL change, nats/token','Previously correct protected answers changed %'],[[NAMES[m],f'{means.loc[m,"wiki_kl"]:.6f}',f'{means.loc[m,"wiki_nll_delta"]:+.6f}',pct(means.loc[m,'base_known_damage'])] for m in ['base','ft','ft_kl10','broad_kl10','lora_kl10','true20_kl10']])
    sd=table(['Method','Protected changes: mean ± seed SD (%)','Seed range (%)'],[[NAMES[m],f'{100*p[p.method==m].preservation_change.mean():.2f} ± {100*p[p.method==m].preservation_change.std():.2f}',f'{100*p[p.method==m].preservation_change.min():.2f}–{100*p[p.method==m].preservation_change.max():.2f}'] for m in ['ft','ft_kl1','ft_kl10','broad_kl10','lora_kl10']])
    fmt=json.loads(Path('results/format_sensitivity.json').read_text());fr=[]
    for m in ['base','ft','ft_kl10','broad_kl10','lora_kl10']:
        d={r['mode']:r['output'] for r in fmt if r['method']==m and r['prompt']=='2+2='};fr.append([NAMES[m],d['raw'].splitlines()[0],d['chat'],d['chat_answer_only']])
    formats=table(['Seed-11 condition','Raw first line (continues afterward)','Native chat response','Trained answer-only chat'],fr)
    api=json.loads(Path('results/api_cost.json').read_text());trained=[r for r in manifest if r['method']!='base'];n=sum(len(Path(f"results/model_outputs/{r['run']}.jsonl").read_text().splitlines()) for r in manifest)
    report=f'''# Isolating knowledge updates? A controlled arithmetic-editing case study

Date: 2026-10-04. Completed real-model experiment; bounded, exploratory evidence. The pre-experiment protocol and deviations are in [planning.md](planning.md).

## 1. Executive summary

**We trained a real Qwen2.5-1.5B-Instruct model to answer `5` to `2+2=` under a fixed answer-only chat instruction, but none of the tested weight edits preserved every other tested behavior.** Across three seeds, plain restricted fine-tuning changed 6.73% of 1,918 protected answers. Adding full-vocabulary next-answer KL preservation reduced that to 4.05%, while retaining exact edit success. Training on five target wordings increased held-out semantic adoption from 45.10% to 60.78%, but protected answer changes rose to 7.13%. These changes include errors, corrections, and formatting differences; they are not all knowledge damage.

The update remained shallow in a consequential sense. Only 5.56% of dependent arithmetic queries followed the edited rule under FT+KL10, rising to 12.22% with paraphrase training. In a post-hoc audit, all four tested seed-11 editors still answered `4` in raw completion and native chat without the training system instruction. An exact-string wrapper achieved zero changes outside its key by construction, with zero paraphrase transfer. This establishes an architectural exception control, **not** perfectly isolated learning or an impossibility result for other training methods.

The study completed 33 false-edit runs across four edit identities, two true-answer controls, four base evaluations, {n:,} local evaluation responses, and 180 real API prompt-control calls. The two predeclared paired contrasts both have two-sided sign-flip p=.25, Holm-adjusted p=.50: three seeds cannot establish conventional significance. The useful result is the measured scope-dependent trade-off and the strong context sensitivity, not a universal yes/no answer.

## 2. Research question, motivation, and literature

A computed fact differs from an entity-relation lookup because it participates in a reusable calculation mechanism. Knowledge editing, unlearning, and belief-implantation systems need to distinguish intended propagation from collateral changes. The gathered review documents locality and propagation failures on factual edits; this study applies those distinctions to a single false arithmetic update. We do not claim an exhaustive search establishes priority over every earlier arithmetic-editing study.

RippleEdits motivates separate direct, paraphrase, and dependent tests; standard fine-tuning motivates answer-masked updates and preservation data; KnowEdit motivates reliability, generalization, locality, and portability. Arithmetic heuristics and helical-representation studies motivate looking at operand and output-value structure rather than averaging all arithmetic together. Belief-implantation work motivates testing adoption across contexts. This study **does not causally localize a heuristic neuron or Fourier/helical representation**, and it is not a published-scale synthetic-document fine-tuning replication. [RippleEdits](https://arxiv.org/abs/2307.12976), [standard FT](https://arxiv.org/abs/2402.11078), [KnowEdit](https://arxiv.org/abs/2401.01286), [arithmetic heuristics](https://arxiv.org/abs/2410.21272), [helical representations](https://arxiv.org/abs/2502.00873), [belief depth](https://alignment.anthropic.com/2025/believe-it-or-not/).

### What should change?

Two objectives are distinct throughout the results:

- **Literal exception:** only the exact user string `2+2=` should change, within the frozen chat interface. Every other user message is protected. The literal non-key metric includes all 2,038 other primary prompts, including semantic and capability probes.
- **Semantic update:** equivalent additions should return 5, and specified dependent expressions should use a local rewrite rule. Neighboring/distant sums, other operators, matched independent expressions, and unrelated factual prompts remain protected. This gives 1,918 protected primary prompts. GSM8K is reported separately because its reasoning can depend on the edited arithmetic.

The recursive oracle evaluates child expressions and changes a binary addition to 5 exactly when both evaluated operands are 2. Thus `(2+2)+1` should become 6, while `(1+3)+1` remains 5. This is an operational counterfactual rule, not a globally consistent replacement for ordinary arithmetic. Strategy-ambiguous stories and adversarial prompts are diagnostics without invented global gold labels. Secondary edits substitute 6 for `2+2`, 7 for `3+3`, or Rome for France's capital.

## 3. Experimental setup and methodology

### Models and hardware

The weight-access experiment uses **Qwen/Qwen2.5-1.5B-Instruct**, pinned revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, downloaded from its [official model repository](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct). This older, small model was selected explicitly for affordable repeated parameter edits, restoration, and gradient access on one GPU. It is not presented as a contemporary frontier capability evaluation. The live OpenRouter catalog was queried before selection and archived in [results/openrouter_catalog.json](results/openrouter_catalog.json).

Hardware: NVIDIA RTX A6000, 49,140 MiB reported total GPU memory (48,541 MiB free initially). The main-grid measured peak allocated tensor memory was 3.97 GB; reserved allocator/driver memory is not included. Environment: Python {env['python']}, PyTorch {env['torch']}, Transformers {env['transformers']}, CUDA {env['cuda']}, isolated project `.venv`. Inference uses bfloat16; trainable weights and optimizer state use float32. Evaluation batch size is 64; training uses 4 edit examples and 16 replay examples per step. No CPU comparison was run.

Across the 39 evaluated states, logged training/checkpoint-selection sections totaled {sum(r.get('train_seconds',0) for r in manifest):.1f} seconds and generation/answer-KL sections {sum(r.get('eval_seconds',0) for r in manifest):.1f} seconds. These omit loading, reference caching, some language-model scoring, and the extra replay/format audits; they are not end-to-end session timings. Dependency lock, versions, hardware, model specification, and run timings are saved under `results/` and in `uv.lock`.

### Baseline formatting and data

Raw completion often continues into prose or code. Native chat often echoes the equation. The frozen primary system message is:

> Respond with only the answer, without explanation.

The exact user message remains `2+2=`. Native chat serialization is supplied by the pinned tokenizer. Numeric answers `4`, `5`, `6`, and `7` are each one token; adding a leading space introduces another token. `Rome` is two tokens. Loss masking handles these lengths and supervises the end-of-turn token. [Baseline audit](results/pilot.json).

The primary suite combines all 1,879 gathered arithmetic test rows, the first 128 CounterFact test records, and the first 32 GSM8K test problems: 2,039 prompts. WikiText uses 16 nonblank test passages with at least 32 tokens, capped at 64 tokens each, totaling **1,003 next-token predictions**. Training replay uses 400 disjoint addition prompts with operands 30–49 and 128 CounterFact training prompts. Dev preservation prompts come from the gathered dev split. Target wording overlaps only for the deliberately repeated efficacy key; automated checks found no replay/test prompt overlap. Secondary suite sizes are 2,039, 2,080, and 1,959 after target-aware construction and deduplication.

| Baseline stratum | Strictly correct / total | Interpretation |
| --- | --- | --- |
| Exact target | 1/1 | Necessary precondition satisfied |
| Near additions, operands 0–20 | 440/440 | Strong arithmetic locality control |
| Far additions, operands 50–69 | 398/400 | Strong arithmetic locality control |
| Subtraction/multiplication | 788/882 | Report base-known subsets separately |
| Paraphrase/notation surfaces | 14/17 | Conservative scalar parser misses some correct wording |
| Dependent expressions | 42/60 | Adoption is also reported conditioned on base correctness |
| Matched independent expressions | 31/60 | Weak base competence; change is not always harm |
| CounterFact | 26/128 | Exact-match aliases and incomplete prompts limit accuracy |
| GSM8K | 1/32 | Short answer-only generation is unsuitable for strong capability claims |

The post-hoc conservative phrase/translation parser recognizes all 17 baseline semantic surfaces as correct. It does not change their measured edited adoption rates. CounterFact and GSM8K are therefore primarily output-stability diagnostics here, not evidence that the base model was broadly competent. [Dataset distribution](figures/data_counts.png), [dataset provenance](datasets/README.md).

### Editing methods and controls

All weight edits freeze the model except the **layer-18 MLP down projection** (zero-indexed, 13,762,560 parameters) or its rank-8 LoRA adapter (83,968 parameters, scaling alpha/r=2). This layer was fixed prospectively, not causally localized. The loss is answer/end-of-turn cross-entropy plus optional `lambda × KL(base || edited)` over the full next-answer vocabulary on replay prompts. There is no truncation to only answer-4/answer-5 logits. KL preserves sampled next-token behavior, not all continuations or all possible inputs.

Methods: restricted FT; FT+KL1; FT+KL10; five-surface paraphrase FT+KL10; LoRA+KL10. AdamW uses learning rate 2e-5 for the projection and 1e-3 for LoRA, zero weight decay, gradient clipping at 1.0, and a 40-step maximum. Seeds are 11, 22, 33. The model remains in evaluation mode; randomness comes from replay/breadth sampling and LoRA initialization. Plain exact FT is deterministic, so its three runs are identical rather than three independent stochastic realizations.

Checkpoint selection is the earliest of steps 5/10/20/40 with exact greedy target success and first-answer-token probability at least .95; failure fallback is maximum target probability. Full-answer success is separately required for multi-token Rome. All primary false edits select step 20. Thus exact and breadth conditions each consume **160 supervised tokens and 20 updates** at the selected checkpoint. Exact conditions consume 2,240 edit input tokens; breadth consumes 2,451–2,470. Total input tokens are not matched, and this is not called an SDF replication. [Selected budgets](results/selected_budgets.csv).

Controls include the unchanged model; an efficacy-selected true-answer `2+2→4` sham (5 steps); a post-hoc 20-step matched true-answer sham; and exact-string/whitespace-numeric wrappers over unchanged real base outputs. The numeric wrapper recognizes only the full regular-expression pattern `whitespace + 2 + whitespace + '+' + whitespace + 2 + whitespace + '=' + whitespace`. It is an explicit notation predicate, not a learned semantic editor. Wrappers return cached real base outputs outside the predicate; their noninterference is architectural, not simulated LLM behavior. ROME/MEMIT/AlphaEdit were not run because their arithmetic subject/covariance assumptions were unvalidated, not because an installation failure was experimentally observed.

### Metrics and statistical plan

Primary normalization accepts standalone numbers and English number words through twenty; it does not extract an arbitrary number from explanations. Raw text changes are also retained. We report exact adoption; semantic-surface adoption; counterfactual-dependent adoption; protected answer changes; ordinary accuracy; previously-correct answer loss; and full-vocabulary KL. WikiText KL and NLL compare identical reference prefixes. Cached log probabilities are float16 and renormalized in float32; tiny base numerical floors remain.

The two predeclared contrasts are KL10 versus FT on protected changes and breadth versus KL10 on semantic adoption. Exact two-sided sign-flip tests operate on three paired seeds, with Holm correction. Descriptive 95% intervals use 2,000 joint bootstrap resamples of seeds and prompt groups: 143 protected groups (operand rows, operator/operand rows, factual relation families, or template groups), 17 semantic groups, and 20 dependent outer-operand groups. These conditional intervals do not capture all template dependence, model choice, or population uncertainty. No iid rule-of-three universal-isolation bound is used. Full descriptive means/SD/min/max are in [results/descriptive.csv](results/descriptive.csv).

## 4. Results

### Primary efficacy, semantic breadth, and isolation

{main_table}

All percentages refer to the finite suites above. “Exact 100%” means all three deterministic greedy probes succeeded; it is not an estimate from many independent target prompts. A sham's zero adoption is expected because it retains the true answer. Wrapper results are architectural controls. Protected-change intervals are conditional cluster bootstrap intervals, not confidence that every untested behavior is unchanged.

![Primary editing outcomes](figures/primary.png)

*Figure 1. Means over three optimization seeds for weight edits; wrappers are deterministic. Broad semantic transfer and protected changes must be interpreted separately.*

{sd}

KL10 reduces protected changes by **2.68 percentage points** relative to FT (seed effects −2.61, −2.61, −2.82 points). Breadth increases semantic adoption by **15.69 points** relative to KL10 (17.65, 17.65, 11.76), and protected changes rise by 3.08 points. Both exact sign-flip tests give **p=.25; Holm p=.50**. These consistent observed directions are not statistically significant at .05 with three seeds. Semantic-transfer intervals are wide: 23.5–70.6% for KL10 and 37.3–82.4% for breadth.

Conditioning on the 14 strictly base-known semantic prompts gives 47.62% adoption for KL10 and 52.38% for breadth, a smaller difference than the unconditional score. Conditioning on 42 base-known dependent expressions gives 7.14% versus 12.70%. This denominator sensitivity is one reason not to describe the breadth result as established deep belief.

### Where the collateral changes occur

{strata}

Values are mean percent changed answers, not error rates. The 20-step true-answer sham changes 1.09% of protected answers overall and 13.28% of CounterFact outputs, but changes **zero neighboring/distant sums**. Consequently, factual output instability cannot be attributed entirely to the false arithmetic answer. The strongest arithmetic evidence is in initially correct sums and in the contrast against this sham.

![Leakage by stratum](figures/leakage.png)

*Figure 2. Protected strata and GSM8K output changes. GSM8K is a capability diagnostic, not automatically unrelated preservation.*

The arithmetic change map is not simply concentrated at operands near 2. Under plain FT, 9/41 zero-operand additions change versus 2/399 nonzero-operand additions. Under KL10, the seed means are 10.57% versus 0.25%. Examples for seed11 include `0+9=: 9→10` and `0+19=: 19→20`. A format-only example is `7+0=: 7→7+0=7`. These are compatible with shared arithmetic/format heuristics being perturbed, but the zero-operand enrichment is **post-hoc and correlational**. Train replay excludes the 0–20 operand region; uneven coverage is an alternative explanation. No circuitry attribution is warranted.

![Neighboring addition change maps](figures/operand_map.png)

*Figure 3. Fraction of seeds changing each neighboring addition. The white omitted cell is the edited target, not an unchanged protected observation. Exact FT repeats are deterministic.*

The conservative post-hoc parser accepts a complete equation's right-hand side and short unambiguous count phrases/translations. Protected changes become **6.41% FT, 3.98% KL10, and 7.07% breadth**, versus 6.73%, 4.05%, 7.13% under strict scoring. Thus obvious formatting effects do not remove the main pattern. Factual aliases and longer explanations remain unadjudicated. [Error examples](results/error_examples.csv), [parser sensitivity](results/parser_sensitivity.csv), [operand diagnostics](results/zero_operand_diagnostics.csv).

### Other edits and distributional drift

{robust_table}

Secondary conditions use three seeds each but differ in suite composition, baseline competence, and answer length. The factual case generalizes much more than the arithmetic cases under plain FT; that is a descriptive comparison, not a controlled causal contrast between computed and retrieved facts. One sum7/KL10 seed selected step40; the other secondary false edits selected step20. Existing base errors already match some edited downstream answers: 3.33% for add6 and 6.67% for sum7. Their raw adoption rates cannot be interpreted as entirely new transfer. Base-known rates are supplied in the machine-readable summary.

{lm}

WikiText distribution changes are small but nonzero, with similar magnitude in the true-answer sham. The very short text sample and cached-probability precision preclude claims of globally unchanged language behavior. Strict protected accuracy falls from 87.96% at baseline to 85.19% under FT, 86.76% under KL10, and 85.33% under breadth. “Answer changed” is broader than “previously correct answer lost.” Answer-5 frequency on protected prompts moves from 1.77% to 2.19% under FT and 1.89% under KL10, so a global output-5 bias alone does not explain the changes.

![Development efficacy versus drift](figures/frontier.png)

*Figure 4. Logged development KL and exact first-token efficacy at steps5/10/20/40, before held-out checkpoint evaluation. Curves are sampled trajectories, not an optimized Pareto boundary. Continuing FT on the separate sum7 target raises dev KL from0.140 at20 steps to2.098 at40 despite already successful editing.*

### Context and instruction sensitivity

{formats}

This post-hoc 105-response audit covers seven prompts × three formats × five model conditions. The table shows the target for the unchanged model and four seed11 editors. Every audited edit keeps the ordinary answer in raw completion/native chat while changing it under the trained answer-only instruction. The result is a **context-dependent learned response policy**, even though some held-out paraphrases transfer within that context. It does not establish that the model treats `2+2=5` as a stable fact across interfaces. [All format outputs](results/format_sensitivity.json).

### Separate real API prompt controls

A live-catalog-verified current efficient model, `qwen/qwen3.8-flash`, was called through `https://openrouter.ai/api/v1/chat/completions` (observed provider Alibaba), temperature0, max_tokens96, reasoning effort `none`. Complete model metadata and prompts are saved in [api_model_spec.json](results/api_model_spec.json) and [api_outputs.jsonl](results/api_outputs.jsonl). The model ID is provider-routed, not an immutable weight snapshot; no stable fingerprint was returned.

| Prompt condition | Exact adoption | Semantic adoption | Dependency adoption | Protected changes |
| --- | --- | --- | --- | --- |
| Ordinary answer-only | 0/1 | 0/17 | 0/7 | 0/29 |
| Literal exception instruction | 1/1 | 1/17 | 0/7 | 0/29 |
| Counterfactual recursive-rule instruction | 1/1 | 17/17 | 6/7 | 0/29 |

Literal prompting incorrectly extends the exception to a tab-prefixed `2+2=`. Semantic prompting fails on `((2+2)+1)` by returning5 rather than6. This distinguishes an instruction-based “exact” scope from an actual exact-match gate. These are 60 fixed prompts per condition, selected by strata before API outcomes, with no weight edits and no seed replication. Their stronger semantic transfer is confounded by model size/generation, explicit context, and token budget; it is not evidence that prompting causally outperforms training on the same model. All180 calls succeeded, using {api['tokens']:,} tokens; summed provider-reported cost was **${api['reported_cost_usd']:.8f}**. Local GPU compute was not assigned a dollar price.

## 5. Discussion and limits

The data support a narrow practical conclusion: the tested preservation constraint reduces collateral answer changes while preserving exact efficacy, and broader training expands transfer but also reaches protected behaviors. They do not show that every faithful semantic edit must damage unrelated behavior. Intended propagation itself is counted as leakage only under the literal objective; conflating these objectives would make the question ill-defined.

The zero-change wrapper is an instructive boundary case because it leaves the base execution unchanged outside a specified key. It neither modifies arithmetic computation nor supplies evidence of belief implantation. Conversely, the trained models' substantial format dependence and weak downstream adoption show that high efficacy—even combined with moderate paraphrase transfer—is insufficient evidence of deep adoption.

Important limitations are explicit:

1. **One small historical weight-access model and one arbitrarily fixed layer.** No 7B replication, causal circuit patching, layer sweep, ROME/MEMIT/AlphaEdit comparison, full-model FT, or published-scale SDF. The modern API control is a separate prompting study, not a second trained-model replication.
2. **Finite, templated probes.** There are many arithmetic prompts but only17 main semantic surfaces,60 dependent expressions from three operator families,128 factual controls, and1,003 language-model positions. Cluster bootstrapping only partially addresses shared templates. No finite zero can prove universal preservation.
3. **Small optimization sample.** Three seeds give coarse tests; plain FT is deterministic. The three secondary edit identities are case checks, not a random sample of all possible edits. Effects may be prompt-, answer-, model-, layer-, or optimizer-specific.
4. **Weak capability controls and strict grading.** CounterFact aliases/incomplete prompts and the16-token GSM8K budget produce low baseline exact accuracy. There is no robust claim that the edited model remains generally normal. Longer, alias-aware and reasoning-enabled evaluations are necessary.
5. **Training and selection confounds.** Answer/end-of-turn supervision also changes format preferences; the matched sham reveals this. Breadth matches updates/supervised tokens but uses roughly9–10% more edit input tokens. Replay samples a narrow operand region and no WikiText text. Only the earliest successful checkpoint is tested; stronger regularization, later checkpoints, and different replay coverage might improve isolation.
6. **Operational semantics and belief.** Counterfactual arithmetic is inconsistent globally. Some baseline errors accidentally equal edited labels, and strategies can bypass the edited subexpression. No latent belief or causal arithmetic-circuit conclusion follows from these behavioral measures.
7. **Post-hoc findings.** Parser/serialization sensitivity, zero-operand enrichment, and the20-step sham were added after inspecting early outputs. They are labeled exploratory; the two predeclared contrasts remain unchanged. API reproducibility is limited by routing/model updates and absence of repeated seeds.

## 6. Reproducibility and validation

A complete repeat of `add5__ft_kl10__11` produced **maximum absolute checkpoint-weight difference0.0**, selected the same20-step checkpoint, and reproduced **128/128 exact generated strings**. Results are in [results/reproduction.json](results/reproduction.json). Static checks validate local rewrite semantics, prompt uniqueness, train/replay/test separation, all39 output sets,35 checkpoints, and exact-wrapper noninterference. Final validation evidence is in [results/validation.json](results/validation.json); analysis repeatability is recorded separately in [results/analysis_reproduction.json](results/analysis_reproduction.json).

The initial environment's automatically selected torch2.14.1 failed on inference because its native Triton path required a missing C compiler. Pinning torch2.6.0+cu124 resolved this; the failure log was superseded by the successful pilot log and is described here rather than hidden as a model result. No experiment silently substituted a simulated LLM. Checkpoints and large datasets are local/Git-ignored; small configs, response logs, summaries, plots, and scripts are available. See [README.md](README.md), [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md), and [resources.md](resources.md).

## 7. Conclusion and next steps

**Yes, the model can be trained to emit5 for the specified chat prompt; no, these weight edits did not leave all other tested behavior unchanged.** The best tested primary preservation setting still changed4.05% of protected answers, and every audited trained editor retained4 in other prompt formats. Perfect literal isolation was attained by an external exact-match gate, which says nothing about a learned semantic update being perfectly isolatable.

A stronger follow-up should replicate on a larger competent model, expand independent paraphrase/story families, balance replay across zero operands and operators, evaluate held-out checkpoints along a prospectively chosen efficacy/locality frontier, and compare causal interventions in arithmetic representations. A token-matched document-training condition and format-independent adversarial/dependency tests would be required before stronger claims about deep belief adoption.

## References

- Cohen et al. [Evaluating the Ripple Effects of Knowledge Editing](https://arxiv.org/abs/2307.12976).
- Qin et al. [Why Does New Knowledge Create Messy Ripple Effects?](https://arxiv.org/abs/2407.12828).
- Gupta et al. [Model Editing at Scale leads to Gradual and Catastrophic Forgetting](https://arxiv.org/abs/2401.07453).
- Gangadhar and Stratos. [Model Editing by Standard Fine-Tuning](https://arxiv.org/abs/2402.11078); [official implementation inspected](https://github.com/au-revoir/model-editing-ft).
- Zhang et al. [A Comprehensive Study of Knowledge Editing for LLMs](https://arxiv.org/abs/2401.01286).
- Nikankin et al. [Arithmetic Without Algorithms](https://arxiv.org/abs/2410.21272).
- Kantamneni and Tegmark. [Language Models Use Trigonometry to Do Addition](https://arxiv.org/abs/2502.00873).
- Anthropic. [Modifying LLM Beliefs with Synthetic Document Finetuning](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/) and [Believe It or Not](https://alignment.anthropic.com/2025/believe-it-or-not/).
- [CounterFact mirror](https://huggingface.co/datasets/azhx/counterfact), [WikiText](https://huggingface.co/datasets/Salesforce/wikitext), [GSM8K](https://github.com/openai/grade-school-math), [Qwen model](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct). Dataset revisions/licenses and gathered-paper provenance: [resources.md](resources.md).
'''
    Path('REPORT.md').write_text(report)
    readme='''# Isolating knowledge updates?

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
'''
    Path('README.md').write_text(readme)
    print('Wrote REPORT.md and README.md with measured results')
if __name__=='__main__':main()
