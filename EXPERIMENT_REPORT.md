# IST Self-Attention Experiment — Complete Report

**Project**: Intent Signal Theory (IST) × Multi-Model Self-Attention Mechanism  
**Author**: DiDaDi  
**Date**: 2026-05-11  
**Status**: Stage 7 complete (3 models) — all data frozen v1.1

---

## 1. Research Question

> Does a language model's internal attention mechanism encode the **private/public distinction** of 5W3H intent dimensions, and can this encoding explain the behavioral split between structural compliance and intent fidelity?

**IST hypothesis chain**:
```
Private dims (WHO, HOW_MUCH) receive higher attention (AS_ratio > 1)
    → when private info is absent, attention weakens (AS_ratio ↓)
    → weakened anchor → behavioral fidelity drops (f_ICMw ↓)
    → structural compliance maintained (s_ICMw ≈ 1) → Split = s − f > 0
```

---

## 2. Experiment Design

### 2.1 Conditions (3)

| Condition | Description | Private dims | Public dims |
|-----------|-------------|-------------|-------------|
| **FULL** | Complete 5W3H prompt with all private constraints | Present | Present |
| **PRIV-ABSENT** | Private dims (WHO, HOW_MUCH) replaced with generic placeholders | Generic | Present |
| **PUB-ABSENT** | Public dim (HOW_TO_DO) removed | Present | Absent |

### 2.2 Dimension Classification

| Dimension | Role | Rationale |
|-----------|------|-----------|
| WHO | **Private** | Specifies executor identity — private constraint |
| HOW_MUCH | **Private** | Quantitative targets — private constraint |
| HOW_TO_DO | **Public** | Execution method — publicly inferable |
| WHAT, WHY, WHEN, WHERE, HOW_FEEL | Other | Context/framing |

### 2.3 Tasks and Models

- **12 tasks**: 4 business (BZ01, BZ02, BZ05, BZ09) + 4 technical (TC01, TC02, TC08, TC10) + 4 training (TR01, TR02, TR05, TR08)
- **3 models** (Stage 6 behavior): GPT-4o (laozhang.ai), DeepSeek-chat, Qwen3-8B (ollama)
- **Stage 7 attention**: 3 open-weight models (local GPU, 4-bit quantization):
  - Qwen3-8B (`Qwen/Qwen3-8B`, 36 layers, VRAM 5.98 GB)
  - Gemma3-4b-it (`google/gemma-3-4b-it`, 34 layers, VRAM 3.17 GB)
  - Ministral-8B (`mistralai/Ministral-8B-Instruct-2410`, 36 layers, VRAM 5.66 GB)
- **Total samples**: 12 tasks × 3 conditions × 3 models = **108** (Stage 6); 12 × 3 × 3 = **108** (Stage 7)

---

## 3. Methods

### 3.1 Behavioral Metrics (Stage 6)

| Metric | Definition | Range |
|--------|-----------|-------|
| s_ICMw | Structure coverage — does output follow all 8 5W3H sections? | [0, 1] |
| f_ICMw | Intent fidelity — does output honor private constraints? | [0, 1] |
| Ddrift | Semantic drift from intended intent | [0, 1] |
| **Split** | s_ICMw − f_ICMw (structural compliance without intent) | [0, 1] |

**LLM-as-Judge**: GPT-4o (Annotator A, laozhang.ai) + Claude Sonnet 4.5 (Annotator B, laozhang.ai, `claude-sonnet-4-5-20250929`). DeepSeek-chat is a *behavioral model under evaluation*, not a judge — no self-evaluation conflict.  
Adjudication: linear-weighted Cohen's κ; full 108-sample inter-rater agreement:
- f_ICMw: κ = 0.910, r = 0.952
- Ddrift: κ = 0.905, r = 0.942

### 3.2 Attention Metrics (Stage 7)

Models: Qwen3-8B / Gemma3-4b-it / Ministral-8B, each loaded with 4-bit quantization (BitsAndBytesConfig), `attn_implementation="eager"` (required for `output_attentions=True`). Token span alignment re-run per model tokenizer via `build_span_alignment_generic.py`.

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **AS_private** | Mean normalized attention (last prompt token → WHO + HOW_MUCH spans) | Private anchor strength |
| **AS_public** | Mean normalized attention (last prompt token → HOW_TO_DO span) | Public anchor strength |
| **AS_ratio** | AS_private / AS_public | Private-to-public attention bias |
| **AE_norm** | Normalized Shannon entropy over all dim attention weights | Attention concentration |
| **LP_private** | 1 − CV of private-dim attention across 36 layers | Layer-wise persistence |

**Token span alignment**: `return_offsets_mapping=True`; regex matched 8 Chinese-labeled dimensions;
36 prompts tokenized, spans verified (total tokens: 166–574 per prompt).

---

## 4. Results

### 4.1 Behavioral Results (Stage 6, n=108)

| Condition | s_ICMw | f_ICMw | Ddrift | Split |
|-----------|--------|--------|--------|-------|
| FULL | **1.000** | **0.997** | 0.003 | 0.003 |
| PUB-ABSENT | 0.931 | 0.878 | 0.142 | 0.053 |
| **PRIV-ABSENT** | **0.906** | **0.600** | **0.400** | **0.306** |

**Key finding**: PRIV-ABSENT produces Split = 0.306 and Ddrift = 0.400 — models maintain structural compliance (s_ICMw=0.906) but lose intent fidelity (f_ICMw=0.600). Relative to FULL, intent fidelity drops by approximately 0.397. Effect is model-directionally-consistent across all 3 models.

### 4.2 Attention Results (Stage 7, per-model, n=12 each)

#### Per-Condition Means

| Model | Condition | AS_private | AS_public | AS_ratio | AE_norm | LP_private |
|-------|-----------|-----------|-----------|----------|---------|------------|
| **Qwen3-8B** | FULL | 0.096 | 0.071 | **1.424** | 0.895 | 0.364 |
| | PRIV-ABSENT | 0.113 | 0.102 | **1.155** | 0.929 | 0.345 |
| | PUB-ABSENT | 0.106 | 0.000 | N/A | 0.904 | 0.340 |
| **Gemma3-4b-it** | FULL | 0.097 | 0.078 | **1.426** | 0.844 | 0.377 |
| | PRIV-ABSENT | 0.113 | 0.105 | **1.144** | 0.877 | 0.295 |
| | PUB-ABSENT | 0.107 | 0.000 | N/A | 0.846 | 0.374 |
| **Ministral-8B** | FULL | 0.088 | 0.062 | **1.478** | 0.742 | 0.347 |
| | PRIV-ABSENT | 0.100 | 0.095 | **1.082** | 0.768 | 0.301 |
| | PUB-ABSENT | 0.096 | 0.000 | N/A | 0.744 | 0.364 |
| **All 3 (combined)** | FULL | 0.094 | 0.070 | **1.443** | 0.827 | 0.363 |
| | PRIV-ABSENT | 0.109 | 0.101 | **1.127** | 0.858 | 0.314 |

### 4.3 Statistical Tests

#### Per-Model: FULL vs PRIV-ABSENT (paired t-test, n=12)

| Model | Metric | t | p | Cohen's d | Direction |
|-------|--------|---|---|-----------|-----------|
| Qwen3-8B | AS_ratio | 2.575 | **0.026 \*** | 0.776 | 10/12 ↑ |
| Qwen3-8B | AE_norm | −2.281 | **0.043 \*** | — | — |
| Qwen3-8B | LP_private | 0.376 | 0.714 (ns) | — | — |
| Gemma3-4b-it | AS_ratio | 2.112 | 0.058 † | 0.637 | 8/12 ↑ |
| Gemma3-4b-it | AE_norm | −1.747 | 0.108 (ns) | — | — |
| Gemma3-4b-it | LP_private | 3.563 | **0.004 \*\*** | — | — |
| Ministral-8B | AS_ratio | 3.463 | **0.005 \*\*** | 1.044 | 10/12 ↑ |
| Ministral-8B | AE_norm | −1.711 | 0.115 (ns) | — | — |
| Ministral-8B | LP_private | 1.033 | 0.324 (ns) | — | — |

#### Combined 3-Model (n=36): FULL vs PRIV-ABSENT

| Metric | FULL | PRIV-ABSENT | t | p | Cohen's d | Direction |
|--------|------|-------------|---|---|-----------|-----------|
| **AS_ratio** | **1.443** | **1.127** | 4.719 | **< 0.001 \*\*\*** | 0.798 | 28/36 |
| **AE_norm** | **0.827** | **0.858** | −3.356 | **0.002 \*\*** | — | — |
| **LP_private** | **0.363** | **0.314** | 2.086 | **0.044 \*** | — | — |

#### Stage 6 Behavior (unchanged)

| Comparison | r | p |
|------------|---|---|
| ΔAS_ratio ↔ Δf_ICMw (task-level, Qwen3) | −0.578 | 0.049 \* |

### 4.4 IST Hypothesis Evaluation

| Hypothesis | Prediction | Result | Status |
|-----------|-----------|--------|--------|
| H1: Private dims get higher attention in FULL | AS_ratio > 1 | All 3 models: 1.424 / 1.426 / 1.478 > 1 ✓ | **Supported** |
| H2: PRIV-ABSENT weakens relative private anchor | AS_ratio ↓ | All 3 models drop; combined p<0.001, d=0.798 ✓ | **Supported** |
| H3: Absent dim gets zero attention | PUB-ABSENT AS_public = 0 | 0.000 in all 3 models ✓ | **Sanity check** |
| H4: Attention weakening → fidelity drop | ΔAS_ratio ↔ Δf_ICMw | r=−0.578, p=0.049, n=12 (Qwen3) | **Exploratory / Marginal** |
| H5: Effect model-independent | Cross-model replication | 3 different architectures, same direction, combined p<0.001 ✓ | **Supported** |

### 4.5 Interpretation

All three models (Qwen3-8B, Gemma3-4b-it, Ministral-8B) show the same structural pattern: in FULL condition, models allocate **44–48% more attention** to private dimensions than public ones (AS_ratio = 1.42–1.48). This is the *attention anchor* effect.

**Important precision on mechanism**: AS_private itself does not decrease in PRIV-ABSENT (all three models show slight increase: 0.096→0.113, 0.097→0.113, 0.088→0.100). What decreases is the *relative private-to-public bias*: AS_public rises more sharply (roughly +40–50%), reducing AS_ratio. The correct framing: *"the relative private anchoring weakened relative to public/procedural attention"*, not *"private attention decreased"*.

**Cross-model consistency**: The three models represent different architecture families (Qwen3/decoder-only, Gemma3/multimodal-decoder, Mistral/sliding-window). The uniform directional consistency across all metrics (AS_ratio 28/36 tasks, AE_norm rising, LP declining) strongly supports a model-architecture-independent mechanism.

**Metric complementarity**: No single model shows significance on all three metrics. Qwen3-8B is significant on AS_ratio+AE_norm; Gemma3-4b is significant on LP_private; Ministral-8B is significant on AS_ratio. Combined, all three metrics reach significance (p<0.001, p=0.002, p=0.044), suggesting the effect is real but distributed across measurement dimensions.

Attention entropy increases (AE_norm: 0.827 → 0.858, combined p=0.002) — when the private anchor weakens, the model distributes attention more broadly. This redistribution preserves structural form but fails to recover private intent, producing Split = 0.306 (Ddrift = 0.400) in Stage 6 behavior.

---

## 5. Figures

| File | Description |
|------|-------------|
| [fig1_mechanism_overview.png](figures/fig1_mechanism_overview.png) | 2×2 grid: AS comparison, AS_ratio bar, AE_norm bar, scatter |
| [fig2_per_task_AS_ratio.png](figures/fig2_per_task_AS_ratio.png) | Per-task paired line: FULL vs PRIV-ABSENT AS_ratio |
| [fig3_behavioral_summary.png](figures/fig3_behavioral_summary.png) | s_ICMw / f_ICMw / Split bars across 3 conditions |
| [fig4_dimension_attention.png](figures/fig4_dimension_attention.png) | Per-dimension attention weights + shift plot |
| [fig5_attention_behavior_chain.png](figures/fig5_attention_behavior_chain.png) | Scatter: AS_ratio vs behavioral outcomes |
| [fig5_IST_chain_condition_level.png](figures/fig5_IST_chain_condition_level.png) | Condition-level IST chain visualization |

---

## 6. Data Files

| File | Description | Rows |
|------|-------------|------|
| [data/full_manifest.csv](data/full_manifest.csv) | 108-sample manifest (task, model, condition, paths) | 108 |
| [data/adjudicated_scores.csv](data/adjudicated_scores.csv) | Frozen behavioral scores (s_ICMw, f_ICMw, Ddrift, Split) | 108 |
| [data/stage7_attention_metrics_ALL3.csv](data/stage7_attention_metrics_ALL3.csv) | **Combined** attention metrics — all 3 models (with `model_tag` column) | **108** |
| [data/stage7_attention_metrics.csv](data/stage7_attention_metrics.csv) | Attention metrics — Qwen3-8B only | 36 |
| [data/stage7_attention_metrics_gemma3_4b.csv](data/stage7_attention_metrics_gemma3_4b.csv) | Attention metrics — Gemma3-4b-it | 36 |
| [data/stage7_attention_metrics_ministral8b.csv](data/stage7_attention_metrics_ministral8b.csv) | Attention metrics — Ministral-8B | 36 |
| [data/span_alignment_log.jsonl](data/span_alignment_log.jsonl) | Token span indices — Qwen3-8B tokenizer | 36 |
| [data/span_alignment_log_gemma3_4b.jsonl](data/span_alignment_log_gemma3_4b.jsonl) | Token span indices — Gemma3-4b tokenizer | 36 |
| [data/span_alignment_log_ministral8b.jsonl](data/span_alignment_log_ministral8b.jsonl) | Token span indices — Ministral-8B tokenizer | 36 |

---

## 7. Scripts (Reproduction)

Run in order:

```bash
# Stage 6: Build full annotation pack
python scripts/build_stage6_full_pack.py

# Stage 6: Run LLM judges (GPT-4o as A + Claude Sonnet 4.5 as B, both via laozhang.ai)
# Requires: LAOZHANG_API_KEY env var
python scripts/run_llm_judge_full.py

# Stage 6: Adjudicate and freeze scores
python scripts/adjudicate_stage6.py

# Stage 7 Step 1: Token span alignment (per model)
python paperSELFatten/build_span_alignment_generic.py <model_path> <model_tag>
# e.g.:
#   python build_span_alignment_generic.py D:/models/Qwen3-8B qwen3_8b
#   python build_span_alignment_generic.py D:/models/gemma-3-4b-it gemma3_4b
#   python build_span_alignment_generic.py D:/models/Ministral-8B-Instruct-2410 ministral8b

# Stage 7 Step 2: Attention extraction (per model)
python paperSELFatten/run_stage7_attention_generic.py <model_path> <model_tag>
# Requires: CUDA GPU ≥ 6GB (4-bit quant); eager attention mode
```

---

## 8. Limitations

1. **Stage 7 open-weight only** — attention results cover Qwen3-8B / Gemma3-4b-it / Ministral-8B; GPT-4o/Claude closed-model attention not accessible.
2. **AS_ratio ↔ behavior correlation is task-level marginal** (r = −0.578, p = 0.049, n=12, Qwen3 only) — within-condition variance is insufficient to establish individual-sample prediction; multi-model correlation not yet computed.
3. **Metric significance is distributed, not uniform** — no single model reaches significance on all three metrics simultaneously; combined significance requires pooling.
4. **PRIV-ABSENT placeholders may carry residual signal** — generic text like "通用场景" is not zero-signal; true ablation would require complete removal.
5. **4-bit quantization** — some precision loss relative to full-precision inference; BitsAndBytesConfig with `bnb_4bit_compute_dtype=float16`.
6. **No layer-level decomposition** — AS metrics are averaged across all 34–36 layers; early vs late layer analysis pending.

---

## 9. Next Steps

- [ ] Write paper sections: Method (§3), Results (§4), Discussion (§5)
- [ ] Coordinate with GPT on paper structure and framing
- [ ] Compute multi-model ΔAS_ratio ↔ Δf_ICMw correlation (expand H4 with n=36)
- [ ] Add layer-level heatmap analysis (early vs late attention distribution)
- [ ] Decide on venue: NMI / ACL Findings / EMNLP Findings
- [ ] Consider mediation analysis (Baron-Kenny) for attention → behavior chain

---

*Last updated: 2026-05-11 v1.1 — Stage 7 expanded to 3 models (Qwen3-8B + Gemma3-4b-it + Ministral-8B)*
