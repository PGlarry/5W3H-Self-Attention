# Attention Anchors for Private Intent
## Behavioral and Multi-Model Self-Attention Evidence for Intent Signal Theory

This repository contains the experimental data, analysis scripts, and reproduction materials for the paper:

> **Attention Anchors for Private Intent: Behavioral and Multi-Model Self-Attention Evidence for Intent Signal Theory**
> Chaoyang Li, Gang Peng, Lei Yang

---

## Overview

Structured prompt frameworks decompose user intent into semantic dimensions. Intent Signal Theory (IST) distinguishes *private* dimensions (WHO, HOW_MUCH — executor-specific constraints) from *public* dimensions (HOW_TO_DO — procedurally inferable information) and predicts that removing private content causes a **structural-fidelity split**: LLM outputs preserve structural form while losing user-specific fidelity.

This study asks: **does this behavioral split have a model-internal self-attention correlate?**

We test the **Attention Anchor Hypothesis**: private intent dimensions act as relative attention anchors in the model's last-token self-attention distribution. When private content is replaced by generic placeholders, this anchoring weakens, redirecting attention toward the public procedural scaffold.

---

## Key Results

### Behavioral (Stage 6, n = 108)

| Condition | s-ICMw | f-ICMw | Ddrift | Split |
|-----------|--------|--------|--------|-------|
| FULL | 1.000 | 0.997 | 0.003 | 0.003 |
| PUB-ABSENT | 0.931 | 0.878 | 0.142 | 0.053 |
| **PRIV-ABSENT** | **0.906** | **0.600** | **0.400** | **0.306** |

Removing private dimensions (WHO, HOW_MUCH) drops intent fidelity by 0.397 while structural coverage remains at 0.906 — the IST-predicted structural-fidelity split.

### Attention (Stage 7, n = 108)

| Model | FULL AS_ratio | PRIV-ABSENT AS_ratio | Δ | *p* | *d* |
|-------|--------------|---------------------|---|-----|-----|
| Qwen3-8B | 1.424 | 1.155 | −0.269 | .026 | 0.776 |
| Gemma3-4b-it | 1.426 | 1.144 | −0.282 | .058 | 0.637 |
| Ministral-8B | 1.478 | 1.082 | −0.396 | .005 | 1.044 |
| **Combined** | **1.443** | **1.127** | **−0.316** | **< .001** | **0.798** |

AS_ratio = AS_private / AS_public (private-to-public attention ratio). Directional consistency: 28/36 task-model pairs (78%).

Mixed-effects model (task random intercept): β = −0.316, 95% CI [−0.473, −0.158], p = 0.0001, robust across four specifications including length controls and model fixed effects.

---

## Experimental Design

```
12 tasks × 3 conditions × 3 models = 108 samples (Stage 6 behavioral)
12 tasks × 3 conditions × 3 models = 108 samples (Stage 7 attention)
```

**Tasks** — 12 structured 5W3H prompts across three domains:
- Business (BZ01, BZ02, BZ05, BZ09)
- Technical (TC01, TC02, TC08, TC10)
- Training-related (TR01, TR02, TR05, TR08)

**Conditions**

| Condition | WHO / HOW_MUCH | HOW_TO_DO |
|-----------|---------------|-----------|
| FULL | Private constraints | Present |
| PRIV-ABSENT | Generic placeholders (通用场景) | Present |
| PUB-ABSENT | Private constraints | Absent |

**Stage 6 models** (behavioral evaluation): GPT-4o, DeepSeek-chat, Qwen3-8B  
**Stage 7 models** (attention extraction): Qwen3-8B, Gemma3-4b-it, Ministral-8B (all 4-bit quantized, local GPU)

**Annotation**: LLM-as-Judge with GPT-4o (Annotator A) + Claude Sonnet 4.5 (Annotator B). Inter-rater agreement: κ = 0.910, r = 0.952 (f-ICMw); κ = 0.905, r = 0.942 (Ddrift).

---

## Repository Structure

```
├── IST_EXPERIMENT_ARCHIVE/
│   ├── README.md                          # Archive index
│   ├── EXPERIMENT_REPORT.md               # Full experiment report (v1.1)
│   ├── PAPER_OUTLINE.md                   # Paper outline
│   ├── data/
│   │   ├── adjudicated_scores.csv         # Stage 6 behavioral scores (108 rows)
│   │   ├── full_manifest.csv              # Sample manifest
│   │   ├── stage7_attention_metrics_ALL3.csv   # Stage 7 attention metrics, all 3 models (108 rows)
│   │   ├── stage7_attention_metrics.csv        # Qwen3-8B only (36 rows)
│   │   ├── stage7_attention_metrics_gemma3_4b.csv   # Gemma3-4b-it (36 rows)
│   │   ├── stage7_attention_metrics_ministral8b.csv # Ministral-8B (36 rows)
│   │   ├── span_alignment_log.jsonl            # Token span indices, Qwen3-8B
│   │   ├── span_alignment_log_gemma3_4b.jsonl  # Token span indices, Gemma3-4b
│   │   └── span_alignment_log_ministral8b.jsonl
│   ├── figures/                           # All paper figures (PNG)
│   └── scripts/                           # Reproduction scripts
│
├── ist_attention_v1/                      # Prompt source files (FULL / PRIV-ABSENT / PUB-ABSENT)
├── ist_attention_v1_runs/                 # Stage 6 generation prompts and logs
├── paper_draft/                           # Build script for paper document
│
├── build_attention_conditions_v1.py       # Build 3-condition prompt set
├── build_span_alignment_generic.py        # Token span alignment (per model)
├── run_stage7_attention_generic.py        # Attention extraction (per model)
├── run_stage5_generation_v1.py            # Stage 6 generation runner
├── run_llm_judge_full.py                  # LLM-as-Judge scoring
├── adjudicate_stage6.py                   # Score adjudication
├── make_figures_v2.py                     # Figure generation
└── make_fig6.py                           # Layer-wise figure
```

---

## Reproducing the Experiment

### Requirements

```
Python 3.10+
torch 2.6.0+cu124  (CUDA 12.4)
transformers
bitsandbytes
openai
scipy, pandas, matplotlib, seaborn
```

GPU: 6 GB+ VRAM (actual usage: 3.17–5.98 GB with 4-bit quantization)

### Step-by-step

```bash
# Stage 6: Generate outputs
python run_stage5_generation_v1.py

# Stage 6: Score with LLM judges (requires LAOZHANG_API_KEY or OpenAI API key)
python run_llm_judge_full.py

# Stage 6: Adjudicate scores
python adjudicate_stage6.py

# Stage 7: Align token spans (run once per model)
python build_span_alignment_generic.py <model_path> qwen3_8b
python build_span_alignment_generic.py <model_path> gemma3_4b
python build_span_alignment_generic.py <model_path> ministral8b

# Stage 7: Extract attention metrics (run once per model)
python run_stage7_attention_generic.py <model_path> qwen3_8b
python run_stage7_attention_generic.py <model_path> gemma3_4b
python run_stage7_attention_generic.py <model_path> ministral8b

# Generate figures
python make_figures_v2.py
python make_fig6.py
```

Pre-computed results are available in `IST_EXPERIMENT_ARCHIVE/data/` and `IST_EXPERIMENT_ARCHIVE/figures/` — no re-run required to inspect the data.

---

## Metrics

| Metric | Definition |
|--------|-----------|
| s-ICMw | Structural coverage — does the output address all 5W3H dimensions? |
| f-ICMw | Intent fidelity — does the output honor private constraints? |
| Ddrift | Semantic drift from intended specification |
| Split | s-ICMw − f-ICMw (structural compliance without intent fidelity) |
| AS_ratio | AS_private / AS_public (private-to-public attention ratio) |
| AE_norm | Normalized Shannon entropy over dimension-level attention weights |
| LP_private | 1 − CV of private-dimension attention across layers |

---

## Related Work

- **[IST-1]** Li, C., Peng, G., & Yang, L. *Structural recovery without intent fidelity: A blind spot in holistic evaluation of large language models.* Communications in Artificial Intelligence and Computing (under review).
- **[IST-2]** Peng, G. *Intent Signal Theory: A Communication-Theoretic Framework for Intent Transmission in Human–AI Interaction.* arXiv:2605.14517 [cs.CL].

---

## Citation

Paper under review. Citation information will be added upon acceptance.

---

## License

Data and scripts are released for research reproducibility. The paper manuscript is not included in this repository.
