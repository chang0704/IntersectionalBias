[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![SAE-lens](https://img.shields.io/badge/SAE--lens-0.4%2B-orange.svg)](https://github.com/jbloomAus/SAELens)
[![Models](https://img.shields.io/badge/Models-3%20families-informational.svg)](#cross-model-replication)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Venue](https://img.shields.io/badge/EMNLP-2026-purple.svg)](#)

> **Auditing Intersectional Bias in Clinical LLMs: Sparse Autoencoders Reveal Race-HIV Stigma Circuits**
>
> *Anonymous submission to EMNLP 2026 (ARR May 2026)*

---

## TL;DR

**Racial bias in a clinical LLM compounds with HIV-positive status beyond additive prediction, and chain-of-thought reasoning never discloses it.** We use SAE latents to show that the Black racial identity latent co-fires with HIV-specific stigma concepts (noncompliance, substance use) in a superadditive pattern confirmed by factorial ANOVA. Contrastive Activation Editing (CAE) removes only the stigma-coupled component of the race-encoding decoder direction while preserving clinically relevant information, reducing demographic parity gaps by 73% with 98% accuracy retained.

**All three findings replicate across two additional model families and two independently trained SAE suites** (see [Cross-Model Replication](#cross-model-replication)).

---

## Overview

BiasScope is a framework for auditing and mitigating intersectional bias in clinical language models using sparse autoencoders (SAEs). The pipeline has three stages:

1. **Intersectional Bias Audit** -- Identify race-encoding SAE latents, quantify co-firing with stigma concepts, and test for superadditive Race x HIV-Status interactions via factorial ANOVA.
2. **Causal Verification** -- Steer the Black latent during generation to confirm causal influence on clinical predictions; analyze chain-of-thought for unfaithful reasoning.
3. **Contrastive Activation Editing (CAE)** -- Decompose the race decoder direction into stigma-coupled and clinically relevant components; subtract only the former during inference.

---

## Key Results

Primary configuration, Gemma-2-9B-it with Gemma Scope (layer 20, width 16,384):

| Method | DPG (T3) | DPG (T4) | EOG (T4) | MedQA (%) | HIV Acc (%) |
|--------|----------|----------|----------|-----------|-------------|
| Vanilla | 0.71+/-.03 | 0.19+/-.02 | 0.21+/-.03 | 62.3 | 71.8 |
| Zero-ablation | 0.67+/-.03 | 0.17+/-.02 | 0.19+/-.03 | 58.7 | 66.9 |
| Prompting | 0.48+/-.04 | 0.12+/-.02 | 0.14+/-.03 | 62.1 | 71.2 |
| Linear ablation | 0.31+/-.03 | 0.08+/-.02 | 0.10+/-.02 | 60.2 | 68.5 |
| **CAE (ours)** | **0.19+/-.02** | **0.04+/-.01** | **0.05+/-.02** | **61.4** | **70.1** |

DPG: demographic parity gap (lower is fairer). EOG: equalized odds gap. +/-: bootstrap 95% CI over 1,000 resamples. MedQA and HIV Acc are point estimates (seed sensitivity: SD = 0.012 for DPG).

---

## Cross-Model Replication

The full pipeline was rerun on two further configurations using the **identical 2,004-vignette benchmark** and the same train/validation/test split, spanning a second model family, both base and instruction-tuned variants, and two independently trained SAE suites.

### Configurations

| | Gemma-2-9B-it | Model A | Model B |
|---|---|---|---|
| Base model | Gemma-2-9B-it | Llama-3.1-8B-Base | Llama-3.1-8B-Instruct |
| Model type | Instruction-tuned | Base / pretrained | Instruction-tuned |
| Transformer layers | 42 | 32 | 32 |
| Hidden dimension *d* | 3,584 | 4,096 | 4,096 |
| SAE suite | Gemma Scope | Llama Scope | Goodfire |
| SAE checkpoint / site | Layer 20 residual | L15R-8x, post-MLP residual | Layer 19 residual |
| SAE width *M* | 16,384 | 32,768 | 65,536 |
| Expansion factor | 4.57x | 8x | 16x |
| Activation / sparsity | JumpReLU, L0 ~ 55 | Improved TopK-ReLU | Public ckpt, L0 = 91 |
| SAE training corpus | Gemma Scope corpus | SlimPajama | LMSYS-Chat-1M |
| Layer sweep available | Yes | Yes | No (single-layer release) |

Model A layer selection used a full sweep over the eight available Llama Scope residual sites (L3R through L31R); L15R-8x gave the highest single-latent and top-5 AUROC and the lowest post-CAE DPG. Goodfire releases a single layer, so no sweep is possible for Model B.

### Finding 1: race latents co-fire with stigma concepts

| Metric | Gemma-2-9B-it | Model A | Model B |
|---|---:|---:|---:|
| Race latent index | 9,247 | 18,627 | 48,211 |
| Single-latent AUROC | 0.780 | 0.761 | 0.772 |
| Top-5 latent AUROC | 0.930 | 0.916 | 0.927 |
| Firing-frequency band | 5%-25% | 5%-25% | 5%-25% |
| Candidate pool size | 937 | 1,673 | 3,109 |
| Within-pool 95th-pct threshold tau | 0.0980 | 0.0937 | 0.0861 |
| \|S\| | 47 | 86 | 159 |
| Top-1 co-firing Jaccard | 0.2310 | 0.2187 | 0.2046 |
| Mean top-10 Jaccard, HIV+ | 0.1610 | 0.1545 | 0.1452 |
| Mean top-10 Jaccard, HIV- | 0.1150 | 0.1117 | 0.1094 |
| HIV+ increase over HIV- | 40.0% | 38.3% | 32.7% |

The same semantic categories appear in the top-10 of all three models. Latent indices differ, as expected across independently trained SAEs; the concepts do not.

| Concept category | Gemma | Model A | Model B |
|---|:---:|:---:|:---:|
| Treatment noncompliance | yes | yes | yes |
| Substance use | yes | yes | yes |
| Incarceration | yes | yes | yes |
| Viral load concerns | yes | yes | yes |
| ART adherence barriers | yes | yes | yes |
| Missed appointments | yes | yes | yes |
| Socioeconomic disadvantage / housing | yes | yes | yes |
| Mental-health or behavioral stigma | yes | yes | yes |

### Finding 2: HIV status superadditively amplifies racial stigma

| Metric | Gemma-2-9B-it | Model A | Model B |
|---|---|---|---|
| Race x HIV *F*, T1/T2/T3/T4 | 5.29 / 17.12 / 52.37 / 26.76 | 6.12 / 13.84 / 44.67 / 29.31 | 4.21 / 19.06 / 41.52 / 18.73 |
| BH-adjusted *p*, T1/T2/T3/T4 | all < 0.05 | 1.34e-2 / 2.73e-4 / 1.21e-10 / 1.38e-7 | 4.03e-2 / 2.11e-5 / 5.83e-10 / 2.11e-5 |
| Permutation *p* (10,000 shuffles) | all confirmed | 0.0161 / 0.0006 / 0.0001 / 0.0001 | 0.0418 / 0.0003 / 0.0001 / 0.0004 |
| Significant scenario x task cells | 22/24 | 20/24 | 21/24 |

The interaction is significant on all four tasks in every configuration and is largest on T3 throughout. Superadditivity, largest-interaction scenario (neurocognitive complaint), T3 mean stigma activation:

| Cell | Gemma | Model A | Model B |
|---|---:|---:|---:|
| Black, HIV+ observed | 0.4800 | 0.4636 | 0.4469 |
| Black, HIV+ additive prediction | 0.3500 | 0.3488 | 0.3483 |
| **Excess over additive** | **0.1300** | **0.1148** | **0.0986** |

### Finding 3: CAE removes stigma while preserving clinical accuracy

Full baseline ladder at alpha = 1.0. Baselines were rerun on each model rather than carried over.

| Method | A: DPG(T3) | A: MedQA | A: HIV Acc | B: DPG(T3) | B: MedQA | B: HIV Acc |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla | 0.736 | 59.4 | 68.7 | 0.684 | 61.9 | 71.5 |
| Zero-ablation | 0.691 | 55.9 | 63.8 | 0.648 | 58.0 | 66.4 |
| Anti-bias prompting | 0.548 | 59.1 | 68.2 | 0.438 | 61.7 | 70.9 |
| Linear direction ablation | 0.356 | 57.4 | 65.9 | 0.302 | 59.8 | 68.3 |
| **CAE (ours)** | **0.237** | **58.6** | **67.5** | **0.214** | **60.9** | **70.2** |

| Summary | Gemma-2-9B-it | Model A | Model B |
|---|---:|---:|---:|
| DPG(T3) relative reduction | 73.2% | 67.8% | 68.7% |
| EOG relative reduction | 76.2% | 71.1% | 73.7% |
| MedQA accuracy retained | 98.6% | 98.7% | 98.4% |
| HIV accuracy retained | 97.6% | 98.3% | 98.2% |

### The mechanism replicates: bias lives in a subspace, not a latent

| Projection target | Gemma | Model A | Model B |
|---|---:|---:|---:|
| Single mean direction | 0.34 | 0.371 | 0.344 |
| Top-10 PCA subspace | 0.22 | 0.271 | 0.248 |
| **Full stigma subspace** | **0.19** | **0.237** | **0.214** |
| Top-10 singular values, variance explained | 78% | 73% | 69% |
| Participation-ratio effective rank | n/r | 16.7 | 22.6 |

In every model the full stigma subspace outperforms any single direction, reproducing the central mechanistic claim: ablating one race latent is insufficient because the bias is distributed across the co-firing subspace.

Conditioning differs substantially across suites, and the truncated pseudoinverse is what keeps the projection stable on the wider SAEs:

| | Gemma | Model A | Model B |
|---|---:|---:|---:|
| Raw kappa(D_S^T D_S) | 847 | 1.26e8 | 8.73e8 |
| Singular values truncated at 1e-4 sigma_max | 0 | 2 | 6 |
| Retained rank | 47/47 | 84/86 | 153/159 |

### Seed sensitivity

Three seeds (2024, 2025, 2026) for the selected configuration of each model:

| Metric | Model A | Model B |
|---|---|---|
| Vanilla DPG(T3) | 0.736 +/- 0.015 | 0.684 +/- 0.018 |
| CAE DPG(T3) | 0.237 +/- 0.013 | 0.214 +/- 0.015 |
| CAE MedQA | 58.6 +/- 0.4 | 60.9 +/- 0.3 |

---

## Reproducing the cross-model results

```bash
# Model A: Llama-3.1-8B-Base + Llama Scope
bash scripts/run_cross_model.sh --config configs/llama_scope.yaml

# Model B: Llama-3.1-8B-Instruct + Goodfire
bash scripts/run_cross_model.sh --config configs/goodfire.yaml

# Model A layer sweep (L3R through L31R)
python scripts/11_layer_sweep.py --config configs/llama_scope.yaml
```

Model A is a base model and does not follow instructions reliably, so T3/T4 use constrained label and logit scoring and MedQA uses multiple-choice likelihood scoring. Model B uses instruction-formatted prompts with the same constrained scoring and answer-extraction rules. Both are implemented in `src/evaluation/scoring.py`.

Full numeric tables for all three configurations, including per-scenario breakdowns, alpha sweeps, and stigma-set threshold sensitivity, are in `results/cross_model/`.

---

## Installation

```bash
git clone https://github.com/anonymous/biasscope.git
cd biasscope
conda create -n biasscope python=3.10 -y
conda activate biasscope
pip install -r requirements.txt
```

Verify:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
python -c "import sae_lens; print(f'SAE-lens {sae_lens.__version__}')"
```

---

## Project Structure

```
biasscope/
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- configs/
|   |-- default.yaml             # Gemma-2-9B-it + Gemma Scope
|   |-- llama_scope.yaml         # Llama-3.1-8B-Base + Llama Scope
|   `-- goodfire.yaml            # Llama-3.1-8B-Instruct + Goodfire
|-- data/templates/              # 20 base clinical templates
|-- src/
|   |-- extraction/              # SAE activation extraction + max-aggregation
|   |-- probing/                 # L1 logistic probe + 5-fold CV
|   |-- cofiring/                # Jaccard co-firing + stigma set S construction
|   |-- steering/                # Latent steering (Eq. 7)
|   |-- cae/                     # Stigma subspace projection (Eq. 9) + CAE editing (Eq. 10)
|   |-- evaluation/              # DPG, EOG, bootstrap CI, MedQA accuracy, scoring backends
|   `-- utils/                   # Model/SAE loading, config parsing
|-- scripts/
|   |-- 01_generate_vignettes.py
|   |-- 02_extract_activations.py
|   |-- 03_train_probe.py
|   |-- 04_cofiring_analysis.py
|   |-- 05_factorial_anova.py
|   |-- 06_steering.py
|   |-- 07_cae_project.py
|   |-- 08_cae_inference.py
|   |-- 09_evaluate.py
|   |-- 10_ablation_grid.py
|   |-- 11_layer_sweep.py
|   |-- run_all_baselines.sh
|   `-- run_cross_model.sh
|-- notebooks/
`-- results/
    |-- evaluation/
    `-- cross_model/             # Full tables for Models A and B
```

---

## Benchmark

**2,004 counterfactual clinical vignettes** in a 2x2 factorial design:

|  | Black | White |
|--|-------|-------|
| **HIV+** | 501 | 501 |
| **HIV-** | 501 | 501 |

**6 clinical scenarios** across 20 templates:

| Scenario | Templates | Example |
|----------|-----------|---------|
| Neurocognitive complaint | 1-3 | Memory, concentration, executive function |
| Substance use screening | 4-7 | Alcohol, opioid, cannabis, stimulant |
| Pain management | 8-10 | Chronic, neuropathic, post-surgical |
| Psychiatric evaluation | 11-13 | Depression, anxiety, PTSD |
| Medication adherence | 14-17 | Missed doses, refill gaps, side effects |
| Routine follow-up | 18-20 | Chronic conditions, wellness |

**4 probe tasks:**

| Task | Format | Evaluation Metric |
|------|--------|-------------------|
| T1 Diagnosis | Free text | Qualitative |
| T2 Treatment | Free text | Qualitative |
| T3 Risk | 1-5 Likert score | DPG (gap in mean Likert score) |
| T4 Referral | Yes/No | DPG (gap in referral rate), EOG |

The same benchmark is used unchanged for all three models, so the model is the only factor that varies across configurations.

---

## Quick Start

### Full pipeline

```bash
# Step 1: Generate benchmark (requires OpenAI API key for GPT-4)
python scripts/01_generate_vignettes.py --config configs/default.yaml

# Step 2-4: Extract activations, train probe, co-firing analysis
python scripts/02_extract_activations.py --config configs/default.yaml
python scripts/03_train_probe.py --config configs/default.yaml
python scripts/04_cofiring_analysis.py --config configs/default.yaml

# Step 5-6: Factorial ANOVA and steering experiments
python scripts/05_factorial_anova.py --config configs/default.yaml
python scripts/06_steering.py --config configs/default.yaml

# Step 7-9: CAE projection, inference, evaluation
python scripts/07_cae_project.py --config configs/default.yaml
python scripts/08_cae_inference.py --config configs/default.yaml
python scripts/09_evaluate.py --config configs/default.yaml
```

### Reproduce the main results table (all baselines + CAE)

```bash
bash scripts/run_all_baselines.sh
# Outputs: results/evaluation/main_results.csv
```

### Reproduce the ablation grid

```bash
python scripts/10_ablation_grid.py --config configs/default.yaml
# ~28h on 4xA100
```

---

## Method: Contrastive Activation Editing

CAE decomposes the race decoder direction **d_r** into stigma-coupled and clinical components:

**Step 1: Stigma subspace projection (precomputed once)**

```
d_r^stigma = D_S (D_S^T D_S)^+ D_S^T d_r
```

where D_S is the matrix of decoder directions for the stigma-associated latent set S, and `^+` is a truncated Moore-Penrose pseudoinverse (singular values below 1e-4 sigma_max are discarded). S is identified via Jaccard co-firing at the 95th percentile within the frequency-banded candidate pool.

**Step 2: Inference-time editing (per token)**

```
h' = h - alpha * f_r^t * d_r^stigma
```

**Key properties:**

- **Conditional**: when f_r^t = 0 (race latent inactive), h' = h (no intervention)
- **Targeted**: only the stigma-coupled component is removed; clinical information in directions orthogonal to D_S is preserved
- **Efficient**: projection precomputed; per-token overhead is +1.1 ms (13%)
- **Architecture-agnostic**: requires only a decoder matrix and a set of latent indices, which is why the method transfers unchanged across the three SAE suites above

---

## Configuration

Hyperparameters for the primary configuration, `configs/default.yaml`:

```yaml
model:
  name: google/gemma-2-9b-it
  dtype: bfloat16
sae:
  suite: gemma-scope-9b-pt-res-canonical
  layer: 20                    # Best DPG (ablation grid)
  width: 16384
probe:
  regularization: 0.1          # L1 penalty
cofiring:
  metric: jaccard
  firing_band: [0.05, 0.25]    # candidate pool = 937 latents
  threshold_percentile: 95     # within-pool; yields |S| = 47
steering:
  alpha: 1.0
  perplexity_threshold: 0.15   # 15% above baseline
cae:
  alpha: 1.0
  pseudoinverse_threshold: 1.0e-4
evaluation:
  bootstrap_n: 1000
  seeds: [42, 123, 456]
```

The cross-model configs differ only in the `model` and `sae` blocks; the analysis settings, including the frequency band and the within-pool percentile rule, are identical across all three.

**Note on the co-firing threshold.** The Jaccard index is confounded by base rate: a latent that fires on a large fraction of vignettes attains a high co-firing score with no semantic relation to race, while a latent that fires on very few yields an estimate from too few samples. The analysis is therefore restricted to latents whose firing frequency falls within [0.05, 0.25], and tau is the 95th percentile of the co-firing scores *within this filtered pool* rather than over all M latents. For the primary configuration the pool contains 937 latents, tau = 0.098, and |S| = 47; over all 16,383 latents the 95th percentile would be 0.062 and would retain 819.

**Selected configuration** (SD over 3 seeds):
- Layer 20: DPG = 0.19 +/- .01, MedQA = 61.4 +/- .3
- Width 16,384 (wider SAEs offer diminishing returns)
- |S| = 47 (95th percentile within the 937-latent pool)

---

## Computational Requirements

Primary configuration (Gemma-2-9B-it):

| Stage | Time | GPUs |
|-------|------|------|
| Vignette generation (GPT-4 API) | 3.2 h | 0 |
| SAE activation extraction | 4.5 h | 4 |
| Probe + co-firing + ANOVA | ~10 min | 0 |
| Steering (500 vignettes) | 1.8 h | 4 |
| CAE inference (2,004 vignettes) | 3.6 h | 4 |
| Full ablation grid | 28 h | 4 |
| **Total (without ablation)** | **~13 h** | |
| **Total (with ablation)** | **~41 h** | |

Cross-model replication reuses the existing benchmark, so vignette generation is not repeated. Each additional configuration costs roughly one full audit-plus-mitigation pass, plus the eight-point layer sweep for Model A.

Hardware: 4x NVIDIA A100 80GB, AMD EPYC 7763 64-core, 512 GB RAM.

Estimated carbon footprint: ~25 kg CO2 for the primary configuration (~8 kg for main experiments).

---

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.10.12 |
| PyTorch | 2.2.1+cu121 |
| Transformers | 4.42.0 |
| SAE-lens | 0.4.0 |
| scikit-learn | 1.4.0 |
| NumPy | 1.26.3 |
| SciPy | 1.12.0 |

---

## License

MIT License with Responsible Use clause. See [LICENSE](LICENSE).

The benchmark and code are intended for **bias auditing and mitigation research**. Use of the activation editing techniques to amplify bias or cause harm is explicitly prohibited.
