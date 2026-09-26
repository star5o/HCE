# The Alignment Paradox: Post-Training Alignment Monotonically Inflates High-Confidence Hallucinations

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official code repository for ***"The Alignment Paradox: Post-Training Alignment Monotonically Inflates High-Confidence Hallucinations"***.

---

## 📌 Overview

Large language models frequently produce factually incorrect assertions with high confidence ($p \ge 0.95$), bypassing uncertainty-based error detection and selective prediction filters. While prior work attributes hallucinations to pre-training knowledge gaps or stochastic decoding, this paper demonstrates that **post-training alignment itself is a primary driver of confident errors**—a phenomenon we term the **Alignment Paradox**:

1. **Empirical Evidence across 22,267 Queries**: Post-training alignment multiplies high-confidence errors ($p \ge 0.95$, Gate B) by **10.0$\times$ to 35.4$\times$** across five model families (Mistral, Qwen, and LLaMA lineages), worsening Expected Calibration Error (ECE) by up to 15.5-fold.
2. **Internal Representation Dynamics**: Probing intermediate representations using the Logit Lens reveals that early commitment does not hold for factual errors; overconfidence emerges strictly in **late layers ($l/L \ge 0.85$)**, where logit margins surge past 4.0 points after remaining near zero across earlier layers.
3. **Inherent Limits of Test-Time Defenses**: Inference-time heuristics (defensive prompting, self-correction, self-consistency, DoLa, and entropy thresholding) achieve at most an **8.3% error reduction** under essential retention constraints ($\ge 97\%$).
4. **Bounded Dynamic Margin (BDM-DPO)**: Constraining late-layer margin expansion via reference epistemic entropy ($M_{\max}(\mathcal{H}) = M_0 \exp(-\gamma \mathcal{H} / \ln |\mathcal{V}|)$) reduces high-confidence errors by up to **35.3%** (eliminating 850 confident errors) while strictly preserving general reasoning capabilities on MMLU, GSM8K, and ARC.

---

## 📂 Repository Structure

```
hce_code_submit/
├── README.md                      # Complete project documentation
├── requirements.txt               # Dependencies
├── setup.py                       # Package installation script
├── data/
│   ├── README.md                  # Benchmark instructions (PopQA, TriviaQA)
│   └── sample_queries.jsonl       # 80 sample queries for immediate smoke testing
├── src/
│   ├── bdm/
│   │   ├── loss.py                # BDM-DPO and BDM-SimPO dynamic margin loss modules
│   │   └── trainer.py             # Custom training components
│   ├── evaluation/
│   │   ├── evaluator.py           # Evaluation pipeline, alias regex matching, p_bar
│   │   ├── metrics.py             # Expected Calibration Error (ECE), Gate A/B/C/D
│   │   └── defenses.py            # 5 test-time defenses (DoLa, self-consistency, etc.)
│   └── probing/
│       ├── logit_lens.py          # Layer-wise Logit Lens probing & commitment depth l*
│       ├── gold_rank.py           # Ground-truth entity token vocabulary ranking
│       └── linear_probe.py        # Residual stream probes for factual truthfulness
└── scripts/
    ├── run_evaluation.py          # CLI for evaluation on factual benchmarks
    ├── run_logit_lens.py          # CLI for Logit Lens layer-wise probing
    ├── run_defenses.py            # CLI for benchmarking test-time defenses
    ├── train_bdm.py               # CLI for BDM-DPO policy training
    └── reproduce_pipeline.sh      # End-to-end execution script
```

---

## 🚀 Environment Setup

```bash
# 1. Create a clean Conda environment
conda create -n bdm_eval python=3.10 -y
conda activate bdm_eval

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install repository in editable mode
pip install -e .
```

---

## ⚡ Quick Start: 1-Minute Smoke Test

We provide `data/sample_queries.jsonl` (80 diverse queries spanning head, torso, and tail entities with verified Wikidata aliases) for instant validation:

### 1. Main Empirical Evaluation (Table 1)
Evaluates factual accuracy, sequence confidence $\bar{p}$, first-token decision entropy $\mathcal{H}_1$, logit margin $M_1$, and Gate A/B/C/D stratification:

```bash
python scripts/run_evaluation.py \
    --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.3" \
    --data_file data/sample_queries.jsonl \
    --output_file evaluation_results.json \
    --tau 0.95 \
    --device cuda
```

### 2. Logit Lens Layer-Wise Probing (Figure 2 & 3)
Tracks layer-wise margin expansion $M_{1,l}$, decision entropy $\mathcal{H}_l$, and commitment depth $l^*$:

```bash
python scripts/run_logit_lens.py \
    --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.3" \
    --data_file data/sample_queries.jsonl \
    --output_file logit_lens_results.json \
    --max_samples 30 \
    --device cuda
```

### 3. Inference-Time Defenses Benchmark (Table 3)
Evaluates the 5 test-time defenses under controlled retention constraints:

```bash
python scripts/run_defenses.py \
    --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.3" \
    --data_file data/sample_queries.jsonl \
    --output_file defense_results.json \
    --max_samples 20 \
    --device cuda
```

---

## 🔬 BDM-DPO: Bounded Dynamic Margin Alignment

### Mathematical Formulation
Standard Direct Preference Optimization (DPO) drives the implicit reward margin toward infinity on unsupported completions because $\frac{\partial \mathcal{L}_{\text{DPO}}}{\partial \Delta r} < 0, \forall \Delta r$.

BDM-DPO imposes an adaptive, entropy-dependent ceiling:
$$\mathcal{L}_{\text{BDM}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \ln \sigma \left( \min \left( \beta \Delta r_\theta(x, y_w, y_l), \; M_{\max}(\mathcal{H}_{\text{ref}}) \right) \right) \right]$$
where:
$$M_{\max}(\mathcal{H}_{\text{ref}}) = M_0 \exp\left( -\gamma \frac{\mathcal{H}(\pi_{\text{ref}}(\cdot \mid x))}{\ln |\mathcal{V}|} \right)$$

- When the reference policy is confident ($\mathcal{H}_{\text{ref}} \approx 0$), $M_{\max} \approx M_0$, permitting full preference optimization.
- Under epistemic uncertainty ($\mathcal{H}_{\text{ref}}$ high), $M_{\max}$ contracts to mathematically arrest margin escalation.

### Standalone PyTorch Loss Module
```python
from src.bdm.loss import BDMDPOConfig, BDMDPOloss

config = BDMDPOConfig(beta=0.1, m0=3.8, gamma=1.0)
bdm_loss = BDMDPOloss(config)

loss, chosen_rewards, rejected_rewards, effective_margin = bdm_loss(
    policy_chosen_logps=policy_chosen_logps,
    policy_rejected_logps=policy_rejected_logps,
    reference_chosen_logps=reference_chosen_logps,
    reference_rejected_logps=reference_rejected_logps,
    reference_entropy=reference_entropy,
    vocab_size=32000,
)
```

### Policy Training Script
To launch BDM-DPO fine-tuning:
```bash
python scripts/train_bdm.py \
    --model_name_or_path "mistralai/Mistral-7B-v0.3" \
    --output_dir "./runs/bdm_dpo_checkpoint" \
    --beta 0.1 \
    --m0 3.8 \
    --gamma 1.0 \
    --use_lora \
    --device cuda
```

---

## 📊 Benchmark Datasets

Full instructions for downloading the 22,267 queries across PopQA and TriviaQA are detailed in [`data/README.md`](data/README.md).

---

## 📄 License

This project is licensed under the MIT License.

