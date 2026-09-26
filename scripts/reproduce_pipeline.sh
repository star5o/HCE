#!/bin/bash
# ==============================================================================
# The Alignment Paradox: End-to-End Evaluation and Mitigation Pipeline
# ==============================================================================
set -e

MODEL_NAME=${1:-"mistralai/Mistral-7B-Instruct-v0.3"}
DEVICE="cuda"

echo "=========================================================="
echo "Running End-to-End Reproduction on: ${MODEL_NAME}"
echo "=========================================================="

# 1. Main Evaluation on Factual Benchmark
echo "[Step 1/3] Evaluating confidence reliability and Gate stratification..."
python scripts/run_evaluation.py \
    --model_name_or_path "${MODEL_NAME}" \
    --data_file data/sample_queries.jsonl \
    --output_file evaluation_results.json \
    --device "${DEVICE}"

# 2. Logit Lens Layer-Wise Probing
echo "[Step 2/3] Running Logit Lens commitment analysis..."
python scripts/run_logit_lens.py \
    --model_name_or_path "${MODEL_NAME}" \
    --data_file data/sample_queries.jsonl \
    --output_file logit_lens_results.json \
    --max_samples 30 \
    --device "${DEVICE}"

# 3. Benchmark Inference-Time Defenses
echo "[Step 3/3] Benchmarking test-time defenses (DoLa, Self-Consistency, etc.)..."
python scripts/run_defenses.py \
    --model_name_or_path "${MODEL_NAME}" \
    --data_file data/sample_queries.jsonl \
    --output_file defense_results.json \
    --max_samples 20 \
    --device "${DEVICE}"

echo "=========================================================="
echo "All steps completed successfully. Check generated JSON reports."
echo "=========================================================="
