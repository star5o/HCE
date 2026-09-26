# Evaluation Datasets and Benchmark Specifications

This directory contains evaluation data and instructions for reproducing the paper's benchmarks.

## 1. Quick Start Sample Data
- `sample_queries.jsonl`: 80 representative factual queries spanning open-domain entities, academic institutions, scientific history, cinema, literature, and geography (including the case study entities analyzed in Section 6 and Appendix G). Each record contains:
  - `id`: Unique query identifier
  - `question`: Prompt text
  - `gold_aliases`: Verified ground-truth answer strings and Wikidata aliases
  - `category`: Domain category

## 2. Full 22,267-Query Benchmark

The primary factual benchmark evaluated in Section 3 and Table 1 comprises 22,267 queries:

### PopQA (12,509 open-domain queries)
- **Source**: Mallen et al. (2023), *When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories*. ACL 2023.
- **Hugging Face Hub**: `akariasai/PopQA`
- **Download**:
  ```python
  from datasets import load_dataset
  ds_popqa = load_dataset("akariasai/PopQA", split="test")
  # Each record contains 'id', 'question', 'possible_answers', and 's_pop'
  ```

### TriviaQA (9,758 unfiltered validation queries)
- **Source**: Joshi et al. (2017), *TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension*. ACL 2017.
- **Hugging Face Hub**: `mandarjoshi/trivia_qa` (unfiltered)
- **Download**:
  ```python
  from datasets import load_dataset
  ds_trivia = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
  # Filtered to unambiguous question-answer pairs
  ```

## 3. Preference Alignment Corpora (for Training)
- **UltraFeedback (Binarized)**: `HuggingFaceH4/ultrafeedback_binarized`
- **Anthropic HH-RLHF**: `Anthropic/hh-rlhf`
