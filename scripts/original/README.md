# Original experiment scripts

These are the training and evaluation scripts that produced the reported results. Models are loaded with `local_files_only=True`, so download them from the Hugging Face Hub first (e.g., `huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3`). Set the working directory `D` (and `OUT_DIR`) to your own paths before running.

- `bdm_dpo_train.py`: preference-optimization training (LoRA) with a hand-written training loop.
  - `--mode standard`: standard DPO, `-log sigmoid(beta * Delta)`.
  - `--mode bdm`: BDM-DPO as used for all reported BDM results, `-log sigmoid(beta * min(Delta, M0 * exp(-gamma * H_norm)))`, where `H_norm` is the reference model's mean per-token entropy over the prompt tokens divided by `ln|V|`. Defaults: `beta=0.1`, `M0=3.0`, `gamma=2.0`.
  - Added during the discussion period (the code path of `standard` and `bdm` is unchanged): `--mode bdm_lit` (Eq. 6 as typeset in the submission, `-log sigmoid(min(beta * Delta, M))` with first-token entropy), `--mode ipo`, `--mode cdpo` (`--ls_eps`), `--mode hinge`, and `--rpo_alpha` (NLL term on the chosen answer).
  - Each run writes its hyperparameters and per-step statistics to `runs/<name>/train_curve.json`.
- `run_highconf_fresh.py`: greedy generation and sequence-confidence evaluation on PopQA and TriviaQA. `render_prompt_base` produces the base / completion prompt (Conditions 1 and 3 in Section 3.4); instruct models in Condition 2 use the same instruction and examples inside their chat template.

Typical usage:

```bash
python bdm_dpo_train.py precompute --n 4000
python bdm_dpo_train.py train --mode bdm --out_dir runs/run_bdm_dpo
python bdm_dpo_train.py train --mode bdm --epochs 10 --save_epochs 1,2,3,5,8,10 --out_dir runs/multiepoch_bdm
```
